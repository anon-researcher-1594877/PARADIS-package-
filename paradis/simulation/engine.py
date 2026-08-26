"""Dispersal simulation engine.

The simulation alternates between:

1. **Dispersal step** (:func:`dispersal_step`) – move density across the
   landscape using sliding windows.
2. **Growth step** – logistic population growth at each occupied pixel.

An optional stochastic component (:mod:`~paradis.simulation.stochastic`) handles
long-distance dispersal and sub-threshold individuals.

Key functions
-------------
:func:`dispersal_step`
    Apply one dispersal step to the full landscape using a sliding-window
    approach.
:func:`run_simulation`
    Full time-stepping loop (dispersal + growth).

Classes
-------
:class:`PopulationSimulator`
    High-level interface that owns both dispersal parameters and the
    simulation state.
"""

from __future__ import annotations

import os
import pathlib
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from paradis._device import device
from paradis.core.adjacency import adjacency_matrix_torch
from paradis.core.dispersal import single_window_dispersal, weighted_adjacency
from paradis.core.growth import carrying_capacity_from_hs, growth_step

# Compile the batched lu_solve once at import time.  On first call TorchInductor
# traces and fuses the float16→float32 cast + solve into a single CUDA kernel;
# subsequent calls hit the compiled graph directly (~2–4× faster on GPU).
# Falls back silently to eager mode on CPU or if compilation is unavailable.
# torch.compile requires Triton (Linux/WSL only on CUDA).  Detect at import
# so we never attempt compilation on platforms where it will fail at runtime.
def _triton_available() -> bool:
    try:
        import triton  # noqa: F401
        return True
    except ImportError:
        return False

if torch.cuda.is_available() and _triton_available():
    _compiled_lu_solve = torch.compile(torch.linalg.lu_solve)
else:
    _compiled_lu_solve = torch.linalg.lu_solve


# ---------------------------------------------------------------------------
# Machine-tuned worker count for the `solver='sparse'` thread pool
# ---------------------------------------------------------------------------

def _sparse_solver_max_workers() -> int:
    """Auto-tuned worker cap for the `solver='sparse'` per-window solve pool.

    Uses PHYSICAL core count (not logical/hyperthreaded) when available —
    SuperLU's back-substitution is compute- and memory-bandwidth-bound, so
    hyperthreaded "extra" logical cores don't provide a second independent
    ALU for this kind of work the way they would for I/O-bound tasks (see
    the discussion this implements). Falls back to `os.cpu_count()`
    (logical) if `psutil` isn't installed, since physical count isn't
    otherwise available from the standard library. Computed once and
    cached at import time — the machine's core count doesn't change
    mid-run.
    """
    try:
        import psutil
        physical = psutil.cpu_count(logical=False)
        if physical:
            return physical
    except ImportError:
        pass
    return os.cpu_count() or 1


_SPARSE_SOLVER_MAX_WORKERS = _sparse_solver_max_workers()


# ---------------------------------------------------------------------------
# Window-level kernel pre-computation
# ---------------------------------------------------------------------------

def _build_wstar_sparse(
    hs_win: torch.Tensor,
    r: float,
    n: float,
    border_flat: torch.Tensor,
    n_flat: int,
) -> torch.Tensor:
    """Build Wstar directly from sparse 4-connected edges — avoids dense adj**r.

    For an N-pixel window, the adjacency matrix has only ~4N non-zero entries.
    Operating on the full N×N dense matrix for the power and row-normalisation
    wastes O(N²) work on zeros.  This function applies r/n only to
    the ~4N edge weights, then scatters them into the dense Wstar in one pass.
    """
    H, W = hs_win.shape
    dev  = hs_win.device

    i_g      = torch.arange(H, device=dev).view(-1, 1).expand(H, W)
    j_g      = torch.arange(W, device=dev).view(1, -1).expand(H, W)
    flat_idx = (i_g * W + j_g)

    # Right edges: (i,j) ↔ (i,j+1)
    src_r = flat_idx[:, :-1].reshape(-1)
    dst_r = flat_idx[:, 1:].reshape(-1)
    w_r   = (hs_win[:, :-1] + hs_win[:, 1:]).reshape(-1) * 0.5

    # Down edges: (i,j) ↔ (i+1,j)
    src_d = flat_idx[:-1, :].reshape(-1)
    dst_d = flat_idx[1:, :].reshape(-1)
    w_d   = (hs_win[:-1, :] + hs_win[1:, :]).reshape(-1) * 0.5

    # Symmetric — both directions for each undirected edge
    src = torch.cat([src_r, dst_r, src_d, dst_d])
    dst = torch.cat([dst_r, src_r, dst_d, src_d])
    w   = torch.cat([w_r,   w_r,   w_d,   w_d])

    # Mirror original: weighted_adjacency(adj**r, n)
    # Step 1: apply r to raw edge weights  →  w_a = adj**r
    w_a = w.clamp(min=1e-7).pow(r)
    # Step 2: weighted_adjacency formula with n
    fW          = w_a.clamp(min=1e-7).pow(n) + 1e-30
    row_sum_fW  = torch.zeros(n_flat, device=dev, dtype=torch.float32)
    row_sum_fW.scatter_add_(0, src, fW)

    Wstar           = torch.zeros((n_flat, n_flat), device=dev, dtype=torch.float32)
    Wstar[src, dst] = w_a * fW / (row_sum_fW[src] + 1e-30)

    # Pin border pixels — zero their rows, set diagonal to 1 (#1 vectorised)
    Wstar[border_flat, :]             = 0.0
    Wstar[border_flat, border_flat]   = 1.0

    return Wstar


def _build_window_lu(
    hs_win: torch.Tensor,
    r: float,
    n: float,
    ewalk: float,
    border_flat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pre-compute and LU-factorise the dispersal linear system for one window.

    Returns
    -------
    LU : torch.Tensor  (float16)
        Packed LU factorisation stored in half-precision to halve cache RAM.
        Upcast to float32 before :func:`torch.linalg.lu_solve`.
    pivots : torch.Tensor  (int32)
        Row-permutation pivot indices.
    """
    n_flat = hs_win.numel()
    dev    = hs_win.device

    Wstar = _build_wstar_sparse(hs_win, r, n, border_flat, n_flat)

    p        = float(ewalk) / (1.0 + float(ewalk))
    constant = 1.0 / (1.0 - p)
    eye      = torch.eye(n_flat, device=dev, dtype=torch.float32)
    Me       = (eye - p * Wstar) * constant

    LU, pivots = torch.linalg.lu_factor(Me.T.contiguous())
    # Always keep kernels on CPU — moves to device only during solve.
    # This bounds VRAM usage to one batch, not the entire kernel set.
    return LU.half().cpu(), pivots.cpu()   # #7: float16 halves cache RAM


# ---------------------------------------------------------------------------
# GMRES iterative solver (alternative to LU — no pre-factorisation needed)
# ---------------------------------------------------------------------------

def _build_wstar_edges(
    hs_win: torch.Tensor,
    r: float,
    n: float,
    border_flat: torch.Tensor,
    n_flat: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build sparse Wstar as (src, dst, vals) edge list stored on CPU.

    O(N) cost vs O(N²) for the dense Wstar and O(N³) for LU.  Each entry
    costs ~64 N bytes vs ~2 N² bytes (float16 LU), a >1000× reduction for
    win_pixels = 71.  Used by the GMRES solver path.
    """
    H, W = hs_win.shape
    dev  = hs_win.device

    i_g      = torch.arange(H, device=dev).view(-1, 1).expand(H, W)
    j_g      = torch.arange(W, device=dev).view(1, -1).expand(H, W)
    flat_idx = i_g * W + j_g

    src_r = flat_idx[:, :-1].reshape(-1)
    dst_r = flat_idx[:, 1:].reshape(-1)
    w_r   = (hs_win[:, :-1] + hs_win[:, 1:]).reshape(-1) * 0.5

    src_d = flat_idx[:-1, :].reshape(-1)
    dst_d = flat_idx[1:, :].reshape(-1)
    w_d   = (hs_win[:-1, :] + hs_win[1:, :]).reshape(-1) * 0.5

    src = torch.cat([src_r, dst_r, src_d, dst_d])
    dst = torch.cat([dst_r, src_r, dst_d, src_d])
    w   = torch.cat([w_r,   w_r,   w_d,   w_d])

    w_a = w.clamp(min=1e-7).pow(r)
    fW  = w_a.clamp(min=1e-7).pow(n) + 1e-30
    row_sum_fW = torch.zeros(n_flat, device=dev, dtype=torch.float32)
    row_sum_fW.scatter_add_(0, src, fW)
    vals = w_a * fW / (row_sum_fW[src] + 1e-30)

    # Border pixels: remove outgoing edges, replace with absorbing self-loops.
    non_border = ~torch.isin(src, border_flat)
    src  = torch.cat([src[non_border],  border_flat])
    dst  = torch.cat([dst[non_border],  border_flat])
    vals = torch.cat([vals[non_border], torch.ones(len(border_flat), device=dev)])

    return src.cpu(), dst.cpu(), vals.cpu()


def _gmres(
    matvec,
    b: torch.Tensor,
    tol: float = 1e-5,
    restart: int = 50,
    max_outer: int = 5,
) -> torch.Tensor:
    """Restarted GMRES(restart) for square non-symmetric linear systems.

    Solves ``A x = b`` where *A* is given as a callable *matvec*.  Uses
    modified Gram-Schmidt Arnoldi followed by a small least-squares
    back-solve at each restart cycle.  No preconditioner is needed: the
    dispersal system ``(I - p W*)`` has eigenvalues bounded away from zero
    for ``p < 1``.

    Parameters
    ----------
    matvec:
        Callable ``x -> A @ x`` (float32 tensor in, float32 tensor out).
    b:
        Right-hand-side vector.
    tol:
        Relative residual tolerance (default ``1e-5``).
    restart:
        Krylov subspace size before restart — GMRES(*restart*).
    max_outer:
        Maximum number of restart cycles.

    Returns
    -------
    torch.Tensor
        Approximate solution, same shape and device as *b*.
    """
    x      = torch.zeros_like(b)
    b_norm = b.norm()
    if b_norm < 1e-30:
        return x

    for _outer in range(max_outer):
        r      = b - matvec(x)
        r_norm = r.norm()
        if r_norm / b_norm < tol:
            break

        m = min(restart, b.numel())
        Q = torch.zeros(b.numel(), m + 1, device=b.device, dtype=b.dtype)
        H = torch.zeros(m + 1,    m,     device=b.device, dtype=b.dtype)
        Q[:, 0] = r / r_norm

        j_stop = m
        for j in range(m):
            w = matvec(Q[:, j])
            for i in range(j + 1):          # modified Gram-Schmidt
                H[i, j] = torch.dot(w, Q[:, i])
                w        = w - H[i, j] * Q[:, i]
            h_next       = w.norm()
            H[j + 1, j]  = h_next
            if h_next > 1e-14:
                Q[:, j + 1] = w / h_next
            if h_next < tol * r_norm:       # happy breakdown
                j_stop = j + 1
                break

        m_eff    = j_stop
        e1       = torch.zeros(m_eff + 1, 1, device=b.device, dtype=b.dtype)
        e1[0, 0] = r_norm
        y = torch.linalg.lstsq(H[:m_eff + 1, :m_eff], e1).solution.squeeze(1)
        x = x + Q[:, :m_eff] @ y

    return x


def _build_sparse_lu(
    hs_win: torch.Tensor,
    r: float,
    n: float,
    ewalk: float,
    border_flat: torch.Tensor,
    n_flat: int,
):
    """Sparse LU factorisation of the dispersal system via scipy SuperLU.

    Builds the sparse W* matrix (only ~4N non-zeros) and calls
    ``scipy.sparse.linalg.splu`` on ``Me.T = constant * (I - p W*).T``.

    Compared with the dense LU path:

    * **Pre-build**: O(N^1.5) instead of O(N^3) — ~100× faster for win=71.
    * **Solve**:     O(N^1.5) instead of O(N^2) — ~70× faster per step.
    * **Storage**:   sparse factors instead of a dense N×N float16 matrix.
    * **Precision**: full float32 — no float16 degradation.

    Returns
    -------
    scipy.sparse.linalg.SuperLU
        Factorised system; call ``.solve(b)`` to back-substitute.
    """
    try:
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla
    except ImportError:
        raise ImportError(
            "scipy is required for solver='sparse'.  "
            "Install with:  pip install scipy"
        )

    H, W = hs_win.shape
    dev  = hs_win.device

    i_g      = torch.arange(H, device=dev).view(-1, 1).expand(H, W)
    j_g      = torch.arange(W, device=dev).view(1, -1).expand(H, W)
    flat_idx = i_g * W + j_g

    src_r = flat_idx[:, :-1].reshape(-1);  dst_r = flat_idx[:, 1:].reshape(-1)
    w_r   = (hs_win[:, :-1] + hs_win[:, 1:]).reshape(-1) * 0.5
    src_d = flat_idx[:-1, :].reshape(-1);  dst_d = flat_idx[1:, :].reshape(-1)
    w_d   = (hs_win[:-1, :] + hs_win[1:, :]).reshape(-1) * 0.5

    src = torch.cat([src_r, dst_r, src_d, dst_d])
    dst = torch.cat([dst_r, src_r, dst_d, src_d])
    w   = torch.cat([w_r,   w_r,   w_d,   w_d])

    w_a = w.clamp(min=1e-7).pow(r)
    fW  = w_a.clamp(min=1e-7).pow(n) + 1e-30
    row_sum_fW = torch.zeros(n_flat, device=dev, dtype=torch.float32)
    row_sum_fW.scatter_add_(0, src, fW)
    vals = w_a * fW / (row_sum_fW[src] + 1e-30)

    non_border = ~torch.isin(src, border_flat)
    src  = torch.cat([src[non_border],  border_flat])
    dst  = torch.cat([dst[non_border],  border_flat])
    vals = torch.cat([vals[non_border], torch.ones(len(border_flat), device=dev)])

    src_np  = src.cpu().numpy().astype(np.int32)
    dst_np  = dst.cpu().numpy().astype(np.int32)
    vals_np = vals.cpu().float().numpy()

    p        = float(ewalk) / (1.0 + float(ewalk))
    constant = 1.0 / (1.0 - p)
    Wstar_sp = sp.csr_matrix((vals_np, (src_np, dst_np)), shape=(n_flat, n_flat))
    I_sp     = sp.eye(n_flat, format='csr', dtype=np.float32)
    MeT_csc  = (constant * (I_sp - p * Wstar_sp)).T.tocsc()

    return spla.splu(MeT_csc)


def _prebuild_kernels(
    valid_centres: list[tuple[int, int]],
    hs: torch.Tensor,
    r: float,
    n: float,
    ewalk: float,
    win_pixels: int,
    mhalf: int,
    xs_map: int,
    ys_map: int,
    border_flat: torch.Tensor,
    kernel_cache,
    solver: str = 'lu',
) -> None:
    """Build kernel entries for all *valid_centres* not yet in *kernel_cache*.

    For ``solver='lu'``:     stores ``(LU, pivots)`` float16 tensors on CPU.
    For ``solver='gmres'``:  stores ``(src, dst, vals)`` sparse edge lists on
        CPU — ~1000× smaller than LU, built in O(N) instead of O(N³).
    For ``solver='sparse'``: stores a scipy ``SuperLU`` object per window —
        built in O(N^1.5) with full float32 precision; solve is O(N^1.5).

    On CPU: uses a thread pool so windows are built in parallel.
    On GPU: sequential (GPU is already parallel).
    """
    missing = [(x0, y0) for x0, y0 in valid_centres if (x0, y0) not in kernel_cache]
    if not missing:
        return

    def _build_one(x0: int, y0: int):
        mx_lo = x0 - mhalf;  mx_hi = x0 + mhalf + 1
        my_lo = y0 - mhalf;  my_hi = y0 + mhalf + 1
        cx_lo = max(0, mx_lo);  cx_hi = min(xs_map, mx_hi)
        cy_lo = max(0, my_lo);  cy_hi = min(ys_map, my_hi)
        wx_lo = cx_lo - mx_lo;  wx_hi = wx_lo + (cx_hi - cx_lo)
        wy_lo = cy_lo - my_lo;  wy_hi = wy_lo + (cy_hi - cy_lo)
        hs_win = torch.zeros((win_pixels, win_pixels), device=device)
        hs_win[wx_lo:wx_hi, wy_lo:wy_hi] = hs[cx_lo:cx_hi, cy_lo:cy_hi]
        if solver == 'gmres':
            return _build_wstar_edges(hs_win, r, n, border_flat, win_pixels ** 2)
        if solver == 'sparse':
            return _build_sparse_lu(hs_win, r, n, ewalk, border_flat, win_pixels ** 2)
        return _build_window_lu(hs_win, r, n, ewalk, border_flat)

    if device.type == "cpu":
        n_workers = min(os.cpu_count() or 1, len(missing), 8)
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futs = {pool.submit(_build_one, x0, y0): (x0, y0) for x0, y0 in missing}
            for f in tqdm(as_completed(futs), total=len(missing),
                          desc="Pre-building kernels", leave=True):
                key = futs[f]
                kernel_cache[key] = f.result()
    else:
        for x0, y0 in tqdm(missing, desc="Pre-building kernels", leave=True):
            kernel_cache[(x0, y0)] = _build_one(x0, y0)


# ---------------------------------------------------------------------------
# Sliding-window dispersal step
# ---------------------------------------------------------------------------

def dispersal_step(
    hs: torch.Tensor,
    distrib: torch.Tensor,
    ewalk: float,
    n: float,
    r: float,
    window_size: int = 70,
    sub_window: int = 31,
    presence_threshold: float = 0.019,
    overdispersion_cap: float = 1e5,
    stochastic: bool = False,
    kernel_cache: dict | None = None,
    valid_centres: list | None = None,
    border_flat: torch.Tensor | None = None,
    _out_distrib: torch.Tensor | None = None,
    _out_borders: torch.Tensor | None = None,
    solver: str = 'lu',
    gmres_tol: float = 1e-5,
    gpu_batch_mem_fraction: float = 0.8,
) -> tuple:
    """Apply one dispersal step via sliding evaluation windows.

    For each grid point spaced *sub_window* apart, the density within the
    sub-window is dispersed across a larger *window_size* × *window_size*
    region.

    **Cross-year kernel cache** – *kernel_cache* maps ``(x0, y0)`` to the LU
    factorisation of the window's dispersal linear system.  Because habitat
    suitability never changes, these factorisations are valid for the entire
    simulation.  On the first encounter of a window the system is factorised
    (O(N³)); on every subsequent year only a triangular back-substitution is
    needed (O(N²)), giving a large speed-up once the occupied area is
    established.  Pass the same dict across all calls to
    :func:`run_simulation` (or store it on the :class:`PopulationSimulator`
    object) to benefit from this caching.

    **Overdispersion loop (NOT recursion)** – any mass that lands on a
    window's BORDER pixels this pass ("border overflow") hasn't actually
    settled yet; it needs to be re-dispersed from wherever it landed,
    potentially requiring several more passes before it decays below
    ``presence_threshold / overdispersion_cap``. This used to be implemented
    as `dispersal_step` calling ITSELF recursively on the leftover border
    mass. That has a real memory cost: a recursive call's `new_distrib`/
    `borders_accum` buffers (each a full-map-sized tensor) can't be freed
    until ALL deeper recursive calls return, because the parent frame is
    still waiting to do `new_distrib + borders_accum` — so with a slow-
    decaying border (e.g. an initial density set too high relative to
    `presence_threshold`, causing very deep "recursion"), memory usage grew
    roughly linearly with depth, with dozens of full-map tensors and
    per-window kernels alive simultaneously. Rewritten here as an explicit
    loop: one working set of buffers is reused every iteration (the
    previous iteration's are simply overwritten, so Python can garbage-
    collect them immediately, independent of how many iterations it takes
    to converge), while a SEPARATE running total accumulates each
    iteration's resolved (interior) contribution. Mathematically identical
    to the old recursive version — same per-iteration computation, same
    convergence check — just without holding every iteration's memory
    alive at once.

    Parameters
    ----------
    hs:
        2-D HS tensor on the compute device.
    distrib:
        2-D density map (CPU).
    ewalk, n, r:
        Dispersal parameters.
    window_size:
        Width of the larger evaluation window (pixels).  Must be even.
    sub_window:
        Width of the inner density extraction window.  Must be odd.
    presence_threshold:
        Relative-abundance value for presence.
    overdispersion_cap:
        Stop the overdispersion loop when border density <
        ``presence_threshold / overdispersion_cap``.
    stochastic:
        If ``True``, accumulate sub-threshold border density separately
        (from the FIRST iteration only — matching the old recursive
        version's behaviour, which likewise only ever returned the
        outermost call's `stoch_density`).
    kernel_cache:
        Mutable dict ``{(x0, y0): (LU, pivots)}`` shared across calls.
        Modified in-place; pass ``{}`` on first use and keep the same object.
    gpu_batch_mem_fraction:
        ``solver='lu'``-only. Every active window's linear solve is
        independent of every other's, so they're solved together in one
        batched GPU call rather than one at a time — this is the fraction
        of currently FREE VRAM that batch is allowed to occupy (default
        0.8 = 80%), directly setting how many windows get solved in
        parallel per batch (`batch_sz = free_VRAM * gpu_batch_mem_fraction
        / bytes_per_window`, uncapped — previously hard-limited to a
        conservative 25%/64-window ceiling regardless of how much memory
        was actually free, leaving real parallelism on the table). Lower
        this if you're running other GPU work alongside the simulation.

    Returns
    -------
    new_distrib : torch.Tensor
        Updated density map (CPU).
    tested_xs, tested_ys : list
        Centres of active windows this step (first overdispersion
        iteration only — see `stochastic`'s note above).
    stoch_density : numpy.ndarray
        Sub-threshold border density (only meaningful when *stochastic=True*).
    """
    if kernel_cache is None:
        kernel_cache = {}

    xs_map, ys_map = hs.shape
    mhalf      = window_size // 2
    win_pixels = 2 * mhalf + 1
    shalf      = sub_window // 2

    def _make_divs(size: int) -> np.ndarray:
        divs = np.arange(shalf, size, sub_window, dtype=int)
        if len(divs) == 0 or int(divs[-1]) + shalf + 1 < size:
            divs = np.append(divs, size - shalf - 1)
        return divs

    x_divs = _make_divs(xs_map)
    y_divs = _make_divs(ys_map)
    xg, yg = np.meshgrid(x_divs, y_divs)

    # ── Border structures (computed once, shared across windows AND across
    # every overdispersion iteration below) ─────────────────────────────────
    # _border_2d : 2-D mask on device — used for masking Dsub after solve
    # border_flat: 1-D flat indices of border pixels — used in _build_window_lu
    if border_flat is None:
        _border_2d  = torch.zeros((win_pixels, win_pixels), device=device, dtype=torch.float32)
        _border_2d[0:win_pixels - 1, 0]               = 1
        _border_2d[0,                0:win_pixels - 1] = 1
        _border_2d[win_pixels - 1,   0:win_pixels - 1] = 1
        _border_2d[0:win_pixels,     win_pixels - 1]   = 1
        border_flat = torch.where(_border_2d.reshape(-1) == 1)[0]
    else:
        _border_2d = torch.zeros((win_pixels, win_pixels), device=device, dtype=torch.float32)
        _border_2d.reshape(-1)[border_flat] = 1.0

    # ── #5: Permanent HS-valid centre filter ────────────────────────────────
    # Windows where every HS pixel is zero can never carry density — skip them
    # permanently.  If valid_centres is pre-computed by run_simulation, reuse it.
    if valid_centres is None:
        valid_centres = []
        for x0, y0 in zip(xg.flatten(), yg.flatten()):
            x0, y0 = int(x0), int(y0)
            cx_lo = max(0, x0 - mhalf);  cx_hi = min(xs_map, x0 + mhalf + 1)
            cy_lo = max(0, y0 - mhalf);  cy_hi = min(ys_map, y0 + mhalf + 1)
            if hs[cx_lo:cx_hi, cy_lo:cy_hi].any():
                valid_centres.append((x0, y0))

    # ── #6: Running total across all overdispersion iterations ─────────────
    # Reuse caller-supplied buffer (zeroed in-place) to avoid repeated VRAM
    # allocation across simulation YEARS; fall back to fresh allocation when
    # not provided. This is the ONLY buffer that persists across iterations
    # — each iteration's own `new_distrib`/`borders_accum` working buffers
    # (below) are freshly allocated and immediately eligible for garbage
    # collection once the next iteration overwrites the local variable, so
    # memory use no longer grows with the number of iterations needed.
    if _out_distrib is not None and _out_distrib.shape == (xs_map, ys_map):
        total_new_distrib = _out_distrib.zero_()
    else:
        total_new_distrib = torch.zeros((xs_map, ys_map), device=device)

    threshold_recurse = presence_threshold / max(overdispersion_cap, 1)
    current_distrib = distrib
    tested_xs: list = []
    tested_ys: list = []
    stoch_density = np.zeros((xs_map, ys_map), dtype=np.float32)
    depth = 0

    while True:
        # ── #6: This iteration's working tensors ────────────────────────────
        if depth == 0 and _out_borders is not None and _out_borders.shape == (xs_map, ys_map):
            borders_accum = _out_borders.zero_()
        else:
            borders_accum = torch.zeros((xs_map, ys_map), device=device)
        new_distrib = torch.zeros((xs_map, ys_map), device=device)

        # ── #3: Collect all active windows, then batch-solve on GPU ─────────
        # "Active" = HS-valid AND has density in its sub-window this iteration.
        active_keys:  list = []
        b_vecs:       list = []
        write_infos:  list = []
        # Local STRONG references to this iteration's kernels, parallel to
        # `active_keys` — used by the solve loops below INSTEAD OF re-reading
        # `kernel_cache[key]`. With a bounded `MemoryKernelCache`, collecting
        # many distinct active windows in this same loop can evict an
        # EARLIER window's kernel before the solve loop gets to read it back
        # — a real `KeyError`, not just a theoretical one. Holding a local
        # reference here keeps every kernel needed THIS iteration alive
        # regardless of what the shared cache evicts in the meantime;
        # `kernel_cache[(x0, y0)] = ...` below still populates the shared
        # cache for cross-call/cross-iteration reuse as before.
        active_kernels: list = []

        with torch.no_grad():
            for x0, y0 in valid_centres:
                sx_lo = max(0, x0 - shalf);  sx_hi = min(xs_map, x0 + shalf + 1)
                sy_lo = max(0, y0 - shalf);  sy_hi = min(ys_map, y0 + shalf + 1)
                sub = current_distrib[sx_lo:sx_hi, sy_lo:sy_hi]
                if not sub.any():
                    continue

                mx_lo = x0 - mhalf;  mx_hi = x0 + mhalf + 1
                my_lo = y0 - mhalf;  my_hi = y0 + mhalf + 1
                cx_lo = max(0, mx_lo);  cx_hi = min(xs_map, mx_hi)
                cy_lo = max(0, my_lo);  cy_hi = min(ys_map, my_hi)
                wx_lo = cx_lo - mx_lo;  wx_hi = wx_lo + (cx_hi - cx_lo)
                wy_lo = cy_lo - my_lo;  wy_hi = wy_lo + (cy_hi - cy_lo)
                dsx_lo = sx_lo - mx_lo;  dsx_hi = dsx_lo + (sx_hi - sx_lo)
                dsy_lo = sy_lo - my_lo;  dsy_hi = dsy_lo + (sy_hi - sy_lo)

                # Lazily build missing kernels (should be empty after _prebuild_kernels)
                if (x0, y0) not in kernel_cache:
                    hs_win = torch.zeros((win_pixels, win_pixels), device=device)
                    hs_win[wx_lo:wx_hi, wy_lo:wy_hi] = hs[cx_lo:cx_hi, cy_lo:cy_hi]
                    if solver == 'gmres':
                        kernel_cache[(x0, y0)] = _build_wstar_edges(
                            hs_win, r, n, border_flat, win_pixels ** 2)
                    elif solver == 'sparse':
                        kernel_cache[(x0, y0)] = _build_sparse_lu(
                            hs_win, r, n, ewalk, border_flat, win_pixels ** 2)
                    else:
                        kernel_cache[(x0, y0)] = _build_window_lu(
                            hs_win, r, n, ewalk, border_flat)
                    del hs_win

                D0e = torch.zeros((win_pixels, win_pixels), device=device)
                D0e[dsx_lo:dsx_hi, dsy_lo:dsy_hi] = sub.to(device)

                active_keys.append((x0, y0))
                active_kernels.append(kernel_cache[(x0, y0)])
                b_vecs.append(D0e.reshape(-1))
                write_infos.append((cx_lo, cx_hi, cy_lo, cy_hi, wx_lo, wx_hi, wy_lo, wy_hi))
                if depth == 0:
                    tested_xs.append(x0)
                    tested_ys.append(y0)

            # ── Solve each active window ─────────────────────────────────────
            if active_keys:
                if solver == 'sparse':
                    # Sparse LU: scipy SuperLU back-substitution — O(N^1.5) per window,
                    # full float32 precision, no GPU memory required for the factors.
                    # This is ALWAYS CPU work regardless of `device` — SciPy's SuperLU
                    # has no CUDA backend at all (that's exactly why `solver='lu'`
                    # exists as the GPU-capable alternative). Every window's solve is
                    # independent of every other's, so — same pattern already used by
                    # `_prebuild_kernels` for the factorisation step — the `.solve()`
                    # calls themselves (the actual CPU-bound cost; SuperLU's C
                    # computation releases the GIL, so this is genuine multi-core
                    # parallelism, not just concurrency) run in a thread pool. The
                    # GPU tensor writes below stay sequential afterwards, since
                    # in-place accumulation into shared `new_distrib`/`borders_accum`
                    # tensors is not itself thread-safe.
                    # `.cpu().numpy()` conversion happens INSIDE the worker, one
                    # window at a time, not pre-materialised into a list up front
                    # — so peak memory for these input vectors is bounded by
                    # `n_workers` (however many solves are actually in flight at
                    # once), not by the total number of active windows this step
                    # (which, on a large map, can be far bigger than n_workers).
                    def _solve_one(i, _kerns=active_kernels, _bv=b_vecs):
                        return i, _kerns[i].solve(_bv[i].cpu().numpy())

                    n_workers = min(_SPARSE_SOLVER_MAX_WORKERS, len(active_keys))
                    if n_workers > 1:
                        with ThreadPoolExecutor(max_workers=n_workers) as pool:
                            solved = dict(pool.map(_solve_one, range(len(active_keys))))
                    else:
                        solved = dict(_solve_one(i) for i in range(len(active_keys)))

                    for i, key in enumerate(active_keys):
                        x_np    = solved[i]
                        Dt_flat = torch.from_numpy(x_np).to(device)
                        wi = write_infos[i]
                        cx_lo, cx_hi, cy_lo, cy_hi, wx_lo, wx_hi, wy_lo, wy_hi = wi
                        Dsub     = Dt_flat.reshape(win_pixels, win_pixels)
                        local_od = Dsub * _border_2d
                        interior = 1.0 - _border_2d[wx_lo:wx_hi, wy_lo:wy_hi]
                        new_distrib[cx_lo:cx_hi, cy_lo:cy_hi]   += (
                            Dsub[wx_lo:wx_hi, wy_lo:wy_hi] * interior)
                        borders_accum[cx_lo:cx_hi, cy_lo:cy_hi] += (
                            local_od[wx_lo:wx_hi, wy_lo:wy_hi])

                elif solver == 'gmres':
                    # GMRES: one iterative solve per window using sparse mat-vec.
                    # No LU factorisation — just edge-list mat-vec O(4N) per iter.
                    p_f      = ewalk / (1.0 + ewalk)
                    constant = 1.0 / (1.0 - p_f)
                    n_flat_w = win_pixels ** 2
                    for i, key in enumerate(active_keys):
                        src_e, dst_e, vals_e = active_kernels[i]
                        src_d  = src_e.to(device)
                        dst_d  = dst_e.to(device)
                        vals_d = vals_e.to(device)

                        # Me.T @ x = constant * (x - p * Wstar.T @ x)
                        # Wstar.T @ x: for edge (src->dst, val), scatter val*x[src] to dst
                        def _mv(x, _s=src_d, _d=dst_d, _v=vals_d,
                                 _nf=n_flat_w, _p=p_f, _c=constant):
                            out = torch.zeros(_nf, device=x.device, dtype=x.dtype)
                            out.scatter_add_(0, _d, _v * x[_s])
                            return _c * (x - _p * out)

                        Dt_flat = _gmres(_mv, b_vecs[i], tol=gmres_tol)
                        del src_d, dst_d, vals_d

                        wi = write_infos[i]
                        cx_lo, cx_hi, cy_lo, cy_hi, wx_lo, wx_hi, wy_lo, wy_hi = wi
                        Dsub     = Dt_flat.reshape(win_pixels, win_pixels)
                        local_od = Dsub * _border_2d
                        interior = 1.0 - _border_2d[wx_lo:wx_hi, wy_lo:wy_hi]
                        new_distrib[cx_lo:cx_hi, cy_lo:cy_hi]   += (
                            Dsub[wx_lo:wx_hi, wy_lo:wy_hi] * interior)
                        borders_accum[cx_lo:cx_hi, cy_lo:cy_hi] += (
                            local_od[wx_lo:wx_hi, wy_lo:wy_hi])

                else:
                    # LU: batched back-substitution — O(N²) per window, O(N³) pre-built.
                    # Kernels are stored on CPU; only one batch lives on device at a time,
                    # so VRAM usage is batch_sz × N² × 4 bytes (float32 during solve).
                    # All windows in `active_keys` are mathematically independent (each
                    # is its own local linear solve) — the batch dimension IS the
                    # parallelism, bounded only by how many windows' worth of solve
                    # memory fit in `gpu_batch_mem_fraction` of currently free VRAM.
                    # The old hardcoded 25%/64-window cap left real GPU parallelism on
                    # the table whenever more memory was actually free. Computed once
                    # per step (not re-measured per batch within the loop below) —
                    # simple and sufficient, since successive batches release their
                    # memory before the next one starts anyway.
                    if device.type == "cuda":
                        free_mb  = torch.cuda.mem_get_info()[0] // (1024 ** 2)
                        lu_mb    = max(1, (win_pixels ** 4) * 4 // (1024 ** 2))
                        batch_sz = max(1, int(free_mb * gpu_batch_mem_fraction / lu_mb))
                    else:
                        batch_sz = 1

                    for start in range(0, len(active_keys), batch_sz):
                        keys    = active_keys[start:start + batch_sz]
                        kerns   = active_kernels[start:start + batch_sz]
                        b_bat = torch.stack(b_vecs[start:start + batch_sz]).unsqueeze(-1)

                        if len(keys) == 1:
                            LU, pivots = kerns[0]
                            LU_dev  = LU.float().to(device)
                            piv_dev = pivots.to(device)
                            res = _compiled_lu_solve(
                                LU_dev, piv_dev, b_bat.squeeze(0)
                            ).squeeze(-1).unsqueeze(0)
                            del LU_dev, piv_dev
                        else:
                            LU_bat  = torch.stack([k[0].float() for k in kerns]).to(device)
                            piv_bat = torch.stack([k[1] for k in kerns]).to(device)
                            res = _compiled_lu_solve(LU_bat, piv_bat, b_bat).squeeze(-1)
                            del LU_bat, piv_bat

                        for i, key in enumerate(keys):
                            wi = write_infos[start + i]
                            cx_lo, cx_hi, cy_lo, cy_hi, wx_lo, wx_hi, wy_lo, wy_hi = wi
                            Dsub     = res[i].reshape(win_pixels, win_pixels)
                            local_od = Dsub * _border_2d
                            interior = 1.0 - _border_2d[wx_lo:wx_hi, wy_lo:wy_hi]
                            new_distrib[cx_lo:cx_hi, cy_lo:cy_hi]   += (
                                Dsub[wx_lo:wx_hi, wy_lo:wy_hi] * interior)
                            borders_accum[cx_lo:cx_hi, cy_lo:cy_hi] += (
                                local_od[wx_lo:wx_hi, wy_lo:wy_hi])

        # Overdispersion: this iteration's resolved (interior) contribution is
        # merged into the running total right away — it does NOT wait for the
        # border mass to finish re-dispersing, unlike the old recursive
        # version where every ancestor frame's `new_distrib` stayed pinned in
        # memory until the deepest call returned.
        total_new_distrib += new_distrib

        if stochastic and depth == 0:
            stoch_density = ((borders_accum < presence_threshold) * borders_accum).cpu().numpy()
            borders_accum = borders_accum * (borders_accum >= presence_threshold)
        elif stochastic:
            borders_accum = borders_accum * (borders_accum >= presence_threshold)

        max_border = float(borders_accum.max().item())
        # Single, continuously-updated status line (carriage return, no
        # newline) — NOT a new print per iteration, so this stays legible
        # even across many overdispersion iterations — showing exactly how
        # close the border mass is to the threshold this loop has to clear
        # before it stops. A border that decays slowly relative to
        # `threshold_recurse` (= presence_threshold / overdispersion_cap —
        # e.g. ~1.7e-6 for presence_threshold=0.17, overdispersion_cap=1e5)
        # needs many iterations — no longer a memory problem (see the
        # docstring above), but still a real compute-time cost worth seeing
        # live.
        print(f"\r[dispersal_step] depth={depth}  border max density="
              f"{max_border:.4g}  vs threshold={threshold_recurse:.4g}  "
              f"{'-> converged' if max_border <= threshold_recurse else '-> recursing...'}"
              + " " * 10, end="", flush=True)

        if max_border <= threshold_recurse:
            total_new_distrib += borders_accum
            break

        current_distrib = borders_accum.cpu()
        depth += 1

    result = total_new_distrib.cpu()   # #6: single CPU transfer
    return result, tested_xs, tested_ys, stoch_density


# ---------------------------------------------------------------------------
# Full simulation
# ---------------------------------------------------------------------------

def run_simulation(
    hs: np.ndarray,
    init_distrib: "np.ndarray | str | pathlib.Path",
    ewalk: float,
    n: float,
    r: float,
    tgrowth: float,
    carrying_capacity_params: tuple,
    presence_threshold: float,
    n_steps: int = 35,
    init_year: int = 1968,
    window_size: int = 70,
    sub_window: int = 31,
    additional_events: dict | None = None,
    breeding_ground: np.ndarray | None = None,
    region_mask: np.ndarray | None = None,
    time_division: int = 1,
    plot: bool = False,
    kernel_cache: dict | None = None,
    solver: str = 'lu',
    gmres_tol: float = 1e-5,
    gpu_batch_mem_fraction: float = 0.8,
    _snapshot_store: list | None = None,
) -> np.ndarray:
    """Run the PARADIS population simulation.

    Parameters
    ----------
    hs:
        2-D habitat-suitability map.
    init_distrib:
        2-D initial abundance map, or a path (``str``/``pathlib.Path``) to
        a ``.npy`` file holding one (loaded via ``numpy.load``).
    ewalk, n, r:
        Dispersal parameters.
    tgrowth:
        Characteristic growth time.
    carrying_capacity_params:
        ``(L, k, x0)`` logistic parameters.
    presence_threshold:
        Relative-abundance value for presence.
    n_steps:
        Number of time steps to simulate.
    init_year:
        Calendar year of the first step (for *additional_events*).
    window_size:
        Evaluation window width (pixels).
    sub_window:
        Sub-window width (pixels).
    additional_events:
        Dict ``{year: (x, y)}`` of reintroduction events.
    breeding_ground:
        Optional binary mask restricting reproduction.
    region_mask:
        Optional binary mask for the study region (cropping only).
    time_division:
        Number of sub-steps per year (for large dispersers).
    plot:
        If ``True``, display an overview plot at the end.
    kernel_cache:
        Mutable dict for the cross-year LU kernel cache.  Pass the same
        object across multiple calls (or store it on
        :class:`PopulationSimulator`) so that factorisations computed in
        earlier years are reused.  If ``None`` a local (call-scoped) dict is
        created and discarded on return.
    gpu_batch_mem_fraction:
        Forwarded to `dispersal_step` — see its docstring. Fraction of
        currently free VRAM the batched ``solver='lu'`` solve is allowed
        to use per batch (default 0.8); all active windows in a batch are
        solved together since they're independent, so this directly
        controls how much of that independence gets exploited in
        parallel.

    Returns
    -------
    numpy.ndarray
        Final 2-D abundance map.
    """
    if not isinstance(init_distrib, np.ndarray):
        init_distrib = np.load(init_distrib)

    L, k, x0 = carrying_capacity_params
    hs_t = torch.tensor(hs, dtype=torch.float32, device=device)
    # Whole-population dispersal model (deliberate simplification, matching
    # paradis.core.growth.equilibrium_distribution): every pixel's entire
    # density disperses every year, not just newly-produced individuals —
    # no Dt/SP (disperser/settled) split, a single state `SP` covers both.
    SP = torch.tensor(init_distrib, dtype=torch.float32)

    a, K_is, _ = carrying_capacity_from_hs(hs_t.cpu(), tgrowth, L, k, x0)
    K_is_gpu = K_is.to(device)
    a_tensor = torch.tensor(a, device=device)

    if kernel_cache is None:
        kernel_cache = {}

    growth_bl: list = []

    if breeding_ground is None:
        bg = (hs_t > 0).cpu()
    else:
        bg = torch.tensor(breeding_ground, dtype=torch.float32)

    # ── Pre-compute window grid geometry (shared with dispersal_step) ────────
    xs_map, ys_map = hs_t.shape
    mhalf      = window_size // 2
    win_pixels = 2 * mhalf + 1
    shalf      = sub_window // 2

    def _make_divs(size: int) -> np.ndarray:
        divs = np.arange(shalf, size, sub_window, dtype=int)
        if len(divs) == 0 or int(divs[-1]) + shalf + 1 < size:
            divs = np.append(divs, size - shalf - 1)
        return divs

    xg, yg = np.meshgrid(_make_divs(xs_map), _make_divs(ys_map))

    # #5: compute HS-valid centres once
    sim_valid_centres = []
    for x0c, y0c in zip(xg.flatten(), yg.flatten()):
        x0c, y0c = int(x0c), int(y0c)
        cx_lo = max(0, x0c - mhalf);  cx_hi = min(xs_map, x0c + mhalf + 1)
        cy_lo = max(0, y0c - mhalf);  cy_hi = min(ys_map, y0c + mhalf + 1)
        if hs_t[cx_lo:cx_hi, cy_lo:cy_hi].any():
            sim_valid_centres.append((x0c, y0c))

    # Border flat indices (shared across all dispersal_step calls)
    sim_border_2d = torch.zeros((win_pixels, win_pixels), device=device, dtype=torch.float32)
    sim_border_2d[0:win_pixels - 1, 0]               = 1
    sim_border_2d[0,                0:win_pixels - 1] = 1
    sim_border_2d[win_pixels - 1,   0:win_pixels - 1] = 1
    sim_border_2d[0:win_pixels,     win_pixels - 1]   = 1
    sim_border_flat = torch.where(sim_border_2d.reshape(-1) == 1)[0]

    # #4: pre-build all kernels before the time loop — SKIPPED for the
    # bounded in-memory `MemoryKernelCache` (the default when `cache_dir`
    # isn't set on `PopulationSimulator`). With a bounded cache, eagerly
    # building every valid window (there can be far more of these than fit
    # in the cache — e.g. tens of thousands on a large map) means later
    # prebuilt windows evict earlier ones BEFORE the step loop even starts,
    # in an arbitrary traversal order unrelated to which windows the
    # simulation will actually touch first (population spreads gradually
    # from the seed, so most of the map's windows may go untouched for many
    # steps, or the whole run). Confirmed to add no real speedup in
    # practice — the on-demand lazy build already in `dispersal_step`'s
    # "collect active windows" loop builds exactly what's actually used,
    # when it's actually used, which plays correctly with LRU eviction
    # instead of fighting it. Kept for `DiskKernelCache` (persists to disk
    # across separate runs — the whole point of prebuilding there) and for
    # a plain/unbounded dict (nothing gets evicted, so prebuild's
    # thread-parallel build cost is pure upside).
    from paradis.simulation.kernel_cache import MemoryKernelCache
    if not isinstance(kernel_cache, MemoryKernelCache):
        _prebuild_kernels(
            sim_valid_centres, hs_t, r, n, ewalk,
            win_pixels, mhalf, xs_map, ys_map, sim_border_flat, kernel_cache,
            solver=solver,
        )

    # Pre-allocate output buffers — zeroed in-place each step, no repeated VRAM alloc
    _buf_distrib = torch.zeros((xs_map, ys_map), device=device)
    _buf_borders = torch.zeros((xs_map, ys_map), device=device)

    snapshots: list[tuple[int, np.ndarray]] = (
        _snapshot_store if _snapshot_store is not None else []
    )
    snapshots.clear()
    snapshots.append((init_year, init_distrib.copy()))

    for t in tqdm(range(n_steps), desc="Simulation steps"):
        year = init_year + t

        if additional_events and year in additional_events:
            for xe, ye in additional_events[year]:
                SP[xe - 5: xe + 5, ye - 5: ye + 5] = presence_threshold

        for _ in range(time_division):
            SP, xs, ys, _ = dispersal_step(
                hs_t, SP, ewalk, n, r,
                window_size=window_size,
                sub_window=sub_window,
                presence_threshold=presence_threshold,
                kernel_cache=kernel_cache,
                valid_centres=sim_valid_centres,
                border_flat=sim_border_flat,
                _out_distrib=_buf_distrib,
                _out_borders=_buf_borders,
                solver=solver, gmres_tol=gmres_tol,
                gpu_batch_mem_fraction=gpu_batch_mem_fraction,
            )

        # Logistic growth applied to the WHOLE post-dispersal density — no
        # residents/newcomers distinction left to make (see `dispersal_step`
        # above: everyone already dispersed this year). `growth_step` itself
        # zeroes growth outside `bg` (breeding_ground) internally; no
        # separate forced-evacuation step is needed here since every pixel,
        # breeding or not, already disperses every year regardless.
        new_dens, growth_bl = growth_step(
            SP,
            a_tensor,
            K_is_gpu,
            xs, ys,
            window_size // 2,
            threshold_abundance=presence_threshold / 1e5,
            blacklisted_points=growth_bl,
            breeding_ground=bg,
            plot=False,
        )
        SP = new_dens.cpu()

        snapshots.append((year + 1, SP.cpu().numpy().copy()))

    result = SP.cpu().numpy()

    if plot:
        _plot_simulation_steps(snapshots, hs_t.cpu().numpy(), presence_threshold)

    return result


def crop_to_mask(
    arrays: "list[np.ndarray]",
    mask: np.ndarray,
    padding: int = 50,
) -> "tuple[list[np.ndarray], tuple[int, int]]":
    """Crop a list of co-registered arrays to the bounding box of *mask* > 0.

    Parameters
    ----------
    arrays:
        List of 2-D arrays (same shape) to crop — e.g. ``[hs, init_distrib]``.
    mask:
        Binary array defining the region of interest (non-zero = inside).
    padding:
        Number of pixels added as margin on every side of the bounding box.

    Returns
    -------
    cropped : list[np.ndarray]
        Each input array cropped to ``[r0:r1, c0:c1]``.
    origin : (int, int)
        ``(r0, c0)`` — the top-left corner of the crop in the original array
        coordinates.  Use this to remap pixel indices::

            new_row = old_row - r0
            new_col = old_col - c0
    """
    rows, cols = np.where(mask > 0)
    if len(rows) == 0:
        raise ValueError("mask contains no non-zero pixels — cannot crop.")
    H, W = mask.shape
    r0 = max(0, int(rows.min()) - padding)
    r1 = min(H, int(rows.max()) + padding + 1)
    c0 = max(0, int(cols.min()) - padding)
    c1 = min(W, int(cols.max()) + padding + 1)
    return [a[r0:r1, c0:c1] for a in arrays], (r0, c0)


def _view_from_mask(mask: np.ndarray, pad: int = 20):
    """Return (col_lo, col_hi, row_hi, row_lo) bounding the non-zero region of *mask*.

    *pad* is a fixed pixel margin added on each side (default 20 km).
    """
    rows, cols = np.where(mask > 0)
    if len(rows) == 0:
        return None
    H, W = mask.shape
    r0 = max(0, int(rows.min()) - pad)
    r1 = min(H, int(rows.max()) + pad)
    c0 = max(0, int(cols.min()) - pad)
    c1 = min(W, int(cols.max()) + pad)
    return c0, c1, r1, r0   # xlim lo/hi, ylim lo/hi (imshow y-axis is inverted)


def _plot_simulation_steps(
    snapshots: list[tuple[int, np.ndarray]],
    hs_bg: np.ndarray,
    presence_threshold: float,
) -> None:
    """Render all simulation snapshots as a subplot grid.

    Parameters
    ----------
    snapshots:
        List of ``(year, abundance_map)`` tuples, one per recorded step
        (including the initial distribution at index 0).
    hs_bg:
        Habitat-suitability background (used as grey underlay).
    presence_threshold:
        Minimum abundance value considered as presence; contours are only
        drawn where the distribution exceeds this threshold.
    """
    n = len(snapshots)
    ncols = int(np.ceil(np.sqrt(n)))
    nrows = int(np.ceil(n / ncols))

    # Shared colour scale: maximum across all non-initial steps.
    densities = [snap for _, snap in snapshots[1:]]
    vmax = float(max(d.max() for d in densities)) if densities else 1.0
    vmax = max(vmax, presence_threshold * 2)   # keep scale visible for sparse runs

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4 * ncols, 4 * nrows),
        constrained_layout=True,
    )
    axes_flat = np.array(axes).flatten()

    for i, (year, snap) in enumerate(snapshots):
        ax = axes_flat[i]

        # HS underlay
        ax.imshow(hs_bg, cmap="Greys", alpha=0.5, interpolation="nearest")

        # Density layer — alpha linear with relative abundance
        alpha_arr = np.clip(snap / (vmax + 1e-12), 0.0, 1.0)
        ax.imshow(
            snap,
            cmap="plasma",
            alpha=alpha_arr,
            vmin=0,
            vmax=vmax,
            interpolation="nearest",
            zorder=5,
        )

        # 3 contour levels at vmax/4, vmax/3, vmax/2
        snap_max = float(snap.max())
        if snap_max > 0:
            ax.contour(
                snap,
                levels=[snap_max / (4 - i) for i in range(3)],
                colors="black",
                linewidths=0.8,
                alpha=0.3,
            )

        ax.set_title(f"Year {year}", fontsize=9, pad=3)
        ax.axis("off")

    # Hide any unused subplot slots
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    # Shared colourbar on the right
    sm = plt.cm.ScalarMappable(cmap="plasma", norm=plt.Normalize(vmin=0, vmax=vmax))
    fig.colorbar(sm, ax=axes_flat[:i + 1], shrink=0.6, label="Relative abundance")

    fig.suptitle("Population distribution – all simulation steps", fontsize=12)
    plt.show()


# ---------------------------------------------------------------------------
# Public visualisation helpers
# ---------------------------------------------------------------------------

def show_expansion(sim: "PopulationSimulator") -> None:
    """Plot the year-by-year population expansion from the last :meth:`run` call.

    Displays every recorded snapshot (initial state + one panel per simulated
    year) as a grid of sub-plots with a shared colour scale, identical to the
    ``plot=True`` behaviour inside :func:`run_simulation`.

    Parameters
    ----------
    sim:
        A :class:`PopulationSimulator` instance that has already been run.

    Raises
    ------
    RuntimeError
        If :meth:`~PopulationSimulator.run` has not been called yet.
    """
    if not sim._snapshots:
        raise RuntimeError(
            "No snapshots available.  Call sim.run() before show_expansion()."
        )
    _plot_simulation_steps(sim._snapshots, sim.hs, sim.presence_threshold)


def show_final(sim: "PopulationSimulator") -> None:
    """Three-panel summary of the last :meth:`run` call.

    Panels
    ------
    0 – Habitat suitability (viridis, NaN outside study area)
    1 – Initial distribution (red overlay on grey HS)
    2 – Final simulated distribution (jet, alpha-blended, with contours)

    Parameters
    ----------
    sim:
        A :class:`PopulationSimulator` instance that has already been run.

    Raises
    ------
    RuntimeError
        If :meth:`~PopulationSimulator.run` has not been called yet.
    """
    if sim._last_result is None:
        raise RuntimeError(
            "No result available.  Call sim.run() before show_final()."
        )

    hs       = sim.hs
    init     = sim._last_init_distrib
    final    = sim._last_result
    thresh   = sim.presence_threshold
    n_steps  = len(sim._snapshots) - 1   # excludes the t=0 entry

    # Use hs > 0 to hide impassable / out-of-extent pixels.
    hs_display = np.where(hs > 0, hs, np.nan)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

    # ── Panel 0: Habitat suitability ────────────────────────────────────────
    axes[0].set_title("Habitat suitability")
    im0 = axes[0].imshow(hs_display, cmap="viridis", interpolation="nearest")
    plt.colorbar(im0, ax=axes[0], shrink=0.8, label="Suitability [0-1]")

    # ── Panel 1: Initial distribution ───────────────────────────────────────
    axes[1].set_title("Initial distribution")
    axes[1].imshow(hs_display, cmap="Greys", alpha=0.5, interpolation="nearest")
    axes[1].imshow(
        np.where(init > 0, init, np.nan),
        cmap="Reds",
        alpha=0.9,
        vmin=0,
        vmax=thresh * 2,
        interpolation="nearest",
    )

    # ── Panel 2: Simulated distribution ─────────────────────────────────────
    axes[2].set_title(f"Simulated distribution (t = {n_steps} years)")
    axes[2].imshow(hs_display, cmap="Greys", alpha=0.5, interpolation="nearest")
    maxval = float(np.nanmax(final)) if np.nanmax(final) > 0 else 1.0
    axes[2].imshow(
        final,
        cmap="plasma",
        alpha=np.clip(final / maxval, 0.0, 1.0),
        vmin=0,
        vmax=maxval,
        interpolation="nearest",
        zorder=5,
    )
    axes[2].contour(
        final,
        levels=[maxval / (4 - i) for i in range(3)],
        colors="black",
        linewidths=0.8,
        alpha=0.3,
    )
    plt.colorbar(
        plt.cm.ScalarMappable(cmap="plasma", norm=plt.Normalize(0, maxval)),
        ax=axes[2],
        shrink=0.8,
        label="Relative abundance",
    )

    for ax in axes:
        ax.axis("off")

    plt.show()


# ---------------------------------------------------------------------------
# Object-oriented wrapper
# ---------------------------------------------------------------------------

class PopulationSimulator:
    """High-level interface for the PARADIS dispersal simulation.

    Parameters
    ----------
    hs:
        Habitat-suitability map.
    ewalk, n, r:
        Dispersal parameters.
    tgrowth:
        Characteristic growth time.
    carrying_capacity_params:
        ``(L, k, x0)``.
    presence_threshold:
        Presence threshold.
    cache_dir:
        Optional path to a directory for the disk-backed kernel cache.
        When set, LU factorisations are written to ``.npz`` files on first
        use and reloaded on subsequent runs — avoiding both RAM overflow on
        large maps and the O(N^3) refactorisation cost on repeated runs.
        If ``None`` (default), a bounded in-memory LRU
        (:class:`~paradis.simulation.kernel_cache.MemoryKernelCache`,
        auto-sized from *ram_fraction*) is used instead — evicted entries
        are silently recomputed, RAM never overflows regardless of map
        size. See :class:`~paradis.simulation.DiskKernelCache` for the
        disk-backed alternative.
    max_mem_windows:
        Maximum LU factorisations held in RAM when *cache_dir* IS set
        (disk-backed cache). Each entry for ``window_size=70`` costs
        ~97 MB; the default of 16 caps RAM at roughly 1.6 GB. Ignored when
        *cache_dir* is ``None`` — see *ram_fraction* instead.
    ram_fraction:
        Only used when *cache_dir* is ``None`` (the default, in-memory
        cache). Fraction of currently FREE RAM the auto-sized
        `MemoryKernelCache` is allowed to occupy (default 0.9 = 90%) —
        computed once at the start of `run()` via
        `kernel_cache._compute_max_mem_windows`, so it adapts to however
        many windows a given map/`window_size` actually needs, unlike
        `max_mem_windows` (a fixed count, only relevant for the disk
        cache). Lower this if you're running other memory-heavy work
        alongside the simulation.

    Examples
    --------
    >>> import numpy as np
    >>> from paradis.simulation import PopulationSimulator
    >>> hs = np.random.rand(200, 200).astype("float32")
    >>> init = np.zeros_like(hs)
    >>> init[90:110, 90:110] = 0.05
    >>> sim = PopulationSimulator(hs, ewalk=50, n=500, r=0.05,
    ...                           tgrowth=7.5, carrying_capacity_params=(0.1, 5, 0.5),
    ...                           presence_threshold=0.02)
    >>> final = sim.run(n_steps=10, init_distrib=init)
    """

    def __init__(
        self,
        hs: "np.ndarray | str | pathlib.Path",
        ewalk: float,
        n: float,
        r: float,
        tgrowth: float,
        carrying_capacity_params: tuple,
        presence_threshold: float,
        cache_dir: str | pathlib.Path | None = None,
        max_mem_windows: int = 16,
        ram_fraction: float = 0.9,
        solver: str = 'lu',
        gmres_tol: float = 1e-5,
    ) -> None:
        if not isinstance(hs, np.ndarray):
            from paradis.io.raster import load_hs
            hs = load_hs(hs)
        self.hs = hs
        self.ewalk = ewalk
        self.n = n
        self.r = r
        self.tgrowth = tgrowth
        self.carrying_capacity_params = carrying_capacity_params
        self.presence_threshold = presence_threshold

        # Disk cache configuration (used lazily inside run()).
        # When None, an adaptive MemoryKernelCache is used instead.
        self._cache_dir = (
            pathlib.Path(cache_dir).expanduser() if cache_dir is not None else None
        )
        self._max_mem_windows = max_mem_windows
        self._ram_fraction = ram_fraction
        self._solver    = solver
        self._gmres_tol = gmres_tol

        # State populated by run() — used by show_expansion / show_final.
        self._snapshots:         list                    = []
        self._last_init_distrib: np.ndarray | None       = None
        self._last_result:       np.ndarray | None       = None

    def suggest_window_params(
        self,
        coverage: float = 0.99,
        min_window: int = 10,
        verbose: bool = True,
    ) -> tuple[int, int]:
        """Recommend *window_size* and *sub_window* for this simulator.

        Delegates to :func:`~paradis.simulation.tuning.suggest_window_params`
        using the dispersal parameters already stored on this object.

        Parameters
        ----------
        coverage:
            Required fraction of dispersal mass captured inside the window.
        min_window:
            Smallest *window_size* to consider.
        verbose:
            Print the recommendation.

        Returns
        -------
        window_size, sub_window : int, int

        Example
        -------
        >>> ws, sw = sim.suggest_window_params()
        >>> final = sim.run(n_steps=10, window_size=ws, sub_window=sw)
        """
        from paradis.simulation.tuning import suggest_window_params as _suggest
        return _suggest(
            self.hs, self.ewalk, self.n, self.r,
            coverage=coverage,
            min_window=min_window,
            verbose=verbose,
        )

    def clear_kernel_cache(self, *, disk: bool = False) -> None:
        """Signal that cached LU factorisations should be discarded.

        The in-memory cache (MemoryKernelCache) is re-created at the start
        of every :meth:`run` call, so it is always clean.  This method is
        useful only when *cache_dir* was set and you want to wipe the
        on-disk ``.npz`` files (e.g. after changing dispersal parameters).

        Parameters
        ----------
        disk:
            When *cache_dir* was provided and ``disk=True``, delete all
            ``.npz`` files in the cache directory.
        """
        if disk and self._cache_dir is not None:
            import shutil
            shutil.rmtree(self._cache_dir, ignore_errors=True)
            self._cache_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        n_steps: int = 35,
        init_distrib: "np.ndarray | str | pathlib.Path | None" = None,
        **kwargs,
    ) -> np.ndarray:
        """Run the simulation.

        Parameters
        ----------
        n_steps:
            Number of time steps.
        init_distrib:
            Initial abundance map — a 2-D array, OR a path (``str``/
            ``pathlib.Path``) to a ``.npy`` file holding one (loaded via
            ``numpy.load``, same convenience `hs` already gets via
            `load_hs` in ``__init__``), e.g. a founder-patch map exported
            from another tool/run. If ``None``, a small central patch at
            the presence threshold is used.
        **kwargs:
            Forwarded to :func:`run_simulation`.

        Returns
        -------
        numpy.ndarray
            Final abundance map.
        """
        if init_distrib is None:
            h, w = self.hs.shape
            init_distrib = np.zeros_like(self.hs)
            init_distrib[h // 2 - 5: h // 2 + 5, w // 2 - 5: w // 2 + 5] = (
                self.presence_threshold
            )
        elif not isinstance(init_distrib, np.ndarray):
            init_distrib = np.load(init_distrib)

        # Build the kernel cache for this run.
        #
        # cache_dir is set → DiskKernelCache: bounded RAM + cross-run
        #   persistence on disk.  .npz files from previous runs are reused.
        #
        # cache_dir is None (default) → MemoryKernelCache: size is chosen
        #   automatically to consume at most 90 % of currently free RAM.
        #   Evicted entries are silently recomputed; RAM never overflows.
        ws = kwargs.get("window_size", 70)   # default matches run_simulation

        if self._cache_dir is not None:
            from paradis.simulation.kernel_cache import DiskKernelCache
            kernel_cache = DiskKernelCache(
                self._cache_dir,
                self.hs,
                self.ewalk,
                self.n,
                self.r,
                ws,
                max_mem_windows=self._max_mem_windows,
            )
        else:
            # Bounded in-memory LRU, auto-sized from `ram_fraction` of
            # currently free RAM — NOT a plain dict (which would hold every
            # window's factorisation simultaneously with no cap: fine for a
            # small map, but silently unbounded RAM growth — potentially
            # gigabytes on a large map with many evaluation windows — on a
            # large one, easily enough to have the process killed by the OS
            # with no Python-level error at all). Evicted entries are safe:
            # `dispersal_step` (see its "#3: Collect all active windows"
            # section) already lazily rebuilds any cache miss on demand
            # (`if (x0, y0) not in kernel_cache: ... kernel_cache[...] = ...`),
            # so eviction only costs a recompute, never a KeyError.
            from paradis.simulation.kernel_cache import (
                MemoryKernelCache, _compute_max_mem_windows,
            )
            n_mem = _compute_max_mem_windows(ws, ram_fraction=self._ram_fraction)
            kernel_cache = MemoryKernelCache(max_mem_windows=n_mem)

        self._last_init_distrib = init_distrib.copy()
        result = run_simulation(
            hs=self.hs,
            init_distrib=init_distrib,
            ewalk=self.ewalk,
            n=self.n,
            r=self.r,
            tgrowth=self.tgrowth,
            carrying_capacity_params=self.carrying_capacity_params,
            presence_threshold=self.presence_threshold,
            n_steps=n_steps,
            kernel_cache=kernel_cache,
            solver=self._solver,
            gmres_tol=self._gmres_tol,
            _snapshot_store=self._snapshots,
            **kwargs,
        )
        self._last_result = result
        return result

    def __repr__(self) -> str:
        return (
            f"PopulationSimulator(hs={self.hs.shape}, "
            f"Ew={self.ewalk:.1f}, n={self.n:.1f}, r={self.r:.5f})"
        )
