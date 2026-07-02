"""Automatic window-parameter selection for the dispersal simulation engine.

Theory
------
The PARADIS dispersal model is a geometric random walk: an individual takes
K ~ Geometric(1/(1+Ew)) steps, each moving to a neighbouring pixel.
For large Ew, the 2-D Green's function of this walk converges to a
*modified Bessel kernel*:

    K(r) ∝ K₀(r / λ),   λ ≈ √Ew

where K₀ decays as exp(-r/λ) / √r for large r.  Integrating the CDF of
this distribution shows that the radius enclosing a fraction f of the
total dispersal mass is approximately

    r_f ≈ λ · g(f),   g(0.95) ≈ 2.5,   g(0.99) ≈ 3.5,   g(0.999) ≈ 5

So the evaluation window must have side ≥ 2 · r_f = 2 · 3.5 · √Ew ≈ 7 √Ew
to capture 99 % of a central point source's dispersal.

When 7 √Ew is small enough (win_pixels ≤ 41, meaning N ≤ 1 681 unknowns),
:func:`suggest_window_params` runs a fast numerical verification: it places a
unit point source at the peak-HS pixel, solves the dispersal system once, and
measures the border-escape fraction.  A linear scan finds the smallest window
that keeps escape below (1 - coverage).  For larger Ew, the analytical bound
is used directly (the numerical test would be too slow to be a
"pre-simulation" step).

sub_window choice
-----------------
``sub_window`` is the stride between window centres.  It acts as the
non-overlapping tile size for *source* pixels: each pixel contributes its
density to exactly one window (the one whose sub-window contains it).
Setting ``sub_window ≈ window_size // 3`` provides generous overlap between
adjacent *destination* regions, ensuring smooth accumulation while keeping
the window count manageable.  Constraints: sub_window must be strictly less
than window_size.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from paradis._device import device
from paradis.core.adjacency import adjacency_matrix_torch
from paradis.core.dispersal import weighted_adjacency


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def suggest_window_params(
    hs: np.ndarray,
    ewalk: float,
    n: float,
    r: float,
    *,
    coverage: float = 0.99,
    min_window: int = 10,
    verbose: bool = True,
) -> tuple[int, int]:
    """Recommend *window_size* and *sub_window* for a dispersal simulation.

    Uses the two-stage approach described in the module docstring.

    Parameters
    ----------
    hs:
        2-D habitat-suitability array (H × W).
    ewalk:
        Expected number of random-walk steps (mean dispersal path length).
    n:
        Risk-avoidance exponent.
    r:
        Risk-scaling exponent.
    coverage:
        Required fraction of a point-source's dispersal mass to capture
        inside the window.  Default 0.99 (99 %).
    min_window:
        Smallest *window_size* to consider (must be even, adjusted if not).
    verbose:
        If ``True``, print the recommended values and a brief justification.

    Returns
    -------
    window_size : int
        Even integer; side length of the evaluation window in pixels.
    sub_window : int
        Integer strictly less than *window_size*; stride between window
        centres.  Approximately ``window_size // 3``.

    Examples
    --------
    >>> import numpy as np
    >>> from paradis.simulation.tuning import suggest_window_params
    >>> hs = np.random.rand(120, 120).astype("float32")
    >>> ws, sw = suggest_window_params(hs, ewalk=1137, n=2094,
    ...                                r=0.016, verbose=False)
    >>> ws >= sw
    True
    """
    H, W = hs.shape

    # Enforce even min_window
    if min_window % 2 != 0:
        min_window += 1

    # Hard cap: the window cannot be larger than the map itself (even)
    map_min = min(H, W)
    win_max = map_min if map_min % 2 == 0 else map_min - 1

    # ------------------------------------------------------------------
    # Stage 1 – Analytical estimate
    # ------------------------------------------------------------------
    lam = math.sqrt(max(ewalk, 1.0))
    # 99th-percentile radius for the Bessel dispersal kernel ≈ 3.5 λ
    # → window side = 2 × radius, rounded up to even.
    win_analytical = 2 * int(math.ceil(3.5 * lam))
    # Clamp and ensure even
    win_analytical = max(win_analytical, min_window)
    if win_analytical % 2 != 0:
        win_analytical += 1
    win_analytical = min(win_analytical, win_max)

    win_pixels_analytical = 2 * (win_analytical // 2) + 1  # = win_analytical + 1

    # ------------------------------------------------------------------
    # Stage 2 – Numerical verification (only when cheap enough)
    # ------------------------------------------------------------------
    # LU factorisation of an N×N system scales as O(N³).
    # For win_pixels = 41, N = 1 681, which factorises in ~10 ms on CPU.
    # For win_pixels = 43, N = 1 849, already ~50 ms per test — skip.
    NUMERICAL_LIMIT = 41   # max win_pixels for numerical search

    if win_analytical >= win_max:
        # Dispersal scale ≥ map size: use the full map as one window.
        window_size = win_max
        source = "dispersal scale exceeds map - using full map"
    elif win_pixels_analytical > NUMERICAL_LIMIT:
        # Analytical estimate already requires large windows; trust it.
        window_size = win_analytical
        source = f"analytical (sqrt(Ew) = {lam:.1f} px)"
    else:
        # Fast numerical search: binary-search the smallest even window
        # whose border-escape fraction ≤ (1 − coverage).
        window_size = _numerical_window_search(
            hs=hs,
            ewalk=ewalk,
            n=n,
            r=r,
            win_lo=min_window,
            win_hi=win_analytical,
            coverage=coverage,
        )
        source = "numerical point-source test"

    # ------------------------------------------------------------------
    # sub_window: ~1/3 of window_size, strictly < window_size
    # ------------------------------------------------------------------
    sub_window = max(3, window_size // 3)
    # Ensure sub_window < window_size with room for at least 2 overlap pixels
    sub_window = min(sub_window, window_size - 2)

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    if verbose:
        n_x = math.ceil(H / sub_window)
        n_y = math.ceil(W / sub_window)
        print(
            f"[paradis] suggest_window_params: window_size={window_size}, "
            f"sub_window={sub_window}"
        )
        print(
            f"       Source: {source}"
        )
        print(
            f"       sqrt(ewalk) = {lam:.1f} px  |  map {H}x{W}  |"
            f"  ~{n_x * n_y} windows/step"
        )

    return window_size, sub_window


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _border_escape_fraction(
    hs_win: torch.Tensor,
    ewalk: float,
    n: float,
    r: float,
    border_2d: torch.Tensor,
) -> float:
    """Return fraction of a central point source that escapes to the border.

    Builds the dispersal linear system for *hs_win*, solves it for a unit
    source at the centre pixel, and returns

        border_density / total_density

    A high value means the window is too small to contain the dispersal.

    Parameters
    ----------
    hs_win:
        ``win_pixels × win_pixels`` float32 HS tensor.
    ewalk, n, r:
        Dispersal parameters.
    border_2d:
        Pre-allocated binary border mask of the same spatial shape.

    Returns
    -------
    float
        Escape fraction in [0, 1].
    """
    n_flat = hs_win.numel()
    dev    = hs_win.device

    # Build dispersal system (same logic as _build_window_lu in engine.py)
    adj   = adjacency_matrix_torch(hs_win)
    Wstar = weighted_adjacency(adj ** r, n)

    n_cols = border_2d.shape[1]
    i_b, j_b = torch.where(border_2d == 1)
    for ib, jb in zip(i_b.tolist(), j_b.tolist()):
        idx = ib * n_cols + jb
        Wstar[idx, :] = 0.0
        Wstar[idx, idx] = 1.0

    p        = float(ewalk) / (1.0 + float(ewalk))
    constant = 1.0 / (1.0 - p)
    eye      = torch.eye(n_flat, device=dev, dtype=torch.float32)
    Me       = (eye - p * Wstar) * constant

    # Unit point source at the window centre
    D0 = torch.zeros(n_flat, device=dev, dtype=torch.float32)
    D0[n_flat // 2] = 1.0

    # Single linear solve (no LU cache needed — we solve once per test window)
    with torch.no_grad():
        Dt_flat = torch.linalg.solve(Me.T, D0.unsqueeze(1)).squeeze(1)

    n_side = int(n_flat ** 0.5)
    Dt = Dt_flat.reshape(n_side, n_side)

    total = float(Dt.sum())
    if total <= 0.0:
        return 1.0
    border_mass = float((Dt * border_2d).sum())
    return border_mass / total


def _numerical_window_search(
    hs: np.ndarray,
    ewalk: float,
    n: float,
    r: float,
    win_lo: int,
    win_hi: int,
    coverage: float,
) -> int:
    """Linear scan for the smallest even window_size in [win_lo, win_hi]
    whose border-escape fraction is ≤ (1 - coverage).

    Tests are performed with a point source at the highest-HS pixel (the
    most dispersive location in the landscape), giving a conservative bound
    that holds everywhere.

    Parameters
    ----------
    hs:
        Full H×W HS array (used to find the peak-HS location and extract
        local patches).
    win_lo, win_hi:
        Search bounds (even integers).
    coverage:
        Required capture fraction.

    Returns
    -------
    int
        Smallest even window_size achieving the requested coverage, or
        *win_hi* if no smaller size suffices.
    """
    H, W = hs.shape
    hs_t = torch.tensor(hs, dtype=torch.float32, device=device)

    # Use the peak-HS pixel as the test centre (maximises dispersal spread).
    peak_idx = int(hs_t.argmax())
    cx, cy   = peak_idx // W, peak_idx % W

    target_escape = 1.0 - coverage

    # Ensure win_lo is even
    if win_lo % 2 != 0:
        win_lo += 1

    for win in range(win_lo, win_hi + 2, 2):   # step by 2 to keep win even
        win = min(win, win_hi)
        mhalf      = win // 2
        win_pixels = 2 * mhalf + 1

        # Extract (possibly padded) HS patch centred on (cx, cy)
        mx_lo = cx - mhalf;  mx_hi = cx + mhalf + 1
        my_lo = cy - mhalf;  my_hi = cy + mhalf + 1
        cx_lo = max(0, mx_lo);  cx_hi = min(H, mx_hi)
        cy_lo = max(0, my_lo);  cy_hi = min(W, my_hi)
        wx_lo = cx_lo - mx_lo;  wx_hi = wx_lo + (cx_hi - cx_lo)
        wy_lo = cy_lo - my_lo;  wy_hi = wy_lo + (cy_hi - cy_lo)

        hs_win = torch.zeros((win_pixels, win_pixels), device=device,
                             dtype=torch.float32)
        hs_win[wx_lo:wx_hi, wy_lo:wy_hi] = hs_t[cx_lo:cx_hi, cy_lo:cy_hi]

        # Border mask (same for every window of this size)
        border_2d = torch.zeros((win_pixels, win_pixels), device=device,
                                dtype=torch.float32)
        border_2d[0:win_pixels - 1, 0]               = 1
        border_2d[0,                0:win_pixels - 1] = 1
        border_2d[win_pixels - 1,   0:win_pixels - 1] = 1
        border_2d[0:win_pixels,     win_pixels - 1]   = 1

        frac = _border_escape_fraction(hs_win, ewalk, n, r, border_2d)

        if frac <= target_escape:
            return win

    return win_hi   # fallback: largest tried size
