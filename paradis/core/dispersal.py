"""Core dispersal kernel computation.

This module implements the random-walk–based dispersal model used in PARADIS.
The key idea is that each individual performs a geometric random walk on the
habitat graph.  The expected number of steps ``Ew`` controls dispersal
distance; the exponent ``Tolerance`` (``n``) controls risk-avoidance
behaviour; and ``r`` scales habitat suitability into a per-step
survival probability.

Key functions
-------------
:func:`row_normalise`
    Compute row sums of a square matrix (helper).
:func:`weighted_adjacency`
    Build the risk-weighted transition matrix ``W*``.
:func:`dispersal_rules` (:func:`Disprule`)
    Invert ``(I - p * W*)`` to obtain the full dispersal kernel.
:func:`core_dispersal` (:func:`core_disp`)
    Apply a pre-computed transition matrix to an initial density vector.
:func:`dispersal_kernel` (:func:`Kdisp2`)
    One-call convenience function returning kernel + mean distance + survival.
:func:`dispersal_kernel_fast` (:func:`Kdisp3`)
    Like :func:`dispersal_kernel` but returns only the kernel matrix (slightly
    faster, suitable for gradient-based optimisation).
:func:`single_window_dispersal` (:func:`disp`)
    Apply dispersal to a single raster window.

Classes
-------
:class:`DispersalKernel`
    Object-oriented wrapper around :func:`dispersal_kernel`.
"""

from __future__ import annotations

import numpy as np
import torch

from paradis._device import device, paradis_compile


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def row_normalise(M: torch.Tensor) -> torch.Tensor:
    """Return a matrix whose ``[i, j]`` entry equals the row-sum of row ``i``.

    Parameters
    ----------
    M:
        Square float tensor of shape ``(n, n)``.

    Returns
    -------
    torch.Tensor
        Shape ``(n, n)`` – each row filled with the sum of that row in *M*.
    """
    M = M.float()
    n = M.shape[0]
    ones = torch.ones((n, 1), device=M.device)
    return M @ ones @ ones.T


def weighted_adjacency(W: torch.Tensor, n: float | torch.Tensor = 0.7) -> torch.Tensor:
    """Build the risk-weighted, row-normalised transition matrix ``W*``.

    The risk-weighted matrix is defined element-wise as::

        W*[i,j] = W[i,j] * W[i,j]^n / sum_k(W[i,k]^n)

    Parameters
    ----------
    W:
        Adjacency matrix with edge weights in ``[0, 1]``.
    n:
        Exponent controlling risk avoidance.  Higher values → more
        selective movement towards high-quality habitat.

    Returns
    -------
    torch.Tensor
        Row-normalised weighted transition matrix.
    """
    # Clamp before raising to `n` to avoid log(0) = -inf in the
    # backward pass (gradient of x^t w.r.t. t is x^t * log(x) → NaN at x=0).
    fW = W.clamp(min=1e-7) ** n + 1e-30
    return W * fW / row_normalise(fW)


@paradis_compile
def dispersal_rules(
    W: torch.Tensor,
    ewalk: float | torch.Tensor,
    n: float | torch.Tensor = 1000.0,
) -> torch.Tensor:
    """Compute the full dispersal transition matrix.

    Solves ``(I - p * W*)``:sup:`-1` * (1 - p), where
    ``p = Ew / (1 + Ew)`` is the step probability and ``W*`` is the
    risk-weighted adjacency matrix.

    Parameters
    ----------
    W:
        Habitat-suitability–weighted adjacency matrix (edge weights in
        ``[0, 1]``).
    ewalk:
        Expected number of random-walk steps (mean dispersal path length).
    n:
        Risk-avoidance exponent.

    Returns
    -------
    torch.Tensor
        Square transition matrix of shape ``(N, N)``.
    """
    Wstar = weighted_adjacency(W, n)
    k = W.shape[0]
    eye = torch.eye(k, k, device=W.device, dtype=W.dtype)
    p = ewalk / (1.0 + ewalk)
    # Small Tikhonov regularisation (1e-5 * I) prevents near-singular inversions
    # on windows where Wstar has a spectral radius close to 1.
    return ((1.0 - p) * torch.linalg.inv(eye * (1.0 + 1e-5) - Wstar * p)).float()


@paradis_compile
def core_dispersal(
    Wstar: torch.Tensor,
    D0: torch.Tensor,
    ewalk: float | torch.Tensor,
) -> torch.Tensor:
    """Apply a single dispersal step via linear solve.

    Instead of building the full kernel matrix, solves the linear system
    ``(I - p * Wstar)^T * Dt = D0`` directly.

    Parameters
    ----------
    Wstar:
        Risk-weighted transition matrix of shape ``(N, N)``.
    D0:
        Initial density row-vector of shape ``(1, N)``.
    ewalk:
        Expected random-walk length.

    Returns
    -------
    torch.Tensor
        Output density vector of shape ``(N, 1)``.
    """
    p = torch.tensor(ewalk / (1.0 + ewalk), device=Wstar.device, dtype=torch.float32)
    constant = 1.0 / (1.0 - p)
    Me = (torch.eye(Wstar.shape[0], device=Wstar.device, dtype=torch.float32) - p * Wstar) * constant
    return torch.linalg.solve(Me.T, D0.to(dtype=torch.float32).T)


@paradis_compile
def dispersal_kernel(
    adj_hs: torch.Tensor | np.ndarray,
    r: float | torch.Tensor = 0.05,
    n: float | torch.Tensor = 1000.0,
    ewalk: float | torch.Tensor = 1000.0,
    dist_matrix: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the dispersal kernel together with mean distance and survival.

    Parameters
    ----------
    adj_hs:
        Adjacency matrix of habitat suitability.
    r:
        Risk scaling exponent (``survival = HS^r`` per step).
    n:
        Risk-avoidance exponent.
    ewalk:
        Expected number of steps.
    dist_matrix:
        Pre-computed pairwise Euclidean distance matrix of shape ``(N, N)``.
        If ``None`` the distances are not computed and ``mean_dist`` is
        returned as ``NaN``.

    Returns
    -------
    kernel : torch.Tensor
        Full ``(N, N)`` dispersal transition matrix.
    mean_dist : torch.Tensor
        Expected dispersal distance (scalar).
    survival : torch.Tensor
        Mean probability of surviving dispersal (scalar).
    """
    if not isinstance(adj_hs, torch.Tensor):
        adj_hs = torch.tensor(adj_hs, dtype=torch.float32, device=device)
    # Clamp before raising to `r` to avoid log(0) NaN in the backward pass.
    W = adj_hs.clamp(min=1e-7) ** r
    kernel = dispersal_rules(W, ewalk, n=n)
    del W, adj_hs
    if dist_matrix is not None:
        mean_dist = (kernel * dist_matrix).sum(dim=1).mean()
    else:
        mean_dist = torch.tensor(float("nan"))
    survival = kernel.sum(dim=1).mean()
    return kernel, mean_dist, survival


@paradis_compile
def dispersal_kernel_fast(
    adj_hs: torch.Tensor | np.ndarray,
    r: float | torch.Tensor = 0.05,
    n: float | torch.Tensor = 1000.0,
    ewalk: float | torch.Tensor = 1000.0,
) -> torch.Tensor:
    """Compute only the dispersal transition matrix (no distance/survival).

    Lighter version of :func:`dispersal_kernel` suitable for use inside
    gradient-based optimisation loops.

    Parameters
    ----------
    adj_hs:
        Adjacency matrix.
    r:
        Risk scaling.
    n:
        Risk-avoidance exponent.
    ewalk:
        Expected number of steps.

    Returns
    -------
    torch.Tensor
        Full ``(N, N)`` dispersal transition matrix.
    """
    if not isinstance(adj_hs, torch.Tensor):
        adj_hs = torch.tensor(adj_hs, dtype=torch.float32, device=device)
    # Clamp before raising to `r` to avoid log(0) NaN in the backward pass.
    W = adj_hs.clamp(min=1e-7) ** r
    kernel = dispersal_rules(W, ewalk, n=n)
    del W, adj_hs
    return kernel


def single_window_dispersal(
    adj_hs: torch.Tensor | np.ndarray,
    D0: torch.Tensor | np.ndarray,
    r: float = 0.05,
    n: float = 1000.0,
    ewalk: float = 1000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Disperse an initial density over a single raster window.

    Parameters
    ----------
    adj_hs:
        Adjacency matrix of the local habitat window.
    D0:
        Initial density (flat array of length ``N``).
    r:
        Risk scaling exponent.
    n:
        Risk-avoidance exponent.
    ewalk:
        Expected number of steps.

    Returns
    -------
    Dt : torch.Tensor
        2-D density map after dispersal, same spatial extent as *adj_hs*.
    border_mask : torch.Tensor
        Binary mask of border pixels (used for overdispersion tracking).
    """
    if not isinstance(adj_hs, torch.Tensor):
        adj_hs = torch.tensor(adj_hs, dtype=torch.float32, device=device)
    if not isinstance(D0, torch.Tensor):
        D0 = torch.tensor(D0, dtype=torch.float32, device=device)

    size = adj_hs.shape[0]
    D0 = D0.reshape(1, size)
    W = adj_hs ** r
    Wstar = weighted_adjacency(W, n)

    nside = int(size ** 0.5)
    border = torch.zeros((nside, nside), device=device, dtype=torch.float32)
    border[0:nside - 1, 0] = 1
    border[0, 0:nside - 1] = 1
    border[nside - 1, 0:nside - 1] = 1
    border[0:nside, nside - 1] = 1
    i_b, j_b = torch.where(border == 1)
    for ib, jb in zip(i_b, j_b):
        idx = int(ib * nside + jb)
        Wstar[idx, :] = 0
        Wstar[idx, idx] = 1

    Dt = core_dispersal(Wstar, D0, ewalk)
    Dt = Dt.reshape(nside, nside)
    return Dt, border

def distance_matrix_from_raster(n: int) -> torch.Tensor:
    """Euclidean pairwise distance matrix for an *n × n* raster.

    Parameters
    ----------
    n:
        Size of one side of the square raster.

    Returns
    -------
    torch.Tensor
        Float32 tensor of shape ``(n², n²)``.
    """
    from scipy.spatial.distance import cdist

    x, y = np.meshgrid(np.arange(n), np.arange(n))
    coords = np.stack([x.ravel(), y.ravel()], axis=1)
    return torch.tensor(cdist(coords, coords, metric="euclidean"), dtype=torch.float32)


# ---------------------------------------------------------------------------
# Object-oriented wrapper
# ---------------------------------------------------------------------------

class DispersalKernel:
    """Object-oriented wrapper for the PARADIS dispersal model.

    Parameters
    ----------
    r:
        Risk scaling exponent (``survival = HS^r``).
    n:
        Risk-avoidance exponent.
    ewalk:
        Expected number of random-walk steps.

    Examples
    --------
    >>> import numpy as np
    >>> from paradis.core.dispersal import DispersalKernel
    >>> hs = np.random.rand(10, 10).astype("float32")
    >>> from paradis.core.adjacency import adjacency_matrix_torch
    >>> import torch
    >>> adj = adjacency_matrix_torch(torch.tensor(hs))
    >>> dk = DispersalKernel(r=0.05, n=500, ewalk=100)
    >>> kernel, mean_dist, survival = dk.compute(adj)
    """

    def __init__(
        self,
        r: float = 0.05,
        n: float = 1000.0,
        ewalk: float = 1000.0,
    ) -> None:
        self.r = r
        self.n = n
        self.ewalk = ewalk

    def compute(
        self,
        adj_hs: torch.Tensor | np.ndarray,
        dist_matrix: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the dispersal kernel for *adj_hs*.

        Parameters
        ----------
        adj_hs:
            Adjacency matrix of habitat suitability.
        dist_matrix:
            Optional pre-computed distance matrix.

        Returns
        -------
        kernel, mean_dist, survival
            See :func:`dispersal_kernel`.
        """
        return dispersal_kernel(
            adj_hs,
            r=self.r,
            n=self.n,
            ewalk=self.ewalk,
            dist_matrix=dist_matrix,
        )

    def __repr__(self) -> str:
        return (
            f"DispersalKernel(r={self.r}, "
            f"n={self.n}, ewalk={self.ewalk})"
        )
