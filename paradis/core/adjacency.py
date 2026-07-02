"""Adjacency matrix construction from 2-D habitat-suitability rasters.

Two implementations are provided:

* :func:`adjacency_matrix` – pure NumPy/SciPy reference implementation.
* :func:`adjacency_matrix_torch` – fully vectorised PyTorch version (GPU-ready).

Each pixel is connected to its four cardinal neighbours (N, S, E, W).  The
edge weight between two adjacent pixels is the average of their habitat-
suitability values.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.sparse import lil_matrix

from paradis._device import device


# ---------------------------------------------------------------------------
# NumPy reference implementation
# ---------------------------------------------------------------------------

def adjacency_matrix(image: np.ndarray) -> np.ndarray:
    """Build a dense adjacency matrix for a 2-D habitat-suitability raster.

    Parameters
    ----------
    image:
        2-D array of habitat-suitability values.  ``NaN`` entries are treated
        as zero (impassable cells, e.g. ocean).

    Returns
    -------
    numpy.ndarray
        Square float32 array of shape ``(N, N)`` where ``N = rows × cols``.
        Entry ``[i, j]`` holds the average suitability of pixels ``i`` and
        ``j`` when they are 4-neighbours, and zero otherwise.
    """
    rows, cols = image.shape
    image = image.copy()
    image[np.isnan(image)] = 0.0
    N = rows * cols
    adj = lil_matrix((N, N))

    def _idx(i: int, j: int) -> int:
        return i * cols + j

    for i in range(rows):
        for j in range(cols):
            node = _idx(i, j)
            neighbours = []
            if j > 0:
                neighbours.append((_idx(i, j - 1), (image[i, j] + image[i, j - 1]) / 2))
            if j < cols - 1:
                neighbours.append((_idx(i, j + 1), (image[i, j] + image[i, j + 1]) / 2))
            if i > 0:
                neighbours.append((_idx(i - 1, j), (image[i, j] + image[i - 1, j]) / 2))
            if i < rows - 1:
                neighbours.append((_idx(i + 1, j), (image[i, j] + image[i + 1, j]) / 2))
            for neighbour, weight in neighbours:
                adj[node, neighbour] = weight
                adj[neighbour, node] = weight

    return adj.toarray().astype(np.float32)


# ---------------------------------------------------------------------------
# PyTorch vectorised implementation
# ---------------------------------------------------------------------------

def adjacency_matrix_torch(image: torch.Tensor) -> torch.Tensor:
    """Vectorised adjacency matrix construction (PyTorch, GPU-ready).

    Parameters
    ----------
    image:
        2-D :class:`torch.Tensor` of habitat-suitability values on any
        device.  ``NaN`` entries are set to zero.

    Returns
    -------
    torch.Tensor
        Dense float32 tensor of shape ``(N, N)`` on the same device as
        *image*.
    """
    dev = image.device
    image = image.clone().float()
    image[torch.isnan(image)] = 0.0

    rows, cols = image.shape
    N = rows * cols

    def _ij_to_idx(i: torch.Tensor, j: torch.Tensor) -> torch.Tensor:
        return i * cols + j

    i_grid = torch.arange(rows, device=dev).view(-1, 1).expand(-1, cols)
    j_grid = torch.arange(cols, device=dev).view(1, -1).expand(rows, -1)
    idx = _ij_to_idx(i_grid, j_grid)

    def _connect(di: int, dj: int):
        i2 = i_grid + di
        j2 = j_grid + dj
        valid = (i2 >= 0) & (i2 < rows) & (j2 >= 0) & (j2 < cols)
        src = idx[valid]
        dst = _ij_to_idx(i2[valid], j2[valid])
        w = (image[i_grid[valid], j_grid[valid]] + image[i2[valid], j2[valid]]) / 2
        return src.flatten(), dst.flatten(), w.flatten()

    all_src, all_dst, all_w = [], [], []
    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        src, dst, w = _connect(di, dj)
        # symmetric: add both directions
        all_src.append(torch.cat([src, dst]))
        all_dst.append(torch.cat([dst, src]))
        all_w.append(torch.cat([w, w]))

    row_idx = torch.cat(all_src)
    col_idx = torch.cat(all_dst)
    weights = torch.cat(all_w).to(dtype=torch.float32, device=dev)

    adj = torch.zeros((N, N), dtype=torch.float32, device=dev)
    adj[row_idx, col_idx] = weights
    return adj
