"""Kernel caches for dispersal-window LU factorisations.

Two cache backends are provided as drop-in replacements for the plain
``dict`` used as *kernel_cache* in
:func:`~paradis.simulation.engine.dispersal_step`:

MemoryKernelCache (default)
    Bounded in-memory LRU.  Size is set automatically to 90% of available
    RAM so the process never freezes due to swap.  Evicted entries are
    silently recomputed on next access — no disk I/O, no overflow.

DiskKernelCache (optional, for cross-run persistence)
    LRU in front of on-disk ``.npz`` files.  Useful when the same
    simulation is run repeatedly with the same parameters and you want to
    skip the O(N^3) LU refactorisation on every run.  Note the large disk
    footprint: ~97 MB per window at ``window_size=70``.
"""

from __future__ import annotations

import hashlib
import pathlib
import shutil
from collections import OrderedDict

import numpy as np
import torch

from paradis._device import device as _device


# ---------------------------------------------------------------------------
# RAM helpers
# ---------------------------------------------------------------------------

def _available_ram_bytes() -> int:
    """Return available (free) physical RAM in bytes.

    Uses :mod:`psutil` when installed; falls back to a conservative 4 GB
    estimate so the package does not hard-depend on psutil.
    """
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except ImportError:
        pass
    return 4 * 1024 ** 3   # conservative fallback: 4 GB


def _lu_bytes_per_window(window_size: int) -> int:
    """Bytes occupied by one LU factorisation (float32) for *window_size*."""
    win_pixels = 2 * (window_size // 2) + 1
    n_flat = win_pixels ** 2
    return n_flat * n_flat * 4   # float32


def _compute_max_mem_windows(window_size: int, ram_fraction: float = 0.9) -> int:
    """How many LU matrices fit in *ram_fraction* of currently available RAM.

    Parameters
    ----------
    window_size:
        Evaluation-window side length in pixels.
    ram_fraction:
        Fraction of free RAM to use (default 0.9 = 90 %).

    Returns
    -------
    int
        Maximum windows to keep in RAM.  Returns 0 when not even one
        window fits (very memory-constrained environment).
    """
    usable = _available_ram_bytes() * ram_fraction
    return max(0, int(usable / _lu_bytes_per_window(window_size)))


# ---------------------------------------------------------------------------
# MemoryKernelCache
# ---------------------------------------------------------------------------

class MemoryKernelCache:
    """Bounded in-memory LRU cache for dispersal-window LU factorisations.

    Automatically evicts the least-recently-used entry when the buffer is
    full.  Evicted entries are dropped silently and recomputed on the next
    access — there is no disk fallback and RAM usage is strictly bounded.

    When *max_mem_windows* is 0 the cache is disabled: ``__setitem__`` is
    a no-op and ``__contains__`` always returns ``False``, so every window
    is recomputed on every step.

    This is the default cache used by :class:`PopulationSimulator`.  Call
    :func:`_compute_max_mem_windows` to size it automatically from free RAM.

    Parameters
    ----------
    max_mem_windows:
        Maximum number of LU factorisations to keep in RAM.
    """

    def __init__(self, max_mem_windows: int = 0) -> None:
        self._max_mem = max_mem_windows
        self._mem: OrderedDict = OrderedDict()

    # ------------------------------------------------------------------
    # dict-like interface
    # ------------------------------------------------------------------

    def __contains__(self, key: tuple) -> bool:
        return key in self._mem

    def __getitem__(self, key: tuple):
        self._mem.move_to_end(key)
        return self._mem[key]

    def __setitem__(self, key: tuple, value) -> None:
        if self._max_mem <= 0:
            return   # caching disabled
        if key in self._mem:
            self._mem.move_to_end(key)
        else:
            if len(self._mem) >= self._max_mem:
                self._mem.popitem(last=False)   # evict LRU
            self._mem[key] = value

    def __bool__(self) -> bool:
        return bool(self._mem)

    def __len__(self) -> int:
        return len(self._mem)

    def clear(self) -> None:
        """Discard all in-memory factorisations."""
        self._mem.clear()


class DiskKernelCache:
    """Disk-backed LRU cache for dispersal-window LU factorisations.

    Behaves like a plain ``dict`` (same ``__contains__`` / ``__getitem__`` /
    ``__setitem__`` / ``__bool__`` interface) so it is a transparent drop-in
    for the *kernel_cache* argument throughout the simulation engine.

    Parameters
    ----------
    cache_dir:
        Root directory for cache files.  A fingerprint-named subdirectory is
        created inside it; different parameter combinations coexist without
        collision and are reused automatically when parameters match.
    hs:
        Habitat-suitability array — used only to compute the fingerprint.
    ewalk, n, r:
        Dispersal parameters (included in the fingerprint).
    window_size:
        Evaluation-window side length in pixels (included in the fingerprint).
    max_mem_windows:
        Maximum LU factorisations kept in RAM simultaneously.  Each entry for
        ``window_size=70`` costs ~97 MB, so the default of 16 caps RAM use at
        roughly 1.6 GB.  Increase for faster within-run access if RAM allows.

    Examples
    --------
    >>> from paradis.simulation import DiskKernelCache, PopulationSimulator
    >>> sim = PopulationSimulator(
    ...     hs=hs, ewalk=1137, n=2094, r=0.016, tgrowth=2.35,
    ...     carrying_capacity_params=(0.023, -6.7, 0.58),
    ...     presence_threshold=0.017,
    ...     cache_dir="~/.paradis_cache/my_species",
    ... )
    """

    def __init__(
        self,
        cache_dir: str | pathlib.Path,
        hs: np.ndarray,
        ewalk: float,
        n: float,
        r: float,
        window_size: int,
        max_mem_windows: int = 16,
    ) -> None:
        self._max_mem = max_mem_windows
        self._mem: OrderedDict = OrderedDict()   # LRU in-memory buffer

        # Fingerprint: short MD5 of HS bytes + all parameters that affect the
        # LU factorisation.  Different parameter sets get separate directories;
        # the same set reuses existing files automatically.
        h = hashlib.md5(hs.tobytes()).hexdigest()[:12]
        fingerprint = (
            f"{h}_ew{ewalk}_n{n}_r{r}_ws{window_size}"
        )

        self._dir = pathlib.Path(cache_dir).expanduser() / fingerprint
        self._dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # dict-like interface expected by dispersal_step / _build_window_lu
    # ------------------------------------------------------------------

    def __contains__(self, key: tuple) -> bool:
        return key in self._mem or self._path(key).exists()

    def __getitem__(self, key: tuple):
        # Fast path: LRU hit — move to most-recently-used end.
        if key in self._mem:
            self._mem.move_to_end(key)
            return self._mem[key]
        # Slow path: load from disk, place in LRU.
        # LU is stored as float16; upcast is done in the solve, not here.
        data   = np.load(str(self._path(key)))
        LU     = torch.from_numpy(data["LU"].copy()).half()   # stays on CPU
        pivots = torch.from_numpy(data["pivots"].copy())
        self._put_mem(key, (LU, pivots))
        return LU, pivots

    def __setitem__(self, key: tuple, value) -> None:
        LU, pivots = value
        # Persist as float16 — halves disk footprint vs float32.
        np.savez(
            str(self._path(key)),
            LU=LU.cpu().half().numpy(),
            pivots=pivots.cpu().numpy(),
        )
        self._put_mem(key, value)

    def __bool__(self) -> bool:
        return bool(self._mem) or any(self._dir.iterdir())

    def __len__(self) -> int:
        return sum(1 for _ in self._dir.glob("*.npz"))

    # ------------------------------------------------------------------
    # House-keeping
    # ------------------------------------------------------------------

    def clear(self, *, disk: bool = False) -> None:
        """Discard cached factorisations.

        Parameters
        ----------
        disk:
            If ``True``, also delete all ``.npz`` files on disk — use this
            when dispersal parameters or *window_size* have changed and the
            on-disk factorisations are stale.  Default ``False`` clears only
            the in-memory LRU buffer.
        """
        self._mem.clear()
        if disk:
            shutil.rmtree(self._dir, ignore_errors=True)
            self._dir.mkdir(parents=True, exist_ok=True)

    @property
    def cache_dir(self) -> pathlib.Path:
        """Absolute path of the directory holding the ``.npz`` files."""
        return self._dir

    def info(self) -> None:
        """Print a brief summary of cache size and disk usage."""
        files  = list(self._dir.glob("*.npz"))
        n_disk = len(files)
        size_gb = sum(f.stat().st_size for f in files) / 1e9
        print(
            f"DiskKernelCache: {n_disk} windows on disk "
            f"({size_gb:.2f} GB),  {len(self._mem)} in LRU "
            f"(max {self._max_mem})"
        )
        print(f"  dir: {self._dir}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _path(self, key: tuple) -> pathlib.Path:
        x0, y0 = key
        return self._dir / f"{x0}_{y0}.npz"

    def _put_mem(self, key: tuple, value) -> None:
        """Insert into the LRU buffer, evicting the least-recently-used entry
        when the buffer is full."""
        if key in self._mem:
            self._mem.move_to_end(key)
        else:
            if len(self._mem) >= self._max_mem:
                self._mem.popitem(last=False)   # evict oldest
            self._mem[key] = value
