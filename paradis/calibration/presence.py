"""Calibration of the relative-abundance presence threshold.

The presence threshold is the minimum ratio ``Obs_sp / Obs_taxa`` that
should be predicted by the model to classify a location as *present*.

The threshold is found by minimising the classification error over a set of
reference points inside and outside the known current range.

Key function
------------
:func:`calibrate_presence_threshold`
    Estimate the optimal relative-abundance presence threshold.
"""

from __future__ import annotations

import numpy as np
from math import factorial
from tqdm import tqdm


def calibrate_presence_threshold(
    current_range: np.ndarray,
    obs: np.ndarray,
    taxa_ref: np.ndarray,
    max_points: int = 3000,
    plot: bool = True,
    verbose: bool = False,
    save_path: str | None = None,
    species_name: str | None = None,
) -> float:
    """Estimate the relative-abundance threshold corresponding to presence.

    For each candidate threshold ``t`` in ``[0, 1]``, the function evaluates
    the expected squared classification error over a balanced sample of
    presence and absence sites.  The threshold minimising this error is
    returned.

    Parameters
    ----------
    current_range:
        2-D binary map – 1 where the species is known to be present.
    obs:
        2-D species observation count map.
    taxa_ref:
        2-D reference-taxa observation count map (sampling effort proxy).
    max_points:
        Maximum number of presence/absence points to include.
    plot:
        If ``True``, display the error curve with the optimal threshold.
    verbose:
        Print progress information.
    save_path:
        If provided, save the figure to this path.
    species_name:
        Used in the figure title and filename.

    Returns
    -------
    float
        Optimal presence threshold in ``(0, 1)``.
    """
    mask_out = (obs <= taxa_ref) & (current_range == 0) & (taxa_ref > 0) & (obs >= 1)
    mask_in  = (obs <= taxa_ref) & (current_range == 1) & (taxa_ref > 0) & (obs >= 1)

    xsp_out, xtaxa_out = obs[mask_out], taxa_ref[mask_out]
    xsp_in,  xtaxa_in  = obs[mask_in],  taxa_ref[mask_in]

    len_out, len_in = len(xsp_out), len(xsp_in)

    # Balance classes
    if len_in > len_out or len_in > max_points:
        size = min(len_out, max_points)
        idx = np.random.choice(len_in, size, replace=False)
        xsp_in, xtaxa_in = xsp_in[idx], xtaxa_in[idx]
    if len_out > len_in or len_out > max_points:
        size = min(len_in, max_points)
        idx = np.random.choice(len_out, size, replace=False)
        xsp_out, xtaxa_out = xsp_out[idx], xtaxa_out[idx]

    xsp_in   = xsp_in.astype(int)
    xtaxa_in = xtaxa_in.astype(int)
    xsp_out   = xsp_out.astype(int)
    xtaxa_out = xtaxa_out.astype(int)

    n_div = 1000
    Rs = np.linspace(1e-6, 1.0, n_div)

    def _cdfs(xsp_arr, xtaxa_arr, label: str):
        cdfs = []
        n_pts = len(xsp_arr)
        for i in tqdm(range(n_pts), desc=f"  Posteriors ({label})",
                      total=n_pts, unit="pts", leave=False):
            xsp, xtaxa = xsp_arr[i], xtaxa_arr[i]
            rtilde = xsp / xtaxa
            dens = np.array([
                factorial(xtaxa) / (factorial(xsp) * factorial(xtaxa - xsp))
                * r ** (rtilde * xtaxa) * (1 - r) ** (xtaxa * (1 - rtilde))
                for r in Rs
            ])
            dens /= np.nansum(dens * (1.0 / n_div))
            cdfs.append([np.cumsum(dens * (1.0 / n_div))])
        return cdfs

    n_in  = len(xsp_in)
    n_out = len(xsp_out)
    print(f"  Computing CDFs: {n_in} presence pts + {n_out} absence pts ...")
    cdfs_in  = _cdfs(xsp_in,  xtaxa_in,  f"presence n={n_in}")
    cdfs_out = _cdfs(xsp_out, xtaxa_out, f"absence  n={n_out}")

    errors = []
    for t_idx in tqdm(range(n_div), desc="  Scanning thresholds", leave=False):
        err = 0.0
        for i in range(len(xsp_in)):
            err += ((1.0 - cdfs_in[i][0][t_idx]) - 1.0) ** 2
            err += ((1.0 - cdfs_out[i][0][t_idx]) - 0.0) ** 2
        errors.append(err)

    t_opt = Rs[np.argmin(errors)]

    if plot or save_path is not None:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(Rs, errors)
        plt.axvline(t_opt, linestyle="--", color="red",
                    label=f"Threshold = {t_opt:.4f}")
        plt.xlabel("Relative abundance threshold")
        plt.ylabel("Classification error")
        plt.title(f"Presence threshold calibration – {species_name or ''}")
        plt.legend()
        plt.grid(linestyle="--", alpha=0.3, color="grey")
        if save_path is not None:
            plt.savefig(save_path, dpi=200)
        if plot:
            plt.show()

    return float(t_opt)
