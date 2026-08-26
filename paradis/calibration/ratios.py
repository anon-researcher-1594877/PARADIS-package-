"""Observed relative-abundance posteriors at calibration sites.

For each calibration window the module computes, at every pixel with
non-zero sampling effort, the Bayesian posterior density of the true
relative abundance ``r = N_sp / N_taxa`` given the observed counts.

Key functions
-------------
:func:`poisson_disk_resample`
    Spatially balanced resampling to equalise presence/absence prevalence.
:func:`compute_site_posteriors`
    Compute per-pixel posteriors for a single calibration window.
:func:`compute_all_posteriors`
    Compute posteriors for all calibration windows (full pipeline).
"""

from __future__ import annotations

import math

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import KDTree
from tqdm import tqdm

from paradis.calibration.sites import _overlay_breeding_range


# ---------------------------------------------------------------------------
# Spatial resampling
# ---------------------------------------------------------------------------

def poisson_disk_resample(
    xs: np.ndarray,
    ys: np.ndarray,
    exclude_xs: np.ndarray,
    exclude_ys: np.ndarray,
    n_target: int,
    max_iter: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Select *n_target* points from ``(xs, ys)`` far from ``(exclude_*)``.

    Uses a binary search on the exclusion radius to hit the target count.

    Parameters
    ----------
    xs, ys:
        Candidate points to select from.
    exclude_xs, exclude_ys:
        Points to keep away from.
    n_target:
        Desired number of selected points.
    max_iter:
        Binary-search iterations.

    Returns
    -------
    sel_xs, sel_ys : numpy.ndarray
        Selected point coordinates.
    """
    pres = np.column_stack((exclude_xs, exclude_ys))
    pres_tree = KDTree(pres) if len(pres) > 0 else None
    pts = np.column_stack((xs, ys))
    lo_r, hi_r = 0.0, 100.0
    selected = []

    for _ in range(max_iter):
        r = 0.5 * (lo_r + hi_r)
        order = np.random.permutation(len(pts))
        selected = []
        abs_tree = None
        for p in pts[order]:
            if pres_tree is not None and len(pres_tree.query_ball_point(p, r)) > 0:
                continue
            if abs_tree is not None and len(abs_tree.query_ball_point(p, r)) > 0:
                continue
            selected.append(p)
            abs_tree = KDTree(np.array(selected))
        if len(selected) == n_target:
            break
        elif len(selected) > n_target:
            lo_r = r
        else:
            hi_r = r

    if len(selected) == 0:
        return np.array([]), np.array([])
    arr = np.array(selected)
    return arr[:, 0], arr[:, 1]


# ---------------------------------------------------------------------------
# Posterior computation
# ---------------------------------------------------------------------------

def compute_site_posteriors(
    obs_window: np.ndarray,
    taxa_window: np.ndarray,
    n_div: int = 200,
) -> tuple[list, np.ndarray]:
    """Compute per-pixel relative-abundance posteriors for one window.

    Parameters
    ----------
    obs_window:
        2-D species observation counts within the calibration window.
    taxa_window:
        2-D reference-taxa counts within the calibration window.
    n_div:
        Number of discrete points in ``[0, 1]`` used to represent each
        posterior density.

    Returns
    -------
    posteriors : list of numpy.ndarray
        One array of length *n_div* per selected pixel.
    selected_mask : numpy.ndarray
        Boolean mask of the same shape as *obs_window* indicating which
        pixels were included (after prevalence balancing).
    """
    Rs = np.linspace(0, 1, n_div)

    list_taxa = taxa_window[taxa_window > 0]
    list_obs  = obs_window[taxa_window > 0]
    mask_pres = list_obs >= 1
    nb_pres   = np.sum(mask_pres)
    mask_abs  = list_obs == 0
    nb_abs    = np.sum(mask_abs)

    # Balance presence/absence
    mask_sel = np.zeros(len(list_taxa), dtype=bool)
    if nb_abs > nb_pres:
        x_abs, y_abs = np.where((obs_window == 0) & (taxa_window > 0))
        x_pres, y_pres = np.where((obs_window >= 1) & (taxa_window > 0))
        x_all, y_all = np.where(taxa_window > 0)
        sel_x, sel_y = poisson_disk_resample(x_abs, y_abs, x_pres, y_pres, nb_pres)
        # Map back to flat indices
        coord2idx = {(x_all[i], y_all[i]): i for i in range(len(x_all))}
        for px, py in zip(sel_x, sel_y):
            idx = coord2idx.get((int(px), int(py)))
            if idx is not None:
                mask_sel[idx] = True
        for idx in np.where(mask_pres)[0]:
            mask_sel[idx] = True
    else:
        x_pres, y_pres = np.where((obs_window >= 1) & (taxa_window > 0))
        x_abs, y_abs   = np.where((obs_window == 0) & (taxa_window > 0))
        x_all, y_all   = np.where(taxa_window > 0)
        sel_x, sel_y   = poisson_disk_resample(x_pres, y_pres, x_abs, y_abs, nb_abs)
        coord2idx = {(x_all[i], y_all[i]): i for i in range(len(x_all))}
        for px, py in zip(sel_x, sel_y):
            idx = coord2idx.get((int(px), int(py)))
            if idx is not None:
                mask_sel[idx] = True
        for idx in np.where(mask_abs)[0]:
            mask_sel[idx] = True

    list_taxa_sel = list_taxa[mask_sel]
    list_obs_sel  = list_obs[mask_sel]

    posteriors = []
    for xtaxa, xsp in zip(list_taxa_sel, list_obs_sel):
        if xsp > xtaxa:
            xsp = xtaxa
        rtilde = xsp / xtaxa
        dens = np.array([
            math.factorial(int(xtaxa))
            / (math.factorial(int(xtaxa * rtilde)) * math.factorial(int(xtaxa - xsp)))
            * r ** (rtilde * xtaxa) * (1 - r) ** (xtaxa * (1 - rtilde))
            for r in Rs
        ])
        dens /= np.sum(dens * (1.0 / n_div))
        posteriors.append(dens)

    return posteriors, mask_sel


def compute_all_posteriors(
    calibration_sites,
    verbose: bool = False,
    save_path: str | None = None,
    species_name: str | None = None,
) -> tuple[list, list]:
    """Compute posteriors for every site in a :class:`~paradis.calibration.sites.CalibrationSites`.

    Parameters
    ----------
    calibration_sites:
        A :class:`~paradis.calibration.sites.CalibrationSites` instance.
    verbose:
        Print progress information.
    save_path:
        If provided, save the resampling overview figure.
    species_name:
        Used in the figure title.

    Returns
    -------
    posteriors : list
        ``posteriors[site_id][pixel_id]`` → 1-D density array.
    selected_masks : list
        ``selected_masks[site_id]`` → boolean mask over taxa>0 pixels.
    """
    n = len(calibration_sites)
    cols = max(1, int(np.ceil(np.sqrt(n))))
    rows = max(1, int(np.ceil(n / cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.array(axes).reshape(-1)

    all_posteriors = []
    all_masks = []

    for i, site_idx in tqdm(
        enumerate(range(n)), total=n, desc="Calibration sites", disable=not verbose
    ):
        obs_win   = calibration_sites.obs_maps[site_idx]
        taxa_win  = calibration_sites.taxa_maps[site_idx]
        hs_win    = calibration_sites.hs_maps[site_idx]

        posts, mask = compute_site_posteriors(obs_win, taxa_win)
        all_posteriors.append(posts)
        all_masks.append(mask)

        ax = axes[i]
        ax.set_title(f"Site {site_idx}")
        ax.imshow(hs_win, cmap="viridis")
        breeding_maps = calibration_sites.breeding_maps
        if site_idx < len(breeding_maps):
            _overlay_breeding_range(ax, breeding_maps[site_idx])

        # Recover 2-D coordinates of all taxa pixels and the selection mask
        x_all, y_all = np.where(taxa_win > 0)
        obs_flat = obs_win[taxa_win > 0]
        is_abs_flat = obs_flat == 0

        sel_abs   = mask & is_abs_flat          # selected absences
        unsel_abs = (~mask) & is_abs_flat       # non-selected absences

        x_pres, y_pres = np.where((obs_win >= 1) & (taxa_win > 0))
        ax.scatter(y_pres,              x_pres,              color="green", marker="o",  s=10, label="Presence")
        ax.scatter(y_all[unsel_abs],    x_all[unsel_abs],    color="red",   marker="x",  s=5,  alpha=0.3, label="Absence (not selected)")
        ax.scatter(y_all[sel_abs],      x_all[sel_abs],      color="blue",  marker="^",  s=15, alpha=0.7, label="Absence (selected)")

    # Turn off any unused axes (when n is not a perfect rows*cols grid)
    for j in range(n, len(axes)):
        axes[j].axis("off")

    # Build a single shared legend for the whole figure from the first
    # subplot's handles (all subplots use the same marker/color scheme),
    # instead of repeating a legend on every tiny panel.
    handles, labels = axes[0].get_legend_handles_labels()
    if any(bm is not None for bm in calibration_sites.breeding_maps):
        handles = handles + [mpatches.Patch(color="red", alpha=0.7, label="Breeding range")]
        labels = labels + ["Breeding range"]
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               bbox_to_anchor=(0.5, -0.02), fontsize=9, markerscale=2)

    fig.suptitle(f"Calibration sites – {species_name or ''}", fontsize=12)
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()

    return all_posteriors, all_masks
