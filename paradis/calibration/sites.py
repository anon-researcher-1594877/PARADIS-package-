"""Calibration site selection.

This module implements the spatial sampling strategy used to identify
suitable calibration sites for parameter estimation.  A calibration site is
a small square window that:

* contains a sufficient number of species observations,
* exhibits spatial contrast in habitat suitability (Otsu separation > 0.5),
* has a balanced proportion of high vs. low suitability pixels.

Key functions
-------------
:func:`spatial_min_distance_filter` (:func:`pdist2D`)
    Greedy spatial thinning of candidate points.
:func:`assess_site_quality`
    Evaluate and filter candidate windows using Otsu-based separation.
:func:`sample_calibration_sites`
    Full pipeline: random sampling → spatial thinning → quality filtering.

Classes
-------
:class:`CalibrationSites`
    Container that stores the selected site windows and their metadata.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance
from skimage import filters


# ---------------------------------------------------------------------------
# Spatial thinning
# ---------------------------------------------------------------------------

def spatial_min_distance_filter(
    xs: np.ndarray,
    ys: np.ndarray,
    distmin: float,
) -> tuple[list, list]:
    """Keep only points separated by at least *distmin* (greedy algorithm).

    Parameters
    ----------
    xs, ys:
        Coordinate arrays of candidate points.
    distmin:
        Minimum allowed distance between retained points.

    Returns
    -------
    filtered_xs, filtered_ys : list
        Coordinates of the retained points.
    """
    filtered_xs: list = []
    filtered_ys: list = []
    points = np.column_stack((xs, ys))
    dist_matrix = distance.squareform(distance.pdist(points))
    np.fill_diagonal(dist_matrix, np.inf)

    blacklist: list = []
    for i in range(len(xs)):
        if i not in blacklist:
            too_close = np.where(dist_matrix[i] < distmin)[0]
            blacklist += list(too_close)
            filtered_xs.append(xs[i])
            filtered_ys.append(ys[i])

    return filtered_xs, filtered_ys


# ---------------------------------------------------------------------------
# Site quality assessment
# ---------------------------------------------------------------------------

def assess_site_quality(
    hs_maps: list,
    obs_maps: list,
    taxa_maps: list,
    centers: list,
    criteria: list | None = None,
    plot: bool = False,
    verbose: bool = False,
) -> tuple[list, list]:
    """Evaluate candidate calibration windows and return those that pass.

    The quality criterion uses Otsu's threshold to measure the between-class
    variance ratio (spatial contrast in HS) and the balance of high/low HS
    pixels.

    Parameters
    ----------
    hs_maps, obs_maps, taxa_maps:
        Parallel lists of 2-D arrays for each candidate window.
    centers:
        List of ``(x, y)`` centre coordinates.
    criteria:
        ``[min_obs, min_separation, balance_tolerance]``.  Defaults to
        ``[10, 0.5, 0.1]``.
    plot:
        If ``True``, display a grid of all candidate windows.

    Returns
    -------
    selected : list
        ``[hs_maps, obs_maps, taxa_maps, centers]`` for accepted sites.
    rejection_counts : list
        ``[n_obs_reject, n_sep_reject, n_balance_reject]`` – counts per
        rejection reason, used to auto-relax criteria in
        :func:`sample_calibration_sites`.
    """
    if criteria is None:
        criteria = [10, 0.5, 0.1]

    selected: list = [[], [], [], []]
    n = len(hs_maps)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    rejection_counts = [0, 0, 0]

    if plot:
        fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
        axes = np.array(axes).reshape(-1)
    else:
        axes = [None] * (rows * cols)

    id_selected = 0
    for i, ax in enumerate(axes):
        if i >= n:
            if plot and ax is not None:
                ax.axis("off")
            continue

        gray = hs_maps[i][hs_maps[i] != 0]
        xs, ys = np.where(obs_maps[i] >= 1)
        xs_taxa, ys_taxa = np.where(taxa_maps[i] >= 1)

        threshold = filters.threshold_otsu(gray)
        m0 = gray <= threshold
        m1 = gray > threshold
        mu0, mu1, muT = gray[m0].mean(), gray[m1].mean(), gray.mean()
        sigT2 = ((gray - muT) ** 2).mean()
        sigW2 = m0.mean() * ((gray[m0] - mu0) ** 2).mean() + m1.mean() * ((gray[m1] - mu1) ** 2).mean()
        separation = (sigT2 - sigW2) / sigT2
        low_extent = m0.sum() / (m0.sum() + m1.sum())

        n_obs = np.nansum(obs_maps[i])
        pass_obs = n_obs > criteria[0]
        pass_sep = separation > criteria[1]
        pass_bal = (0.5 - criteria[2]) <= low_extent <= (0.5 + criteria[2])

        if verbose:
            status = "OK " if (pass_obs and pass_sep and pass_bal) else "REJ"
            cx, cy = centers[i]
            reasons = []
            if not pass_obs: reasons.append(f"obs={n_obs:.0f}<{criteria[0]}")
            if not pass_sep: reasons.append(f"sep={separation:.3f}<{criteria[1]}")
            if not pass_bal: reasons.append(f"bal={low_extent:.2f}")
            reason_str = "  " + ", ".join(reasons) if reasons else ""
            print(f"  [{status}] ({cx:4d},{cy:4d})  "
                  f"sB2/sT2={separation:.3f}  "
                  f"obs={n_obs:.0f}  "
                  f"lo={low_extent*100:.1f}%"
                  f"{reason_str}")

        if plot:
            ax.imshow(hs_maps[i], cmap="viridis")
            ax.contour(hs_maps[i], levels=[threshold], colors="black", linestyles="--")
            ax.scatter(ys_taxa, xs_taxa, color="blue", marker="^", alpha=0.3)
            ax.set_title(
                r"$\frac{\sigma_{1,2}^2}{\sigma_T^2}= $" + str(round(separation, 2))
                + "  low= " + str(round(low_extent * 100, 2)) + " %"
            )

        if pass_obs and pass_sep and pass_bal:
            if plot:
                ax.set_ylabel(f"ID{id_selected}  n_obs={n_obs:.0f}")
                ax.scatter(ys, xs, color="green", marker="+")
            id_selected += 1
            selected[0].append(hs_maps[i])
            selected[1].append(obs_maps[i])
            selected[2].append(taxa_maps[i])
            selected[3].append(centers[i])
        else:
            if not pass_sep:
                rejection_counts[1] += 1
            if not pass_obs:
                rejection_counts[0] += 1
            if not pass_bal:
                rejection_counts[2] += 1
            if plot:
                ax.scatter(ys, xs, color="red", marker="+")

    if plot:
        plt.tight_layout()
        plt.show()

    return selected, rejection_counts


# ---------------------------------------------------------------------------
# Main sampling function
# ---------------------------------------------------------------------------

def sample_calibration_sites(
    hs: np.ndarray,
    obs: np.ndarray,
    taxa_ref: np.ndarray,
    n_samples: int = 50,
    window_size: int = 50,
    distmin: float = 100.0,
    min_obs: int = 10,
    time_budget: float = 60.0,
    plot: bool = True,
    verbose: bool = False,
) -> "CalibrationSites":
    """Sample and select calibration sites from the study region.

    The function iterates random draws of candidate points until at least 10
    well-separated, high-quality windows are found (or the time budget is
    exhausted).  Acceptance criteria are progressively relaxed when too few
    sites are found.

    Parameters
    ----------
    hs:
        Full habitat-suitability map.
    obs:
        Species observation count map.
    taxa_ref:
        Reference-taxa observation count map.
    n_samples:
        Number of random candidate centres drawn each iteration.
    window_size:
        Side length of each square window (pixels).
    distmin:
        Minimum distance between selected site centres (pixels).
    min_obs:
        Minimum number of observations required within a window.
    time_budget:
        Maximum CPU time allowed (seconds).
    plot:
        Display the selected sites on the HS map.
    verbose:
        Print progress messages.

    Returns
    -------
    CalibrationSites
        Container with the selected windows and metadata.
    """
    half = window_size // 2
    best: list = [[], [], [], []]
    all_rejected_centers: list = []
    criteria = [float(min_obs), 0.5, 0.1]
    rejection_counts = None
    t0 = time.time()

    while len(best[3]) < 10 and (time.time() - t0) < time_budget:
        # Auto-relax the tightest criterion
        if rejection_counts is not None:
            argmax = int(np.argmax(rejection_counts))
            loopnb = 0
            while (
                (argmax == 0 and criteria[0] <= 1)
                or (argmax == 1 and criteria[1] <= 0.1)
                or (argmax == 2 and criteria[2] >= 0.2)
            ):
                rejection_counts[argmax] = 0
                argmax = int(np.argmax(rejection_counts))
                loopnb += 1
                if loopnb > 2:
                    break
            if argmax == 0:
                criteria[0] = max(1.0, criteria[0] - 1)
                if verbose:
                    print(f"Relaxing min_obs → {criteria[0]}")
            elif argmax == 1:
                criteria[1] = max(0.1, criteria[1] - 0.1)
                if verbose:
                    print(f"Relaxing separation → {criteria[1]}")
            else:
                criteria[2] = min(0.2, criteria[2] + 0.05)
                if verbose:
                    print(f"Relaxing balance tolerance → {criteria[2]}")

        # Draw random candidate centres
        x_obs, y_obs = np.where(obs >= 1)
        selected_idx = random.sample(range(len(x_obs)), min(n_samples, len(x_obs)))
        xs = np.array([x_obs[i] for i in selected_idx])
        ys = np.array([y_obs[i] for i in selected_idx])
        xs, ys = spatial_min_distance_filter(xs, ys, distmin)

        hs_wins, obs_wins, taxa_wins, valid_centers = [], [], [], []
        for xc, yc in zip(xs, ys):
            xc, yc = int(xc), int(yc)
            if np.nansum(obs[xc - half:xc + half, yc - half:yc + half]) >= min_obs:
                hs_wins.append(hs[xc - half:xc + half, yc - half:yc + half])
                obs_wins.append(obs[xc - half:xc + half, yc - half:yc + half])
                taxa_wins.append(taxa_ref[xc - half:xc + half, yc - half:yc + half])
                valid_centers.append((xc, yc))

        selected_batch, rejection_counts = assess_site_quality(
            hs_wins, obs_wins, taxa_wins, valid_centers,
            criteria=criteria, plot=False, verbose=verbose,
        )

        accepted_set = set(map(tuple, selected_batch[3]))
        for c in valid_centers:
            if tuple(c) not in accepted_set:
                all_rejected_centers.append(c)

        for idx in range(len(selected_batch[3])):
            center = selected_batch[3][idx]
            include = all(
                np.sqrt((center[0] - ec[0]) ** 2 + (center[1] - ec[1]) ** 2) >= distmin
                for ec in best[3]
            )
            if include:
                best[0].append(selected_batch[0][idx])
                best[1].append(selected_batch[1][idx])
                best[2].append(selected_batch[2][idx])
                best[3].append(center)

    elapsed = time.time() - t0
    if verbose:
        print(f"Found {len(best[3])} calibration sites in {elapsed:.1f}s")

    if plot and len(best[3]) > 0:
        # ── Overview: location of all windows on the HS map ──────────────────
        plt.figure(figsize=(10, 10))
        plt.imshow(hs, cmap="viridis")
        plt.colorbar(label="Habitat suitability", shrink=0.8)
        for iD, center in enumerate(best[3]):
            x0, y0 = center
            plt.plot(
                [y0 - half, y0 + half, y0 + half, y0 - half, y0 - half],
                [x0 - half, x0 - half, x0 + half, x0 + half, x0 - half],
                color="red",
            )
            plt.text(y0, x0, str(iD), color="white", fontsize=8)
        plt.title(f"{len(best[3])} calibration sites selected")
        plt.show()

        # ── Grid: each window with Otsu σ_B²/σ_T² ratio in title ────────────
        # Matches the original grid_plots2(…, plot=True) call.
        assess_site_quality(
            best[0], best[1], best[2], best[3],
            criteria=criteria, plot=True, verbose=False,
        )

    return CalibrationSites(
        hs_maps=best[0],
        obs_maps=best[1],
        taxa_maps=best[2],
        centers=best[3],
        rejected_centers=all_rejected_centers,
        window_size=window_size,
    )


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class CalibrationSites:
    """Container for a set of selected calibration windows.

    Attributes
    ----------
    hs_maps:
        List of 2-D HS windows.
    obs_maps:
        List of 2-D species observation windows.
    taxa_maps:
        List of 2-D reference-taxa windows.
    centers:
        List of ``(row, col)`` centre coordinates in the full map.
    window_size:
        Side length of each window (pixels).
    """

    hs_maps: List[np.ndarray] = field(default_factory=list)
    obs_maps: List[np.ndarray] = field(default_factory=list)
    taxa_maps: List[np.ndarray] = field(default_factory=list)
    centers: List[Tuple[int, int]] = field(default_factory=list)
    rejected_centers: List[Tuple[int, int]] = field(default_factory=list)
    window_size: int = 50

    def __len__(self) -> int:
        return len(self.hs_maps)

    def __repr__(self) -> str:
        return f"CalibrationSites(n={len(self)}, window_size={self.window_size})"
