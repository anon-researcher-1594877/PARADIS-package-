"""High-level one-call calibration pipeline.

:func:`calibrate`
    Run the complete PARADIS calibration pipeline from raw raster arrays
    to estimated dispersal and growth parameters in a single call.

Example
-------
>>> import paradis, numpy as np
>>> results = paradis.calibrate(hs, obs, taxa_ref, mdd=21.0,
...                             species_name="Elanus caeruleus", plot=True)
>>> print(results["Ew"], results["n"], results["r"], results["Tg"])
"""

from __future__ import annotations

import math

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from skimage import filters as skfilters

from paradis.calibration.presence          import calibrate_presence_threshold
from paradis.calibration.carrying_capacity import estimate_carrying_capacity, logistic
from paradis.calibration.sites             import sample_calibration_sites
from paradis.calibration.ratios            import compute_all_posteriors
from paradis.learning.optimizer            import ParameterLearner
from paradis.io.raster                     import load_hs, load_obs, load_mask


def _as_array(
    x: "np.ndarray | str | pathlib.Path",
    kind: str,
) -> "np.ndarray":
    """Return *x* unchanged if it is already an array, else load from .tif.

    Parameters
    ----------
    x:
        Either a NumPy array or a path to a ``.tif`` file.
    kind:
        One of ``'hs'``, ``'obs'``, ``'mask'`` — selects the appropriate
        loader (normalisation, NODATA handling) for that raster type.
    """
    import pathlib
    if isinstance(x, np.ndarray):
        return x
    loaders = {"hs": load_hs, "obs": load_obs, "mask": load_mask}
    if kind not in loaders:
        raise ValueError(f"Unknown kind '{kind}'. Use 'hs', 'obs', or 'mask'.")
    return loaders[kind](pathlib.Path(x))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def calibrate(
    hs: "np.ndarray | str | pathlib.Path",
    obs: "np.ndarray | str | pathlib.Path",
    taxa_ref: "np.ndarray | str | pathlib.Path",
    mdd: float,
    current_range: "np.ndarray | str | pathlib.Path | None" = None,
    valid_mask: "np.ndarray | str | pathlib.Path | None" = None,
    hs_bin_width: float = 0.05,
    window_size: int = 70,
    distmin: float = 40.0,
    min_obs: int = 5,
    n_calibration_sites: int = 5,
    n_random_sites: int = 3,
    n_candidate_samples: int = 200,
    time_budget: float = 30.0,
    max_iter: int = 500,
    plot: bool = True,
    save_path: str | None = None,
    species_name: str = "",
    verbose: bool = False,
) -> dict:
    """Run the complete PARADIS calibration pipeline in one call.

    Executes the following steps automatically:

    1. Derive current range from ``obs > 0`` (if not supplied).
    2. Calibrate presence threshold.
    3. Estimate carrying-capacity logistic parameters ``(L, k, x0)``.
    4. Select calibration windows.
    5. Compute per-window relative-abundance posteriors.
    6. Fit dispersal + growth parameters ``(Ew, n, r, Tg)`` via Adam SGD.
    7. Produce a summary figure (when ``plot=True`` or ``save_path`` is set).

    Parameters
    ----------
    hs : np.ndarray
        Habitat-suitability map, values in ``[0, 1]``.
    obs : np.ndarray
        Species observation count map (same shape as *hs*).
    taxa_ref : np.ndarray
        Reference-taxa observation count map.
    mdd : float
        Prior mean dispersal distance (km; same units as raster pixels).
    current_range : np.ndarray | None
        Binary map of the known current range.
        Derived from ``obs > 0`` when ``None``.
    valid_mask : np.ndarray | None
        Boolean mask of pixels with valid HS data (used for the map display
        only).  Defaults to ``hs > 0`` when ``None``.
    hs_bin_width : float
        Width of HS bins for carrying-capacity estimation (default 0.05).
    window_size : int
        Side length of calibration windows in pixels (default 40).
    distmin : float
        Minimum distance between calibration window centres (default 40).
    min_obs : int
        Minimum species observations required in a window (default 2).
    n_calibration_sites : int
        Number of calibration sites used for learning (default 5).
    n_random_sites : int
        Mini-batch size per gradient step (default 3).
    n_candidate_samples : int
        Random candidate windows drawn per sampling iteration (default 200).
    time_budget : float
        Time budget for site selection in seconds (default 30).
    max_iter : int
        Adam optimisation steps (default 500).
    plot : bool
        Display the summary figure (default True).
    save_path : str | None
        If provided, save the figure to this file path.
    species_name : str
        Species label used in figure titles and legends.
    verbose : bool
        Print intermediate diagnostics (default False).

    Returns
    -------
    dict
        ``Ew``, ``n``, ``r``, ``Tg``  — dispersal / growth parameters.
        ``L``, ``k``, ``x0``          — carrying-capacity logistic params.
        ``presence_threshold``         — calibrated presence cut-off.
        ``calib_sites``               — :class:`~paradis.calibration.sites.CalibrationSites`.
        ``all_costs``                 — per-run loss histories.
    """
    # ------------------------------------------------------------------
    # Step 0 – Load rasters if paths were supplied
    # ------------------------------------------------------------------
    import pathlib
    hs       = _as_array(hs,       "hs")
    obs      = _as_array(obs,      "obs")
    taxa_ref = _as_array(taxa_ref, "obs")
    if current_range is not None:
        current_range = _as_array(current_range, "mask")
    if valid_mask is not None:
        valid_mask = _as_array(valid_mask, "mask")

    # ------------------------------------------------------------------
    # Step 1 – Current range & mean HS
    # ------------------------------------------------------------------
    if current_range is None:
        current_range = (obs > 0).astype(np.float32)
    hmean = float(hs[current_range > 0].mean())

    if valid_mask is None:
        valid_mask = hs > 0

    # ------------------------------------------------------------------
    # Step 2 – Presence threshold
    # ------------------------------------------------------------------
    presence_threshold = calibrate_presence_threshold(
        current_range, obs, taxa_ref,
        plot=False, verbose=verbose,
        species_name=species_name,
    )

    # ------------------------------------------------------------------
    # Step 3 – Carrying-capacity parameters
    # ------------------------------------------------------------------
    L, k, x0, khs_bins = estimate_carrying_capacity(
        hs, obs, taxa_ref, current_range,
        hs_bin_width       = hs_bin_width,
        presence_threshold = presence_threshold,
        plot               = False,
        species_name       = species_name,
        return_bins        = True,
    )
    carrying_capacity_params = (L, k, x0)

    # ------------------------------------------------------------------
    # Step 4 – Calibration windows
    # ------------------------------------------------------------------
    calib_sites = sample_calibration_sites(
        hs, obs, taxa_ref,
        n_samples   = n_candidate_samples,
        window_size = window_size,
        distmin     = distmin,
        min_obs     = min_obs,
        time_budget = time_budget,
        plot        = False,
        verbose     = verbose,
    )
    if len(calib_sites) == 0:
        raise RuntimeError(
            "No calibration sites found. Try reducing distmin or min_obs."
        )

    # ------------------------------------------------------------------
    # Step 5 – Posteriors
    # ------------------------------------------------------------------
    posteriors_and_masks = compute_all_posteriors(calib_sites, verbose=verbose)

    # ------------------------------------------------------------------
    # Step 6 – Gradient-descent parameter learning
    # ------------------------------------------------------------------
    n_use   = min(n_calibration_sites, len(calib_sites))
    learner = ParameterLearner(mdd=mdd, max_iter=max_iter)
    Ew, n_param, r, Tg, all_costs, _weights, _endpoints = learner.fit(
        calib_sites,
        hmean                    = hmean,
        posteriors_and_masks     = posteriors_and_masks,
        carrying_capacity_params = carrying_capacity_params,
        n_learning_sites         = n_use,
        all_together             = True,
        n_random_sites           = n_random_sites,
        plot_summary             = False,
        verbose                  = verbose,
    )

    # ------------------------------------------------------------------
    # Step 7 – Summary figure
    # ------------------------------------------------------------------
    if plot or save_path:
        _plot_calibration_summary(
            hs               = hs,
            obs              = obs,
            valid_mask       = valid_mask,
            current_range    = current_range,
            calib_sites      = calib_sites,
            khs_bins         = khs_bins,
            presence_threshold = presence_threshold,
            L=L, k=k, x0=x0,
            all_costs        = all_costs,
            Ew=Ew, n=n_param, r=r, Tg=Tg,
            species_name     = species_name,
            save_path        = save_path,
            show             = plot,
        )

    return dict(
        Ew                 = Ew,
        n                  = n_param,
        r                  = r,
        Tg                 = Tg,
        L                  = L,
        k                  = k,
        x0                 = x0,
        presence_threshold = presence_threshold,
        calib_sites        = calib_sites,
        all_costs          = all_costs,
        khs_bins           = khs_bins,
        hs                 = hs,
        obs                = obs,
        valid_mask         = valid_mask,
    )


# ---------------------------------------------------------------------------
# Public re-plot helper
# ---------------------------------------------------------------------------

def plot_calibration_summary(
    results: dict,
    species_name: str = "",
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """Re-draw the calibration summary figure from a saved ``results`` dict.

    This lets you regenerate or save the figure after calling
    :func:`calibrate` without re-running the full pipeline.

    Parameters
    ----------
    results : dict
        The dict returned by :func:`calibrate`.  Must contain the keys
        ``hs``, ``obs``, ``valid_mask``, ``calib_sites``, ``khs_bins``,
        ``presence_threshold``, ``L``, ``k``, ``x0``, ``all_costs``,
        ``Ew``, ``n``, ``r``, ``Tg``.
    species_name : str
        Overrides the species label in the figure title.
        Defaults to an empty string.
    save_path : str | None
        If provided, save the figure to this file path.
    show : bool
        Display the figure interactively (default True).
    """
    _plot_calibration_summary(
        hs                 = results["hs"],
        obs                = results["obs"],
        valid_mask         = results["valid_mask"],
        current_range      = (results["obs"] > 0).astype(np.float32),
        calib_sites        = results["calib_sites"],
        khs_bins           = results["khs_bins"],
        presence_threshold = results["presence_threshold"],
        L                  = results["L"],
        k                  = results["k"],
        x0                 = results["x0"],
        all_costs          = results["all_costs"],
        Ew                 = results["Ew"],
        n                  = results["n"],
        r                  = results["r"],
        Tg                 = results["Tg"],
        species_name       = species_name,
        save_path          = save_path,
        show               = show,
    )


# ---------------------------------------------------------------------------
# Internal plotting helper
# ---------------------------------------------------------------------------

def _plot_calibration_summary(
    hs, obs, valid_mask, current_range, calib_sites,
    khs_bins, presence_threshold, L, k, x0,
    all_costs, Ew, n, r, Tg,
    species_name="", save_path=None, show=True,
):
    """Produce the single-page calibration summary figure."""

    n_sites   = len(calib_sites)
    half      = calib_sites.window_size // 2
    n_cols_th = min(4, n_sites)
    n_rows_th = math.ceil(n_sites / n_cols_th)
    palette   = plt.cm.tab10(np.linspace(0, 0.9, min(n_sites, 10)))

    title_suffix = f" · {species_name}" if species_name else ""
    fig = plt.figure(figsize=(20, 12), facecolor="white")
    fig.suptitle(
        f"PARADIS  ·  Parameter calibration{title_suffix}",
        fontsize=13, fontweight="bold", y=0.995,
    )

    # ── Top-level grid: HS map (left) | thumbnails + plots (right) ──────
    gs_root = gridspec.GridSpec(
        1, 2,
        figure       = fig,
        width_ratios = [1.0, 1.9],
        wspace       = 0.08,
        left=0.04, right=0.97, top=0.96, bottom=0.10,
    )
    ax_map = fig.add_subplot(gs_root[0, 0])

    gs_right = gridspec.GridSpecFromSubplotSpec(
        2, 1,
        subplot_spec  = gs_root[0, 1],
        height_ratios = [3.8, 1.0],
        hspace        = 0.28,
    )
    # First row is an invisible spacer that reserves room for the section
    # header text above the thumbnails (GridSpecFromSubplotSpec does not
    # accept top/bottom margin kwargs, so we fake the padding with a row).
    gs_thumb = gridspec.GridSpecFromSubplotSpec(
        n_rows_th + 1, n_cols_th,
        subplot_spec  = gs_right[0, 0],
        hspace        = 0.12,
        wspace        = 0.04,
        height_ratios = [0.06] + [1.0] * n_rows_th,
    )
    gs_bottom = gridspec.GridSpecFromSubplotSpec(
        1, 2,
        subplot_spec = gs_right[1, 0],
        wspace       = 0.32,
    )
    ax_conv = fig.add_subplot(gs_bottom[0, 0])
    ax_khs  = fig.add_subplot(gs_bottom[0, 1])

    # ── HS map with presence overlay and site boxes ──────────────────────
    hs_disp = np.where(valid_mask, hs, np.nan)
    im = ax_map.imshow(hs_disp, cmap="viridis", origin="upper",
                       vmin=0, vmax=1, interpolation="none")

    obs_r, obs_c = np.where(obs > 0)
    ax_map.scatter(obs_c, obs_r, s=12, c="lime", marker="o",
                   alpha=0.55, linewidths=0, rasterized=True, label="Observations")

    if calib_sites.rejected_centers:
        rej_r = [c[0] for c in calib_sites.rejected_centers]
        rej_c = [c[1] for c in calib_sites.rejected_centers]
        ax_map.scatter(rej_c, rej_r, s=8, c="red", marker="x",
                       linewidths=0.8, alpha=0.6, zorder=3, label="Rejected")

    for i, (xc, yc) in enumerate(calib_sites.centers):
        col  = palette[i % len(palette)]
        rect = Rectangle(
            (yc - half, xc - half),
            calib_sites.window_size, calib_sites.window_size,
            linewidth=1.8, edgecolor=col, facecolor="none", zorder=3,
        )
        ax_map.add_patch(rect)
        ax_map.text(yc, xc - half - 4, str(i),
                    color=col, fontsize=7.5, ha="center", va="bottom",
                    fontweight="bold", zorder=4)

    cbar = fig.colorbar(im, ax=ax_map, fraction=0.035, pad=0.02)
    cbar.set_label("Habitat suitability", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    ax_map.set_title("Calibration sites on HS map",
                     fontsize=10, fontweight="bold", pad=5)
    ax_map.set_xlabel("x (km)", fontsize=8)
    ax_map.set_ylabel("y (km)", fontsize=8)
    ax_map.tick_params(labelsize=7)
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="lime",
               markersize=4, label="Observations", alpha=0.55),
        Line2D([0], [0], marker="x", color="red", markersize=4,
               markeredgewidth=0.8, linestyle="None", label="Rejected", alpha=0.6),
    ]
    ax_map.legend(handles=legend_handles, loc="lower right", fontsize=7,
                  framealpha=0.75, handletextpad=0.4)
    ax_map.set_aspect("equal", adjustable="box")

    # ── Calibration-window thumbnails ────────────────────────────────────
    pos_th = gs_right[0, 0].get_position(fig)
    fig.text(
        (pos_th.x0 + pos_th.x1) / 2,
        pos_th.y1 + 0.003,
        f"Calibration windows  ({n_sites} sites · "
        f"{calib_sites.window_size}×{calib_sites.window_size} km)  "
        r"—  viridis HS · Otsu contour · ▲ taxa · ● obs",
        ha="center", va="bottom", fontsize=8.5, fontweight="bold",
        transform=fig.transFigure,
    )

    for i in range(n_rows_th * n_cols_th):
        ax_t = fig.add_subplot(gs_thumb[i // n_cols_th + 1, i % n_cols_th])

        if i < n_sites:
            col    = palette[i % len(palette)]
            hs_win = calib_sites.hs_maps[i]
            ob_win = calib_sites.obs_maps[i]
            tx_win = calib_sites.taxa_maps[i]

            ax_t.imshow(hs_win, cmap="viridis", origin="upper", vmin=0, vmax=1)

            gray = hs_win[hs_win != 0]
            title = f"Site {i}"
            if len(gray) > 1:
                thr = skfilters.threshold_otsu(gray)
                try:
                    ax_t.contour(hs_win, levels=[thr], colors="black",
                                 linestyles="--", linewidths=0.7)
                except Exception:
                    pass
                m0, m1 = gray <= thr, gray > thr
                if m0.any() and m1.any():
                    muT   = gray.mean()
                    sigT2 = ((gray - muT) ** 2).mean()
                    sigW2 = (
                        m0.mean() * ((gray[m0] - gray[m0].mean()) ** 2).mean()
                        + m1.mean() * ((gray[m1] - gray[m1].mean()) ** 2).mean()
                    )
                    sep   = (sigT2 - sigW2) / (sigT2 + 1e-12)
                    low   = m0.sum() / (m0.sum() + m1.sum()) * 100
                    title = (r"$\frac{\sigma_{1,2}^2}{\sigma_T^2}$="
                             + f"{sep:.2f}  lo={low:.0f}%")

            xs_tx, ys_tx = np.where(tx_win >= 1)
            if xs_tx.size:
                ax_t.scatter(ys_tx, xs_tx, color="blue", marker="^",
                             alpha=0.35, s=5, linewidths=0)
            xs_ob, ys_ob = np.where(ob_win >= 1)
            if xs_ob.size:
                ax_t.scatter(ys_ob, xs_ob, color="lime", marker="o",
                             s=12, linewidths=0, alpha=0.7)

            ax_t.set_title(title, fontsize=6.0, pad=1.0)
            for sp in ax_t.spines.values():
                sp.set_edgecolor(col)
                sp.set_linewidth(1.8)
            ax_t.text(0.04, 0.96, str(i), transform=ax_t.transAxes,
                      color=col, fontsize=7, fontweight="bold",
                      va="top", ha="left")
        else:
            ax_t.axis("off")

        ax_t.set_xticks([])
        ax_t.set_yticks([])

    # ── Convergence curves ───────────────────────────────────────────────
    for i, costs in enumerate(all_costs):
        arr  = np.asarray(costs, float)
        norm = (arr - arr.min()) / (arr.max() - arr.min() + 1e-9)
        ax_conv.plot(norm, lw=1.5, alpha=0.85, label=f"Run {i + 1}")
    ax_conv.set_xlabel("Optimisation step", fontsize=9)
    ax_conv.set_ylabel("Normalised cost", fontsize=9)
    ax_conv.set_title("Learning convergence  (Adam, mini-batch SGD)",
                      fontsize=9.5, fontweight="bold")
    ax_conv.set_ylim(-0.05, 1.05)
    ax_conv.legend(fontsize=8, ncol=2, framealpha=0.7)
    ax_conv.grid(linestyle="--", linewidth=0.5, alpha=0.4)
    ax_conv.tick_params(labelsize=8)

    # ── K(HS) with bin data ──────────────────────────────────────────────
    hs_line = np.linspace(0.0, 1.0, 300)
    K_line  = logistic(hs_line, L, k, x0)

    for j in range(len(khs_bins["hs_vals"])):
        ax_khs.plot(
            [khs_bins["hs_vals"][j], khs_bins["hs_vals"][j]],
            [khs_bins["ci_lo"][j],   khs_bins["ci_hi"][j]],
            color="grey", linestyle="--", linewidth=0.9,
            label="95 % CI" if j == 0 else "",
            zorder=1,
        )
        ax_khs.text(
            khs_bins["hs_vals"][j], khs_bins["ab_vals"][j],
            str(khs_bins["n_pts"][j]),
            fontsize=6, ha="center", va="bottom", color="#555555",
        )
    ax_khs.scatter(khs_bins["hs_vals"], khs_bins["ab_vals"],
                   color="steelblue", s=28, zorder=3, label="Mode")
    ax_khs.axhline(presence_threshold, color="green", linestyle="--",
                   linewidth=1.1,
                   label=f"Threshold ({presence_threshold:.4f})", zorder=2)
    ax_khs.plot(hs_line, K_line, color="red", lw=2.0, alpha=0.85,
                label="Fitted logistic", zorder=4)
    ax_khs.fill_between(hs_line, 0, K_line, alpha=0.08, color="red", zorder=0)
    ax_khs.set_xlabel("Habitat suitability", fontsize=9)
    ax_khs.set_ylabel(r"Relative abundance $N_{sp}/N_{taxa}$", fontsize=8.5)
    ax_khs.set_title("Carrying capacity  K(HS)", fontsize=9.5, fontweight="bold")
    ax_khs.set_xlim(0, 1)
    ax_khs.set_ylim(0, K_line.max() * 2.0)
    ax_khs.legend(fontsize=7.5, framealpha=0.7)
    ax_khs.grid(linestyle="--", linewidth=0.5, alpha=0.4)
    ax_khs.tick_params(labelsize=8)

    # ── Parameter banner ─────────────────────────────────────────────────
    param_str = (
        f"Ew = {Ew:,.1f}    n = {n:,.1f}    r = {r:.6f}    Tg = {Tg:.3f}"
        "          "
        f"L = {L:.5f}    k = {k:.4f}    x₀ = {x0:.4f}"
    )
    fig.text(
        0.5, 0.012, param_str,
        ha="center", va="bottom",
        fontsize=10, fontweight="bold", family="monospace",
        bbox=dict(
            boxstyle="round,pad=0.45",
            facecolor="#fffbe6",
            edgecolor="#c8a800",
            linewidth=1.3,
            alpha=0.93,
        ),
        transform=fig.transFigure,
    )

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
