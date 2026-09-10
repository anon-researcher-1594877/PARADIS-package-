"""Visualisation utilities for dispersal kernels and simulations.

Key functions
-------------
:func:`plot_dispersal_kernel`
    Overlay a dispersal kernel on a habitat-suitability map.
:func:`plot_calibration_grid`
    Display a grid of calibration windows.
:func:`plot_simulation_result`
    Plot the final simulated distribution.
:func:`plot_parameter_space_3d`
    Interactive 3-D visualisation of the parameter posterior.
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import torch
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap

# `core` has no dependency on `visualization` (growth.py only imports numpy,
# torch, matplotlib, and paradis._device), so importing the shared seeding
# helper here creates no circularity — it lets `plot_calibration_grid` draw
# an illustrative seed mask using EXACTLY the same logic
# `equilibrium_distribution` uses internally, instead of duplicating the
# `torch.rand(...) < seed_fraction` draw.
from paradis.core.growth import seed_mask


def plot_dispersal_kernel(
    kernel: np.ndarray,
    hs_background: np.ndarray,
    mean_dist: float | torch.Tensor | None = None,
    survival: float | torch.Tensor | None = None,
    cmap: str = "viridis",
    save_path: str | None = None,
) -> None:
    """Overlay a dispersal kernel on the habitat background.

    Parameters
    ----------
    kernel:
        2-D dispersal probability map (same shape as *hs_background*).
    hs_background:
        2-D habitat-suitability map used as a grey backdrop.
    mean_dist:
        Mean dispersal distance in pixels – drawn as a red circle.
    survival:
        Mean survival probability – displayed as text.
    cmap:
        Colour map for the kernel.
    save_path:
        If provided, save the figure.
    """
    plt.figure()
    bg = hs_background.copy()
    bg[np.isnan(bg)] = 0.0
    plt.imshow(bg, cmap="Greys", alpha=1.0)

    K = kernel.copy()
    K[K < 0] = 0.0
    Knorm = K / (np.nanmax(K) + 1e-10)
    Knorm[np.isnan(Knorm)] = 0.0
    plt.imshow(K, alpha=Knorm ** (1 / 4), cmap=cmap)
    plt.contour(K, levels=10, cmap=cmap)

    for cut in np.linspace(0, np.nanmax(K), 100):
        if np.nansum(K[K < cut]) > 0.05:
            break
    plt.contour(K, levels=[cut], linestyles="--", colors="white")

    if mean_dist is not None:
        r = float(mean_dist) if isinstance(mean_dist, torch.Tensor) else mean_dist
        cx, cy = K.shape[0] // 2, K.shape[1] // 2
        theta = np.linspace(0, 2 * np.pi, 100)
        plt.plot(
            cy + r * np.cos(theta),
            cx + r * np.sin(theta),
            color="red", linestyle="--", linewidth=1,
            label=f"Mean dispersal distance ({r:.1f} px)",
        )

    if survival is not None:
        s = float(survival) if isinstance(survival, torch.Tensor) else survival
        plt.text(5, 5, f"Survival = {s:.3f}", color="white")

    plt.legend(loc="lower left")
    plt.colorbar(label="Dispersal probability", shrink=0.8)
    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.show()


def plot_calibration_grid(
    hs_maps: list,
    obs_maps: list,
    cmap: str = "plasma",
    title: str = "",
    show_seed_example: bool = True,
    seed_fraction: float = 0.25,
) -> None:
    """Display a grid of calibration windows with observation overlays.

    Parameters
    ----------
    hs_maps:
        List of 2-D HS arrays.
    obs_maps:
        List of 2-D observation arrays.
    cmap:
        Colour map for HS.
    title:
        Figure title.
    show_seed_example:
        If ``True`` (default), also display a SECOND grid, below the main
        HS-map grid, showing — for each site — one illustrative draw of
        the random seeding mask that
        :func:`~paradis.core.growth.equilibrium_distribution` uses to
        build its sparse starting density (see that function's
        ``random_seed_init`` docstring for the full rationale: pixels not
        selected by the mask start at density 0 and can only become
        populated through dispersal, which is what makes the fitted
        equilibrium actually sensitive to the dispersal parameters). This
        is purely illustrative — a single fixed draw per site, shown once,
        for visual sanity-checking that the sparsity/coverage looks
        reasonable on real HS maps. This now matches the actual training
        loop even more closely than before: training draws ONE fixed mask
        per site at the start of a calibration run (via
        :func:`~paradis.core.growth.seed_mask`) and reuses that exact same
        mask for every gradient step / grid point evaluated for that site
        (passed into ``equilibrium_distribution`` as
        ``seed_mask_override``), rather than redrawing fresh on every
        call. This plot's draw is still independent of whichever mask
        training happens to fix for a given run (it's a fresh, one-off
        example draw for visualisation purposes, not literally the same
        mask object training will use), but it is produced by the exact
        same helper `equilibrium_distribution` itself calls — so this
        preview can't drift out of sync with the real seeding logic.
    seed_fraction:
        Fraction of pixels seeded (default 0.25 = 25 %), forwarded to
        :func:`~paradis.core.growth.seed_mask`. Keep this in sync with
        whatever ``seed_fraction`` is actually used for training/inference
        (``equilibrium_distribution``'s own default is also 0.25) so the
        preview stays representative.
    """
    n = len(hs_maps)
    cols = max(1, math.ceil(math.sqrt(n)))
    rows = max(1, math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.array(axes).reshape(-1)
    for i, ax in enumerate(axes):
        if i < n:
            ax.imshow(hs_maps[i], cmap=cmap)
            xs, ys = np.where(obs_maps[i] >= 1)
            ax.scatter(ys, xs, color="red", marker="+", zorder=2)
            ax.contour(hs_maps[i], colors="black", zorder=1, levels=5)
        ax.axis("off")
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    plt.show()

    if show_seed_example:
        seed_cmap = ListedColormap(["#222222", "#f2c744"])  # empty, seeded
        fig2, axes2 = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
        axes2 = np.array(axes2).reshape(-1)
        for i, ax in enumerate(axes2):
            if i < n:
                hs_arr = np.asarray(hs_maps[i])
                mask = seed_mask(hs_arr.shape, seed_fraction=seed_fraction).numpy()
                ax.imshow(mask, cmap=seed_cmap, vmin=0, vmax=1)
                ax.set_title(
                    f"seed example ({mask.mean() * 100:.0f}% shown)", fontsize=8
                )
            ax.axis("off")
        fig2.suptitle(
            (title + " — " if title else "")
            + f"illustrative random seed mask (seed_fraction={seed_fraction:g}, "
            f"one draw per site — redrawn fresh every call in training)"
        )
        plt.tight_layout()
        plt.show()


def plot_simulation_result(
    result: np.ndarray,
    hs_background: np.ndarray,
    presence_threshold: float,
    region_mask: np.ndarray | None = None,
    title: str = "Simulated distribution",
    save_path: str | None = None,
) -> None:
    """Plot the final simulated species distribution.

    Parameters
    ----------
    result:
        2-D abundance map.
    hs_background:
        2-D HS map for the grey background.
    presence_threshold:
        Threshold above which a pixel is considered *present*.
    region_mask:
        Optional binary mask of the study region boundary.
    title:
        Figure title.
    save_path:
        If provided, save the figure.
    """
    plt.figure(figsize=(10, 10))
    bg = hs_background.copy()
    if region_mask is not None:
        bg = bg * region_mask
    plt.imshow(bg, cmap="Greys", alpha=0.5)

    presence = result > presence_threshold
    plotval = np.where(presence, result, np.nan)
    maxval = np.nanmax(plotval) if np.any(~np.isnan(plotval)) else 1.0
    plt.imshow(
        plotval,
        cmap="plasma",
        alpha=np.where(~np.isnan(plotval), plotval / maxval, 0),
        vmin=0,
        vmax=maxval,
        zorder=5,
    )
    plt.contour(
        np.nan_to_num(plotval),
        levels=[maxval / (4 - i) for i in range(3)],
        colors="black",
        linewidths=0.8,
        alpha=0.3,
    )

    if region_mask is not None:
        plt.contour(region_mask, levels=[0.5], colors="black", linewidths=0.5, zorder=6)

    plt.axis("off")
    plt.title(title)
    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.show()


def plot_parameter_space_3d(
    r_vals: np.ndarray,
    n_vals: np.ndarray,
    tg_vals: np.ndarray,
    weights: np.ndarray | None = None,
    trajectories: list | None = None,
    labels: list | None = None,
) -> None:
    """Interactive 3-D scatter of parameter estimates.

    Parameters
    ----------
    r_vals, n_vals, tg_vals:
        Arrays of endpoint values for each calibration site.
    weights:
        Optional per-point weights (controls marker size).
    trajectories:
        Optional list of ``(r_traj, n_traj, tg_traj)`` tuples for
        plotting optimisation paths.
    labels:
        Optional list of labels for each endpoint.
    """
    sizes = (np.ones_like(r_vals) * 8) if weights is None else weights * 20 + 5

    scatter = go.Scatter3d(
        x=r_vals, y=n_vals, z=tg_vals,
        mode="markers+text" if labels is not None else "markers",
        marker=dict(size=sizes, color="red"),
        text=labels,
        textposition="top center",
        name="Endpoints",
    )
    data = [scatter]

    if trajectories is not None:
        for r_tr, n_tr, tg_tr in trajectories:
            data.append(go.Scatter3d(
                x=r_tr, y=n_tr, z=tg_tr,
                mode="lines",
                line=dict(color="lightgrey", width=1),
            ))

    fig = go.Figure(data=data)
    fig.update_layout(
        scene=dict(
            xaxis_title="r (risk scaling)",
            yaxis_title="n (risk avoidance)",
            zaxis_title="S_crit (survival threshold)",
        ),
        title="Dispersal parameter space",
        width=900, height=700,
    )
    fig.show()
