"""Gradient-based dispersal parameter learning.

Ported faithfully from the PARADIS working script
(PARADIS_learn_dispersal_params_birds.py) with two key speed-ups:

1. Adjacency matrices are precomputed once before the optimisation loop
   (they only depend on the fixed HS data, not on the learned parameters).
2. Per-pixel carrying capacities K_is are precomputed once (they only depend
   on L, k, x0 which are fixed; only the growth coefficient `a` changes with
   the learned Tg and is trivially recomputed each step).

These two changes eliminate repeated O(N^2) adjacency builds and O(N)
logistic evaluations from every gradient step, leaving only the unavoidable
O(N^3) matrix inverse (dispersal kernel) and O(N^2 * n_iter) equilibrium
iteration.

Key functions
-------------
:func:`cost_function`
    Negative average log-likelihood over a mini-batch of calibration sites.
:func:`learn_dispersal_parameters`
    Full Adam optimisation loop.

Classes
-------
:class:`ParameterLearner`
    Object-oriented wrapper.
"""

from __future__ import annotations

import math
import os
import random

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import plotly.graph_objects as go
from sklearn.neighbors import KernelDensity

from paradis._device import device
from paradis.core.adjacency import adjacency_matrix_torch
from paradis.core.dispersal import dispersal_kernel_fast
from paradis.core.growth import equilibrium_distribution


# ---------------------------------------------------------------------------
# Internal helpers (logistic carried capacity, re-parameterisation)
# ---------------------------------------------------------------------------

def _logistic(x, L: float, k: float, x0: float):
    """Shifted logistic so that f(0) = 0 (numpy or torch)."""
    def g(t):
        if isinstance(t, torch.Tensor):
            z = torch.clamp(k * (t - x0), -700.0, 700.0)
            return L / (1.0 + torch.exp(z))
        z = np.clip(k * (t - x0), -700.0, 700.0)
        return L / (1.0 + np.exp(z))
    return g(x) - g(0)


def _tanh_rescale(x: torch.Tensor, low: float, high: float) -> torch.Tensor:
    """Map unconstrained -> [low, high] via tanh (wider gradient than sigmoid)."""
    return low + (high - low) / 2.0 * (1.0 + torch.tanh(x))


def _logistic_rescale(x: torch.Tensor, low: float, high: float) -> torch.Tensor:
    """Map unconstrained -> [low, high] via sigmoid (used for reading endpoints)."""
    return low + (high - low) * torch.sigmoid(x)


def extract_density(
    posteriors: list,
    site_idx: int,
    pixel_idx: int,
    r_sim: torch.Tensor,
) -> torch.Tensor:
    """Linear interpolation of the posterior density at simulated abundance *r_sim*.

    Parameters
    ----------
    posteriors:
        Nested list ``posteriors[site][pixel][r_value]``.
    site_idx, pixel_idx:
        Indices into *posteriors*.
    r_sim:
        Simulated relative abundance (scalar tensor).

    Returns
    -------
    torch.Tensor
        Interpolated density, clamped to ``>= 0.001`` to avoid ``log(0)``.
    """
    density = posteriors[site_idx][pixel_idx]
    n_pts   = len(density)

    # Guard NaN / Inf before indexing
    r_sim = torch.nan_to_num(r_sim, nan=0.0, posinf=1.0, neginf=0.0)
    r_sim = torch.clamp(r_sim, 0.0, 1.0)

    pos  = r_sim * (n_pts - 1)
    idx0 = torch.floor(pos).long()
    idx1 = torch.clamp(idx0 + 1, max=n_pts - 1)
    w    = pos - idx0.float()

    dens_t = torch.tensor(density, dtype=torch.float32, device=r_sim.device)
    densr  = (1.0 - w) * dens_t[idx0] + w * dens_t[idx1]
    return torch.clamp(densr, min=0.001)


# ---------------------------------------------------------------------------
# Cost function
# ---------------------------------------------------------------------------

def cost_function(
    mdd: float,
    posteriors_and_masks: tuple,
    calibration_sites,
    params: tuple,
    batch_indices: list,
    carrying_capacity_params: tuple,
    hmean: float,
    adj_mats: list,
    K_is_list: list,
    plot: bool = False,
    verbose: bool = False,
) -> torch.Tensor:
    """Negative average log-likelihood over *batch_indices* calibration sites.

    Parameters
    ----------
    mdd:
        Prior mean dispersal distance (same units as raster pixels).
    posteriors_and_masks:
        Output of :func:`~paradis.calibration.ratios.compute_all_posteriors`.
    calibration_sites:
        :class:`~paradis.calibration.sites.CalibrationSites` instance.
    params:
        ``(n_scaled, r_scaled, Tg_scaled)`` – current parameter values.
    batch_indices:
        Indices of calibration sites to include in this gradient step.
    carrying_capacity_params:
        Fixed ``(L, k, x0)`` logistic parameters.
    hmean:
        Mean HS within the known current range.
    adj_mats:
        Precomputed adjacency matrices, one per calibration site.
    K_is_list:
        Precomputed flat K_is tensors, one per calibration site.
        (Only the growth coefficient `a = 0.05^(1/Tg)` is recomputed here.)
    plot, verbose:
        Diagnostic flags.

    Returns
    -------
    torch.Tensor
        Scalar loss value.
    """
    posteriors, selected_masks = posteriors_and_masks
    n_param, r_param, Tg_param = params
    L, k, x0 = carrying_capacity_params
    size_site = calibration_sites.hs_maps[0].shape[0]

    # All scalar tensors created on device so autograd stays on-device.
    mdd_t   = torch.tensor(float(mdd),   device=device, dtype=torch.float32)
    hmean_t = torch.tensor(float(hmean), device=device, dtype=torch.float32)

    # Derive Ew from MDD, hmean, and the learned r.
    # Clamp denom away from zero (both below and above) so that Ew stays in
    # a range where p = Ew/(1+Ew) is safely < 1 and (I - p·W*) stays
    # invertible.  Without an upper clamp, denom → 0+ gives Ew → ∞,
    # p → 1, and linalg.inv returns NaN.
    C   = (2.0 * torch.exp(-1.11 / mdd_t)) / (1.0 + torch.exp(-2.0 * 1.11 / mdd_t))
    denom = hmean_t ** r_param - C
    denom = torch.where(denom >= 0, denom.clamp(min=1e-4), denom.clamp(max=-1e-4))
    Ew  = torch.clamp(C / denom, min=1e-3, max=1e4)

    # Growth coefficient from learned Tg (trivial recomputation)
    pt05 = torch.tensor(0.05, device=device, dtype=torch.float32)
    linear_growth = pt05 ** (1.0 / Tg_param)

    total_cost = torch.tensor(0.0, dtype=torch.float32)

    for site_idx in batch_indices:
        hs_t = torch.tensor(
            calibration_sites.hs_maps[site_idx], dtype=torch.float32, device=device
        )
        if torch.isnan(hs_t).any():
            if verbose:
                print(f"NaN in hs for site {site_idx}, skipping.")
            continue

        K_is = K_is_list[site_idx]
        Kd = dispersal_kernel_fast(
            adj_mats[site_idx], r=r_param, n=n_param, ewalk=Ew
        )
        N_inf = equilibrium_distribution(
            K_is.to(device), Kd, linear_growth, plot=plot, verbose=verbose
        )
        N_inf = N_inf.reshape(size_site, size_site)

        if plot:
            plt.figure()
            plt.imshow(N_inf.cpu().detach().numpy(), cmap="viridis")
            plt.colorbar()
            plt.title(f"Equilibrium – site {site_idx}  n={n_param.item():.0f}"
                      f"  r={r_param.item():.5f}  Tg={Tg_param.item():.2f}")
            plt.show()

        # Pixels where we have reference-taxa data and the selected mask
        simulated = N_inf[calibration_sites.taxa_maps[site_idx] > 0][
            selected_masks[site_idx]
        ]
        n_locs = len(simulated)
        if n_locs == 0:
            continue

        site_loss = torch.tensor(0.0, dtype=torch.float32)
        for loc_idx, r_sim in enumerate(simulated):
            dens      = extract_density(posteriors, site_idx, loc_idx, r_sim)
            dens      = torch.clamp(dens, min=1e-12)
            site_loss = site_loss + (-(1.0 / n_locs) * torch.log(dens))

        total_cost = total_cost + site_loss

    return total_cost / max(len(batch_indices), 1)


# ---------------------------------------------------------------------------
# Main optimisation loop
# ---------------------------------------------------------------------------

def learn_dispersal_parameters(
    calibration_sites,
    hmean: float,
    mdd: float,
    posteriors_and_masks: tuple,
    carrying_capacity_params: tuple,
    priors: list | None = None,
    max_iter: int = 500,
    n_learning_sites: int = 5,
    all_together: bool = True,
    n_random_sites: int = 3,
    average_method: str = "same",
    plot: bool = False,
    plot_summary: bool = True,
    verbose: bool = False,
    save_fig_folder: str | None = None,
    species_name: str | None = None,
    seed: int | None = None,
) -> tuple:
    """Estimate dispersal parameters by Adam gradient descent.

    Parameters
    ----------
    calibration_sites:
        :class:`~paradis.calibration.sites.CalibrationSites` instance.
    hmean:
        Mean HS within the known current range.
    mdd:
        Prior mean dispersal distance (pixels = km at 1 km/px).
    posteriors_and_masks:
        Output of :func:`~paradis.calibration.ratios.compute_all_posteriors`.
    carrying_capacity_params:
        Fixed ``(L, k, x0)`` from carrying-capacity estimation.
    priors:
        ``[(n_min, n_max), (r_min, r_max), (Tg_min, Tg_max)]``.
        Defaults: n in [0, 3000], r in [0, r_max], Tg in [0, 10].
    max_iter:
        Adam steps per optimisation run.
    n_learning_sites:
        Number of calibration sites to use when *all_together* is False.
    all_together:
        If True use mini-batches from all sites (stochastic gradient). Default.
    n_random_sites:
        Mini-batch size when *all_together* is True.
    average_method:
        ``"same"`` | ``"size"`` | ``"Likelihood"`` | ``"BIC"``.
    plot, verbose:
        Diagnostic flags.
    seed:
        If given, seeds ``numpy``/``random``/``torch``(+CUDA) before any
        stochastic step, for reproducible runs (mini-batch site sampling,
        Adam initialisation/step order).

    Returns
    -------
    Ew, n, r, Tg : float
        Estimated parameters.
    list_costs : list
        Loss history per run.
    weights : numpy.ndarray
        Averaging weights.
    endpoints : list
        ``[r_vals, n_vals, Tg_vals]`` at convergence.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    L, k, x0 = carrying_capacity_params

    # ------------------------------------------------------------------
    # Priors
    # ------------------------------------------------------------------
    if priors is None:
        C    = (2.0 * np.exp(-1.11 / mdd)) / (1.0 + np.exp(-2.0 * 1.11 / mdd))
        rmax = (1.0 / np.log(hmean)) * np.log(C)
        priors = [(0, 3000), (0, rmax), (0, 10)]

    (nmin, nmax), (rmin, rmax), (tgmin, tgmax) = priors

    # ------------------------------------------------------------------
    # PRECOMPUTATION (outside the optimisation loop)
    # These quantities only depend on the fixed HS data and (L, k, x0).
    # ------------------------------------------------------------------
    print("[precompute] Building adjacency matrices and K_is ...")
    n_sites_total = len(calibration_sites)
    adj_mats  = []
    K_is_list = []

    for hs_map in tqdm(calibration_sites.hs_maps, desc="  Sites", leave=False):
        hs_t    = torch.tensor(hs_map, dtype=torch.float32, device=device)
        adj_mat = adjacency_matrix_torch(hs_t)
        adj_mats.append(adj_mat)

        K_is_flat = torch.tensor(
            _logistic(hs_map.flatten().astype(np.float32), L, k, x0),
            dtype=torch.float32,
        )
        K_is_list.append(K_is_flat)

    print(f"[precompute] Done — {n_sites_total} sites.")

    # ------------------------------------------------------------------
    # Learning sites selection
    # ------------------------------------------------------------------
    if all_together:
        learning_runs = [0]          # one run over random mini-batches
    else:
        n_use = min(n_learning_sites, n_sites_total)
        learning_runs = list(
            np.random.choice(n_sites_total, size=n_use, replace=False)
        )

    # GPU availability — critical for reasonable speed.
    if torch.cuda.is_available():
        print(f"[learning] GPU: {torch.cuda.get_device_name(0)}  "
              f"(memory: {torch.cuda.get_device_properties(0).total_memory // 2**20} MB)")
    else:
        print("[learning] WARNING: CUDA not available — running on CPU.\n"
              "           Matrix inverse of a 70x70 window (~4900x4900) is O(N^3)\n"
              "           and is very slow on CPU.  GPU is strongly recommended.\n"
              "           Expected speed on CPU: ~1 step per several seconds.")

    # ------------------------------------------------------------------
    # Optimisation
    # ------------------------------------------------------------------
    all_costs, all_full_cost, all_n, all_r, all_tg = [], [], [], [], []
    n_runs = len(learning_runs)
    print("\033[93m[Optimization steps will be printed every 100 steps ... please wait]\033[0m")

    for run_idx, site_loop_idx in enumerate(
        tqdm(learning_runs, desc="Optimisation run", position=0, leave=True)
    ):
        # Parameters created ON device so all autograd stays on GPU when available.
        params    = [torch.nn.Parameter(torch.tensor(0.0, device=device, requires_grad=True))
                     for _ in range(3)]
        optimizer = torch.optim.Adam(params)
        losses, ln, lr, ltg = [], [], [], []
        full_cost_hist: list[tuple[int, float]] = []   # (step, full-dataset cost)

        run_label = (
            f"Steps (run {run_idx+1}/{n_runs})" if n_runs > 1 else "Steps"
        )
        step_bar = tqdm(
            range(max_iter),
            desc=f"  {run_label}",
            position=1,
            leave=True,
        )

        for step in step_bar:
            # Mini-batch selection
            if all_together:
                batch = list(
                    np.random.choice(
                        n_sites_total,
                        size=min(n_random_sites, n_sites_total),
                        replace=False,
                    )
                )
            else:
                batch = [site_loop_idx]

            n_raw, r_raw, tg_raw = params
            n_s  = _tanh_rescale(n_raw,  nmin,  nmax)
            r_s  = _tanh_rescale(r_raw,  rmin,  rmax)
            tg_s = _tanh_rescale(tg_raw, tgmin, tgmax)

            do_plot = plot and (step % 50 == 0)
            loss = cost_function(
                mdd, posteriors_and_masks, calibration_sites,
                (n_s, r_s, tg_s),
                batch,
                carrying_capacity_params,
                hmean,
                adj_mats,
                K_is_list,
                plot=do_plot,
                verbose=verbose,
            )

            optimizer.zero_grad()
            loss.backward()

            # Read gradients BEFORE clipping (to show raw signal magnitude).
            grads = [p.grad.item() if p.grad is not None else float("nan")
                     for p in params]

            # NaN gradient guard
            if any(math.isnan(g) for g in grads):
                raise RuntimeError(
                    f"NaN gradient at step {step}  "
                    f"params={[p.item() for p in params]}  grads={grads}"
                )

            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            torch.nn.utils.clip_grad_value_(params, clip_value=5.0)
            optimizer.step()

            with torch.no_grad():
                loss_val = loss.item()
                n_val    = _logistic_rescale(params[0], nmin, nmax).item()
                r_val    = _logistic_rescale(params[1], rmin, rmax).item()
                tg_val   = _logistic_rescale(params[2], tgmin, tgmax).item()
                losses.append(loss_val)
                ln.append(n_val)
                lr.append(r_val)
                ltg.append(tg_val)

            # Update progress bar: parameters + raw gradient magnitudes.
            step_bar.set_postfix(
                loss=f"{loss_val:.4f}",
                n=f"{n_val:.0f}",
                r=f"{r_val:.5f}",
                Tg=f"{tg_val:.2f}",
                gn=f"{grads[0]:+.3f}",
                gr=f"{grads[1]:+.3f}",
                gT=f"{grads[2]:+.3f}",
            )

            # Full-dataset cost every 100 steps (visualisation only, no gradient).
            if step % 100 == 0:
                with torch.no_grad():
                    full_loss = cost_function(
                        mdd, posteriors_and_masks, calibration_sites,
                        (n_s, r_s, tg_s),
                        list(range(n_sites_total)),
                        carrying_capacity_params,
                        hmean,
                        adj_mats,
                        K_is_list,
                        plot=False,
                        verbose=False,
                    )
                full_cost_hist.append((step, full_loss.item()))

            if step % 100 == 0:
                print(f"  step {step:4d} | loss={loss_val:.5f}"
                      f" | n={n_val:7.1f}  r={r_val:.6f}  Tg={tg_val:.3f}"
                      f" | grads: n={grads[0]:+.4f}  r={grads[1]:+.4f}"
                      f"  Tg={grads[2]:+.4f}"
                      f" | batch={list(batch)}")
                print(f"\033[93m  step {step:4d} | full-dataset cost="
                      f"{full_loss.item():.5f}  (n={n_val:7.1f}  r={r_val:.6f}"
                      f"  Tg={tg_val:.3f})\033[0m")

        step_bar.close()
        all_costs.append(losses)
        all_full_cost.append(full_cost_hist)
        all_n.append(ln)
        all_r.append(lr)
        all_tg.append(ltg)

    # ------------------------------------------------------------------
    # Aggregate endpoints
    # ------------------------------------------------------------------
    r_ends  = np.array([lr[-1]  for lr  in all_r])
    n_ends  = np.array([ln[-1]  for ln  in all_n])
    tg_ends = np.array([ltg[-1] for ltg in all_tg])

    if average_method == "Likelihood":
        final_ll = np.array([-c[-1] for c in all_costs])
        adj_ll   = final_ll - final_ll.max()
        exp_adj  = np.exp(adj_ll)
        weights  = exp_adj / exp_adj.sum()

    elif average_method == "BIC":
        posteriors, selected_masks = posteriors_and_masks
        n_locs = np.array([m.sum() for m in selected_masks])[learning_runs]
        final_ll = np.array([-c[-1] * nl for c, nl in zip(all_costs, n_locs)])
        bic      = -2.0 * final_ll + 3.0 * np.log(np.maximum(n_locs, 1))
        diff_bic = np.exp(-0.5 * (bic - bic.min()))
        weights  = diff_bic / diff_bic.sum()

    elif average_method == "size":
        posteriors, selected_masks = posteriors_and_masks
        sizes   = np.array([m.sum() for m in selected_masks])[learning_runs]
        sizes   = sizes / sizes.max()
        weights = sizes

    else:   # "same"
        weights = np.ones(len(r_ends)) / len(r_ends)

    mean_r  = float(np.sum(r_ends  * weights))
    mean_n  = float(np.sum(n_ends  * weights))
    mean_tg = float(np.sum(tg_ends * weights))

    C            = (2.0 * np.exp(-1.11 / mdd)) / (1.0 + np.exp(-2.0 * 1.11 / mdd))
    estimated_ew = float(C / (hmean ** mean_r - C))

    # ------------------------------------------------------------------
    # Summary plots  (suppressed when plot_summary=False)
    # ------------------------------------------------------------------
    if plot_summary:
        prefix = f"{species_name}_" if species_name else ""

        def _save_show(filename: str) -> None:
            if save_fig_folder:
                plt.savefig(os.path.join(save_fig_folder, filename),
                            dpi=150, bbox_inches="tight")
            plt.show()

        # Normalised cost curves
        plt.figure()
        for idx in range(len(all_costs)):
            arr  = np.array(all_costs[idx])
            norm = (arr - arr.min()) / (arr.max() - arr.min() + 1e-9)
            label = (f"run {learning_runs[idx]}" if not all_together
                     else f"mini-batch run {idx+1}")
            plt.plot(norm, alpha=0.5, label=label)

            # Full-dataset cost overlay (red "+" markers + red curve)
            full_hist = all_full_cost[idx]
            if full_hist:
                fc_steps = np.array([s for s, _ in full_hist])
                fc_vals  = np.array([v for _, v in full_hist])
                fc_norm  = (fc_vals - arr.min()) / (arr.max() - arr.min() + 1e-9)
                plt.plot(fc_steps, fc_norm, color="red", linewidth=1.2,
                         label="full-dataset cost")
                plt.plot(fc_steps, fc_norm, "+", color="red", markersize=8)

        plt.xlabel("Optimisation step")
        plt.ylabel("Normalised cost")
        plt.title("Cost function evolution during optimisation")
        plt.legend()
        plt.grid(linestyle="--", color="grey", linewidth=0.2, alpha=0.5)
        _save_show(f"{prefix}cost.png")

        # Endpoint scatter (n vs r, r vs Tg, n vs Tg)
        sizes_scatter = (
            np.ones(len(r_ends)) * 100
            if average_method in ("Likelihood", "same")
            else weights * 200
        )
        for xlabel, ylabel, x, y in [
            ("r (alpha)", "n (tolerance)", r_ends, n_ends),
            ("r (alpha)", "Tg (tgrowth)",  r_ends, tg_ends),
            ("n (tolerance)", "Tg (tgrowth)", n_ends, tg_ends),
        ]:
            plt.figure()
            plt.scatter(x, y, s=sizes_scatter, color="blue")
            plt.scatter(
                (mean_r if "r" in xlabel else mean_n),
                (mean_n if "n" in ylabel else mean_tg),
                color="red", s=150, marker="X",
                label="Weighted estimate",
            )
            for idx in range(len(all_costs)):
                px = np.array(all_r[idx] if "r" in xlabel else all_n[idx])
                py = np.array(all_n[idx] if "n" in ylabel else all_tg[idx])
                plt.plot(px, py, color="lightgrey", alpha=0.5)
                plt.text(x[idx], y[idx], str(learning_runs[idx] if not all_together else idx),
                         fontsize=8)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title(f"Endpoint projections: {ylabel} vs {xlabel}")
            plt.grid(linestyle="--", color="grey", linewidth=0.2, alpha=0.5)
            plt.legend()
            slug = f"{ylabel.split()[0].lower()}_vs_{xlabel.split()[0].lower()}"
            _save_show(f"{prefix}scatter_{slug}.png")

        # 3-D interactive plot (Plotly)
        _kde_3d_plotly(
            r_ends, n_ends, tg_ends,
            priors=priors,
            list_traj=[all_n, all_r, all_tg],
            weights=weights,
            sizes_scatter=weights * 20,
            learning_runs=learning_runs,
            all_together=all_together,
        )

    return estimated_ew, mean_n, mean_r, mean_tg, all_costs, weights, [r_ends, n_ends, tg_ends]


# ---------------------------------------------------------------------------
# 3-D KDE endpoint visualisation (Plotly)
# ---------------------------------------------------------------------------

def _kde_3d_plotly(
    r_ends: np.ndarray,
    n_ends: np.ndarray,
    tg_ends: np.ndarray,
    priors: list,
    list_traj: list,
    weights: np.ndarray,
    sizes_scatter,
    learning_runs: list,
    all_together: bool,
    grid_size: int = 30,
):
    """Adaptive 3-D KDE of the endpoint cloud (Plotly Volume + trajectories)."""
    (nmin, nmax), (rmin, rmax), (tgmin, tgmax) = priors

    # Trajectories
    data_lines = []
    for ln, lr, ltg in zip(list_traj[0], list_traj[1], list_traj[2]):
        data_lines.append(go.Scatter3d(
            x=lr, y=ln, z=ltg,
            mode="lines", line=dict(color="lightgrey", width=1),
        ))

    # Endpoint scatter
    pts = go.Scatter3d(
        x=r_ends, y=n_ends, z=tg_ends,
        mode="markers+text",
        marker=dict(size=sizes_scatter, color="red"),
        text=[str(lr if not all_together else i)
              for i, lr in enumerate(learning_runs)],
        textposition="middle right",
        textfont=dict(color="black", size=8),
        name="Endpoints",
    )

    if not all_together and len(r_ends) > 1:
        # Adaptive KDE
        xyz  = np.vstack([r_ends, n_ends, tg_ends]).T
        sigma = np.std(xyz, axis=0) + 1e-9
        bw    = len(xyz) ** (-1.0 / (3 + 4))
        kde   = KernelDensity(bandwidth=bw, kernel="gaussian")
        kde.fit(xyz / sigma, sample_weight=weights)

        xg = np.linspace(r_ends.min(), r_ends.max(), grid_size)
        yg = np.linspace(n_ends.min(), n_ends.max(), grid_size)
        zg = np.linspace(tg_ends.min(), tg_ends.max(), grid_size)
        X, Y, Z = np.meshgrid(xg, yg, zg)
        gp      = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
        density = np.exp(kde.score_samples(gp / sigma))
        mode    = gp[np.argmax(density)]

        vol = go.Volume(
            x=gp[:, 0], y=gp[:, 1], z=gp[:, 2],
            value=density,
            isomin=np.percentile(density, 70),
            isomax=density.max(),
            opacity=0.15,
            surface_count=20,
            caps=dict(x_show=False, y_show=False, z_show=False),
        )
        mode_pt = go.Scatter3d(
            x=[mode[0]], y=[mode[1]], z=[mode[2]],
            mode="markers+text",
            marker=dict(size=3, color="black", symbol="x"),
            text=["MODE"], textposition="top center",
            name="KDE mode",
        )
        fig_data = [vol, pts, mode_pt] + data_lines
        title = f"3D KDE (adaptive bandwidth = {bw:.3f})"
    else:
        fig_data = [pts] + data_lines
        title    = "3D parameter trajectory"

    fig = go.Figure(data=fig_data)
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="r (alpha)",
            yaxis_title="n (tolerance)",
            zaxis_title="Tg (growth time)",
            xaxis=dict(range=[rmin, rmax]),
            yaxis=dict(range=[nmin, nmax]),
            zaxis=dict(range=[tgmin, tgmax]),
        ),
        width=900, height=700,
    )
    fig.show()


# ---------------------------------------------------------------------------
# Object-oriented wrapper
# ---------------------------------------------------------------------------

class ParameterLearner:
    """Gradient-based learner for PARADIS dispersal parameters.

    Parameters
    ----------
    mdd:
        Prior mean dispersal distance (pixels = km at 1 km/px).
    priors:
        ``[(n_min, n_max), (r_min, r_max), (Tg_min, Tg_max)]``.
    max_iter:
        Adam steps per optimisation run.

    Examples
    --------
    >>> from paradis.learning import ParameterLearner
    >>> learner = ParameterLearner(mdd=9.0, max_iter=300)
    >>> Ew, n, r, Tg, *_ = learner.fit(
    ...     calibration_sites, hmean, posteriors_and_masks, (L, k, x0),
    ...     all_together=True, n_random_sites=3,
    ... )
    """

    def __init__(
        self,
        mdd: float,
        priors: list | None = None,
        max_iter: int = 500,
    ) -> None:
        self.mdd      = mdd
        self.priors   = priors
        self.max_iter = max_iter

    def fit(
        self,
        calibration_sites,
        hmean: float,
        posteriors_and_masks: tuple,
        carrying_capacity_params: tuple,
        **kwargs,
    ) -> tuple:
        """Run parameter estimation (keyword arguments forwarded to
        :func:`learn_dispersal_parameters`)."""
        return learn_dispersal_parameters(
            calibration_sites        = calibration_sites,
            hmean                    = hmean,
            mdd                      = self.mdd,
            posteriors_and_masks     = posteriors_and_masks,
            carrying_capacity_params = carrying_capacity_params,
            priors                   = self.priors,
            max_iter                 = self.max_iter,
            **kwargs,
        )

    def __repr__(self) -> str:
        return f"ParameterLearner(mdd={self.mdd}, max_iter={self.max_iter})"
