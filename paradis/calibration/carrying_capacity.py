"""Carrying-capacity estimation from habitat suitability and observations.

The carrying capacity ``K(HS)`` is modelled as a logistic function of the
habitat-suitability score.  The parameters are estimated by fitting the
relationship between HS and the per-pixel relative abundance
``r = Obs_sp / Obs_taxa`` within the known current range.

Key functions
-------------
:func:`logistic`
    Logistic curve used to model ``K(HS)``.
:func:`fit_logistic`
    Maximum-likelihood fit of the logistic curve to binned HS–abundance data.
:func:`estimate_carrying_capacity`
    Full pipeline: data extraction → Bayesian credible intervals → logistic fit.
"""

from __future__ import annotations

import math
import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Logistic helpers
# ---------------------------------------------------------------------------

def logistic(x, L: float, k: float, x0: float):
    """Shifted logistic curve such that ``f(0) = 0``.

    Parameters
    ----------
    x:
        Input array or scalar.
    L:
        Asymptotic maximum.
    k:
        Steepness.
    x0:
        Inflection point.

    Returns
    -------
    float or numpy.ndarray
    """
    def g(t):
        z = np.clip(k * (t - x0), -700, 700)
        return L / (1.0 + np.exp(z))

    return g(x) - g(0)


def _neg_log_likelihood(theta, x, y, sigma):
    L, k, x0 = theta
    mu = logistic(x, L, k, x0)
    res = (y - mu) / sigma
    return 0.5 * np.nansum(res * res + np.log(2 * np.pi * sigma * sigma))


def _approx_hessian(fun, x0, eps=None, args=()):
    x0 = np.asarray(x0, dtype=float)
    n = x0.size
    if eps is None:
        eps = np.sqrt(np.finfo(float).eps)

    def grad(x):
        g = np.zeros_like(x)
        fx = fun(x, *args)
        for i in range(n):
            h = eps * max(1.0, abs(x[i]))
            xp = x.copy()
            xp[i] += h
            g[i] = (fun(xp, *args) - fx) / h
        return g

    H = np.zeros((n, n))
    for i in range(n):
        h = eps * max(1.0, abs(x0[i]))
        xp, xm = x0.copy(), x0.copy()
        xp[i] += h
        xm[i] -= h
        H[:, i] = (grad(xp) - grad(xm)) / (2.0 * h)
    return 0.5 * (H + H.T)


def fit_logistic(
    hs_values: np.ndarray,
    abundance_means: np.ndarray,
    abundance_sigmas: np.ndarray,
    unit_interval: bool = False,
    verbose: bool = False,
) -> dict:
    """Fit a logistic curve to binned HS–abundance data.

    Parameters
    ----------
    hs_values:
        Array of bin-centre HS values.
    abundance_means:
        Mean relative abundances per bin.
    abundance_sigmas:
        Standard deviations (or half-widths of 95 % CI / 3.92).
    unit_interval:
        If ``True``, constrain ``L`` and ``x0`` to ``[0, 1]``.
    verbose:
        Print optimiser messages.

    Returns
    -------
    dict
        ``{"L": ..., "k": ..., "x0": ...}``
    """
    x = np.asarray(hs_values, dtype=float)
    y = np.asarray(abundance_means, dtype=float)
    sigma = np.asarray(abundance_sigmas, dtype=float)
    if sigma.ndim == 0:
        sigma = np.full_like(y, float(sigma))

    L0 = max(np.nanmax(y), 1e-3)
    k0 = -1.0  # K(HS) must be increasing → requires k < 0 in the shifted logistic
    x0_0 = float(np.median(x))

    if unit_interval:
        lower = [0.0, -100.0, 0.0]
        upper = [1.0,  100.0, 1.0]
        x_init = np.array([np.clip(L0, 0, 1), k0, np.clip(x0_0, 0, 1)])
    else:
        pad = np.ptp(x) if np.ptp(x) > 0 else 1.0
        lower = [0.0, -1e3, np.min(x) - pad]
        upper = [np.inf, 1e3, np.max(x) + pad]
        x_init = np.array([L0, k0, x0_0])

    obj = lambda th: _neg_log_likelihood(th, x, y, sigma)
    bounds = list(zip(lower, upper))
    res = minimize(obj, x_init, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 20000})
    if verbose:
        print(res.message)
    if not res.success:
        res = minimize(obj, res.x, method="Nelder-Mead",
                       options={"maxiter": 20000})
        if verbose:
            print("Fallback:", res.message)

    L, k, x0 = res.x
    return {"L": float(L), "k": float(k), "x0": float(x0)}


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def estimate_carrying_capacity(
    hs: np.ndarray,
    obs: np.ndarray,
    taxa_ref: np.ndarray,
    current_range: np.ndarray,
    hs_bin_width: float = 0.025,
    presence_threshold: float | None = None,
    plot: bool = False,
    save_path: str | None = None,
    species_name: str | None = None,
    return_bins: bool = False,
) -> tuple:
    """Estimate the logistic K(HS) parameters from observation data.

    For each HS bin, the function builds the combined Bayesian posterior of
    the relative abundance from all pixels within the known range, computes
    the 95 % credible interval, and fits a logistic curve to the bin means.

    Parameters
    ----------
    hs:
        Full habitat-suitability map.
    obs:
        Species observation counts.
    taxa_ref:
        Reference-taxa observation counts.
    current_range:
        Binary map of the known current range.
    hs_bin_width:
        Width of HS bins used to aggregate data.
    presence_threshold:
        Optional presence threshold to overlay on the plot.
    plot:
        If ``True``, display the fitted curve.
    save_path:
        If provided, save the figure.
    species_name:
        Used in figure labels.

    Returns
    -------
    L, k, x0 : float
        Logistic parameters.
    """
    bins = np.arange(0, 1, hs_bin_width)
    hs_vals, ab_vals, sigmas, ci_lo, ci_hi, n_pts = [], [], [], [], [], []
    n_div = 3000
    Rs = np.linspace(0, 1, n_div)

    for hs_bin in tqdm(bins, desc="HS bins", leave=False):
        mask = (hs >= hs_bin) & (hs < hs_bin + hs_bin_width) & (current_range > 0)
        xsp_c = obs * mask
        xtaxa_c = taxa_ref * mask
        valid = (xsp_c > 0) & (xtaxa_c >= xsp_c)
        xsp_v = xsp_c[valid]
        xtaxa_v = xtaxa_c[valid]

        if len(xsp_v) == 0:
            continue

        if len(xsp_v) > 200:
            idx = np.random.choice(len(xsp_v), 200, replace=False)
            xsp_v = xsp_v[idx]
            xtaxa_v = xtaxa_v[idx]

        posteriors = []
        for xtaxa, xsp in zip(xtaxa_v, xsp_v):
            if xtaxa > 2000:
                continue
            rtilde = xsp / xtaxa
            dens = np.array([
                math.factorial(int(xtaxa))
                / (math.factorial(int(xtaxa * rtilde)) * math.factorial(int(xtaxa - xsp)))
                * r ** (rtilde * xtaxa) * (1 - r) ** (xtaxa * (1 - rtilde))
                for r in Rs
            ])
            dens /= np.sum(dens * (1.0 / n_div))
            posteriors.append(dens)

        if not posteriors:
            continue

        combined = np.prod(posteriors, axis=0, dtype=np.float64)
        combined /= np.sum(combined * (1.0 / n_div))
        cdf = np.cumsum(combined * (1.0 / n_div))

        lo = Rs[np.argmax(cdf >= 0.025)]
        hi = Rs[np.argmax(cdf >= 0.975)]
        mode = Rs[np.argmax(combined)]

        hs_vals.append(hs_bin + hs_bin_width / 2)
        ab_vals.append(mode)
        sigmas.append((hi - lo) / (2 * 1.96))
        ci_lo.append(lo)
        ci_hi.append(hi)
        n_pts.append(len(posteriors))

    if not hs_vals:
        # No HS bin contained enough valid presence pixels to compute a
        # posterior. This typically means the species has too few observations
        # to populate even a single HS bin inside the current range.
        # Raising ValueError (instead of just warning) lets the batch pipeline
        # catch it, record the cause, and skip to the next species cleanly.
        raise ValueError(
            "Not enough observations for the species to construct K=f(HS): "
            "no valid HS bin found inside the current range."
        )

    centers = 0.5 * (np.array(ci_lo) + np.array(ci_hi))
    params = fit_logistic(hs_vals, centers, sigmas, verbose=False)
    L, k, x0 = params["L"], params["k"], params["x0"]

    ab_vals_arr = np.array(ab_vals)
    bin_data = dict(
        hs_vals = np.array(hs_vals),
        ab_vals = ab_vals_arr,
        ci_lo   = np.array(ci_lo),
        ci_hi   = np.array(ci_hi),
        n_pts   = list(n_pts),
        min_maxposterior = float(np.min(ab_vals_arr)),
        min_maxposterior_hs = float(np.array(hs_vals)[np.argmin(ab_vals_arr)]),
    )

    if plot or save_path is not None:
        plot_carrying_capacity(
            bin_data, L, k, x0,
            presence_threshold=presence_threshold,
            species_name=species_name,
            save_path=save_path,
            show=plot,
        )

    if return_bins:
        return L, k, x0, bin_data
    return L, k, x0


def plot_carrying_capacity(
    bin_data: dict,
    L: float,
    k: float,
    x0: float,
    presence_threshold: float | None = None,
    species_name: str | None = None,
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """Plot the fitted K=f(HS) logistic curve with optional presence threshold.

    Separated from :func:`estimate_carrying_capacity` so the plot can be
    regenerated with a corrected presence threshold without re-running the
    heavy HS-bin computation.

    Parameters
    ----------
    bin_data:
        Dict with keys ``hs_vals``, ``ab_vals``, ``ci_lo``, ``ci_hi``,
        ``n_pts`` as returned by ``estimate_carrying_capacity(return_bins=True)``.
    L, k, x0:
        Logistic parameters from the same call.
    presence_threshold:
        Horizontal line drawn on the plot (the corrected threshold).
    species_name:
        Used in the figure title.
    save_path:
        If provided, save the figure to this path.
    show:
        If True, call plt.show().
    """
    hs_vals = bin_data["hs_vals"]
    ab_vals = bin_data["ab_vals"]
    ci_lo   = bin_data["ci_lo"]
    ci_hi   = bin_data["ci_hi"]
    n_pts   = bin_data["n_pts"]

    # Lowest per-HS-bin posterior mode ("max a posteriori") across all HS
    # chunks -- a candidate lower bound for K, independent of the logistic
    # fit itself (unlike L, which is the fitted asymptote).
    min_maxposterior = float(np.min(ab_vals))
    min_maxposterior_hs = float(hs_vals[np.argmin(ab_vals)])

    print(f"\033[92m[{species_name or 'species'}] carrying capacity: "
          f"L={L:.6g}  k={k:.6g}  x0={x0:.6g}  "
          f"min_maxposterior={min_maxposterior:.6g} (at HS={min_maxposterior_hs:.3g})\033[0m")

    X = np.linspace(0, 1, 100)
    Y = logistic(X, L, k, x0)

    # Computed here (BEFORE plotting the per-bin `n_pts` labels below) so
    # those labels can be skipped when they'd land outside the y-range --
    # see the full rationale in the ylim block further down.
    K1 = float(logistic(np.array([1.0]), L, k, x0)[0])
    y_top = 1.5 * K1 if np.isfinite(K1) and K1 > 0 else float(np.max(ab_vals))

    plt.figure()
    plt.scatter(hs_vals, ab_vals, label="Mode", color="steelblue")
    for i in range(len(hs_vals)):
        plt.plot([hs_vals[i], hs_vals[i]], [ci_lo[i], ci_hi[i]],
                 color="grey", linestyle="--",
                 label="95 % CI" if i == 0 else "")
        # Skip the n_pts label when the mode itself sits above the visible
        # range -- it would otherwise render outside the axes (or get
        # silently clipped), which is confusing rather than informative;
        # the upward arrow already signals "there's more data here".
        if ab_vals[i] <= y_top:
            plt.text(hs_vals[i], ab_vals[i], str(n_pts[i]), fontsize=7)
    plt.plot(X, Y, color="red", alpha=0.7, linewidth=2.5, label="Fitted logistic")

    if presence_threshold is not None:
        plt.axhline(presence_threshold, color="green", linestyle="--",
                    label=f"Presence threshold ({presence_threshold:.3f})")

    plt.axhline(min_maxposterior, color="black", linestyle=":",
                label=f"Min. max-posterior ({min_maxposterior:.3f} at HS={min_maxposterior_hs:.2g})")

    plt.xlabel("Habitat Suitability")
    plt.ylabel(r"Relative abundance $\frac{N_{sp}}{N_{taxa}}$")
    plt.title(f"Carrying capacity – {species_name or ''}")
    plt.legend()
    plt.grid(linestyle="--", alpha=0.3, color="grey")
    plt.xlim(0, 1)

    # Fix the y-range to ~1.5x K(HS=1) -- the logistic's own value at HS=1,
    # NOT necessarily L itself (only equal when x0 is far enough from 0 --
    # see the Kmax note elsewhere in this codebase) -- so the FIT stays the
    # readable focal point of the plot. Deliberately NOT stretched to fit
    # every data point/CI: a handful of chunks can sit far above the fit
    # (small-n bins, sampling noise) and letting those drive the axis would
    # squash the logistic curve itself into an unreadable sliver. Points
    # that fall outside this range are instead flagged with a small upward
    # arrow (see below) rather than silently cropped. (`K1`/`y_top` were
    # already computed above, before the per-bin labels were drawn.)
    plt.ylim(0, y_top)

    # Small black upward arrows for any chunk whose mode (or CI whisker)
    # actually extends above `y_top` -- signals "there's more data up
    # here" without letting it distort the y-scale itself.
    ax = plt.gca()
    arrow_y0 = y_top * 0.90
    arrow_y1 = y_top * 0.985
    for i in range(len(hs_vals)):
        if ab_vals[i] > y_top or ci_hi[i] > y_top:
            ax.annotate(
                "", xy=(hs_vals[i], arrow_y1), xytext=(hs_vals[i], arrow_y0),
                arrowprops=dict(arrowstyle="-|>", color="black", linewidth=1.3),
                zorder=10,
            )

    if save_path is not None:
        plt.savefig(save_path, dpi=200)
    if show:
        plt.show()
