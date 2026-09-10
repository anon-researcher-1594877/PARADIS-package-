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
import scipy.ndimage
import torch
import matplotlib.pyplot as plt
from tqdm.auto import tqdm   # auto-picks ipywidgets bars in Jupyter/VSCode
                              # notebooks, plain console \r bars elsewhere —
                              # plain `tqdm` (console-only) doesn't render
                              # correctly in a notebook, especially with the
                              # nested position=0/position=1 bars used below.
import plotly.graph_objects as go
from sklearn.neighbors import KernelDensity

from paradis._device import device
from paradis.core.adjacency import adjacency_matrix_torch
from paradis.core.dispersal import dispersal_kernel_fast
from paradis.core.growth import equilibrium_distribution, seed_mask


# ---------------------------------------------------------------------------
# Internal helpers (logistic carried capacity, re-parameterisation)
# ---------------------------------------------------------------------------

def _build_breeding_masks_list(calibration_sites, n_sites_total: int) -> list:
    """Per-site `breeding_ground` masks for `cost_function`/
    `equilibrium_distribution`, built from `calibration_sites.breeding_maps`
    (see `sample_calibration_sites`'s `breeding_mask` parameter) — one
    entry per site, `None` wherever that site has no breeding-range window
    (unconstrained growth there, identical to the behaviour before this
    parameter existed). Safe against `CalibrationSites` instances built
    without `breeding_maps` at all (older code, or constructed by hand) —
    falls back to `None` for every site in that case, rather than raising.
    """
    breeding_maps = getattr(calibration_sites, "breeding_maps", None)
    if not breeding_maps or len(breeding_maps) != n_sites_total:
        return [None] * n_sites_total
    return [
        # Flattened — `Un`/`growth` inside `equilibrium_distribution` are
        # flat (N,), same convention as `K_is_flat`/`seed_mask`, NOT the
        # raw 2-D window shape `breeding_maps` stores.
        (torch.tensor(np.asarray(bm).flatten(), dtype=torch.float32) if bm is not None else None)
        for bm in breeding_maps
    ]


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
    """Read-only alias for `_tanh_rescale` — MUST stay mathematically
    identical to it (same tanh formula), since every readout of the
    "current" n/r/Tg (progress-bar postfix, periodic prints, full-dataset
    checkpoint evaluations, the final returned endpoint) uses this
    function while the actual optimisation (the forward pass whose loss
    is backpropagated) uses `_tanh_rescale` directly. A previous version
    used `sigmoid(x)` here instead — mathematically DIFFERENT from
    `_tanh_rescale`'s `(1+tanh(x))/2` (they only agree at x=0) — which
    silently made every reported/returned parameter value wrong relative
    to what Adam was actually optimising (confirmed: a single Adam step,
    raw parameter moving by exactly the expected ~lr, could read back as
    an enormous, unrelated jump in the reported n/r/Tg)."""
    return low + (high - low) / 2.0 * (1.0 + torch.tanh(x))


def _inv_tanh_rescale(target: float, low: float, high: float) -> float:
    """Inverse of `_tanh_rescale`: raw value x such that
    _tanh_rescale(x, low, high) == target, for initialising Adam's raw
    parameter at a specific chosen point in [low, high] instead of always
    at the box's midpoint (x=0)."""
    frac = (target - low) / (high - low) * 2.0 - 1.0
    frac = float(np.clip(frac, -0.999999, 0.999999))
    return float(np.arctanh(frac))


# ---------------------------------------------------------------------------
# (s, d) reparametrisation of (n, r) — see module-level note below.
#
# `n` and `r` only enter the dispersal kernel's SELECTIVITY pattern (the
# relative weight assigned to each neighbouring pixel, before the outer
# multiplicative HS^r "survival" factor — see `weighted_adjacency`) through
# their PRODUCT: substituting y_i = r*ln(hs_i) turns
# `hs_i^(n*r) / sum_j hs_j^(n*r)` into `softmax(n*y)_i`, a softmax of
# "temperature" 1/n over VALUES y_i that already contain r. So (n, r) are
# only identifiable, as far as this selectivity pattern is concerned,
# through s := log10(n) + log10(r) = log10(n*r) — moving along a curve of
# constant s (i.e. varying n and r inversely so their product stays fixed)
# leaves the selectivity pattern *exactly* unchanged, and only perturbs the
# model through the residual r-only "survival" terms that live outside the
# softmax (the HS^r multiplicative factor in `weighted_adjacency`, and
# Ew(r) — both are the same "survival per step" mechanism, since Ew is
# itself derived from hmean^r). That makes s the INFORMATIVE axis and
# d := log10(n) - log10(r) (motion strictly along a fixed-s curve) the
# near-flat, weakly-identifiable one — the two are orthogonal by
# construction (grad(s) = (1,1), grad(d) = (1,-1) in (log10 n, log10 r)
# space, dot product 0), so sampling/learning directly in (s, d) instead
# of independently in (log10 n, log10 r) aligns the search with the
# model's actual identifiable structure instead of fighting a ~45-degree
# ridge in the original axes.
#
# The characteristic scale of this softmax's "uniform -> argmax" crossover
# (see `_s_center`) sits at n*r = 1/Delta, where Delta = ln(hs_1/hs_2) is
# the LOG-ratio gap between two competing habitat values — for small
# relative differences delta = hs_1 - hs_2 << hs, Delta ~= delta/hs, so
# Delta's typical (population) scale is well approximated by the ratio of
# the mean adjacent-pixel HS difference to the mean HS level itself:
# Delta_bar ~= delta_H_bar / hs_bar. Hence n*r ~= hs_bar / delta_H_bar,
# giving a data-driven centring point for s independent of any arbitrarily
# chosen r (see `_s_center`).
# ---------------------------------------------------------------------------

def _hs_local_contrast(calibration_sites) -> tuple[float, float]:
    """Estimate the mean HS level and mean adjacent-pixel HS contrast from
    the calibration sites' HS windows, for use in `_s_center`.

    Parameters
    ----------
    calibration_sites:
        Any object exposing `.hs_maps` (a sequence of 2-D HS arrays), e.g.
        :class:`~paradis.calibration.sites.CalibrationSites`.

    Returns
    -------
    hs_bar : float
        Mean HS value across all valid (finite) pixels of all sites.
    delta_h_bar : float
        Mean absolute HS difference between 4-connected adjacent pixel
        pairs, across all sites — the empirical analogue of `bar{Delta H}`
        in the (n, r) sensitivity-centre derivation.
    """
    levels: list[np.ndarray] = []
    diffs: list[np.ndarray] = []
    for hs_map in calibration_sites.hs_maps:
        hs = np.asarray(hs_map, dtype=np.float64)
        valid = np.isfinite(hs)
        levels.append(hs[valid])

        d_right = np.abs(hs[:, :-1] - hs[:, 1:])
        v_right = valid[:, :-1] & valid[:, 1:]
        diffs.append(d_right[v_right])

        d_down = np.abs(hs[:-1, :] - hs[1:, :])
        v_down = valid[:-1, :] & valid[1:, :]
        diffs.append(d_down[v_down])

    hs_bar = float(np.concatenate(levels).mean())
    delta_h_bar = float(np.concatenate(diffs).mean())
    return hs_bar, delta_h_bar


def _s_center(hs_bar: float, delta_h_bar: float) -> float:
    """Data-driven centre of the informative axis `s = log10(n) + log10(r)
    = log10(n*r)`, from the mean HS level and mean adjacent-pixel HS
    contrast (see `_hs_local_contrast`): `s* = log10(hs_bar / delta_h_bar)`.

    Note `n* * r = 1/Delta_bar` (see module note above) does NOT depend on
    r itself — it's a property of the habitat data alone — so this is a
    single fixed value, not a function of the currently-explored r.
    """
    delta_h_bar = max(delta_h_bar, 1e-6)   # guard against a degenerate (flat) HS map
    return float(np.log10(hs_bar / delta_h_bar))


def _Scrit_box() -> tuple[float, float]:
    """Fixed box `(1/3, 1.0)` for the learning/sampling variable `S_crit`
    — the critical PER-DISPERSAL-STEP survival probability below which NO
    growth rate `g` can sustain a locally-growing population. This is the
    module's growth-timescale learning axis, replacing both the old
    geometrically-motivated `ell`/`_ell_box` (characteristic penetration
    distance) AND an intermediate `x_crit = h_crit**r` candidate (see the
    derivation below for why that candidate was abandoned). `g` (NOT
    `Tg`/`a` — those are not used anywhere in this package any more) is
    the model's actual growth coefficient, used directly in
    `~paradis.core.growth.equilibrium_distribution`/`growth_step`.

    Derivation. Consider a locally-homogeneous patch with per-step
    survival probability `S` (whatever its origin — `S` itself, not a
    habitat-quality proxy for it, is the quantity that actually enters
    the invasion criterion below). A population can grow from rare in
    this patch iff the combined dispersal-then-growth map, linearised
    near zero density, has slope > 1:

        S*(1 + g) > 1

    Solving `S*(1+g) = 1` for the CRITICAL survival probability (the
    threshold below which even this specific `g` cannot sustain growth):

        S_crit = 1 / (1 + g)                                [forward formula]

    Inverting (solve for `g` given a chosen `S_crit`, same equation run
    backwards):

        g(S_crit) = (1 - S_crit) / S_crit = 1/S_crit - 1     [inverse formula]

    The upper bound on `g` (hence the lower bound on `S_crit`) comes from
    the discrete logistic recursion's own dynamical stability, NOT from
    any Tg-based parametrisation: `Un_new = Un + g*Un*(1-Un/K)` has a
    fixed point at `Un=K`, with linear stability `f'(K) = 1 - g`, so
    `|f'(K)| < 1` (the fixed point genuinely attracting, no
    oscillation/period-doubling) requires `0 < g < 2`. `g -> 0+` gives
    `S_crit -> 1-` (infinitesimally slow growth needs an arbitrarily high
    per-step survival probability to persist), and `g -> 2-` (the
    dynamical-stability limit) gives `S_crit -> 1/(1+2) = 1/3+`. So:

        S_crit in (1/3, 1.0)

    `g in [2, 3)` still converges (oscillating: period-2, then a
    period-doubling cascade toward chaos) but is excluded here to keep
    the sampled region strictly within the well-behaved, non-oscillatory
    regime; `g >= 3` makes the UNCLAMPED recursion diverge outright (the
    package clamps density to `[0, 1]` regardless, so this never produces
    `NaN`/`inf` in practice, but is a numerically nonsensical regime to
    sample from — a population violently oscillating between 0 and
    saturation every single step is not a meaningful biological answer).

    Crucially — and this is the whole point of using `S_crit` rather than
    `x_crit = h_crit**r` (an earlier candidate that was tried and
    abandoned) — this box is a FIXED UNIVERSAL CONSTANT, `(1/3, 1.0)`,
    the SAME for every species, with NO dependency whatsoever on `mdd`,
    `r`, or `n`. The `x_crit` candidate's box was `(x_min, 1.0)` with
    `x_min = 2*C / (1 + C)`, `C = 2*exp(-alpha/mdd) / (1 + exp(-2*alpha/mdd))`
    — and `C -> 1` as `mdd` grows, which pushes `x_min -> 1` too, i.e. the
    box COLLAPSES to a vanishingly thin sliver for large-`mdd` species.
    Confirmed numerically pathological for a real example with
    `mdd=73.33`: the resulting `x_crit` box had width ~1e-5, making the
    tanh-bounded raw parameter's usable range (and any grid/MCMC sampling
    resolution within it) absurdly, uselessly narrow. `S_crit` sidesteps
    this failure mode entirely — its box is always exactly `(1/3, 1.0)`,
    well-scaled and numerically benign regardless of species, since it
    comes directly from the model's `S*(1+g)>1` invasion/viability
    dynamics rather than from any per-species habitat-quality quantity.
    No per-species computation is needed for the box bounds themselves —
    this function takes no arguments.

    Returns
    -------
    (1/3, 1.0) : tuple[float, float]
    """
    return 1.0 / 3.0, 1.0


def _Scrit_to_Tg(S_crit):
    """`S_crit -> Tg`, via the inverse survival-threshold formula (see
    `_Scrit_box`'s docstring for the full derivation):

        g = (1 - S_crit) / S_crit
        a = 1 - g = (2*S_crit - 1) / S_crit
        Tg = ln(20) / (-ln(a))

    (`a = 0.05**(1/Tg) => Tg = ln(0.05)/ln(a) = -ln(20)/ln(a) =
    ln(20)/(-ln(a))`.) Unlike the old `_ell_to_Tg`/`_xcrit_to_Tg`, this
    conversion needs NO `mdd`/`alpha` at all — `S_crit` is already a
    per-step survival probability, not a habitat-quality proxy that first
    has to be converted through `C`/`Ew`, so there is nothing
    species-specific left in this formula. Works for floats, numpy
    arrays, and torch tensors — branches on `-log(a)` (torch tensors need
    `torch.log`, not `math.log`/`np.log`) so autograd flows through when
    `S_crit` is a tensor with `requires_grad=True`, exactly mirroring how
    `_ell_to_Tg`/`_xcrit_to_Tg` used to branch on tensor vs. float.
    """
    g = (1.0 - S_crit) / S_crit
    a = 1.0 - g
    if isinstance(S_crit, torch.Tensor):
        neg_log_a = -torch.log(a)
    else:
        neg_log_a = -np.log(a)
    return math.log(20.0) / neg_log_a


def _Tg_to_Scrit(Tg):
    """`Tg -> S_crit`, the inverse of `_Scrit_to_Tg`: `a = 0.05**(1/Tg)`,
    `g = 1 - a`, `S_crit = 1/(1+g)` (the forward formula from
    `_Scrit_box`'s docstring). Needed for `init_params`/`init_point` APIs
    that still accept a `Tg` starting value. Works for floats, numpy
    arrays, or torch tensors (`**` and `/` are polymorphic across all of
    them, so no isinstance branching is needed here, unlike
    `_Scrit_to_Tg`).
    """
    a = 0.05 ** (1.0 / Tg)
    g = 1.0 - a
    return 1.0 / (1.0 + g)


def _s_range(s_center: float, s_halfwidth: float = 3.0) -> tuple[float, float]:
    """Data-driven range for the informative axis `s = log10(n) + log10(r)
    = log10(n*r)`, centred on `s_center` (see `_s_center`) and spanning
    `s_halfwidth` decades either side — the full "uniform -> argmax"
    softmax crossover is essentially complete within ~1-2 decades of its
    centre, see the module note above `_hs_local_contrast`.

    Extracted out of `_sd_box` so it can be evaluated on its own, before
    `n`'s box is known — `s_center` (and hence this range) depends only on
    the calibration sites' HS data (via `_hs_local_contrast`/`_s_center`),
    never on `n` or `r`, so it can and should be computed first and then
    used to derive a data-driven `n` box (see `_n_box_from_s_and_r`)
    instead of the other way around.

    Returns
    -------
    (s_min, s_max)
    """
    return s_center - s_halfwidth, s_center + s_halfwidth


def _sd_box(
    nmin: float, nmax: float, log_rmin: float, log_rmax: float,
    s_center: float, s_halfwidth: float = 3.0,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Bounding box in (s, d) space, combining a data-driven range for the
    informative axis `s` (centred on `s_center`, spanning
    `s_halfwidth` decades either side — see `_s_range`) with the full
    range for the near-flat axis `d` inherited from the ORIGINAL
    independent (n, r) priors box — `d` has no data-driven centre of its
    own, so there's no reason to narrow it beyond what (n, r) were already
    allowed to span.

    Returns
    -------
    (s_min, s_max), (d_min, d_max)
    """
    s_min, s_max = _s_range(s_center, s_halfwidth)
    # Corners of the (log10 n, log10 r) box that maximise/minimise
    # d = log10(n) - log10(r).
    d_min = nmin - log_rmax
    d_max = nmax - log_rmin
    return (s_min, s_max), (d_min, d_max)


def _n_box_from_s_and_r(
    smin: float, smax: float, log_rmin: float, log_rmax: float,
) -> tuple[float, float]:
    """Data-driven bounding box `(nmin, nmax)` for `log10(n)`, derived from
    the already data-driven `s` box (`_s_center`/`_s_range` — itself
    independent of any `n` prior, see the module note above
    `_hs_local_contrast`) and the already data-driven `r` box (from
    `_find_r_min`/the survival-derived `rmax`), replacing the old fixed
    `(0.0, 3.5)` literal that used to stand in for `n`'s prior regardless
    of species.

    Derivation: since `s = log10(n) + log10(r)`, and `n` ranges over
    `[nmin, nmax]` (log10 units) while `log10(r)` ranges independently
    over `[log_rmin, log_rmax]`, the achievable range of `s` given BOTH
    boxes is exactly the SUM of the two intervals:
    `[nmin + log_rmin, nmax + log_rmax]`. Setting this equal to the
    already-computed `s` box `[smin, smax]` and solving for `nmin`/`nmax`
    gives the TIGHTEST `n` box consistent with the already-chosen `s`
    box:

        nmin = smin - log_rmin
        nmax = smax - log_rmax

    `nmin` is additionally floored at 0.0 since `n >= 1` (i.e.
    `log10(n) >= 0`) is a hard physical constraint, not merely a
    data-driven preference.

    Returns
    -------
    (nmin, nmax)
    """
    nmin = max(smin - log_rmin, 0.0)
    nmax = smax - log_rmax
    return nmin, nmax


def _sd_to_logn_logr(s, d):
    """(s, d) -> (log10 n, log10 r). Inverse of `_logn_logr_to_sd`."""
    return (s + d) / 2.0, (s - d) / 2.0


def _logn_logr_to_sd(log_n, log_r):
    """(log10 n, log10 r) -> (s, d). Inverse of `_sd_to_logn_logr`."""
    return log_n + log_r, log_n - log_r


def _geometric_median(points: np.ndarray, eps: float = 1e-6, max_iter: int = 100) -> np.ndarray:
    """Multivariate geometric median via Weiszfeld's algorithm — the point
    minimising the SUM OF EUCLIDEAN DISTANCES to every row of `points`
    (shape (n_points, dim)), as opposed to the arithmetic mean which
    minimises the sum of SQUARED distances (and is far more sensitive to
    a single outlier point). Byzantine-robust: breakdown point ~50%, i.e.
    up to half the points can be arbitrarily bad without arbitrarily
    corrupting the result — unlike the mean, whose breakdown point is 0%.

    Used to combine per-site gradient vectors (one 3-D point per site:
    d(loss)/d(n_raw, r_raw, Scrit_raw)) into a single robust update direction
    for a mini-batch/full-dataset SGD step, instead of the plain average —
    a site whose gradient disagrees sharply with the rest gets naturally
    down-weighted by construction, without needing to identify it explicitly.
    """
    y = points.mean(axis=0)
    for _ in range(max_iter):
        dist = np.linalg.norm(points - y, axis=1)
        dist = np.maximum(dist, eps)  # avoid division by zero at points coinciding with y
        w = 1.0 / dist
        y_new = (points * w[:, None]).sum(axis=0) / w.sum()
        if np.linalg.norm(y_new - y) < eps:
            y = y_new
            break
        y = y_new
    return y


def _survival(r, hmean: float, alpha: float, mdd: float):
    """Probability of being alive when dispersal stops, derived from the
    process: at each round, continue (prob p=Ew/(1+Ew)) or stop (prob
    1-p, safe — no step taken); if continuing, survive this step with
    probability hmean**r or die with probability 1-hmean**r.
    Recursively, S = (1-p) + p*hmean**r*S  =>  S = (1-p)/(1 - p*hmean**r),
    which simplifies (substituting p=Ew/(1+Ew)) to:

        S(r) = 1 / (1 + Ew*(1 - hmean**r))

    with Ew = C / (hmean**r - C), C = 2*exp(-alpha/mdd)/(1+exp(-2*alpha/mdd))
    — the same C/Ew used elsewhere (cost_function, rmax). Unlike the
    naive ``1/(1+Ew*(hmean**r - 1))`` (wrong sign), this stays in [0, 1]
    for the ENTIRE valid domain r in (0, rmax): (1 - hmean**r) > 0 always
    (hmean < 1), so the denominator is always >= 1, with S(0)=1 and
    S(rmax-)=0 (Ew -> +infinity there) — monotonically decreasing, no
    blow-up/sign-flip like the wrong-sign version has well before rmax.
    """
    C = 2.0 * np.exp(-alpha / mdd) / (1.0 + np.exp(-2.0 * alpha / mdd))
    x = hmean ** np.asarray(r, dtype=np.float64)
    Ew = C / (x - C)
    return 1.0 / (1.0 + Ew * (1.0 - x))


def _find_r_min(hmean: float, alpha: float, mdd: float, tol: float = 0.005) -> float:
    """r such that survival S(r) = 1 - tol (closed form, not a scan).

    Solving S(r) = 1/(1 + Ew*(1-x)) = T for x = hmean**r (with T = 1-tol,
    Ew = C/(x-C)):

        Ew*(1-x) = (1-T)/T
        C*(1-x)/(x-C) = tol/(1-tol)
        x = C / (tol + C*(1-tol))            [solving for x]
        r_min = log(x) / log(hmean)

    S(r) is monotonically decreasing from S(0)=1 to S(rmax-)=0 over the
    model's valid domain (0, rmax) (see `_survival`), so this x is unique
    and always lands in (0, 1) for tol in (0, 1) — no scan/search needed.
    `r_min` is the point below which decreasing r further changes survival
    by less than `tol` (default: 0.5 percentage points), i.e. the model
    can no longer meaningfully distinguish r from 0 — a principled lower
    bound for r's log-space box, instead of an arbitrary value or the
    float precision floor.
    """
    C = 2.0 * np.exp(-alpha / mdd) / (1.0 + np.exp(-2.0 * alpha / mdd))
    x = C / (tol + C * (1.0 - tol))
    return float(np.log(x) / np.log(hmean))


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
    adaptive: bool = True,
    convergence_ratio_tol: float = 0.0001,
    max_iter: int = 500,
    return_iters: bool = False,
    seed_masks_list: list | None = None,
    breeding_masks_list: list | None = None,
) -> torch.Tensor | tuple[torch.Tensor, list]:
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
        ``(n_scaled, r_scaled, S_crit_scaled)`` – current parameter values.
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
        (Only the growth coefficient `g = 1/S_crit - 1` is recomputed here
        — no `Tg`/`a` intermediate.)
    plot, verbose:
        Diagnostic flags.
    adaptive, convergence_ratio_tol, max_iter:
        Forwarded to :func:`~paradis.core.growth.equilibrium_distribution`.
        Default ``adaptive=True``: iterate until the equilibrium has
        genuinely converged (change-ratio < `convergence_ratio_tol`,
        default 0.01%) rather than a fixed 10 steps — this closes a real
        artifact where a small `g` (slow dynamics) could stop well short
        of a true equilibrium, appearing to fit better than it genuinely
        does. Confirmed empirically: forcing true convergence flipped a
        previously "better" slow-growth point to substantially worse
        (mean cost 0.83 vs 0.51 across sites) — i.e. this was pure
        artifact exploitation, not genuine fit. Costs more compute
        (variable extra iterations, worst where the artifact would have
        been exploited) — pass ``adaptive=False`` to restore the old
        fixed-10-iteration behaviour if needed.
    return_iters:
        If ``True``, also return the list of per-site iteration counts
        actually used by `equilibrium_distribution` to reach convergence
        (only meaningful with ``adaptive=True`` — with ``adaptive=False``
        every site trivially uses the fixed `n_iter`). Useful for
        diagnosing whether the current (n, r, S_crit) is landing in a
        slow-to-converge region (e.g. near-singular kernel inversion),
        which directly costs wall-clock time per `cost_function` call
        regardless of how many outer SGD steps are taken.
    seed_masks_list:
        Optional precomputed sparse seeding masks, one per calibration
        site (same indexing as `adj_mats`/`K_is_list`), each drawn ONCE
        for the whole run and reused for every evaluation of that site —
        passed through to `equilibrium_distribution` as
        `seed_mask_override` instead of letting it draw a fresh random
        mask on every call. `None` (default) falls back to the old
        per-call fresh-draw behaviour.
    breeding_masks_list:
        Optional per-site breeding-range masks (same indexing as
        `adj_mats`/`K_is_list`/`seed_masks_list`), passed through to
        `equilibrium_distribution` as `breeding_ground` — constrains
        positive growth (NOT decline — see that function's docstring) to
        the breeding range at sites that have one. A site whose own entry
        is `None` (e.g. no breeding-range file matched for this species,
        or this whole list is `None`) gets unconstrained growth, matching
        every existing call site's behaviour before this parameter
        existed.

    Returns
    -------
    torch.Tensor
        Scalar loss value, or ``(loss, iters_per_site)`` if
        ``return_iters=True`` (``iters_per_site`` is a plain list of
        ints, one per site actually evaluated — sites skipped for
        NaN HS or zero selected locations are simply absent from it).
    """
    posteriors, selected_masks = posteriors_and_masks
    n_param, r_param, Scrit_param = params
    L, k, x0 = carrying_capacity_params
    size_site = calibration_sites.hs_maps[0].shape[0]

    # All scalar tensors created on device so autograd stays on-device.
    mdd_t   = torch.tensor(float(mdd),   device=device, dtype=torch.float32)
    hmean_t = torch.tensor(float(hmean), device=device, dtype=torch.float32)

    # ONE hard physical boundary on r, enforced centrally HERE so it
    # applies to every caller (batchSGD, refine_from_point, run_mala, the
    # grid scan) without each of them needing its own copy of this logic:
    # beyond r_max, Ew(r)/the dispersal kernel become ill-defined — the
    # process can no longer be matched to the target MDD at all (see
    # `_survival`'s docstring). n_min/n_max/r_min are soft PRIOR choices,
    # not physical impossibilities, and are NOT enforced here (or
    # anywhere else in the package — see the identical reasoning in
    # `_scan_cost_grid_sd`'s and `run_mala`'s validity checks). Clamping
    # (rather than raising/returning NaN) keeps this differentiable —
    # gradients simply vanish past the boundary, exactly like a one-sided
    # ReLU, which is standard and safe for gradient-based optimisation.
    C_np = (2.0 * np.exp(-1.11 / mdd)) / (1.0 + np.exp(-2.0 * 1.11 / mdd))
    r_max_t = torch.tensor(
        float((1.0 / np.log(hmean)) * np.log(C_np)), device=device, dtype=torch.float32,
    )
    r_param = torch.clamp(r_param, max=r_max_t)

    # Derive Ew from MDD, hmean, and the learned r.
    # Clamp denom away from zero (both below and above) so that Ew stays in
    # a range where p = Ew/(1+Ew) is safely < 1 and (I - p·W*) stays
    # invertible.  Without an upper clamp, denom → 0+ gives Ew → ∞,
    # p → 1, and linalg.inv returns NaN.
    C   = (2.0 * torch.exp(-1.11 / mdd_t)) / (1.0 + torch.exp(-2.0 * 1.11 / mdd_t))
    denom = hmean_t ** r_param - C
    denom = torch.where(denom >= 0, denom.clamp(min=1e-4), denom.clamp(max=-1e-4))
    Ew  = torch.clamp(C / denom, min=1e-3, max=1e4)

    # Growth coefficient directly from learned S_crit -- no Tg/a intermediate.
    g_param = 1.0 / Scrit_param - 1.0

    total_cost = torch.tensor(0.0, dtype=torch.float32)
    iters_per_site: list[int] = []

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
        seed_mask_override = (
            seed_masks_list[site_idx] if seed_masks_list is not None else None
        )
        breeding_ground = (
            breeding_masks_list[site_idx] if breeding_masks_list is not None else None
        )
        if return_iters:
            N_inf, changes = equilibrium_distribution(
                K_is.to(device), Kd, g_param, plot=plot, verbose=verbose,
                adaptive=adaptive, convergence_ratio_tol=convergence_ratio_tol,
                max_iter=max_iter, return_history=True,
                seed_mask_override=seed_mask_override,
                breeding_ground=breeding_ground,
            )
            iters_per_site.append(len(changes))
        else:
            N_inf = equilibrium_distribution(
                K_is.to(device), Kd, g_param, plot=plot, verbose=verbose,
                adaptive=adaptive, convergence_ratio_tol=convergence_ratio_tol,
                max_iter=max_iter,
                breeding_ground=breeding_ground,
                seed_mask_override=seed_mask_override,
            )
        N_inf = N_inf.reshape(size_site, size_site)

        if plot:
            plt.figure()
            plt.imshow(N_inf.cpu().detach().numpy(), cmap="viridis")
            plt.colorbar()
            plt.title(f"Equilibrium – site {site_idx}  n={n_param.item():.0f}"
                      f"  r={r_param.item():.5f}  S_crit={Scrit_param.item():.3f}")
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

    loss = total_cost / max(len(batch_indices), 1)
    if return_iters:
        return loss, iters_per_site
    return loss


# ---------------------------------------------------------------------------
# Per-site cost (no gradient — cheap, one site at a time)
# ---------------------------------------------------------------------------

def _per_site_costs(
    mdd: float,
    posteriors_and_masks: tuple,
    calibration_sites,
    theta: tuple,
    carrying_capacity_params: tuple,
    hmean: float,
    adj_mats: list,
    K_is_list: list,
    n_sites_total: int,
    seed_masks_list: list | None = None,
    breeding_masks_list: list | None = None,
) -> list:
    """Cost at *theta* for EVERY calibration site individually (no
    autograd graph — cheap, one site's dense matrices in memory at a time).
    Lets the full-dataset checkpoint report the MEDIAN across sites instead
    of just the mean: with a right-skewed per-site cost distribution (a
    few hard sites, most sites easy), the mean is pulled up by the rare
    hard sites and sits near the *top* of where most mini-batch draws land,
    while the median reflects the "typical" site's cost.

    `theta` is `(n_val, r_val, Scrit_val)` — handed straight to
    `cost_function`, which itself expects `(n, r, S_crit)` directly, no
    `Tg`/`a` intermediate.
    """
    n_val, r_val, Scrit_val = theta
    costs = []
    with torch.no_grad():
        n_p  = torch.tensor(n_val,  dtype=torch.float32, device=device)
        r_p  = torch.tensor(r_val,  dtype=torch.float32, device=device)
        scrit_p = torch.tensor(Scrit_val, dtype=torch.float32, device=device)
        for site_idx in range(n_sites_total):
            c = cost_function(
                mdd, posteriors_and_masks, calibration_sites,
                (n_p, r_p, scrit_p), [site_idx],
                carrying_capacity_params, hmean, adj_mats, K_is_list,
                plot=False, verbose=False,
                seed_masks_list=seed_masks_list, breeding_masks_list=breeding_masks_list,
            )
            costs.append(c.item())
    return costs


# ---------------------------------------------------------------------------
# Full-dataset loss/gradient via gradient accumulation over site chunks
# ---------------------------------------------------------------------------

def _full_dataset_grad(
    mdd: float,
    posteriors_and_masks: tuple,
    calibration_sites,
    theta: tuple,
    carrying_capacity_params: tuple,
    hmean: float,
    adj_mats: list,
    K_is_list: list,
    n_sites_total: int,
    chunk_size: int = 3,
    seed_masks_list: list | None = None,
    breeding_masks_list: list | None = None,
) -> tuple[float, list]:
    """Exact full-dataset loss and gradient w.r.t. (n, r, S_crit) at *theta*,
    computed by splitting the n_sites_total calibration sites into chunks
    of at most `chunk_size` and accumulating loss.backward() calls (same
    leaf tensors, gradients sum automatically) instead of one single
    backward pass over every site's dense NxN matrix-inverse graph at once.
    Memory for the largest live graph is bounded by `chunk_size`, not
    n_sites_total, at the cost of more (cheaper) forward/backward passes.
    Each chunk's graph is freed before the next one is built.

    `theta` is `(n_val, r_val, Scrit_val)` — the autograd LEAF is `Scrit_p`
    (the physical penetration-distance variable), handed DIRECTLY to
    `cost_function` (no `Tg`/`a` intermediate), so backprop gives
    `Scrit_p.grad = d(loss)/d(S_crit)` directly.
    """
    n_val, r_val, Scrit_val = theta
    n_p   = torch.tensor(n_val,   requires_grad=True, dtype=torch.float32, device=device)
    r_p   = torch.tensor(r_val,   requires_grad=True, dtype=torch.float32, device=device)
    Scrit_p = torch.tensor(Scrit_val, requires_grad=True, dtype=torch.float32, device=device)

    all_indices = list(range(n_sites_total))
    total_loss_val = 0.0
    for start in range(0, n_sites_total, chunk_size):
        chunk = all_indices[start:start + chunk_size]
        # Weight so summing the chunks' contributions reconstructs the
        # correct overall mean over ALL sites (not just this chunk's mean).
        weight = len(chunk) / n_sites_total
        partial = cost_function(
            mdd, posteriors_and_masks, calibration_sites,
            (n_p, r_p, Scrit_p), chunk,
            carrying_capacity_params, hmean, adj_mats, K_is_list,
            plot=False, verbose=False,
            seed_masks_list=seed_masks_list, breeding_masks_list=breeding_masks_list,
        )
        scaled = partial * weight
        scaled.backward(retain_graph=True)
        total_loss_val += scaled.item()
        del partial, scaled
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    grad = [n_p.grad.item(), r_p.grad.item(), Scrit_p.grad.item()]
    del n_p, r_p, Scrit_p
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return total_loss_val, grad


# ---------------------------------------------------------------------------
# Hessian at a point (finite differences of the first-order gradient)
# ---------------------------------------------------------------------------

def _hessian_at_point(
    mdd: float,
    posteriors_and_masks: tuple,
    calibration_sites,
    theta0: tuple,
    carrying_capacity_params: tuple,
    hmean: float,
    adj_mats: list,
    K_is_list: list,
    n_sites_total: int,
    eps: tuple = (1.0, 1e-4, 0.05),
    chunk_size: int = 3,
) -> dict:
    """Hessian of cost_function w.r.t. (n, r, S_crit) at theta0, over ALL
    calibration sites, via central finite differences of the first-order
    autograd gradient (each gradient evaluation itself computed by
    `_full_dataset_grad`'s chunked accumulation, to stay memory-bounded).
    Returns a dict with `eigvals`/`eigvecs`/`H`.

    `theta0` is `(n_val, r_val, Scrit_val)` — `eps`'s third entry perturbs
    `S_crit`, not `Tg` (its default, 0.05, was tuned for the old Tg-space
    scale and likely needs re-tuning to S_crit's pixel-distance scale before
    use — same caveat as the `compute_hessian` default of False elsewhere
    in this module).
    """
    theta0_arr = np.array(theta0, dtype=np.float64)
    eps_arr = np.array(eps, dtype=np.float64)

    def _grad_at(n_val, r_val, Scrit_val):
        _, grad = _full_dataset_grad(
            mdd, posteriors_and_masks, calibration_sites,
            (n_val, r_val, Scrit_val), carrying_capacity_params, hmean,
            adj_mats, K_is_list, n_sites_total, chunk_size=chunk_size,
        )
        return np.array(grad)

    H = np.zeros((3, 3))
    for j in range(3):
        step = np.zeros(3)
        step[j] = eps_arr[j]
        g_plus  = _grad_at(*(theta0_arr + step))
        g_minus = _grad_at(*(theta0_arr - step))
        H[:, j] = (g_plus - g_minus) / (2.0 * eps_arr[j])

    H = 0.5 * (H + H.T)   # symmetrize (finite-diff noise breaks exact symmetry)
    eigvals, eigvecs = np.linalg.eigh(H)

    return {"theta0": theta0_arr, "H": H, "eigvals": eigvals, "eigvecs": eigvecs}


def _full_dataset_grad_sd(
    mdd: float,
    posteriors_and_masks: tuple,
    calibration_sites,
    theta: tuple,
    carrying_capacity_params: tuple,
    hmean: float,
    adj_mats: list,
    K_is_list: list,
    n_sites_total: int,
    chunk_size: int = 3,
    seed_masks_list: list | None = None,
    breeding_masks_list: list | None = None,
) -> tuple[float, list]:
    """Exact full-dataset loss and gradient w.r.t. `(s, d, S_crit)`
    DIRECTLY — the actual physically-meaningful, well-conditioned axes
    this package learns in (see the module-level note above
    `_hs_local_contrast`) — rather than `(n, r, S_crit)` like
    `_full_dataset_grad`. `theta = (s_val, d_val, Scrit_val)`; the
    autograd LEAVES are `s_p`, `d_p`, `Scrit_p` themselves (raw values,
    no tanh-bounding — this is a diagnostic at a single already-chosen
    point, not an optimisation step, so there's no box to stay inside of
    beyond what the caller already ensured). `n`/`r`/`Tg` are recovered
    from these leaves via `_sd_to_logn_logr`/`_Scrit_to_Tg` INSIDE the
    autograd graph (not detached), so `s_p.grad`/`d_p.grad`/
    `Scrit_p.grad` are exactly `d(loss)/d(s)`, `d(loss)/d(d)`,
    `d(loss)/d(S_crit)` — chunked the same way as `_full_dataset_grad`
    (memory bounded by `chunk_size`, `s_p`/`d_p`/`Scrit_p` recomputed
    fresh into `n_s`/`r_s`/`tg_s` each chunk to avoid a "backward through
    the graph a second time" error, gradients accumulate automatically
    since the leaves themselves are reused across chunks).
    """
    s_val, d_val, Scrit_val = theta
    s_p     = torch.tensor(s_val,     requires_grad=True, dtype=torch.float32, device=device)
    d_p     = torch.tensor(d_val,     requires_grad=True, dtype=torch.float32, device=device)
    Scrit_p = torch.tensor(Scrit_val, requires_grad=True, dtype=torch.float32, device=device)

    all_indices = list(range(n_sites_total))
    total_loss_val = 0.0
    for start in range(0, n_sites_total, chunk_size):
        chunk = all_indices[start:start + chunk_size]
        weight = len(chunk) / n_sites_total
        log_n_s, log_r_s = _sd_to_logn_logr(s_p, d_p)
        n_s  = 10.0 ** log_n_s
        r_s  = 10.0 ** log_r_s
        partial = cost_function(
            mdd, posteriors_and_masks, calibration_sites,
            (n_s, r_s, Scrit_p), chunk,
            carrying_capacity_params, hmean, adj_mats, K_is_list,
            plot=False, verbose=False,
            seed_masks_list=seed_masks_list, breeding_masks_list=breeding_masks_list,
        )
        scaled = partial * weight
        scaled.backward(retain_graph=True)
        total_loss_val += scaled.item()
        del n_s, r_s, partial, scaled
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    grad = [s_p.grad.item(), d_p.grad.item(), Scrit_p.grad.item()]
    del s_p, d_p, Scrit_p
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return total_loss_val, grad


def _hessian_at_point_sd(
    mdd: float,
    posteriors_and_masks: tuple,
    calibration_sites,
    theta0: tuple,
    carrying_capacity_params: tuple,
    hmean: float,
    adj_mats: list,
    K_is_list: list,
    n_sites_total: int,
    scrit_box: tuple,
    eps: tuple = (0.05, 0.05, 0.01),
    chunk_size: int = 3,
    seed_masks_list: list | None = None,
    breeding_masks_list: list | None = None,
) -> dict:
    """Hessian of the full-dataset cost w.r.t. `(s, d, S_crit)` at
    `theta0`, via central finite differences of `_full_dataset_grad_sd`'s
    autograd gradient — used to tell whether a converged point (e.g.
    `refine_from_point`'s endpoint) is a genuine LOCAL MINIMUM of the
    cost surface (a true "equilibrium" the optimiser can't improve on in
    ANY direction) or merely a stationary point the gradient happens to
    vanish at while the surface still decreases in some other direction
    (a saddle point).

    Unlike the old `(n, r, S_crit)`-space `_hessian_at_point`, `(s, d,
    S_crit)` are all comparably-scaled, O(1)-ish quantities (a handful of
    decades for `s`/`d`, a probability in `[0.5, 1]` for `S_crit`) — so a
    SINGLE fixed `eps` triple works reasonably well regardless of
    species, unlike `n`/`r` which can span many orders of magnitude and
    would need per-species rescaling of `eps` to mean the same relative
    step everywhere.

    Classification of the returned eigenvalues:
    - ALL POSITIVE: positive-definite curvature in every direction —
      a genuine local minimum, a real local equilibrium of the cost
      surface.
    - ANY NEGATIVE: a saddle point — the gradient vanished here, but the
      surface still decreases along at least one direction (the
      corresponding eigenvector), so this is NOT a true local minimum.
    - NEAR ZERO (within noise of `eps`): a flat/degenerate direction —
      curvature too weak to classify confidently at this `eps` (try a
      larger `eps` along that eigenvector before trusting either
      conclusion).

    `scrit_box` = `(SCRIT_MIN, SCRIT_MAX)` — perturbed `S_crit` values are
    clamped strictly inside this box (with a small margin) before
    evaluating the gradient there, since `_Scrit_to_Tg` is singular at
    both exact endpoints.
    """
    theta0_arr = np.array(theta0, dtype=np.float64)
    eps_arr = np.array(eps, dtype=np.float64)
    scrit_lo, scrit_hi = scrit_box[0] + 1e-6, scrit_box[1] - 1e-6

    def _grad_at(s_val, d_val, Scrit_val):
        Scrit_val = float(np.clip(Scrit_val, scrit_lo, scrit_hi))
        _, grad = _full_dataset_grad_sd(
            mdd, posteriors_and_masks, calibration_sites,
            (s_val, d_val, Scrit_val), carrying_capacity_params, hmean,
            adj_mats, K_is_list, n_sites_total, chunk_size=chunk_size,
            seed_masks_list=seed_masks_list, breeding_masks_list=breeding_masks_list,
        )
        return np.array(grad)

    H = np.zeros((3, 3))
    for j in range(3):
        step = np.zeros(3)
        step[j] = eps_arr[j]
        g_plus  = _grad_at(*(theta0_arr + step))
        g_minus = _grad_at(*(theta0_arr - step))
        H[:, j] = (g_plus - g_minus) / (2.0 * eps_arr[j])

    H = 0.5 * (H + H.T)   # symmetrize (finite-diff noise breaks exact symmetry)
    eigvals, eigvecs = np.linalg.eigh(H)

    return {"theta0": theta0_arr, "H": H, "eigvals": eigvals, "eigvecs": eigvecs}


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
    gpu_memory_fraction: float | None = 0.8,
    r_min: float | None = None,
    r_survival_tol: float = 0.005,
    r_survival_alpha: float = 1.11,
    compute_hessian: bool = False,
    method: str = "batchSGD",
    points_per_axis: int = 10,
    grid_n_range: tuple | None = None,
    grid_r_range: tuple | None = None,
    grid_tg_range: tuple | None = None,
    init_params: dict | None = None,
    robust_gradient: bool = False,
    batch_chunk_size: int | None = None,
    seed_fraction: float = 0.25,
    s_halfwidth: float = 3.0,
    d_max_r_survival_tol: float = 1e-12,
    precise_gridscan_center: tuple | None = None,
    precise_gridscan_s_window: float = 1.0,
    precise_gridscan_d_window: float = 1.0,
    precise_gridscan_Scrit_window: float = 0.1,
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
        ``[(sn_min, sn_max), (r_min, r_max), (sTg_min, sTg_max)]``.
        n, r, AND Tg are ALL learned in LOG10 space (not linear) — an
        Adam step of fixed size now corresponds to a roughly constant
        MULTIPLICATIVE change in each, matching their actual
        (order-of-magnitude) effect, instead of a constant additive change
        that means very different things near the low vs. high end of a
        linear box. n is parametrized as ``n = 10**sn`` with ``sn``
        tanh-bounded to ``[sn_min, sn_max]`` (default ``[0, 3.5]``, i.e.
        n in [1, ~3162]). Tg is parametrized the same way, ``Tg = 10**sTg``
        with ``sTg`` tanh-bounded to ``[sTg_min, sTg_max]`` (default
        ``[-2, 2]``, i.e. Tg in [0.01, 100]). r's slot is still given as
        actual (r_min, r_max) values (not their logs) for API convenience —
        internally it's tanh-bounded on ``[log10(r_min), log10(r_max)]``
        for the same reason (dividing r by 2 halves the per-step mortality
        rate — a multiplicative effect). r_max is the hard physical cap
        (beyond it, Ew is undefined) and r_min defaults to the value
        computed by :func:`_find_r_min` (see ``r_min``/``r_survival_tol``
        below) rather than an arbitrary floor or 0 (which log can't reach
        anyway).
    r_min:
        Lower bound of r's log-space box. If ``None`` (default), computed
        automatically via :func:`_find_r_min` from ``hmean``, ``mdd``, and
        ``r_survival_tol``/``r_survival_alpha`` — the largest r below which
        further decreases change mean survival probability by less than
        ``r_survival_tol``, i.e. the point past which the model can no
        longer distinguish r from 0. Ignored if `priors` is given
        explicitly (its r-slot's low value is used instead).
    r_survival_tol, r_survival_alpha:
        Only used to auto-compute ``r_min`` when it and `priors` are both
        left at their defaults. ``r_survival_tol`` (default 0.005 = 0.5
        percentage points) is the survival-probability tolerance defining
        "no longer distinguishable from r=0". ``r_survival_alpha`` is the
        same constant used elsewhere for Ew (default 1.11, matching
        ``cost_function``'s C).
    compute_hessian:
        If True, compute (and print) the Hessian eigenvalues/eigenvectors
        of the full-dataset cost at every 100-step checkpoint. Default
        False — currently disabled: the finite-difference `eps` in
        `_hessian_at_point` were tuned for the old, much wider linear
        parameter boxes and can give erratic, unstable eigenvalue
        estimates at the current (narrower) log-space scale. Re-enable
        once `eps` has been re-calibrated.
    method:
        ``"batchSGD"`` (default) — the stochastic mini-batch Adam run
        described above. ``"grid"`` — brute-force grid scan of the
        full-dataset cost (no gradients at all), directly visualising the
        cost surface instead of optimising it; see `grid_*` below.
        ``"precise_gridscan"`` — a small, ZOOMED-IN grid scan around a
        chosen ``(s, d, S_crit)`` point (e.g. `refine_from_point`'s
        converged endpoint), to visually verify the cost surface's local
        shape as an independent check on whether that point is a genuine
        local minimum — see `_run_precise_gridscan` and the
        `precise_gridscan_*` parameters below. Ignores every
        batchSGD-only parameter above.
    precise_gridscan_center, precise_gridscan_s_window,
    precise_gridscan_d_window, precise_gridscan_Scrit_window:
        ``method="precise_gridscan"``-only. `precise_gridscan_center`
        (required, ``(s, d, S_crit)``) is the point to zoom in on.
        `precise_gridscan_s_window`/`_d_window`/`_Scrit_window` (defaults
        ``1.0``, ``1.0``, ``0.1``) are the half-widths of the local scan
        window around that point, in each axis's own units. Resolution
        uses the SAME `points_per_axis` as ``method="grid"`` (see
        below) — since the window is much smaller than the full prior
        box, this gives a MUCH finer resolution per unit than a full
        ``method="grid"`` scan at the same point count.
    points_per_axis, grid_n_range, grid_r_range, grid_tg_range:
        ``method="grid"`` or ``"precise_gridscan"``-only. By default runs
        a single scan (``points_per_axis``^3 points) over the full prior
        box (``"grid"``) or the local `precise_gridscan_*` window
        (``"precise_gridscan"``); pass any of
        `grid_n_range`/`grid_r_range`/`grid_tg_range` (actual n/r/Tg
        units, not logs, ``method="grid"``-only) to scan an exact custom
        box instead. n and r are sampled log-spaced within their range
        (matching how they're learned/interpreted everywhere else), Tg
        linear-spaced. Saves 2-D heatmap slices (log axes) and an
        interactive 3-D volume (Plotly) to `save_fig_folder`. The
        returned (Ew, n, r, Tg) is the grid's minimum location — a coarse
        estimate, not a converged optimum; use
        ``method="batchSGD"``/`refine_from_point` for that. Each grid
        point costs about one full-dataset forward pass (~1s typically),
        so e.g. ``points_per_axis=10`` (default, 1000 points) costs
        roughly 15-20 minutes for a full ``"grid"`` scan (much less for
        ``"precise_gridscan"``'s small window).
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
    gpu_memory_fraction:
        Caps PyTorch's CUDA caching allocator at this fraction of total GPU
        memory (default 0.8 = 80%), so this process can't grow into and
        exhaust the whole card — leaves headroom for the OS/other
        processes and fails fast with an OOM instead of silently starving
        other GPU users. Set to ``None`` to leave PyTorch's default
        (effectively 100%) unchanged. No effect without CUDA.
    init_params:
        ``method="batchSGD"``-only. Optional custom starting point,
        ``{"s": ..., "d": ..., "S_crit": ...}`` — given directly in this
        function's OWN learning space (NOT ``(n, r, Tg)`` any more), since
        that's what's actually optimised. Each key is optional; any
        missing one defaults to its own box's centre (matching the
        ``init_params=None`` default of starting exactly at the box
        centre for all three). Values are clamped into the (possibly
        auto-computed) box if given outside it — same convention as
        `refine_from_point`'s ``init_point``, just as a dict here rather
        than a tuple. Unlike `refine_from_point`, this is still the
        regular STOCHASTIC mini-batch run (noisy gradient steps, not the
        exact full-dataset gradient) — just started from a chosen point
        instead of the defaults.
    robust_gradient:
        ``method="batchSGD"``-only. If True, each site in the mini-batch
        gets its OWN gradient computed separately (instead of averaging
        their losses together before a single backward pass), and the
        per-step update uses the GEOMETRIC MEDIAN of these per-site
        gradient vectors (`_geometric_median`, Weiszfeld's algorithm)
        rather than their mean. Byzantine-robust: a site whose gradient
        disagrees sharply with the rest (e.g. pushing Tg in the opposite
        direction) is naturally down-weighted, without needing to identify
        it explicitly — breakdown point ~50% vs. 0% for the plain average.
        Costs ``len(batch)`` separate forward/backward passes per step
        instead of 1 (same total forward compute, just not fused into a
        single averaged loss). No effect when the mini-batch has only 1
        site (``all_together=False``, or ``n_random_sites=1``).
    batch_chunk_size:
        ``method="batchSGD"``-only, and only affects the default
        (``robust_gradient=False``) forward pass — that path normally
        builds ONE autograd graph over the WHOLE mini-batch (every site's
        dense N×N matrix-inverse graph held in memory simultaneously) and
        calls ``.backward()`` once. With a large ``n_random_sites`` (e.g.
        set close to or above the total number of calibration sites, to
        train on all of them every step) this can exceed GPU memory. If
        given, the batch is instead split into chunks of at most
        ``batch_chunk_size`` sites, with gradients accumulated across
        chunked ``.backward()`` calls — same technique as
        `_full_dataset_grad`/`refine_from_point`'s ``chunk_size`` — so
        memory stays bounded by ``batch_chunk_size`` sites at a time
        regardless of the mini-batch size. Default ``None`` — the
        original single-graph behaviour, unchanged for small batches.
        (``robust_gradient=True`` never needs this: it already computes
        one site's gradient at a time.)
    seed_fraction:
        Fraction of pixels seeded per site's sparse initial density mask,
        drawn ONCE per site at the start of this run and FIXED for every
        evaluation thereafter (not redrawn per step). Default 0.25.

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
    if method == "grid_default":
        # Fully automatic grid scan: no grid_*_range/points_per_axis
        # need to be supplied by the caller — every axis's range is filled
        # in from the SAME computed priors used by batchSGD/refine/MALA
        # (n, r from the (hmean, mdd)-derived box; S_crit/Tg from the
        # fixed universal `_Scrit_box`, see `_run_grid`),
        # at a fixed resolution of 10 points/axis. Equivalent to calling
        # `method="grid"` with every `grid_*_range` left at `None` and
        # `points_per_axis=10` — just without having to remember to
        # leave them all unset (and without whatever custom values a
        # caller may have accumulated in their own script from a previous,
        # now-stale, manual scan).
        return _run_grid(
            calibration_sites=calibration_sites, hmean=hmean, mdd=mdd,
            posteriors_and_masks=posteriors_and_masks,
            carrying_capacity_params=carrying_capacity_params,
            priors=priors,
            points_per_axis=10,
            grid_n_range=None, grid_r_range=None, grid_tg_range=None,
            save_fig_folder=save_fig_folder, species_name=species_name,
            r_min=r_min, r_survival_tol=r_survival_tol, r_survival_alpha=r_survival_alpha,
            s_halfwidth=s_halfwidth, d_max_r_survival_tol=d_max_r_survival_tol,
        )
    elif method == "grid":
        return _run_grid(
            calibration_sites=calibration_sites, hmean=hmean, mdd=mdd,
            posteriors_and_masks=posteriors_and_masks,
            carrying_capacity_params=carrying_capacity_params,
            priors=priors,
            points_per_axis=points_per_axis,
            grid_n_range=grid_n_range, grid_r_range=grid_r_range, grid_tg_range=grid_tg_range,
            save_fig_folder=save_fig_folder, species_name=species_name,
            r_min=r_min, r_survival_tol=r_survival_tol, r_survival_alpha=r_survival_alpha,
            s_halfwidth=s_halfwidth, d_max_r_survival_tol=d_max_r_survival_tol,
        )
    elif method == "precise_gridscan":
        if precise_gridscan_center is None:
            raise ValueError(
                "method='precise_gridscan' requires "
                "precise_gridscan_center=(s, d, S_crit)."
            )
        return _run_precise_gridscan(
            calibration_sites=calibration_sites, hmean=hmean, mdd=mdd,
            posteriors_and_masks=posteriors_and_masks,
            carrying_capacity_params=carrying_capacity_params,
            center=precise_gridscan_center,
            s_window=precise_gridscan_s_window,
            d_window=precise_gridscan_d_window,
            Scrit_window=precise_gridscan_Scrit_window,
            points_per_axis=points_per_axis,
            save_fig_folder=save_fig_folder, species_name=species_name,
            seed_fraction=seed_fraction,
        )
    elif method != "batchSGD":
        raise ValueError(
            f"Unknown method={method!r}. Use 'batchSGD', 'grid', "
            f"'grid_default', or 'precise_gridscan'."
        )

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
        if r_min is None:
            r_min = _find_r_min(hmean, r_survival_alpha, mdd, tol=r_survival_tol)
            print(f"[priors] Auto r_min={r_min:.6g} (survival within "
                  f"{r_survival_tol*100:.2g}pp of its r->0 limit below this).")
        log_rmin, log_rmax = np.log10(r_min), np.log10(rmax)
        hs_bar, delta_h_bar = _hs_local_contrast(calibration_sites)
        s_center = _s_center(hs_bar, delta_h_bar)
        smin, smax = _s_range(s_center, s_halfwidth)
        nmin, nmax = _n_box_from_s_and_r(smin, smax, log_rmin, log_rmax)
        priors = [(nmin, nmax), (r_min, rmax), (-2.0, 2.0)]
        print(f"[priors] r range for this species: r_min={r_min:.6g}  "
              f"r_max={rmax:.6g}  (hmean={hmean:.4g}, mdd={mdd:.4g})")

    # n and r are learned in LOG10 space (see docstring): `nmin`/`nmax`
    # here are bounds on log10(n) directly; `rmin`/`rmax` are r's actual
    # (r_min, r_max) box, log10'd below right before use. Tg is learned
    # via `S_crit` instead (see `_Scrit_box`/`_Scrit_to_Tg`/`_Tg_to_Scrit`) — not
    # in log10 space like n/r, and not Tg itself: `tgmin`/`tgmax` from
    # `priors` are unused here (a fixed, universal `S_crit` box — always
    # `(0.5, 1.0)`, the same for every species — replaces them for this
    # function).
    (nmin, nmax), (rmin, rmax), (_tgmin_unused, _tgmax_unused) = priors
    log_rmin, log_rmax = np.log10(rmin), np.log10(rmax)

    # d_max EXTENSION via a SEPARATE, much smaller r floor (see the
    # module-level note above `_n_box_from_s_and_r`/`_sd_box`) — d_min
    # (via n_min/r_min above) stays computed with the MODERATE
    # `r_survival_tol` (the "no longer meaningful effect on survival"
    # threshold), unchanged from before. Only d_max is pushed out further,
    # using an independently, extremely small `d_max_r_survival_tol`
    # r-floor — species whose true optimum needs a very small r (near-zero
    # dispersal mortality, confirmed for e.g. wolf/wild boar) can still
    # reach it, without shifting d_min or the default starting point's
    # (conservative) box centre.
    r_min_for_dmax = _find_r_min(hmean, r_survival_alpha, mdd, tol=d_max_r_survival_tol)
    log_rmin_for_d = np.log10(r_min_for_dmax)

    # (n, r) -> (s, d) reparametrisation (see the module-level note above
    # `_hs_local_contrast`): n and r are only identifiable, as far as the
    # dispersal kernel's SELECTIVITY pattern is concerned, through their
    # product s = log10(n*r) — d = log10(n/r) (motion at fixed n*r) barely
    # changes that pattern at all. Learning directly in (s, d) instead of
    # independently in (log10 n, log10 r) aligns Adam's two coordinates
    # with the model's actual informative vs. near-flat directions, instead
    # of both directions being a ~45-degree mix of the two as they are in
    # the original (log10 n, log10 r) axes.
    #
    # Printed in this exact order (r -> s -> n -> d) since n's own box is
    # now DERIVED from the s box + r box (see `_n_box_from_s_and_r`), and
    # d's box in turn depends on n's — each quantity is only meaningful
    # once the one before it has been shown.
    hs_bar, delta_h_bar = _hs_local_contrast(calibration_sites)
    s_center = _s_center(hs_bar, delta_h_bar)
    (smin, smax), (dmin, dmax_conservative) = _sd_box(nmin, nmax, log_rmin, log_rmax, s_center, s_halfwidth)
    # d_max is now recomputed using `log_rmin_for_d` (the SEPARATE,
    # extremely small r floor from `d_max_r_survival_tol`, computed above)
    # instead of the moderate `log_rmin` — extends d_max far out for
    # species whose true optimum needs near-zero dispersal mortality,
    # while `dmin`/`dmax_conservative` (computed with the moderate r
    # floor) are kept around for the default init point's box centre
    # (see below) and for reference/printing. The corner achieving d_max
    # is (log_n=n_max, log_r=log_rmin_for_d): d increases AND s decreases
    # there relative to the conservative corner (s = n_max + log_rmin_for_d
    # < n_max + log_rmin), i.e. the newly reachable region skews toward
    # the UPPER-LEFT (low s, high d) — the opposite of widening n_max,
    # which would skew the same corner toward the upper-RIGHT.
    dmax = nmax - log_rmin_for_d
    print(f"[priors] s prior for this species: hs_bar={hs_bar:.4g}  "
          f"delta_h_bar={delta_h_bar:.4g}  s*=log10(n*r)={s_center:.4g}  "
          f"s box=[{smin:.4g}, {smax:.4g}]  (s_halfwidth={s_halfwidth:.3g})")
    print(f"[priors] n range for this species (data-driven from s box + "
          f"r box): n_min={10**nmin:.4g}  n_max={10**nmax:.4g}  "
          f"(log10: [{nmin:.4g}, {nmax:.4g}])")
    print(f"[priors] d prior for this species (from n box + r box): "
          f"conservative d box=[{dmin:.4g}, {dmax_conservative:.4g}] "
          f"(r_survival_tol={r_survival_tol:.3g}, used for the default init "
          f"point's box centre)  ->  LEARNING d box=[{dmin:.4g}, {dmax:.4g}] "
          f"(d_max extended via d_max_r_survival_tol={d_max_r_survival_tol:.3g})")

    # S_crit reparametrisation of Tg (see the module-level growth-timescale
    # note and `_Scrit_box`): `S_crit`, not Tg itself, is the physically
    # well-motivated learning variable — the critical per-dispersal-step
    # survival probability below which no growth rate can sustain a
    # locally-growing population. Its box is a FIXED universal constant,
    # (0.5, 1.0), independent of mdd/r/n (see `_Scrit_box`'s docstring),
    # so unlike the old `ell`/`x_crit` boxes there is nothing
    # species-specific to precompute here.
    SCRIT_MIN, SCRIT_MAX = _Scrit_box()
    print(f"[priors] S_crit (survival-threshold) reparametrisation for "
          f"this species: S_crit box=[{SCRIT_MIN:.4g}, {SCRIT_MAX:.4g}]  "
          f"(interpretation: S_crit = critical per-dispersal-step survival "
          f"probability below which NO growth rate can sustain a local "
          f"population — S_crit=0.5 is the g->1 (fastest possible growth, "
          f"Tg->0) limit, S_crit->1 is the g->0 (Tg->inf) limit)")

    # ------------------------------------------------------------------
    # PRECOMPUTATION (outside the optimisation loop)
    # These quantities only depend on the fixed HS data and (L, k, x0).
    # ------------------------------------------------------------------
    print("[precompute] Building adjacency matrices and K_is ...")
    n_sites_total = len(calibration_sites)
    adj_mats  = []
    K_is_list = []
    seed_masks_list = []

    for hs_map in tqdm(calibration_sites.hs_maps, desc="  Sites", leave=False):
        hs_t    = torch.tensor(hs_map, dtype=torch.float32, device=device)
        adj_mat = adjacency_matrix_torch(hs_t)
        adj_mats.append(adj_mat)

        K_is_flat = torch.tensor(
            _logistic(hs_map.flatten().astype(np.float32), L, k, x0),
            dtype=torch.float32,
        )
        K_is_list.append(K_is_flat)
        seed_masks_list.append(seed_mask(
            K_is_flat.shape, seed_fraction=seed_fraction,
            device=K_is_flat.device, dtype=K_is_flat.dtype,
        ))

    print(f"[precompute] Done — {n_sites_total} sites.")
    breeding_masks_list = _build_breeding_masks_list(calibration_sites, n_sites_total)
    print(f"[precompute] Seeded initial densities: {seed_fraction*100:.0f}% of "
          f"pixels per site, FIXED for the whole run (not redrawn per step).")

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
        if gpu_memory_fraction is not None:
            torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction, device=0)
            print(f"[learning] Capping GPU memory usage at "
                  f"{gpu_memory_fraction * 100:.0f}% of total.")
    else:
        print("[learning] WARNING: CUDA not available — running on CPU.\n"
              "           Matrix inverse of a 70x70 window (~4900x4900) is O(N^3)\n"
              "           and is very slow on CPU.  GPU is strongly recommended.\n"
              "           Expected speed on CPU: ~1 step per several seconds.")

    # ------------------------------------------------------------------
    # Optimisation
    # ------------------------------------------------------------------
    all_costs, all_full_cost, all_n, all_r, all_Scrit = [], [], [], [], []
    all_site_cost_hist = []   # per-run list of (step, [21 per-site costs])
    n_runs = len(learning_runs)
    print("\033[93m[Optimization steps will be printed every 100 steps ... please wait]\033[0m")

    for run_idx, site_loop_idx in enumerate(
        tqdm(learning_runs, desc="Optimisation run", position=0, leave=True)
    ):
        # Parameters created ON device so all autograd stays on GPU when
        # available. n/r/Tg are all now tanh-bounded IN LOG10 SPACE (see
        # forward pass below) — raw=0.0 would land at each box's LOG-space
        # midpoint (the geometric mean of the box, e.g. n~56 for sn in
        # [0,3.5]), not the historical starting points (n=1500, r=rmax/2,
        # Tg=5). Explicit inverse-tanh initialisation restores those, or
        # uses the caller-supplied init_params={"n":.., "r":.., "Tg":..}
        # instead if given — same clamp-into-box convention as
        # refine_from_point's init_point.
        #
        # DEFAULT (init_params=None): start EXACTLY at the (s, d, Tg) box
        # CENTRE — raw=0 for all three — instead of the old historical
        # anchor (n=1500, r=rmax/2, Tg=5). `smin`/`smax` are already built
        # around the data-driven `s_center` (= s*, see `_sd_box`), so
        # raw_s=0 lands precisely on the softmax's most sensitive point
        # w.r.t. `hs_bar`/`delta_h_bar` — no historical n/r anchor is
        # meaningful now that n has no prior of its own ("n libre"), and
        # starting at the box centre also means the tanh's own steepest,
        # most-responsive region (around raw=0, where tanh'(0)=1 is
        # maximal) coincides with this high-sensitivity zone, instead of
        # initialising somewhere the tanh derivative may already be small.
        # `d`'s CENTRE uses the CONSERVATIVE box (dmin, dmax_conservative —
        # computed with the moderate `r_survival_tol`), not the extended
        # LEARNING box (dmin, dmax) — d_max was pushed out far via a much
        # smaller, separate r floor (`d_max_r_survival_tol`) specifically
        # so species that need it CAN reach it during optimisation, but
        # the default/unspecified starting point should stay in the
        # well-supported, moderate region, not already halfway to the
        # extreme edge. `s_raw_init`/`Scrit_raw_init` still just use their
        # own (unextended) box centres via raw=0.
        d_center_conservative = (dmin + dmax_conservative) / 2.0
        if init_params is None:
            s_raw_init, Scrit_raw_init = 0.0, 0.0
            d_raw_init = _inv_tanh_rescale(d_center_conservative, dmin, dmax)
        else:
            # Caller supplies a starting point directly in this function's
            # OWN learning space, (s, d, S_crit) — not (n, r, Tg) — since
            # that's what's actually optimised (see the module-level
            # notes above `_hs_local_contrast`/`_Scrit_box`). Each key is
            # optional; any missing one defaults to its own box centre
            # (`d`'s to the CONSERVATIVE centre, see above).
            s_init     = init_params.get("s",      (smin + smax) / 2.0)
            d_init     = init_params.get("d",      d_center_conservative)
            Scrit_init = init_params.get("S_crit", (SCRIT_MIN + SCRIT_MAX) / 2.0)
            s_init     = min(max(s_init, smin), smax)
            d_init     = min(max(d_init, dmin), dmax)
            Scrit_init = min(max(Scrit_init, SCRIT_MIN), SCRIT_MAX)
            s_raw_init     = _inv_tanh_rescale(s_init, smin, smax)
            d_raw_init     = _inv_tanh_rescale(d_init, dmin, dmax)
            Scrit_raw_init = _inv_tanh_rescale(Scrit_init, SCRIT_MIN, SCRIT_MAX)
        # params[0] = s_raw (informative axis, ~log10(n*r)),
        # params[1] = d_raw (near-flat axis, ~log10(n/r)),
        # params[2] = Scrit_raw (tanh-bounded to the S_crit box, see
        # `_Scrit_box` above — handed directly to `cost_function`, no
        # Tg/a intermediate).
        params = [
            torch.nn.Parameter(torch.tensor(s_raw_init,   device=device, dtype=torch.float32, requires_grad=True)),
            torch.nn.Parameter(torch.tensor(d_raw_init,   device=device, dtype=torch.float32, requires_grad=True)),
            torch.nn.Parameter(torch.tensor(Scrit_raw_init, device=device, dtype=torch.float32, requires_grad=True)),
        ]
        optimizer = torch.optim.Adam(params)
        losses, ln, lr, lScrit = [], [], [], []
        full_cost_hist: list[tuple[int, float, float]] = []   # (step, mean, median)
        site_cost_hist: list[tuple[int, list]] = []   # (step, [21 per-site costs])

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

            do_plot = plot and (step % 50 == 0)

            if robust_gradient and len(batch) > 1:
                # Each site gets its OWN forward+backward pass, separately
                # — NOT fused into one averaged loss — so we can combine
                # the resulting per-site gradient VECTORS via the
                # geometric median instead of their mean (see
                # `_geometric_median`'s docstring for why this is
                # Byzantine-robust to a disagreeing site).
                site_grads = []
                loss_val_accum = 0.0
                batch_iters: list[int] = []
                for site_idx in batch:
                    s_raw, d_raw, Scrit_raw = params
                    s_s  = _tanh_rescale(s_raw, smin, smax)
                    d_s  = _tanh_rescale(d_raw, dmin, dmax)
                    log_n_s, log_r_s = _sd_to_logn_logr(s_s, d_s)
                    n_s  = 10.0 ** log_n_s
                    r_s  = 10.0 ** log_r_s
                    Scrit_s = _tanh_rescale(Scrit_raw, SCRIT_MIN, SCRIT_MAX)
                    site_loss, site_iters = cost_function(
                        mdd, posteriors_and_masks, calibration_sites,
                        (n_s, r_s, Scrit_s), [site_idx],
                        carrying_capacity_params, hmean, adj_mats, K_is_list,
                        plot=False, verbose=verbose, return_iters=True,
                        seed_masks_list=seed_masks_list, breeding_masks_list=breeding_masks_list,
                    )
                    optimizer.zero_grad()
                    site_loss.backward()
                    site_grads.append([p.grad.item() for p in params])
                    loss_val_accum += site_loss.item()
                    batch_iters.extend(site_iters)

                loss_val = loss_val_accum / len(batch)
                combined_grad = _geometric_median(np.array(site_grads))
                avg_iters = float(np.mean(batch_iters)) if batch_iters else float("nan")

                if step % 100 == 0:
                    print(f"\033[95m  step {step:4d} | robust_gradient — per-site "
                          f"grads (raw tanh-space s_raw, d_raw, Scrit_raw) | "
                          f"mean equilibrium iterations used: {avg_iters:.1f}\033[0m")
                    for site_idx, g in zip(batch, site_grads):
                        print(f"\033[95m    site {site_idx:4d}: "
                              f"s={g[0]:+.4f}  d={g[1]:+.4f}  S_crit={g[2]:+.4f}\033[0m")
                    print(f"\033[95m    geometric median: "
                          f"s={combined_grad[0]:+.4f}  d={combined_grad[1]:+.4f}"
                          f"  S_crit={combined_grad[2]:+.4f}"
                          f"  (vs. mean: s={np.mean(site_grads,axis=0)[0]:+.4f}"
                          f"  d={np.mean(site_grads,axis=0)[1]:+.4f}"
                          f"  S_crit={np.mean(site_grads,axis=0)[2]:+.4f})\033[0m")

                optimizer.zero_grad()
                for p, g in zip(params, combined_grad):
                    p.grad = torch.tensor(float(g), device=device, dtype=torch.float32)
                grads = list(combined_grad)
            else:
                s_raw, d_raw, Scrit_raw = params
                # s = log10(n*r) (informative axis) and d = log10(n/r)
                # (near-flat axis): both tanh-bounded directly in (s, d)
                # space (see the (n,r)->(s,d) reparametrisation note above
                # `_hs_local_contrast`), THEN converted back to
                # (log10 n, log10 r) and exponentiated — a fixed-size Adam
                # step in s now corresponds to genuinely informative
                # exploration (crossing the softmax's uniform/argmax
                # crossover), while a step in d mostly explores the
                # near-degenerate n<->r trade-off at fixed selectivity.
                # Tg: tanh-bounded via S_crit (see `_Scrit_box` above) instead
                # of Tg directly — S_crit is tanh-bounded to
                # [SCRIT_MIN, SCRIT_MAX], handed straight to `cost_function`
                # (no `Tg`/`a` intermediate).
                s_s  = _tanh_rescale(s_raw, smin, smax)
                d_s  = _tanh_rescale(d_raw, dmin, dmax)
                log_n_s, log_r_s = _sd_to_logn_logr(s_s, d_s)
                n_s  = 10.0 ** log_n_s
                r_s  = 10.0 ** log_r_s
                Scrit_s = _tanh_rescale(Scrit_raw, SCRIT_MIN, SCRIT_MAX)

                if batch_chunk_size is not None and batch_chunk_size < len(batch):
                    # Memory-bounded path — same chunked-accumulation
                    # technique as `_full_dataset_grad`: split the batch,
                    # accumulate gradients across several smaller
                    # `.backward()` calls instead of holding every site's
                    # dense graph in memory at once. n_s/r_s/Scrit_s are
                    # recomputed FRESH from (s_raw, d_raw, Scrit_raw) inside
                    # each chunk (cheap scalar ops) rather than reusing the
                    # single copy computed above — reusing it would share
                    # the same upstream tanh_rescale graph nodes across
                    # chunks, which `.backward()` frees after the first
                    # call (RuntimeError on the second chunk without
                    # retain_graph=True); recomputing avoids that entirely.
                    optimizer.zero_grad()
                    loss_val = 0.0
                    batch_iters = []
                    for start in range(0, len(batch), batch_chunk_size):
                        chunk = batch[start:start + batch_chunk_size]
                        weight = len(chunk) / len(batch)
                        s_c  = _tanh_rescale(s_raw, smin, smax)
                        d_c  = _tanh_rescale(d_raw, dmin, dmax)
                        log_n_c, log_r_c = _sd_to_logn_logr(s_c, d_c)
                        n_c  = 10.0 ** log_n_c
                        r_c  = 10.0 ** log_r_c
                        Scrit_c = _tanh_rescale(Scrit_raw, SCRIT_MIN, SCRIT_MAX)
                        partial, chunk_iters = cost_function(
                            mdd, posteriors_and_masks, calibration_sites,
                            (n_c, r_c, Scrit_c), chunk,
                            carrying_capacity_params, hmean, adj_mats, K_is_list,
                            plot=False, verbose=verbose, return_iters=True,
                            seed_masks_list=seed_masks_list, breeding_masks_list=breeding_masks_list,
                        )
                        scaled = partial * weight
                        scaled.backward()
                        loss_val += scaled.item()
                        batch_iters.extend(chunk_iters)
                    avg_iters = float(np.mean(batch_iters)) if batch_iters else float("nan")
                else:
                    loss, batch_iters = cost_function(
                        mdd, posteriors_and_masks, calibration_sites,
                        (n_s, r_s, Scrit_s),
                        batch,
                        carrying_capacity_params,
                        hmean,
                        adj_mats,
                        K_is_list,
                        plot=do_plot,
                        verbose=verbose,
                        return_iters=True,
                        seed_masks_list=seed_masks_list, breeding_masks_list=breeding_masks_list,
                    )
                    avg_iters = float(np.mean(batch_iters)) if batch_iters else float("nan")

                    optimizer.zero_grad()
                    loss.backward()
                    loss_val = loss.item()

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
                s_val    = _logistic_rescale(params[0], smin, smax).item()
                d_val    = _logistic_rescale(params[1], dmin, dmax).item()
                log_n_val, log_r_val = _sd_to_logn_logr(s_val, d_val)
                n_val    = 10.0 ** log_n_val
                r_val    = 10.0 ** log_r_val
                Scrit_val  = _logistic_rescale(params[2], SCRIT_MIN, SCRIT_MAX).item()
                losses.append(loss_val)
                ln.append(n_val)
                lr.append(r_val)
                lScrit.append(Scrit_val)

            # Update progress bar: parameters (actual n, r, S_crit — S_crit
            # is the headline learned quantity now, Tg is not shown) + raw
            # gradient magnitudes (in (s, d, S_crit) space — gs/gd, not
            # gn/gr, since that's the space Adam is actually stepping in
            # now).
            step_bar.set_postfix(
                step=f"{step+1}/{max_iter}",
                loss=f"{loss_val:.4f}",
                n=f"{n_val:.0f}",
                r=f"{r_val:.5f}",
                S_crit=f"{Scrit_val:.3g}",
                gs=f"{grads[0]:+.3f}",
                gd=f"{grads[1]:+.3f}",
                gL=f"{grads[2]:+.3f}",
            )

            # Full-dataset cost AND gradient every 100 steps, over ALL sites
            # (unlike `grads` above, from a random mini-batch — noisy step
            # to step even near convergence). Computed w.r.t. the actual
            # (n, r, S_crit) values directly, via chunked gradient accumulation
            # (_full_dataset_grad) so at most `n_random_sites` sites' dense
            # matrix-inverse graphs are held in memory at once — a single
            # backward pass over ALL sites simultaneously was exceeding
            # 8 GB GPU memory even before the Hessian's own extra passes.
            if step % 100 == 0:
                full_loss_val, full_grad = _full_dataset_grad(
                    mdd, posteriors_and_masks, calibration_sites,
                    (n_val, r_val, Scrit_val), carrying_capacity_params, hmean,
                    adj_mats, K_is_list, n_sites_total,
                    chunk_size=n_random_sites,
                    seed_masks_list=seed_masks_list, breeding_masks_list=breeding_masks_list,
                )
                # Median across sites (not the mean) for the checkpoint
                # history/plot — with a right-skewed per-site cost
                # distribution (a few hard sites, most sites easy), the
                # mean is pulled up by the rare hard sites and lands near
                # the *top* of where most mini-batch draws fall, while the
                # median reflects the "typical" site's cost, which is what
                # you actually want to compare the mini-batch curve against.
                site_costs = _per_site_costs(
                    mdd, posteriors_and_masks, calibration_sites,
                    (n_val, r_val, Scrit_val), carrying_capacity_params, hmean,
                    adj_mats, K_is_list, n_sites_total,
                    seed_masks_list=seed_masks_list, breeding_masks_list=breeding_masks_list,
                )
                full_median_val = float(np.median(site_costs))
                full_cost_hist.append((step, full_loss_val, full_median_val))
                site_cost_hist.append((step, list(site_costs)))

            if step % 100 == 0:
                print(f"  step {step:4d} | loss={loss_val:.5f}"
                      f" | n={n_val:7.1f}  r={r_val:.6f}  S_crit={Scrit_val:.3g}"
                      # `grads` here are w.r.t. the RAW (s, d, S_crit) tanh
                      # parameters Adam actually steps in — NOT w.r.t.
                      # (n, r, S_crit) directly (unlike `full_grad` below,
                      # which genuinely is a (n, r, S_crit)-space gradient).
                      f" | raw grads (s,d,S_crit): s={grads[0]:+.4f}  d={grads[1]:+.4f}"
                      f"  S_crit={grads[2]:+.4f}"
                      # Mean number of equilibrium_distribution iterations
                      # needed for THIS mini-batch — a direct diagnostic
                      # of wall-clock cost per step (high values mean the
                      # current (n, r, S_crit) sits in a slow-to-converge
                      # region, e.g. a near-singular kernel inversion).
                      f" | mean eq. iters={avg_iters:.1f}"
                      f" | batch={list(batch)}")
                print(f"\033[93m  step {step:4d} | n={n_val:7.1f}  r={r_val:.6f}"
                      f"  S_crit={Scrit_val:.3g}"
                      f"  | global cost (mean)={full_loss_val:.5f}"
                      f"  (median)={full_median_val:.5f}"
                      f"  | full-dataset grad: n={full_grad[0]:+.4g}"
                      f"  r={full_grad[1]:+.4g}  S_crit={full_grad[2]:+.4g}\033[0m")

                # Hessian disabled by default (compute_hessian=False) — the
                # finite-difference eps values need re-tuning to the
                # current log-space parametrization's scale before its
                # eigenvalues can be trusted step to step (large erratic
                # swings observed otherwise). Extra cost when enabled: 6
                # more full-dataset gradient evaluations per checkpoint, on
                # top of the 1 already spent on full_loss/full_grad above.
                if compute_hessian:
                    hess_now = _hessian_at_point(
                        mdd, posteriors_and_masks, calibration_sites,
                        (n_val, r_val, Scrit_val), carrying_capacity_params, hmean,
                        adj_mats, K_is_list, n_sites_total,
                        chunk_size=n_random_sites,
                    )
                    eig_str = "  ".join(f"lambda_{i}={ev:+.4g}" for i, ev in enumerate(hess_now["eigvals"]))
                    print(f"\033[93m  step {step:4d} | Hessian eigvals: {eig_str}\033[0m")
                    # Eigenvectors — the direction (in n/r/S_crit units) each
                    # eigenvalue's curvature is measured along. A near-zero
                    # eigenvalue with an eigenvector like (n, r, S_crit) =
                    # (0.7, 0.7, 0.0) means the cost is flat along that
                    # particular combined n-r direction (a ridge), not
                    # along n or r individually.
                    for i, (ev, vec) in enumerate(zip(hess_now["eigvals"], hess_now["eigvecs"].T)):
                        print(f"\033[93m    eigvec_{i} (lambda={ev:+.4g}): "
                              f"n={vec[0]:+.3f}  r={vec[1]:+.3f}  S_crit={vec[2]:+.3f}\033[0m")

                    # Release the Hessian's own retained result before
                    # resuming mini-batch steps (the 6 finite-diff
                    # evaluations already freed their own graphs
                    # incrementally inside _grad_at).
                    del hess_now
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        step_bar.close()
        all_costs.append(losses)
        all_full_cost.append(full_cost_hist)
        all_site_cost_hist.append(site_cost_hist)
        all_n.append(ln)
        all_r.append(lr)
        all_Scrit.append(lScrit)

    # ------------------------------------------------------------------
    # Aggregate endpoints
    # ------------------------------------------------------------------
    r_ends   = np.array([lr[-1]   for lr   in all_r])
    n_ends   = np.array([ln[-1]   for ln   in all_n])
    Scrit_ends = np.array([lScrit[-1] for lScrit in all_Scrit])

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

    mean_r   = float(np.sum(r_ends   * weights))
    mean_n   = float(np.sum(n_ends   * weights))
    mean_Scrit = float(np.sum(Scrit_ends * weights))

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

        # Raw cost curves (mini-batch cost and full-dataset cost plotted on
        # their own true scale — NOT normalised. An earlier version
        # normalised the full-dataset cost using the mini-batch cost's own
        # min/max, but these are two different quantities (a 3-site vs. an
        # all-sites average) with no reason to share a common scale, which
        # made the full-dataset overlay appear mismatched/off-scale
        # relative to the mini-batch curve's true trajectory.
        plt.figure()
        for idx in range(len(all_costs)):
            arr = np.array(all_costs[idx])
            label = (f"run {learning_runs[idx]}" if not all_together
                     else f"mini-batch ({n_random_sites} sites)")
            plt.plot(arr, alpha=0.5, label=label)

            # Full-dataset cost overlay: BOTH mean and median across sites.
            # With a right-skewed per-site cost distribution (a few hard
            # sites, most sites easy), the mean is pulled up by the rare
            # hard sites and sits near the *top* of where mini-batch draws
            # land, while the median reflects the "typical" site's cost —
            # showing both makes that skew directly visible on the plot.
            full_hist = all_full_cost[idx]
            if full_hist:
                fc_steps  = np.array([s for s, _, _ in full_hist])
                fc_mean   = np.array([m for _, m, _ in full_hist])
                fc_median = np.array([md for _, _, md in full_hist])
                plt.plot(fc_steps, fc_mean, color="red", linewidth=1.2,
                         label="full-dataset cost (mean)")
                plt.plot(fc_steps, fc_median, color="darkorange", linewidth=1.2,
                         label="full-dataset cost (median)")

            # Percentile bands across the 21 per-site costs at each
            # checkpoint, instead of just the raw min/max — overlapping
            # blue fills (2.5-97.5, 5-95, 10-90, 25-75) so the shading
            # naturally darkens toward the center of the distribution,
            # giving a visual sense of density/concentration rather than
            # just the bare extremes.
            site_hist = all_site_cost_hist[idx] if idx < len(all_site_cost_hist) else []
            if site_hist:
                sc_steps = np.array([s for s, _ in site_hist])
                sc_costs = [np.array(c) for _, c in site_hist]
                for lo_q, hi_q in [(2.5, 97.5), (5, 95), (10, 90), (25, 75)]:
                    lo = np.array([np.percentile(c, lo_q) for c in sc_costs])
                    hi = np.array([np.percentile(c, hi_q) for c in sc_costs])
                    plt.fill_between(
                        sc_steps, lo, hi, color="blue", alpha=0.15,
                        label=f"{lo_q:g}-{hi_q:g}% (sites)",
                    )

        plt.xlabel("Optimisation step")
        plt.ylabel("Cost")
        plt.title("Cost function evolution during optimisation")
        plt.legend()
        plt.grid(linestyle="--", color="grey", linewidth=0.2, alpha=0.5)
        _save_show(f"{prefix}cost.png")

        # Per-site cost distribution at each 100-step checkpoint — direct
        # empirical check of whether the per-site costs are right-skewed
        # (a few hard sites pulling the mean above the median, as the
        # mean/median overlay above suggests) rather than inferring it
        # indirectly from the mini-batch curve's shape. Same x-axis range
        # across all subplots so the checkpoints are directly comparable.
        for idx in range(len(all_site_cost_hist)):
            site_hist = all_site_cost_hist[idx]
            if not site_hist:
                continue
            n_checkpoints = len(site_hist)
            all_vals = np.concatenate([np.array(c) for _, c in site_hist])
            xlo, xhi = float(all_vals.min()), float(all_vals.max())

            ncols = 5
            nrows = int(np.ceil(n_checkpoints / ncols))
            fig, axes = plt.subplots(
                nrows, ncols, figsize=(3.2 * ncols, 2.4 * nrows),
                sharex=True, sharey=True,
            )
            axes = np.atleast_1d(axes).flatten()
            for ax, (step, costs) in zip(axes, site_hist):
                costs = np.array(costs)
                ax.hist(costs, bins=15, range=(xlo, xhi), color="steelblue", alpha=0.8)
                ax.axvline(np.mean(costs), color="red", linewidth=1.2, label="mean")
                ax.axvline(np.median(costs), color="darkorange", linewidth=1.2, label="median")
                ax.set_title(f"step {step}", fontsize=9)
            for ax in axes[n_checkpoints:]:
                ax.axis("off")
            axes[0].legend(fontsize=7)
            fig.suptitle(
                f"Per-site cost distribution over training"
                + (f" (run {learning_runs[idx]})" if not all_together else ""),
            )
            fig.supxlabel("Cost")
            fig.supylabel("Count (out of 21 sites)")
            fig.tight_layout()
            run_suffix = f"_run{learning_runs[idx]}" if not all_together else ""
            _save_show(f"{prefix}site_cost_histograms{run_suffix}.png")

        # Endpoint scatter (n vs r, r vs S_crit, n vs S_crit) — `S_crit` is
        # the headline learned quantity, used directly (no Tg anywhere).
        sizes_scatter = (
            np.ones(len(r_ends)) * 100
            if average_method in ("Likelihood", "same")
            else weights * 200
        )
        for xlabel, ylabel, x, y in [
            ("r (alpha)", "n (tolerance)", r_ends, n_ends),
            ("r (alpha)", "S_crit (survival threshold)",  r_ends, Scrit_ends),
            ("n (tolerance)", "S_crit (survival threshold)", n_ends, Scrit_ends),
        ]:
            plt.figure()
            plt.scatter(x, y, s=sizes_scatter, color="blue")
            plt.scatter(
                (mean_r if "r" in xlabel else mean_n),
                (mean_n if "n" in ylabel else mean_Scrit),
                color="red", s=150, marker="X",
                label="Weighted estimate",
            )
            for idx in range(len(all_costs)):
                px = np.array(all_r[idx] if "r" in xlabel else all_n[idx])
                py = np.array(all_n[idx] if "n" in ylabel else all_Scrit[idx])
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

        # 3-D interactive plot (Plotly) — plotted in `S_crit` (the actual
        # learned/sampled variable), not `Tg`.
        _kde_3d_plotly(
            r_ends, n_ends, Scrit_ends,
            priors=priors,
            list_traj=[all_n, all_r, all_Scrit],
            weights=weights,
            sizes_scatter=weights * 20,
            learning_runs=learning_runs,
            all_together=all_together,
        )

    # The 4th return value is `mean_Scrit` directly (NOT `Tg` — Tg is
    # invalid/NaN for S_crit <= 0.5, a now-reachable region, so it can no
    # longer be reported at all). Callers unpacking
    # `Ew, n, r, Tg, *_ = learn_dispersal_parameters(...)` (see batch.py)
    # must be updated to `Ew, n, r, S_crit, *_ = ...`.
    return (estimated_ew, mean_n, mean_r, mean_Scrit, all_costs, weights,
            [r_ends, n_ends, Scrit_ends], mean_Scrit, Scrit_ends)


# ---------------------------------------------------------------------------
# Deterministic full-dataset refinement from an explicit starting point
# ---------------------------------------------------------------------------

def refine_from_point(
    calibration_sites,
    hmean: float,
    mdd: float,
    posteriors_and_masks: tuple,
    carrying_capacity_params: tuple,
    init_point: tuple | None = None,
    priors: list | None = None,
    max_iter: int = 300,
    lr: float = 0.05,
    chunk_size: int = 3,
    check_every: int = 10,
    plot_summary: bool = True,
    save_fig_folder: str | None = None,
    species_name: str | None = None,
    r_min: float | None = None,
    r_survival_tol: float = 0.005,
    r_survival_alpha: float = 1.11,
    gpu_memory_fraction: float | None = 0.8,
    robust_gradient: bool = False,
    seed_fraction: float = 0.25,
    s_halfwidth: float = 3.0,
    d_max_r_survival_tol: float = 1e-12,
    compute_hessian: bool = False,
    hessian_eps: tuple = (0.05, 0.05, 0.01),
) -> tuple:
    """Deterministic full-dataset Adam refinement starting from an explicit
    ``(s, d, S_crit)`` point — no mini-batching, no stochasticity: every
    step's gradient is the EXACT full-dataset gradient (all calibration
    sites, via chunked accumulation — see `_full_dataset_grad` — so memory
    stays bounded by `chunk_size` sites at a time, not all of them at once).

    Useful to directly check whether the optimizer moves in a sensible
    (cost-decreasing) direction from a given starting point — e.g. the
    endpoint of a stochastic `learn_dispersal_parameters` run, or any other
    point of interest — without any mini-batch noise muddying the picture.

    Parameters
    ----------
    init_point:
        ``(s0, d0, S_crit0)`` — the exact starting point, given directly
        in this function's OWN learning space (NOT ``(n, r, Tg)`` any
        more), since that's what's actually optimised. Values are
        clamped into the (possibly auto-computed) box if given outside
        it. Default ``None`` — start EXACTLY at each box's centre
        (``raw=0`` for all three), matching
        `learn_dispersal_parameters`'s ``init_params=None`` default.
    priors, r_min, r_survival_tol, r_survival_alpha:
        Same meaning as in :func:`learn_dispersal_parameters` — n and r are
        learned in log10 space, same auto ``r_min`` computation applies.
    max_iter, lr, chunk_size:
        Number of deterministic Adam steps, its (fixed, undecayed) learning
        rate, and the site-chunk size for `_full_dataset_grad` (smaller =
        less peak memory, more forward/backward passes per step).
    check_every:
        Print progress every this many steps.
    robust_gradient:
        If True, EVERY step computes each of the `n_sites_total` sites'
        gradient SEPARATELY (not the chunked weighted-sum of
        `_full_dataset_grad`), and the actual Adam update uses the
        (Byzantine-robust) GEOMETRIC MEDIAN of these per-site gradient
        vectors instead of their mean — so a site whose gradient disagrees
        sharply with the rest is naturally down-weighted at every single
        step, not just diagnosed after the fact. Costs `n_sites_total`
        separate forward/backward passes per step (same total forward
        compute as the chunked approach, just not fused/summed). Default
        False (unchanged exact-mean behaviour).
    seed_fraction:
        Fraction of pixels seeded per site's sparse initial density mask,
        drawn ONCE per site at the start of this run and FIXED for every
        evaluation thereafter (not redrawn per step). Default 0.25.
    compute_hessian:
        If True, after the loop finishes, compute the Hessian of the
        full-dataset cost w.r.t. `(s, d, S_crit)` at the FINAL converged
        point (see `_hessian_at_point_sd`) and print its eigenvalues with
        an interpretation: all positive means the endpoint is a genuine
        local minimum (a true local equilibrium of the cost surface —
        the optimiser cannot improve on it in ANY direction); any
        negative eigenvalue means it's a SADDLE POINT instead (the
        gradient vanished, but the surface still decreases along the
        corresponding eigenvector — not a real equilibrium, worth
        perturbing along that direction and re-running `refine_from_point`
        to see if a genuinely better point exists nearby). Costs
        `2 * 3 = 6` extra full-dataset gradient evaluations (one
        `_full_dataset_grad_sd` call in each direction, for each of the
        3 axes) — noticeably more compute than a single step, so left
        off by default.
    hessian_eps:
        ``compute_hessian=True``-only. The `(eps_s, eps_d, eps_Scrit)`
        finite-difference step sizes forwarded to `_hessian_at_point_sd`.
        Default ``(0.05, 0.05, 0.01)`` — reasonable for all species since
        `s`/`d`/`S_crit` are comparably-scaled axes (unlike `n`/`r`,
        which would need per-species rescaling).

    Returns
    -------
    Ew, n, r, Tg : float
        Final point after `max_iter` steps.
    cost_hist : list[tuple[int, float]]
        ``(step, full-dataset cost)`` at every step.
    """
    if gpu_memory_fraction is not None and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction, device=0)

    L, k, x0 = carrying_capacity_params

    if priors is None:
        C    = (2.0 * np.exp(-1.11 / mdd)) / (1.0 + np.exp(-2.0 * 1.11 / mdd))
        rmax = (1.0 / np.log(hmean)) * np.log(C)
        if r_min is None:
            r_min = _find_r_min(hmean, r_survival_alpha, mdd, tol=r_survival_tol)
            print(f"[priors] Auto r_min={r_min:.6g} (survival within "
                  f"{r_survival_tol*100:.2g}pp of its r->0 limit below this).")
        log_rmin, log_rmax = np.log10(r_min), np.log10(rmax)
        hs_bar, delta_h_bar = _hs_local_contrast(calibration_sites)
        s_center = _s_center(hs_bar, delta_h_bar)
        smin, smax = _s_range(s_center, s_halfwidth)
        nmin, nmax = _n_box_from_s_and_r(smin, smax, log_rmin, log_rmax)
        priors = [(nmin, nmax), (r_min, rmax), (-2.0, 2.0)]
        print(f"[priors] r range for this species: r_min={r_min:.6g}  "
              f"r_max={rmax:.6g}  (hmean={hmean:.4g}, mdd={mdd:.4g})")

    (nmin, nmax), (rmin, rmax), (_tgmin_unused, _tgmax_unused) = priors
    log_rmin, log_rmax = np.log10(rmin), np.log10(rmax)

    # d_max EXTENSION via a SEPARATE, much smaller r floor (see
    # `learn_dispersal_parameters` for the full derivation) — d_min stays
    # computed with the MODERATE `r_survival_tol` above; only d_max is
    # pushed out further via this independent, extremely small
    # `d_max_r_survival_tol` r-floor.
    r_min_for_dmax = _find_r_min(hmean, r_survival_alpha, mdd, tol=d_max_r_survival_tol)
    log_rmin_for_d = np.log10(r_min_for_dmax)

    print("[precompute] Building adjacency matrices and K_is ...")
    n_sites_total = len(calibration_sites)
    all_indices = list(range(n_sites_total))
    adj_mats, K_is_list, seed_masks_list = [], [], []
    for hs_map in tqdm(calibration_sites.hs_maps, desc="  Sites", leave=False):
        hs_t    = torch.tensor(hs_map, dtype=torch.float32, device=device)
        adj_mats.append(adjacency_matrix_torch(hs_t))
        K_is_flat = torch.tensor(
            _logistic(hs_map.flatten().astype(np.float32), L, k, x0),
            dtype=torch.float32,
        )
        K_is_list.append(K_is_flat)
        seed_masks_list.append(seed_mask(
            K_is_flat.shape, seed_fraction=seed_fraction,
            device=K_is_flat.device, dtype=K_is_flat.dtype,
        ))
    print(f"[precompute] Done — {n_sites_total} sites.")
    breeding_masks_list = _build_breeding_masks_list(calibration_sites, n_sites_total)
    print(f"[precompute] Seeded initial densities: {seed_fraction*100:.0f}% of "
          f"pixels per site, FIXED for the whole run (not redrawn per step).")

    # (n, r) -> (s, d) reparametrisation — same rationale/derivation as
    # `learn_dispersal_parameters` (see the module-level note above
    # `_hs_local_contrast`): s=log10(n*r) is the informative axis (controls
    # the dispersal kernel's selectivity), d=log10(n/r) the near-flat one.
    # Learning directly in (s, d) here too (instead of independently in
    # (log10 n, log10 r)) fixes the exact zigzag/drift pattern this
    # function's own diagnostic plots showed: full-dataset cost converging
    # quickly while n and r kept drifting in lockstep along the flat ridge
    # for hundreds more steps with no further cost improvement. Tg is
    # learned via `S_crit` (see `_Scrit_box`) for the same reason as in
    # `learn_dispersal_parameters`.
    #
    # Printed in this exact order (r -> s -> n -> d), see
    # `learn_dispersal_parameters` for why.
    hs_bar, delta_h_bar = _hs_local_contrast(calibration_sites)
    s_center = _s_center(hs_bar, delta_h_bar)
    (smin, smax), (dmin, dmax_conservative) = _sd_box(nmin, nmax, log_rmin, log_rmax, s_center, s_halfwidth)
    # d_max is now recomputed using `log_rmin_for_d` (the SEPARATE,
    # extremely small r floor from `d_max_r_survival_tol`, computed above)
    # instead of the moderate `log_rmin` — extends d_max far out for
    # species whose true optimum needs near-zero dispersal mortality,
    # while `dmin`/`dmax_conservative` (computed with the moderate r
    # floor) are kept around for the default init point's box centre
    # (see below) and for reference/printing. The corner achieving d_max
    # is (log_n=n_max, log_r=log_rmin_for_d): d increases AND s decreases
    # there relative to the conservative corner (s = n_max + log_rmin_for_d
    # < n_max + log_rmin), i.e. the newly reachable region skews toward
    # the UPPER-LEFT (low s, high d) — the opposite of widening n_max,
    # which would skew the same corner toward the upper-RIGHT.
    dmax = nmax - log_rmin_for_d
    print(f"[priors] s prior for this species: hs_bar={hs_bar:.4g}  "
          f"delta_h_bar={delta_h_bar:.4g}  s*=log10(n*r)={s_center:.4g}  "
          f"s box=[{smin:.4g}, {smax:.4g}]  (s_halfwidth={s_halfwidth:.3g})")
    print(f"[priors] n range for this species (data-driven from s box + "
          f"r box): n_min={10**nmin:.4g}  n_max={10**nmax:.4g}  "
          f"(log10: [{nmin:.4g}, {nmax:.4g}])")
    print(f"[priors] d prior for this species (from n box + r box): "
          f"conservative d box=[{dmin:.4g}, {dmax_conservative:.4g}] "
          f"(r_survival_tol={r_survival_tol:.3g}, used for the default init "
          f"point's box centre)  ->  LEARNING d box=[{dmin:.4g}, {dmax:.4g}] "
          f"(d_max extended via d_max_r_survival_tol={d_max_r_survival_tol:.3g})")

    # S_crit reparametrisation of Tg (see the module-level growth-timescale
    # note and `_Scrit_box`) — a fixed universal (0.5, 1.0) box, no
    # species-specific precompute needed.
    SCRIT_MIN, SCRIT_MAX = _Scrit_box()
    print(f"[priors] S_crit (survival-threshold) reparametrisation for "
          f"this species: S_crit box=[{SCRIT_MIN:.4g}, {SCRIT_MAX:.4g}]  "
          f"(interpretation: S_crit = critical per-dispersal-step survival "
          f"probability below which NO growth rate can sustain a local "
          f"population — S_crit=0.5 is the g->1 (fastest possible growth, "
          f"Tg->0) limit, S_crit->1 is the g->0 (Tg->inf) limit)")

    # Initialise raw params (via inverse-tanh) at the exact requested point,
    # instead of the box's midpoint — clamped into the (s, d) / S_crit
    # boxes if given a point outside them. `init_point` is given directly
    # in this function's OWN learning space, (s, d, S_crit) — not
    # (n, r, Tg) any more — since that's what's actually optimised.
    # `init_point=None` (default): start EXACTLY at each box's centre —
    # same convention as `learn_dispersal_parameters`'s `init_params=None`
    # default, for consistency between the two functions. `d`'s centre
    # uses the CONSERVATIVE box (dmin, dmax_conservative — moderate
    # `r_survival_tol`), not the extended LEARNING box (dmin, dmax) — see
    # the module-level note above `_n_box_from_s_and_r`/`_sd_box`: d_max
    # is pushed out far via a separate, much smaller `d_max_r_survival_tol`
    # floor so species that need it CAN reach it, but the default starting
    # point should stay in the well-supported region, not already halfway
    # to that extreme edge.
    d_center_conservative = (dmin + dmax_conservative) / 2.0
    if init_point is None:
        s0, d0, Scrit0 = (smin + smax) / 2.0, d_center_conservative, (SCRIT_MIN + SCRIT_MAX) / 2.0
        s_raw_init     = 0.0
        d_raw_init     = _inv_tanh_rescale(d0, dmin, dmax)
        Scrit_raw_init = 0.0
    else:
        s0, d0, Scrit0 = init_point
        s0 = min(max(s0, smin), smax)
        d0 = min(max(d0, dmin), dmax)
        Scrit0 = min(max(Scrit0, SCRIT_MIN), SCRIT_MAX)
        s_raw_init     = _inv_tanh_rescale(s0, smin, smax)
        d_raw_init     = _inv_tanh_rescale(d0, dmin, dmax)
        Scrit_raw_init = _inv_tanh_rescale(Scrit0, SCRIT_MIN, SCRIT_MAX)
    # n0/r0 recovered only for the human-readable print below.
    log_n0, log_r0 = _sd_to_logn_logr(s0, d0)
    n0, r0 = 10.0 ** log_n0, 10.0 ** log_r0
    # params[0] = s_raw (informative axis, ~log10(n*r)),
    # params[1] = d_raw (near-flat axis, ~log10(n/r)),
    # params[2] = Scrit_raw (tanh-bounded to the S_crit box, see `_Scrit_box`
    # above — converted to actual Tg via `_Scrit_to_Tg` immediately before
    # being handed to `cost_function`).
    params = [
        torch.nn.Parameter(torch.tensor(s_raw_init,   device=device, dtype=torch.float32, requires_grad=True)),
        torch.nn.Parameter(torch.tensor(d_raw_init,   device=device, dtype=torch.float32, requires_grad=True)),
        torch.nn.Parameter(torch.tensor(Scrit_raw_init, device=device, dtype=torch.float32, requires_grad=True)),
    ]
    optimizer = torch.optim.Adam(params, lr=lr)

    print(f"\033[96m[refine_from_point] Starting deterministic full-dataset "
          f"refinement from s={s0:.4g}  d={d0:.4g}  S_crit={Scrit0:.3g}\033[0m")

    cost_hist: list[tuple[int, float]] = []
    ln, lr_hist, lScrit, ls, ld = [], [], [], [], []
    last_cost_val = float("nan")  # last known cost, for the progress-bar postfix on non-print steps
    step_bar = tqdm(range(max_iter), desc="  Deterministic steps", leave=True)
    for step in step_bar:
        site_grads_arr = None  # populated below when robust_gradient=True; reused for the check_every print

        if robust_gradient:
            # Each of the n_sites_total sites gets its OWN forward+backward
            # pass, separately — the actual Adam update below uses the
            # GEOMETRIC MEDIAN of these per-site gradient vectors instead
            # of their mean (see `_geometric_median`'s docstring). The
            # arithmetic-mean COST (total_loss_val) is only a comparison
            # figure — not what's being optimised (the update uses the
            # median gradient, not the mean loss) — so it's only computed
            # at check_every print steps, not every step.
            compute_cost = (step % check_every == 0)
            site_grads = []
            loss_val_accum = 0.0
            for site_idx in all_indices:
                s_raw, d_raw, Scrit_raw = params
                s_s  = _tanh_rescale(s_raw, smin, smax)
                d_s  = _tanh_rescale(d_raw, dmin, dmax)
                log_n_s, log_r_s = _sd_to_logn_logr(s_s, d_s)
                n_s  = 10.0 ** log_n_s
                r_s  = 10.0 ** log_r_s
                Scrit_s = _tanh_rescale(Scrit_raw, SCRIT_MIN, SCRIT_MAX)
                site_loss = cost_function(
                    mdd, posteriors_and_masks, calibration_sites,
                    (n_s, r_s, Scrit_s), [site_idx],
                    carrying_capacity_params, hmean, adj_mats, K_is_list,
                    plot=False, verbose=False,
                    seed_masks_list=seed_masks_list, breeding_masks_list=breeding_masks_list,
                )
                optimizer.zero_grad()
                site_loss.backward()
                site_grads.append([p.grad.item() for p in params])
                if compute_cost:
                    loss_val_accum += site_loss.item()
                del n_s, r_s, site_loss
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            total_loss_val = (loss_val_accum / n_sites_total) if compute_cost else None
            site_grads_arr = np.array(site_grads)
            combined_grad = _geometric_median(site_grads_arr)
            optimizer.zero_grad()
            for p, g in zip(params, combined_grad):
                p.grad = torch.tensor(float(g), device=device, dtype=torch.float32)
            grads = list(combined_grad)
        else:
            optimizer.zero_grad()
            total_loss_val = 0.0
            for start in range(0, n_sites_total, chunk_size):
                chunk = all_indices[start:start + chunk_size]
                weight = len(chunk) / n_sites_total
                s_raw, d_raw, Scrit_raw = params
                s_s  = _tanh_rescale(s_raw, smin, smax)
                d_s  = _tanh_rescale(d_raw, dmin, dmax)
                log_n_s, log_r_s = _sd_to_logn_logr(s_s, d_s)
                n_s  = 10.0 ** log_n_s
                r_s  = 10.0 ** log_r_s
                Scrit_s = _tanh_rescale(Scrit_raw, SCRIT_MIN, SCRIT_MAX)
                partial = cost_function(
                    mdd, posteriors_and_masks, calibration_sites,
                    (n_s, r_s, Scrit_s), chunk,
                    carrying_capacity_params, hmean, adj_mats, K_is_list,
                    plot=False, verbose=False,
                    seed_masks_list=seed_masks_list, breeding_masks_list=breeding_masks_list,
                )
                scaled = partial * weight
                scaled.backward()
                total_loss_val += scaled.item()
                del n_s, r_s, partial, scaled
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            grads = [p.grad.item() if p.grad is not None else float("nan") for p in params]

        if any(math.isnan(g) for g in grads):
            raise RuntimeError(f"NaN gradient at step {step}  grads={grads}")
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        torch.nn.utils.clip_grad_value_(params, clip_value=5.0)
        optimizer.step()

        with torch.no_grad():
            s_val = _logistic_rescale(params[0], smin, smax).item()
            d_val = _logistic_rescale(params[1], dmin, dmax).item()
            log_n_val, log_r_val = _sd_to_logn_logr(s_val, d_val)
            n_val  = 10.0 ** log_n_val
            r_val  = 10.0 ** log_r_val
            Scrit_val = _logistic_rescale(params[2], SCRIT_MIN, SCRIT_MAX).item()
        if total_loss_val is not None:
            cost_hist.append((step, total_loss_val))
            last_cost_val = total_loss_val
        ln.append(n_val); lr_hist.append(r_val); lScrit.append(Scrit_val)
        ls.append(s_val); ld.append(d_val)

        step_bar.set_postfix(cost=f"{last_cost_val:.5f}", s=f"{s_val:.3g}",
                              d=f"{d_val:.3g}", S_crit=f"{Scrit_val:.3g}")
        if step % check_every == 0:
            print(f"  step {step:4d} | cost={total_loss_val:.5f}"
                  f" | s={s_val:+.4g}  d={d_val:+.4g}  S_crit={Scrit_val:.3g}"
                  f" | raw grads (s,d,S_crit): s={grads[0]:+.4g}  d={grads[1]:+.4g}  S_crit={grads[2]:+.4g}")

            if robust_gradient:
                # Already computed above for the actual update this step —
                # just report mean vs. the geometric median actually used,
                # no extra computation needed.
                mean_grad = site_grads_arr.mean(axis=0)
                print(f"\033[95m    per-site raw grad (s,d,S_crit) (n={n_sites_total}): "
                      f"mean=[s={mean_grad[0]:+.4g}  d={mean_grad[1]:+.4g}  S_crit={mean_grad[2]:+.4g}]"
                      f"   geometric median (USED)=[s={grads[0]:+.4g}  d={grads[1]:+.4g}"
                      f"  S_crit={grads[2]:+.4g}]\033[0m")
            else:
                # Diagnostic only (does NOT change the update above, which
                # still uses the exact chunked full-dataset mean gradient)
                # — per-site gradients at the CURRENT point, so you can
                # compare the mean (what's actually used) against the
                # Byzantine-robust geometric median (what robust_gradient=
                # True would use instead), to see how much a disagreeing
                # site is influencing the applied update. The autograd
                # LEAVES here are `s_p`/`d_p`/`Scrit_p` directly (not
                # `n_p`/`r_p`/`tg_p`), matching `(s, d, S_crit)` being the
                # actual learned variables now.
                site_grads = []
                for site_idx in all_indices:
                    s_p   = torch.tensor(s_val,   requires_grad=True, dtype=torch.float32, device=device)
                    d_p   = torch.tensor(d_val,   requires_grad=True, dtype=torch.float32, device=device)
                    Scrit_p = torch.tensor(Scrit_val, requires_grad=True, dtype=torch.float32, device=device)
                    log_n_p, log_r_p = _sd_to_logn_logr(s_p, d_p)
                    n_p, r_p = 10.0 ** log_n_p, 10.0 ** log_r_p
                    site_loss = cost_function(
                        mdd, posteriors_and_masks, calibration_sites,
                        (n_p, r_p, Scrit_p), [site_idx],
                        carrying_capacity_params, hmean, adj_mats, K_is_list,
                        plot=False, verbose=False,
                        seed_masks_list=seed_masks_list, breeding_masks_list=breeding_masks_list,
                    )
                    site_loss.backward()
                    site_grads.append([s_p.grad.item(), d_p.grad.item(), Scrit_p.grad.item()])
                    del s_p, d_p, Scrit_p, n_p, r_p, site_loss
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                site_grads = np.array(site_grads)
                mean_grad = site_grads.mean(axis=0)
                median_grad = _geometric_median(site_grads)
                print(f"\033[95m    per-site raw grad (s,d,S_crit) (n={n_sites_total}): "
                      f"mean (USED)=[s={mean_grad[0]:+.4g}  d={mean_grad[1]:+.4g}  S_crit={mean_grad[2]:+.4g}]"
                      f"   geometric median=[s={median_grad[0]:+.4g}  d={median_grad[1]:+.4g}"
                      f"  S_crit={median_grad[2]:+.4g}]\033[0m")

    n_final, r_final, Scrit_final = ln[-1], lr_hist[-1], lScrit[-1]
    s_final, d_final = ls[-1], ld[-1]
    C_final = (2.0 * np.exp(-1.11 / mdd)) / (1.0 + np.exp(-2.0 * 1.11 / mdd))
    estimated_ew = float(C_final / (hmean ** r_final - C_final))
    print(f"\033[92m[refine_from_point] Done: s={s_final:.4g}  d={d_final:.4g}"
          f"  S_crit={Scrit_final:.3g}  Ew={estimated_ew:.1f}  "
          f"cost {cost_hist[0][1]:.5f} -> {cost_hist[-1][1]:.5f}\033[0m")

    if compute_hessian:
        print(f"\033[96m[refine_from_point] Computing Hessian at the endpoint "
              f"(s={s_final:.4g}, d={d_final:.4g}, S_crit={Scrit_final:.3g}) "
              f"to check whether this is a genuine local minimum...\033[0m")
        hess_info = _hessian_at_point_sd(
            mdd, posteriors_and_masks, calibration_sites,
            (s_final, d_final, Scrit_final), (L, k, x0), hmean,
            adj_mats, K_is_list, n_sites_total,
            scrit_box=(SCRIT_MIN, SCRIT_MAX), eps=hessian_eps,
            chunk_size=chunk_size, seed_masks_list=seed_masks_list, breeding_masks_list=breeding_masks_list,
        )
        eigvals = hess_info["eigvals"]
        if np.all(eigvals > 0):
            verdict = "\033[92mLOCAL MINIMUM (genuine equilibrium — all eigenvalues positive)\033[0m"
        elif np.any(eigvals < 0):
            verdict = "\033[91mSADDLE POINT (NOT a true equilibrium — at least one negative eigenvalue)\033[0m"
        else:
            verdict = "\033[93mDEGENERATE (near-zero eigenvalue(s) — inconclusive at this eps)\033[0m"
        # np.linalg.eigh returns eigenvalues in ASCENDING order — NOT in
        # (s, d, S_crit) axis order — and each eigenvalue's direction is a
        # potentially-mixed eigenvector (a linear combination of s, d,
        # S_crit), not necessarily a pure axis. Print each eigenvector's
        # per-axis components alongside its eigenvalue, and flag which
        # axis dominates it (largest |component|), so this doesn't have
        # to be guessed at from the eigenvalues alone.
        axis_names = ["s", "d", "S_crit"]
        eigvecs = hess_info["eigvecs"]   # column i = eigenvector for eigvals[i]
        print(f"[refine_from_point] Hessian eigenvalues + eigenvectors "
              f"(s, d, S_crit space; ascending order):")
        for i in range(3):
            vec = eigvecs[:, i]
            dominant = axis_names[int(np.argmax(np.abs(vec)))]
            composition = "  ".join(f"{axis_names[k]}={vec[k]:+.3f}" for k in range(3))
            print(f"    eigval[{i}] = {eigvals[i]:+.4g}   "
                  f"dominant axis: {dominant}   eigvec=({composition})")
        print(f"[refine_from_point] Verdict: {verdict}")

    if plot_summary:
        prefix = f"{species_name}_" if species_name else ""

        def _save_show(filename: str) -> None:
            if save_fig_folder:
                plt.savefig(os.path.join(save_fig_folder, filename),
                            dpi=150, bbox_inches="tight")
            plt.show()

        steps = [s for s, _ in cost_hist]
        costs = [c for _, c in cost_hist]
        # 2x3 grid: cost, then (s, d, S_crit) — the actual learned axes —
        # and (n, r) — the original-units reference view, shown here only
        # (not in the console prints, which report (s, d, S_crit) alone).
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes[0, 0].plot(steps, costs, color="darkred")
        axes[0, 0].set_title("Full-dataset cost (deterministic)")
        axes[0, 0].set_xlabel("Step"); axes[0, 0].set_ylabel("Cost")
        axes[0, 1].plot(steps, ls, color="mediumpurple")
        axes[0, 1].set_title("s = log10(n*r)  [informative]"); axes[0, 1].set_xlabel("Step")
        axes[0, 2].plot(steps, ld, color="peru")
        axes[0, 2].set_title("d = log10(n/r)  [near-flat]"); axes[0, 2].set_xlabel("Step")
        axes[1, 0].plot(steps, lScrit, color="darkorange")
        axes[1, 0].set_title("S_crit (survival threshold)"); axes[1, 0].set_xlabel("Step")
        axes[1, 1].plot(steps, ln, color="steelblue")
        axes[1, 1].set_title("n  [reference only]"); axes[1, 1].set_xlabel("Step")
        axes[1, 2].plot(steps, lr_hist, color="seagreen")
        axes[1, 2].set_title("r  [reference only]"); axes[1, 2].set_xlabel("Step")
        for ax in axes.flat:
            ax.grid(linestyle="--", color="grey", linewidth=0.2, alpha=0.5)
        fig.suptitle(f"Deterministic refinement from s={s0:.3g} d={d0:.3g} S_crit={Scrit0:.3g}"
                     f"  (n={n0:.1f} r={r0:.5f})")
        fig.tight_layout()
        _save_show(f"{prefix}refine_from_point.png")

    # The 4th return value is `Scrit_final` directly (NOT `Tg` — Tg is
    # invalid/NaN for S_crit <= 0.5, which is now a reachable region, so
    # it can no longer be reported at all). Callers unpacking
    # `Ew, n, r, Tg, *_ = refine_from_point(...)` (see batch.py) must be
    # updated to `Ew, n, r, S_crit, *_ = ...`.
    return estimated_ew, n_final, r_final, Scrit_final, cost_hist, Scrit_final


# ---------------------------------------------------------------------------
# Full joint posterior via MALA (Metropolis-Adjusted Langevin Algorithm),
# in (s, d, S_crit) space
# ---------------------------------------------------------------------------

def run_mala(
    calibration_sites,
    hmean: float,
    mdd: float,
    posteriors_and_masks: tuple,
    carrying_capacity_params: tuple,
    priors: list | None = None,
    num_samples: int = 500,
    warmup_steps: int = 200,
    num_chains: int = 1,
    step_size: float | None = None,
    target_accept_prob: float = 0.574,
    chunk_size: int = 3,
    r_min: float | None = None,
    r_survival_tol: float = 0.005,
    r_survival_alpha: float = 1.11,
    gpu_memory_fraction: float | None = 0.8,
    plot_summary: bool = True,
    save_fig_folder: str | None = None,
    species_name: str | None = None,
    seed: int | None = None,
    seed_fraction: float = 0.25,
    s_halfwidth: float = 3.0,
    d_max_r_survival_tol: float = 1e-12,
    init_point: tuple | None = None,
    disperse_chains: bool = True,
    pre_optimize_steps: int = 50,
) -> dict:
    """Full joint posterior over ``(s, d, S_crit)`` via MALA (Metropolis-Adjusted
    Langevin Algorithm) — a hand-rolled, gradient-informed Metropolis-Hastings
    sampler, no external MCMC library required. ``S_crit`` is the critical
    per-dispersal-step survival probability below which no growth rate can
    sustain a locally-growing population (see the module-level
    growth-timescale note); ``Tg`` is derived from it via `_Scrit_to_Tg`
    wherever still needed for reference.

    Why MALA instead of a plain (symmetric) random-walk Metropolis or NUTS:
    the Langevin proposal ``theta' = theta + (eps^2/2)*grad(log pi(theta)) +
    eps*xi`` (``xi ~ N(0, I)``) is deliberately ASYMMETRIC — it's centred on
    theta shifted a little towards higher posterior density, not on theta
    itself — so it mixes far better than a symmetric random walk (which
    wanders with no sense of direction) while costing only ONE gradient
    evaluation per proposal, unlike NUTS's several leapfrog steps per
    sample. Because the proposal is asymmetric, the Hastings correction
    term (the ``log_q_bwd - log_q_fwd`` below) is required to keep the
    chain's stationary distribution exactly the target posterior —
    without it this would silently sample from the wrong distribution.

    Sampling happens directly in raw ``(s, d, S_crit)`` (NOT ``n, r`` — see the
    module-level note above `_hs_local_contrast`) with a HARD box
    constraint standing in for the same uniform priors used everywhere
    else in this module (`learn_dispersal_parameters`, `refine_from_point`,
    `_run_grid`): any proposal landing outside the ``(s, d)``/``S_crit`` box is
    rejected outright (log-prior = -inf there), no unconstrained-space
    transform needed (unlike NUTS/Pyro, which needs one internally for
    bounded priors) — MALA/Metropolis handle bounded uniform priors
    natively via rejection.

    All chains start at the SAME point: the centre of the informative
    zone — ``s = s*`` (the data-driven centre from `_s_center`), ``d`` and
    ``S_crit`` at their own box midpoints — exactly the default starting point
    `learn_dispersal_parameters` uses (raw=0 for all three axes there).
    This matters specifically because of the low-cost-gradient "collapse"
    region identified at small (n, r): starting IN the informative zone
    (not at an arbitrary/neutral point, and not by chance inside the
    collapse region) sidesteps the risk of the chain being trapped there
    from the very first step, where MALA's gradient-informed drift term
    vanishes and it degrades to a slow, purely diffusive random walk (see
    the discussion this function follows from).

    The step size ``eps`` is adapted during `warmup_steps` (simple
    Robbins-Monro-style multiplicative adaptation, shared scale across all
    three dimensions) to hit `target_accept_prob` — MALA's theoretically
    optimal acceptance rate is ~0.574 (Roberts & Rosenthal 1998), notably
    higher than NUTS's ~0.8 or a plain random walk's ~0.234 (in high
    dimension) — the algorithms aren't interchangeable at the same target.

    Memory for the full-dataset log-posterior gradient (needed once per
    proposal, not once per leapfrog sub-step) is bounded via the same
    `torch.utils.checkpoint`-based chunking as `run_nuts` used to.

    Parameters
    ----------
    num_samples, warmup_steps:
        Posterior draws kept, and adaptation steps discarded beforehand.
    num_chains:
        Independent chains (sequential, not parallel — same reasoning as
        `run_nuts`: keeps this simple and avoids any multiprocessing
        pickling issues). See `disperse_chains` below for where each
        chain starts.
    init_point:
        ``(s, d, S_crit)`` to start every chain at EXACTLY (clamped into
        the valid box if needed) — overrides `disperse_chains` entirely,
        since an explicit point is an explicit request, not something to
        randomize away. ``None`` (default): see `disperse_chains`.
    disperse_chains:
        Only relevant when `init_point` is ``None``. If ``True`` (default)
        and `num_chains` > 1, each chain draws its OWN starting point
        uniformly at random from the valid `(s, d, S_crit)` region (using
        the SAME global `seed` for reproducibility, if given) instead of
        all starting at the identical conservative box centre. This
        matters for R-hat: with identical starting points, split-R-hat can
        only catch a chain drifting during its own run — it can never
        catch a mode/valley/plateau some chains reach and others never do,
        because nothing ever sends different chains toward different
        regions in the first place. Random dispersed starts fix that blind
        spot, at the cost of losing the (weaker) guarantee that every
        chain begins in the well-supported "conservative" region — a chain
        can start further out in the wider, less-supported extended
        (learning) box. With `num_chains=1`, or `disperse_chains=False`,
        the single deterministic conservative-box-centre start is used
        (matching `refine_from_point`'s own default convention).
    pre_optimize_steps:
        Only used when `disperse_chains` is actually active (see above).
        After each chain's random start is drawn, run this many steps of
        a short, noise-free, Metropolis-free gradient ASCENT (backtracking
        line search, NOT part of the sampled chain) to slide it off flat/
        high-cost terrain onto the nearest ridge or valley floor before
        real (noisy) warmup begins — a uniform random draw over a big box
        is very likely to land far from a narrow or elongated valley, and
        letting MALA's own noisy warmup random-walk there instead would
        waste real warmup/sampling steps just travelling. This does NOT
        undo the dispersion itself — each chain still slides toward
        whichever nearby high-density region ITS OWN random draw was
        closest to, so different chains can still land in different parts
        of a long valley or different local modes; it only removes each
        start's easily-fixed local badness. Default 50; set to 0 to
        disable and use the raw random draw directly.
    step_size:
        Initial ``eps`` (scalar, applied to all three dimensions before
        adaptation). ``None`` auto-initialises it to 5% of each
        dimension's own box width (so the very first proposals are already
        roughly comparable in scale across s/d/S_crit despite their different
        box widths).
    target_accept_prob:
        MALA's step-size adaptation target (default ~0.574, see above).
    chunk_size:
        Sites per gradient-checkpointed chunk — bounds peak GPU memory.
    seed_fraction:
        Fraction of pixels seeded per site's sparse initial density mask,
        drawn ONCE per site at the start of this run and FIXED for every
        evaluation thereafter (not redrawn per step). Default 0.25.

    Returns
    -------
    dict
        ``{"samples": {"s":..., "d":..., "S_crit":..., "Tg":..., "n":..., "r":...},
        "ci_95": {"s": (lo, hi), "d": ..., "S_crit": ..., "Tg": ..., "n": ..., "r": ...},
        "accept_rate": float}`` — ``samples`` are numpy arrays, shape
        ``(num_chains, num_samples)`` if ``num_chains>1`` else
        ``(num_samples,)``; ``ci_95`` is the equal-tailed [2.5%, 97.5%]
        empirical credible interval per parameter (the same values
        printed at the end of the run). ``s``, ``d``, and ``S_crit`` (the
        critical per-dispersal-step survival probability) are the PRIMARY
        sampled variables — they are the actual raw MCMC chain values, the
        space MALA samples in. ``n``, ``r``, and ``Tg`` are SECONDARY,
        derived per-sample from ``s``/``d``/``S_crit`` (``Tg`` via
        `_Scrit_to_Tg`) and kept only for backward compatibility with
        callers that still expect "n"/"r"/"Tg" keys (e.g. `batch.py`).
    """
    from torch.utils.checkpoint import checkpoint

    if gpu_memory_fraction is not None and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction, device=0)
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    L, k, x0 = carrying_capacity_params

    if priors is None:
        C    = (2.0 * np.exp(-1.11 / mdd)) / (1.0 + np.exp(-2.0 * 1.11 / mdd))
        rmax = (1.0 / np.log(hmean)) * np.log(C)
        if r_min is None:
            r_min = _find_r_min(hmean, r_survival_alpha, mdd, tol=r_survival_tol)
            print(f"[priors] Auto r_min={r_min:.6g} (survival within "
                  f"{r_survival_tol*100:.2g}pp of its r->0 limit below this).")
        log_rmin, log_rmax = np.log10(r_min), np.log10(rmax)
        hs_bar, delta_h_bar = _hs_local_contrast(calibration_sites)
        s_center = _s_center(hs_bar, delta_h_bar)
        smin, smax = _s_range(s_center, s_halfwidth)
        nmin, nmax = _n_box_from_s_and_r(smin, smax, log_rmin, log_rmax)
        priors = [(nmin, nmax), (r_min, rmax), (-2.0, 2.0)]
        print(f"[priors] r range for this species: r_min={r_min:.6g}  "
              f"r_max={rmax:.6g}  (hmean={hmean:.4g}, mdd={mdd:.4g})")

    (nmin, nmax), (rmin, rmax), (_tgmin_unused, _tgmax_unused) = priors
    log_rmin, log_rmax = np.log10(rmin), np.log10(rmax)

    # d_max EXTENSION via a SEPARATE, much smaller r floor (see
    # `learn_dispersal_parameters` for the full derivation).
    r_min_for_dmax = _find_r_min(hmean, r_survival_alpha, mdd, tol=d_max_r_survival_tol)
    log_rmin_for_d = np.log10(r_min_for_dmax)

    # Printed in this exact order (r -> s -> n -> d), see
    # `learn_dispersal_parameters` for why.
    hs_bar, delta_h_bar = _hs_local_contrast(calibration_sites)
    s_center = _s_center(hs_bar, delta_h_bar)
    (smin, smax), (dmin, dmax_conservative) = _sd_box(nmin, nmax, log_rmin, log_rmax, s_center, s_halfwidth)
    # d_max is now recomputed using `log_rmin_for_d` (the SEPARATE,
    # extremely small r floor from `d_max_r_survival_tol`, computed above)
    # instead of the moderate `log_rmin` — extends d_max far out for
    # species whose true optimum needs near-zero dispersal mortality,
    # while `dmin`/`dmax_conservative` (computed with the moderate r
    # floor) are kept around for the default init point's box centre
    # (see below) and for reference/printing. The corner achieving d_max
    # is (log_n=n_max, log_r=log_rmin_for_d): d increases AND s decreases
    # there relative to the conservative corner (s = n_max + log_rmin_for_d
    # < n_max + log_rmin), i.e. the newly reachable region skews toward
    # the UPPER-LEFT (low s, high d) — the opposite of widening n_max,
    # which would skew the same corner toward the upper-RIGHT.
    dmax = nmax - log_rmin_for_d
    print(f"[priors] s prior for this species: hs_bar={hs_bar:.4g}  "
          f"delta_h_bar={delta_h_bar:.4g}  s*=log10(n*r)={s_center:.4g}  "
          f"s box=[{smin:.4g}, {smax:.4g}]  (s_halfwidth={s_halfwidth:.3g})")
    print(f"[priors] n range for this species (data-driven from s box + "
          f"r box): n_min={10**nmin:.4g}  n_max={10**nmax:.4g}  "
          f"(log10: [{nmin:.4g}, {nmax:.4g}])")
    print(f"[priors] d prior for this species (from n box + r box): "
          f"conservative d box=[{dmin:.4g}, {dmax_conservative:.4g}] "
          f"(r_survival_tol={r_survival_tol:.3g}, used for the default init "
          f"point's box centre)  ->  LEARNING d box=[{dmin:.4g}, {dmax:.4g}] "
          f"(d_max extended via d_max_r_survival_tol={d_max_r_survival_tol:.3g})")

    # S_crit reparametrisation of Tg — sampling happens directly in raw
    # S_crit (see the module-level growth-timescale note and
    # `_Scrit_box` — a fixed universal (0.5, 1.0) box, no species-specific
    # precompute needed), with the SAME hard-box-rejection mechanism as
    # (s, d) below (no tanh transform here — MALA samples the raw box
    # directly).
    SCRIT_MIN, SCRIT_MAX = _Scrit_box()
    print(f"[priors] S_crit (survival-threshold) reparametrisation for "
          f"this species: S_crit box=[{SCRIT_MIN:.4g}, {SCRIT_MAX:.4g}]  "
          f"(interpretation: S_crit = critical per-dispersal-step survival "
          f"probability below which NO growth rate can sustain a local "
          f"population — S_crit=0.5 is the g->1 (fastest possible growth, "
          f"Tg->0) limit, S_crit->1 is the g->0 (Tg->inf) limit)")

    print("[precompute] Building adjacency matrices and K_is ...")
    n_sites_total = len(calibration_sites)
    all_indices = list(range(n_sites_total))
    adj_mats, K_is_list, seed_masks_list = [], [], []
    for hs_map in tqdm(calibration_sites.hs_maps, desc="  Sites", leave=False):
        hs_t = torch.tensor(hs_map, dtype=torch.float32, device=device)
        adj_mats.append(adjacency_matrix_torch(hs_t))
        K_is_flat = torch.tensor(
            _logistic(hs_map.flatten().astype(np.float32), L, k, x0),
            dtype=torch.float32,
        )
        K_is_list.append(K_is_flat)
        seed_masks_list.append(seed_mask(
            K_is_flat.shape, seed_fraction=seed_fraction,
            device=K_is_flat.device, dtype=K_is_flat.dtype,
        ))
    print(f"[precompute] Done — {n_sites_total} sites.")
    breeding_masks_list = _build_breeding_masks_list(calibration_sites, n_sites_total)
    print(f"[precompute] Seeded initial densities: {seed_fraction*100:.0f}% of "
          f"pixels per site, FIXED for the whole run (not redrawn per step).")

    def _chunk_nll(n_val: torch.Tensor, r_val: torch.Tensor, Scrit_val: torch.Tensor,
                    chunk: list) -> torch.Tensor:
        # `seed_masks_list` reaches here the same way `adj_mats`/`K_is_list`
        # already do — via closure capture, not as an explicit `checkpoint()`
        # arg — since it's only ever indexed inside `cost_function`, never
        # differentiated through.
        partial = cost_function(
            mdd, posteriors_and_masks, calibration_sites,
            (n_val, r_val, Scrit_val), chunk,
            carrying_capacity_params, hmean, adj_mats, K_is_list,
            plot=False, verbose=False,
            seed_masks_list=seed_masks_list, breeding_masks_list=breeding_masks_list,
        )
        return partial * len(chunk)

    def _log_post_and_grad(theta: torch.Tensor) -> tuple:
        """theta = (s, d, Tg) raw tensor (no grad attached yet). Returns
        (log_posterior: float tensor (detached), grad: (3,) tensor
        (detached)) — log_posterior = -inf, grad = None if outside the box
        (uniform prior support), since there's nothing to differentiate.

        TWO separate checks, not one: (1) the (s, d, S_crit) box itself
        (soft prior choices), and (2) the ONE hard physical boundary,
        `r > r_max` — since the (s, d) rectangle is the smallest
        AXIS-ALIGNED box containing the (rotated, diamond-shaped) image of
        the true (log10 n, log10 r) box (see `_sd_box`), its own corners
        overshoot that diamond and can land at `r > r_max`, where
        `Ew(r)`/the dispersal kernel become genuinely ill-defined (not
        just "outside a prior" — the process can no longer be matched to
        the target MDD at all, see `_survival`'s docstring). Checking only
        `(s,d) in box` let the sampler wander into that overshooting
        corner and find a spurious second mode there — confirmed directly
        from a real run's (log n, log r) scatter showing a separate
        cluster exactly there. `n_min`/`n_max`/`r_min` are NOT enforced
        here (soft prior choices, not physical impossibilities — see the
        identical reasoning in `_scan_cost_grid_sd`'s validity check,
        which this mirrors for consistency across the whole package).
        """
        # theta[2] is raw S_crit, NOT Tg — box-checked against
        # [SCRIT_MIN, SCRIT_MAX] (with a small numerical margin at BOTH
        # ends: S_crit=0.5 is the g->1/Tg->0 limit and S_crit=1.0 is the
        # g->0/Tg->inf limit, both singular in `_Scrit_to_Tg`, so neither
        # exact endpoint may be reached) and converted to an actual Tg
        # value (via `_Scrit_to_Tg`) only right before being handed to
        # `cost_function`.
        s_val, d_val, Scrit_val = theta[0].item(), theta[1].item(), theta[2].item()
        if not (smin <= s_val <= smax and dmin <= d_val <= dmax
                and SCRIT_MIN + 1e-6 <= Scrit_val <= SCRIT_MAX - 1e-6):
            return torch.tensor(-float("inf")), None
        # ONLY hard physical boundary: r > r_max (Ew/the dispersal kernel
        # become ill-defined beyond it — see `_survival`'s docstring).
        # n_min/n_max and r_min are soft PRIOR choices, not physical
        # impossibilities, and n/r themselves can never be negative in
        # this log-parametrisation (n=10**log_n, r=10**log_r always > 0)
        # — see the identical reasoning in `_scan_cost_grid_sd`'s
        # validity check, which this mirrors for consistency across the
        # whole package.
        log_n_val, log_r_val = _sd_to_logn_logr(s_val, d_val)
        if log_r_val > log_rmax:
            return torch.tensor(-float("inf")), None
        theta_g = theta.clone().detach().requires_grad_(True)
        s, d, S_crit = theta_g[0], theta_g[1], theta_g[2]
        log_n, log_r = _sd_to_logn_logr(s, d)
        n_val, r_val = 10.0 ** log_n, 10.0 ** log_r
        nll = torch.zeros((), device=device)
        for start in range(0, n_sites_total, chunk_size):
            chunk = all_indices[start:start + chunk_size]
            nll = nll + checkpoint(_chunk_nll, n_val, r_val, S_crit, chunk, use_reentrant=False)
        log_post = -nll   # uniform prior contributes a constant (0) inside the box
        log_post.backward()
        return log_post.detach(), theta_g.grad.detach().clone()

    box_widths = torch.tensor([smax - smin, dmax - dmin, SCRIT_MAX - SCRIT_MIN], dtype=torch.float32, device=device)
    eps0 = (0.05 * box_widths) if step_size is None else torch.full((3,), float(step_size), device=device)

    # `s* ` (from `_s_center`, purely data-driven from hs_bar/delta_h_bar)
    # is NOT guaranteed to fall inside the range of s actually achievable
    # within the true physical (n, r) box — the two are derived
    # independently. If it doesn't, clamp it to the nearest achievable
    # value (with a warning) instead of starting from an impossible
    # point. A small inward margin avoids landing exactly on the
    # boundary, where the achievable d-range below would be a single
    # (zero-width) point.
    achievable_s_min, achievable_s_max = nmin + log_rmin, nmax + log_rmax
    margin = 0.01 * (achievable_s_max - achievable_s_min)

    if init_point is not None:
        # Caller supplied an explicit starting point (e.g. a converged
        # optimum from `refine_from_point`/`learn_dispersal_parameters`) —
        # use it directly instead of the data-driven sensitivity-zone
        # centre, just clamped into the valid box on each axis so a
        # slightly-out-of-box point (numerical edge case) doesn't produce
        # an immediate -inf log-posterior.
        s0_req, d0_req, Scrit0_req = init_point
        s0 = min(max(s0_req, achievable_s_min + margin), achievable_s_max - margin)
        d_lo = max(dmin, 2.0 * nmin - s0, s0 - 2.0 * log_rmax)
        d_hi = min(dmax, 2.0 * nmax - s0, s0 - 2.0 * log_rmin)
        if d_lo > d_hi:
            raise ValueError(
                f"No valid d exists at s=s0={s0:.4g} (from init_point) — "
                f"the (n, r) and (s,d) boxes don't overlap there. Check "
                f"the priors (r_min/r_max, n bounds) or init_point itself."
            )
        d0 = min(max(d0_req, d_lo), d_hi)
        Scrit0 = min(max(Scrit0_req, SCRIT_MIN), SCRIT_MAX)
        if (s0, d0, Scrit0) != (s0_req, d0_req, Scrit0_req):
            print(f"\033[91m[run_mala] Warning: init_point=({s0_req:.4g}, "
                  f"{d0_req:.4g}, {Scrit0_req:.4g}) fell outside the valid "
                  f"box — clamped to ({s0:.4g}, {d0:.4g}, {Scrit0:.4g}).\033[0m")
        print(f"\033[96m[run_mala] Starting MALA: {num_chains} chain(s), "
              f"{warmup_steps} warmup + {num_samples} samples, "
              f"all starting at the supplied init_point "
              f"s={s0:.4g}  d={d0:.4g}  S_crit={Scrit0:.4g}\033[0m")
    else:
        # `init_point=None` (default): start EXACTLY at the CONSERVATIVE
        # box's centre — same convention as `refine_from_point`'s/
        # `learn_dispersal_parameters`'s own `init_point=None`/
        # `init_params=None` defaults, for consistency across all three.
        # `d`'s centre uses the CONSERVATIVE box (dmin, dmax_conservative —
        # moderate `r_survival_tol`), NOT the extended LEARNING box (dmin,
        # dmax) that MALA still samples/explores across — see the
        # module-level note above `_n_box_from_s_and_r`/`_sd_box`: d_max is
        # pushed out far via a separate, much smaller
        # `d_max_r_survival_tol` floor so species that need it CAN reach
        # it during sampling, but the default STARTING point should stay
        # in the well-supported region, not already halfway to that
        # extreme edge.
        s0 = min(max((smin + smax) / 2.0, achievable_s_min + margin), achievable_s_max - margin)
        d_center_conservative = (dmin + dmax_conservative) / 2.0
        d_lo = max(dmin, 2.0 * nmin - s0, s0 - 2.0 * log_rmax)
        d_hi = min(dmax, 2.0 * nmax - s0, s0 - 2.0 * log_rmin)
        if d_lo > d_hi:
            raise ValueError(
                f"No valid d exists at s=s0={s0:.4g} (conservative box "
                f"centre) — the (n, r) and (s,d) boxes don't overlap "
                f"there. Check the priors (r_min/r_max, n bounds)."
            )
        d0 = min(max(d_center_conservative, d_lo), d_hi)
        Scrit0 = (SCRIT_MIN + SCRIT_MAX) / 2.0
        print(f"\033[96m[run_mala] Starting MALA: {num_chains} chain(s), "
              f"{warmup_steps} warmup + {num_samples} samples, "
              f"all starting at the conservative box centre "
              f"s={s0:.4g}  d={d0:.4g} (valid range at this s: [{d_lo:.4g}, {d_hi:.4g}])  "
              f"S_crit={Scrit0:.4g}  — the extended learning box "
              f"(d up to {dmax:.4g}) is still fully sampled during MCMC.\033[0m")

    def _random_valid_start() -> tuple:
        """Uniformly draw an `(s, d, S_crit)` starting point from the
        valid region (achievable `s` range, then the `d` range actually
        valid AT that drawn `s`, then the full `S_crit` box) — used to
        disperse chains (see `disperse_chains`'s docstring) so split-R-hat
        can catch chains stuck in different plateaus/valleys instead of
        every chain starting from the identical point.
        """
        s_r = np.random.uniform(achievable_s_min + margin, achievable_s_max - margin)
        d_lo_r = max(dmin, 2.0 * nmin - s_r, s_r - 2.0 * log_rmax)
        d_hi_r = min(dmax, 2.0 * nmax - s_r, s_r - 2.0 * log_rmin)
        d_r = np.random.uniform(d_lo_r, d_hi_r)
        Scrit_r = np.random.uniform(SCRIT_MIN, SCRIT_MAX)
        return float(s_r), float(d_r), float(Scrit_r)

    def _pre_optimize(theta: torch.Tensor, n_steps: int) -> torch.Tensor:
        """Short, noise-free, Metropolis-free gradient ASCENT on the
        log-posterior — NOT part of the actual MCMC chain, just a cheap
        way to slide a random dispersed start (which is very likely to
        land on flat, high-cost terrain far from a narrow/long valley,
        especially in a high-dimensional or elongated posterior) onto the
        nearest ridge/valley floor before real (noisy, Metropolis-
        corrected) warmup begins. This preserves the WHOLE point of
        dispersion — different random starts still end up in different
        parts of the valley/different local modes, since gradient ascent
        only removes each start's local, easily-fixed badness (being on a
        bad plateau) without erasing the global diversity between starts
        — while not wasting real warmup/sampling steps on a slow random
        walk across flat terrain the gradient could have crossed in a
        handful of steps. Uses a simple backtracking line search (halving
        the step on a rejected/invalid move, same coordinate-independent
        `eps0`-scaled step as MALA's own initial step size) so it can't
        wander outside the valid `(n, r)` region or diverge.
        """
        step = 0.5 * eps0
        logpost, grad = _log_post_and_grad(theta)
        for _ in range(n_steps):
            if not torch.isfinite(logpost) or grad is None:
                break
            improved = False
            for _ in range(6):   # backtracking: halve the step up to 6x
                theta_new = theta + step * grad
                logpost_new, grad_new = _log_post_and_grad(theta_new)
                if torch.isfinite(logpost_new) and logpost_new >= logpost:
                    theta, logpost, grad = theta_new, logpost_new, grad_new
                    improved = True
                    break
                step = step * 0.5
            if not improved:
                break
        return theta.detach()

    disperse = disperse_chains and init_point is None and num_chains > 1
    if disperse:
        print(f"\033[96m[run_mala] disperse_chains=True: each of the "
              f"{num_chains} chains draws its OWN random point in the "
              f"valid (s, d, S_crit) region, then a short "
              f"{pre_optimize_steps}-step gradient ascent (no noise, no "
              f"Metropolis step — just to reach the nearest ridge/valley "
              f"floor) before real warmup — not all starting at the "
              f"conservative box centre printed above.\033[0m")

    all_s, all_d, all_Scrit, accept_rates = [], [], [], []
    total_steps = warmup_steps + num_samples

    for chain_idx in range(num_chains):
        if disperse:
            s0_c, d0_c, Scrit0_c = _random_valid_start()
            if pre_optimize_steps > 0:
                theta_pre = torch.tensor([s0_c, d0_c, Scrit0_c], dtype=torch.float32, device=device)
                theta_pre = _pre_optimize(theta_pre, pre_optimize_steps)
                s0_c, d0_c, Scrit0_c = (float(theta_pre[0]), float(theta_pre[1]), float(theta_pre[2]))
        else:
            s0_c, d0_c, Scrit0_c = s0, d0, Scrit0
        if num_chains > 1:
            print(f"\033[96m[run_mala] Chain {chain_idx+1}/{num_chains} "
                  f"starting at s={s0_c:.4g}  d={d0_c:.4g}  "
                  f"S_crit={Scrit0_c:.4g}\033[0m")

        theta = torch.tensor([s0_c, d0_c, Scrit0_c], dtype=torch.float32, device=device)
        eps = eps0.clone()
        cur_logpost, cur_grad = _log_post_and_grad(theta)

        chain_s, chain_d, chain_Scrit = [], [], []
        n_accept = 0
        step_bar = tqdm(range(total_steps), desc="  MALA steps", leave=True)
        for it in step_bar:
            drift = 0.5 * eps ** 2 * cur_grad
            noise = eps * torch.randn(3, device=device)
            theta_prop = theta + drift + noise

            prop_logpost, prop_grad = _log_post_and_grad(theta_prop)
            if torch.isfinite(prop_logpost):
                # Hastings correction for the asymmetric Langevin proposal:
                # q(theta | theta') = N(theta; theta' + 0.5*eps^2*grad(theta'), eps^2 I)
                # q(theta' | theta) = N(theta'; theta + 0.5*eps^2*grad(theta), eps^2 I)
                fwd_mean = theta + drift
                bwd_mean = theta_prop + 0.5 * eps ** 2 * prop_grad
                log_q_fwd = -0.5 * torch.sum(((theta_prop - fwd_mean) / eps) ** 2)
                log_q_bwd = -0.5 * torch.sum(((theta - bwd_mean) / eps) ** 2)
                log_alpha = (prop_logpost - cur_logpost) + (log_q_bwd - log_q_fwd)
                accept = bool(torch.log(torch.rand((), device=device)) < log_alpha)
            else:
                accept = False   # outside the box — always rejected

            if accept:
                theta = theta_prop.detach()
                cur_logpost, cur_grad = prop_logpost, prop_grad
                n_accept += 1

            if it < warmup_steps:
                # Simple Robbins-Monro-style multiplicative step-size
                # adaptation, decaying adaptation rate — one shared scale
                # factor applied to all 3 dimensions (not per-dimension,
                # to keep this simple), targeting `target_accept_prob`.
                adapt_rate = 1.0 / (it + 1) ** 0.6
                eps = eps * float(np.exp(adapt_rate * ((1.0 if accept else 0.0) - target_accept_prob)))
            else:
                chain_s.append(theta[0].item())
                chain_d.append(theta[1].item())
                chain_Scrit.append(theta[2].item())

            if it % 20 == 0 or it == total_steps - 1:
                phase = "warmup" if it < warmup_steps else "sample"
                step_bar.set_postfix(
                    phase=phase, accept_rate=f"{n_accept/(it+1):.2f}",
                    eps=f"{eps.mean().item():.3g}",
                    s=f"{theta[0].item():.3g}", d=f"{theta[1].item():.3g}",
                    S_crit=f"{theta[2].item():.3g}",
                )

        all_s.append(np.array(chain_s)); all_d.append(np.array(chain_d)); all_Scrit.append(np.array(chain_Scrit))
        accept_rates.append(n_accept / total_steps)
        print(f"\033[92m[run_mala] Chain {chain_idx+1}/{num_chains} done — "
              f"overall accept rate={accept_rates[-1]:.3f} "
              f"(target={target_accept_prob:.3f})\033[0m")

    if num_chains > 1:
        s_samp   = np.stack(all_s, axis=0)
        d_samp   = np.stack(all_d, axis=0)
        Scrit_samp = np.stack(all_Scrit, axis=0)
    else:
        s_samp, d_samp, Scrit_samp = all_s[0], all_d[0], all_Scrit[0]
    # `all_Scrit`/`chain_Scrit` collected the RAW sampled `S_crit` values
    # directly (see `_log_post_and_grad` — theta[2] IS S_crit, the actual
    # learned/sampled variable) — the primary result from here on, no
    # `Tg` anywhere.
    log_n_samp, log_r_samp = _sd_to_logn_logr(s_samp, d_samp)
    n_samp = 10.0 ** log_n_samp
    r_samp = 10.0 ** log_r_samp

    corr_sd = float(np.corrcoef(s_samp.flatten(), d_samp.flatten())[0, 1])

    def _ci(arr: np.ndarray) -> tuple:
        # Equal-tailed 95% credible interval — the direct empirical
        # quantiles of the posterior samples, not a mean+/-std Gaussian
        # approximation (which would be misleading for a skewed or
        # bounded marginal, e.g. S_crit pushed up against its box edge).
        lo, hi = np.percentile(arr, [2.5, 97.5])
        return float(lo), float(hi)

    s_ci, d_ci, Scrit_ci = _ci(s_samp), _ci(d_samp), _ci(Scrit_samp)
    n_ci, r_ci = _ci(n_samp), _ci(r_samp)

    print(f"\033[92m[run_mala] Done. Posterior summary "
          f"(median [2.5%, 97.5%] CI over {s_samp.size} samples) — "
          f"THE result is in (s, d, S_crit):\033[0m")
    print(f"  s   = {np.median(s_samp):.4g}  [{s_ci[0]:.4g}, {s_ci[1]:.4g}]")
    print(f"  d   = {np.median(d_samp):.4g}  [{d_ci[0]:.4g}, {d_ci[1]:.4g}]")
    print(f"  S_crit = {np.median(Scrit_samp):.4g}  [{Scrit_ci[0]:.4g}, {Scrit_ci[1]:.4g}]")
    print(f"  corr(s, d) = {corr_sd:+.3f}")
    print(f"\033[90m  [for reference only — (n, r) marginals: "
          f"n={np.median(n_samp):.4g} [{n_ci[0]:.4g}, {n_ci[1]:.4g}]  "
          f"r={np.median(r_samp):.4g} [{r_ci[0]:.4g}, {r_ci[1]:.4g}]]\033[0m")

    # Convergence diagnostics — reuse the same split-R-hat / effective
    # sample size machinery Pyro provides (`pyro.ops.stats`), without
    # needing Pyro's samplers themselves — only this small ops module.
    try:
        import pyro.ops.stats as pyro_stats

        def _chain_tensor(arr: np.ndarray) -> torch.Tensor:
            t = torch.as_tensor(arr, dtype=torch.float32)
            return t if t.dim() == 2 else t.unsqueeze(0)

        print(f"\033[93m[run_mala] Convergence diagnostics "
              f"({num_chains} chain(s) x {num_samples} samples):\033[0m")
        for name, arr in [("s", s_samp), ("d", d_samp), ("S_crit", Scrit_samp)]:
            t = _chain_tensor(arr)
            rhat = float(pyro_stats.split_gelman_rubin(t).item())
            n_eff = float(pyro_stats.effective_sample_size(t).item())
            total_draws = t.shape[0] * t.shape[1]
            flag = "OK" if rhat < 1.01 else ("borderline" if rhat < 1.1 else "BAD — do not trust this run")
            print(f"    {name:>2s}: R-hat={rhat:.4f} [{flag}]   "
                  f"n_eff={n_eff:.0f}/{total_draws} ({n_eff/total_draws*100:.0f}%)")
    except ImportError:
        print("\033[93m[run_mala] (pyro not installed — skipping R-hat/n_eff; "
              "`pip install pyro-ppl` for these diagnostics, not needed for "
              "sampling itself.)\033[0m")

    if plot_summary:
        prefix = f"{species_name}_" if species_name else ""

        def _save_show(filename: str) -> None:
            if save_fig_folder:
                plt.savefig(os.path.join(save_fig_folder, filename),
                             dpi=150, bbox_inches="tight")
            plt.show()

        s_flat, d_flat, Scrit_flat = s_samp.flatten(), d_samp.flatten(), Scrit_samp.flatten()
        n_flat, r_flat = n_samp.flatten(), r_samp.flatten()

        fig, axes = plt.subplots(2, 4, figsize=(19, 8))
        axes[0, 0].plot(s_flat, alpha=0.7, linewidth=0.5); axes[0, 0].set_title("s trace")
        axes[0, 1].plot(d_flat, alpha=0.7, linewidth=0.5); axes[0, 1].set_title("d trace")
        axes[0, 2].plot(Scrit_flat, alpha=0.7, linewidth=0.5); axes[0, 2].set_title("S_crit trace")
        axes[0, 3].scatter(s_flat, d_flat, s=4, alpha=0.3, color="darkred")
        axes[0, 3].set_xlabel("s"); axes[0, 3].set_ylabel("d")
        axes[0, 3].set_title(f"(s,d) joint  corr={corr_sd:+.2f}")
        axes[1, 0].hist(s_flat, bins=40, color="steelblue"); axes[1, 0].set_title("s posterior")
        axes[1, 1].hist(d_flat, bins=40, color="darkred"); axes[1, 1].set_title("d posterior")
        axes[1, 2].hist(Scrit_flat, bins=40, color="darkorange"); axes[1, 2].set_title("S_crit posterior")
        axes[1, 3].axis("off")
        for ax in axes.flat:
            ax.grid(linestyle="--", color="grey", linewidth=0.2, alpha=0.5)
        fig.suptitle(f"MALA joint posterior in (s, d, S_crit) — THE result — {species_name or ''}")
        fig.tight_layout()
        _save_show(f"{prefix}mala_sd_posterior.png")

        fig2, axes2 = plt.subplots(1, 2, figsize=(11, 4.5))
        axes2[0].scatter(np.log10(n_flat), np.log10(r_flat), s=4, alpha=0.3, color="teal")
        axes2[0].set_xlabel("log10(n)"); axes2[0].set_ylabel("log10(r)")
        axes2[0].set_title("Same posterior, converted to (log n, log r)")
        axes2[1].hist(n_flat, bins=40, color="teal", alpha=0.7)
        axes2[1].set_title("n marginal"); axes2[1].set_xlabel("n")
        for ax in axes2.flat:
            ax.grid(linestyle="--", color="grey", linewidth=0.2, alpha=0.5)
        fig2.suptitle(f"(n, r) — for reference/interpretation only — {species_name or ''}")
        fig2.tight_layout()
        _save_show(f"{prefix}mala_nr_reference.png")

    # `S_crit` is the actual sampled/reported variable (see the module-level
    # growth-timescale note) — no `Tg` key anywhere: `Tg` is invalid/NaN
    # for `S_crit <= 0.5`, a reachable region now that the box goes down
    # to 1/3, so it can no longer be reported at all. Callers expecting a
    # "Tg" key (e.g. `batch.py`'s `samples[k] for k in ("n", "r", "Tg")`)
    # must be updated to use "S_crit" instead.
    return {
        "samples": {
            "s": s_samp, "d": d_samp, "S_crit": Scrit_samp,
            "n": n_samp, "r": r_samp,
        },
        # Equal-tailed 95% credible intervals, (low, high) per parameter —
        # the same values printed above, for programmatic use without
        # re-deriving them from the raw samples.
        "ci_95": {"s": s_ci, "d": d_ci, "S_crit": Scrit_ci, "n": n_ci, "r": r_ci},
        "accept_rate": float(np.mean(accept_rates)),
    }


# ---------------------------------------------------------------------------
# Post-hoc analysis of a saved posterior-sample .npz (autocorrelation,
# effective thinning, 3-D KDE) — works on the output of `run_mala` (or
# any dict with the same "s"/"d"/"Tg" keys), no live model/data needed.
# ---------------------------------------------------------------------------

def analyze_posterior_samples(
    npz_path: str,
    acf_max_lag: int = 200,
    acf_threshold: float = 0.05,
    kde_grid_size: int = 30,
    r_max_filter: float | None = None,
    thin_interval: int | None = None,
    plot: bool = True,
    save_fig_folder: str | None = None,
    species_name: str | None = None,
) -> dict:
    """Load a ``*_samples.npz`` file (as saved by `run_mala`/`BatchLearner`)
    and analyse the (s, d, S_crit) chain: autocorrelation, an
    automatically chosen thinning interval, and a 3-D kernel density
    estimate of the full posterior.

    Sampling space is always (s, d, S_crit) — same convention as
    everywhere else in this module (see the module-level note above
    `_hs_local_contrast` and `_Scrit_box`'s docstring) — NOT (n, r, Tg);
    `n`/`r`/`Tg` are loaded too (for reference — `Tg` derived from
    `S_crit` via `_Scrit_to_Tg`) but not analysed here. Analysing `S_crit`
    rather than `Tg` isn't just cosmetic: MALA's random walk actually
    happens in raw `S_crit` space (see `run_mala`), so the autocorrelation
    of `S_crit` is what genuinely reflects the chain's own mixing —
    `Tg`'s autocorrelation would be a nonlinearly-distorted view of the
    same underlying walk (through `_Scrit_to_Tg`'s log/rational mapping),
    not the quantity MALA's proposal/acceptance mechanics actually acted on.

    Parameters
    ----------
    npz_path:
        Path to the ``.npz`` file (e.g.
        ``output/Castor_fiber_samples.npz``).
    acf_max_lag:
        Largest lag (in steps) to compute the autocorrelation function
        out to. Should comfortably exceed the true decorrelation time —
        if the reported `decorrelation_lag` below lands at exactly
        `acf_max_lag`, the chain is still correlated past the window
        checked and this should be raised.
    acf_threshold:
        The chain is considered "decorrelated" at the first lag whose
        |autocorrelation| drops below this and stays there. Default 0.05
        (a common rule of thumb — not a hard theoretical cutoff).
    kde_grid_size:
        Grid resolution per axis for the 3-D KDE volume (cost is cubic in
        this — 30 gives a 27 000-point grid, already fairly slow to
        render interactively at much higher values).
    r_max_filter:
        DIAGNOSTIC/EXPLORATORY ONLY — NOT statistically valid. If given,
        drops every sample with ``r > r_max_filter`` (e.g. to strip out
        an old run's leaked-outside-the-box cluster — see the
        `run_mala` fix for the box-overshoot bug this was tracking down)
        before doing anything else. Deleting samples out of a Markov
        chain like this breaks its temporal structure (like removing
        random frames from a video) — the autocorrelation/decorrelation
        results computed afterwards are no longer rigorous, only
        illustrative of "what would this look like without that
        cluster". Re-running `run_mala` with the box-overshoot fix is
        the correct way to get a clean chain — this is just for a quick
        look at an already-expensive 5000-sample run without re-doing it.
        Default ``None`` (no filtering).
    thin_interval:
        Override the automatically-detected `decorrelation_lag` — keep
        every `thin_interval`-th draw instead of letting the joint ACF
        pick the interval. Useful to try a specific value directly (e.g.
        to sanity-check the auto-detected one, or to force a coarser/finer
        thinning for a downstream use with different requirements).
        Default ``None`` (use the auto-detected decorrelation lag).
    plot:
        If True, show the ACF curves and the 3-D KDE volume.

    Returns
    -------
    dict
        ``{"acf": {"s":..., "d":..., "S_crit":..., "joint":...},
        "decorrelation_lag": int,
        "thinned_samples": {"s":..., "d":..., "S_crit":..., "Tg":...,
        "n":..., "r":...},
        "n_thinned": int}`` — `acf` arrays are indexed by lag
        (``acf[k]`` = autocorrelation at lag ``k``, ``acf[0]==1`` always);
        `joint` is the 3-D (vector-valued) autocorrelation, treating each
        (s, d, S_crit) draw as a single point in 3-D rather than analysing
        each coordinate separately (the `decorrelation_lag` used for
        thinning is derived from this joint curve, not the per-dimension
        ones, so a single thinning interval serves all three coordinates
        at once). ``thinned_samples`` keeps every `decorrelation_lag`-th
        draw, starting at index 0 (``Tg`` included as a derived/secondary
        array, computed from the thinned ``S_crit``, not analysed itself).
    """
    data = np.load(npz_path)
    s_raw, d_raw, Scrit_raw = data["s"], data["d"], data["S_crit"]
    n_raw, r_raw = data["n"], data["r"]

    # Multiple chains (shape (num_chains, num_samples)) are analysed
    # PER CHAIN and averaged — concatenating chains directly would create
    # a spurious jump at each chain boundary and corrupt the ACF.
    def _as_chain_list(arr) -> list:
        if isinstance(arr, list):
            return arr   # already a per-chain list (e.g. after r_max_filter, ragged sizes)
        return list(arr) if arr.ndim == 2 else [arr]

    n_chains_list = _as_chain_list(n_raw)
    r_chains_list = _as_chain_list(r_raw)

    if r_max_filter is not None:
        print(f"\033[91m[analyze_posterior_samples] r_max_filter={r_max_filter:.4g} "
              f"active — dropping every sample with r > this value. NOT "
              f"statistically valid (breaks the chain's temporal order) — "
              f"exploratory only. See `run_mala`'s box-overshoot fix for "
              f"the correct way to avoid this cluster in a fresh run.\033[0m")
        s_f, d_f, Scrit_f, n_f, r_f = [], [], [], [], []
        n_before = n_after = 0
        for sc, dc, Sc, nc, rc in zip(
            _as_chain_list(s_raw), _as_chain_list(d_raw), _as_chain_list(Scrit_raw),
            n_chains_list, r_chains_list,
        ):
            keep = rc <= r_max_filter
            n_before += rc.size
            n_after  += int(keep.sum())
            s_f.append(sc[keep]); d_f.append(dc[keep]); Scrit_f.append(Sc[keep])
            n_f.append(nc[keep]); r_f.append(rc[keep])
        print(f"[analyze_posterior_samples] Kept {n_after}/{n_before} samples "
              f"({n_after/n_before*100:.1f}%).")
        s_raw, d_raw, Scrit_raw = s_f, d_f, Scrit_f   # already lists of (possibly ragged) chains
        n_chains_list, r_chains_list = n_f, r_f

    s_chains  = _as_chain_list(s_raw)
    d_chains  = _as_chain_list(d_raw)
    Scrit_chains = _as_chain_list(Scrit_raw)
    n_chains_data = len(s_chains)
    sizes = [c.size for c in s_chains]
    print(f"[analyze_posterior_samples] Loaded {npz_path}: "
          f"{n_chains_data} chain(s), "
          f"{sizes[0] if len(set(sizes)) == 1 else sizes} samples "
          f"{'each' if len(set(sizes)) == 1 else 'per chain'}.")

    def _acf_1d(x: np.ndarray, max_lag: int) -> np.ndarray:
        x = x - x.mean()
        var = np.dot(x, x)
        max_lag = min(max_lag, len(x) - 1)
        out = np.empty(max_lag + 1)
        out[0] = 1.0
        for k in range(1, max_lag + 1):
            out[k] = np.dot(x[:-k], x[k:]) / var if var > 0 else 0.0
        return out

    def _acf_3d(X: np.ndarray, max_lag: int) -> np.ndarray:
        # X: (n_samples, 3) — the chain treated as a single 3-D vector
        # walk; autocorrelation via the normalised inner product between
        # a draw and the draw `k` steps later, exactly the 1-D formula's
        # natural vector generalisation (dot product instead of product).
        Xc = X - X.mean(axis=0, keepdims=True)
        total_var = np.sum(Xc ** 2)
        max_lag = min(max_lag, X.shape[0] - 1)
        out = np.empty(max_lag + 1)
        out[0] = 1.0
        for k in range(1, max_lag + 1):
            out[k] = np.sum(Xc[:-k] * Xc[k:]) / total_var if total_var > 0 else 0.0
        return out

    # Use the SMALLEST chain's size (not just the first) — chains can be
    # ragged after `r_max_filter` drops a different number of samples
    # from each; `_acf_1d`/`_acf_3d` need one shared `max_lag` so every
    # chain's ACF array comes out the same length and can be averaged.
    max_lag = min(acf_max_lag, min(c.size for c in s_chains) - 1)
    acf_s     = np.mean([_acf_1d(c, max_lag) for c in s_chains],     axis=0)
    acf_d     = np.mean([_acf_1d(c, max_lag) for c in d_chains],     axis=0)
    acf_Scrit = np.mean([_acf_1d(c, max_lag) for c in Scrit_chains], axis=0)
    acf_joint = np.mean(
        [_acf_3d(np.stack([sc, dc, Sc], axis=1), max_lag)
         for sc, dc, Sc in zip(s_chains, d_chains, Scrit_chains)],
        axis=0,
    )

    if thin_interval is not None:
        decorrelation_lag = int(thin_interval)
        print(f"[analyze_posterior_samples] thin_interval={decorrelation_lag} "
              f"given explicitly — skipping auto-detection from the ACF.")
    else:
        # First lag (>0) at which the JOINT autocorrelation drops below the
        # threshold and never exceeds it again for the rest of the window
        # checked (avoids picking an early lag where the curve dips below
        # the threshold only transiently before rising back up).
        below = np.abs(acf_joint) < acf_threshold
        decorrelation_lag = max_lag
        for k in range(1, len(below)):
            if below[k:].all():
                decorrelation_lag = k
                break
        else:
            print(f"\033[91m[analyze_posterior_samples] Warning: joint ACF "
                  f"never drops below {acf_threshold} within "
                  f"acf_max_lag={acf_max_lag} — the chain may still be "
                  f"correlated past this window. Re-run with a larger "
                  f"acf_max_lag before trusting the thinning below, or "
                  f"pass an explicit thin_interval to bypass this "
                  f"detection.\033[0m")
        print(f"[analyze_posterior_samples] Decorrelation lag (joint s,d,S_crit "
              f"ACF < {acf_threshold}): {decorrelation_lag} steps.")

    # Thin each chain independently by this interval, then concatenate.
    thin_s, thin_d, thin_Scrit, thin_n, thin_r = [], [], [], [], []
    for sc, dc, Sc, nc, rc in zip(
        s_chains, d_chains, Scrit_chains, n_chains_list, r_chains_list
    ):
        sl = slice(None, None, max(1, decorrelation_lag))
        thin_s.append(sc[sl]); thin_d.append(dc[sl]); thin_Scrit.append(Sc[sl])
        thin_n.append(nc[sl]); thin_r.append(rc[sl])
    thin_s     = np.concatenate(thin_s)
    thin_d     = np.concatenate(thin_d)
    thin_Scrit = np.concatenate(thin_Scrit)
    thin_n     = np.concatenate(thin_n)
    thin_r     = np.concatenate(thin_r)
    print(f"[analyze_posterior_samples] Thinned {sum(c.size for c in s_chains)} "
          f"-> {thin_s.size} samples (keeping every {decorrelation_lag}-th draw).")

    if plot:
        prefix = f"{species_name}_" if species_name else ""

        def _save_show(filename: str) -> None:
            if save_fig_folder:
                plt.savefig(os.path.join(save_fig_folder, filename),
                             dpi=150, bbox_inches="tight")
            plt.show()

        lags = np.arange(max_lag + 1)
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(lags, acf_joint, color="black", linewidth=2, label="joint (s,d,S_crit)")
        ax.plot(lags, acf_s,     color="steelblue", alpha=0.7, label="s")
        ax.plot(lags, acf_d,     color="darkred",   alpha=0.7, label="d")
        ax.plot(lags, acf_Scrit, color="darkorange", alpha=0.7, label="S_crit")
        ax.axhline(acf_threshold, color="grey", linestyle="--", linewidth=1)
        ax.axhline(-acf_threshold, color="grey", linestyle="--", linewidth=1)
        ax.axvline(decorrelation_lag, color="green", linestyle=":", linewidth=1.5,
                   label=f"decorrelation lag = {decorrelation_lag}")
        ax.set_xlabel("Lag (steps from the considered point)")
        ax.set_ylabel("Autocorrelation")
        ax.set_title(f"Autocorrelation of the (s,d,S_crit) walk — {species_name or ''}")
        ax.legend()
        ax.grid(linestyle="--", color="grey", linewidth=0.2, alpha=0.5)
        fig.tight_layout()
        _save_show(f"{prefix}posterior_acf.png")

        # 3-D KDE volume of the FULL posterior (not the thinned chain —
        # thinning is for downstream use where independence matters, e.g.
        # summary statistics assuming i.i.d. draws; the density estimate
        # itself is more accurate using all available samples). All
        # samples in grey (alpha=0.5); the thinned/resampled subset
        # overlaid in solid black, so it's visible which points survive
        # decorrelation-based thinning.
        s_all     = np.concatenate(s_chains)
        d_all     = np.concatenate(d_chains)
        Scrit_all = np.concatenate(Scrit_chains)

        def _kde_volume_figure(
            x_all, y_all, z_all, x_thin, y_thin, z_thin,
            axis_titles: tuple, title: str, filename: str,
        ) -> None:
            xyz = np.vstack([x_all, y_all, z_all]).T
            sigma = np.std(xyz, axis=0) + 1e-9
            bw = len(xyz) ** (-1.0 / (3 + 4))   # Scott/Silverman-type rule of thumb
            kde = KernelDensity(bandwidth=bw, kernel="gaussian")
            kde.fit(xyz / sigma)

            xg = np.linspace(x_all.min(), x_all.max(), kde_grid_size)
            yg = np.linspace(y_all.min(), y_all.max(), kde_grid_size)
            zg = np.linspace(z_all.min(), z_all.max(), kde_grid_size)
            X, Y, Z = np.meshgrid(xg, yg, zg)
            gp = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
            density = np.exp(kde.score_samples(gp / sigma))

            vol = go.Volume(
                x=gp[:, 0], y=gp[:, 1], z=gp[:, 2], value=density,
                isomin=np.percentile(density, 30), isomax=density.max(),
                opacity=0.15, surface_count=20, colorscale="Viridis",
                caps=dict(x_show=False, y_show=False, z_show=False),
            )
            pts_all = go.Scatter3d(
                x=x_all, y=y_all, z=z_all, mode="markers",
                marker=dict(size=2, color="grey", opacity=0.5),
                name="all samples",
            )
            pts_thin = go.Scatter3d(
                x=x_thin, y=y_thin, z=z_thin, mode="markers",
                marker=dict(size=3, color="black", opacity=1.0),
                name="resampled (decorrelated)",
            )
            fig3d = go.Figure(data=[vol, pts_all, pts_thin])
            fig3d.update_layout(
                title=f"{title} (bandwidth={bw:.3f}) — {species_name or ''}",
                scene=dict(
                    xaxis_title=axis_titles[0], yaxis_title=axis_titles[1],
                    zaxis_title=axis_titles[2],
                    # Equal-sized cube regardless of each axis's own data
                    # range — without this, plotly stretches each axis
                    # independently to fill the scene, which visually
                    # distorts distances/correlations between axes.
                    aspectmode="cube",
                ),
                width=900, height=700,
            )
            if save_fig_folder:
                fig3d.write_html(os.path.join(save_fig_folder, filename))
            fig3d.show()

        _kde_volume_figure(
            s_all, d_all, Scrit_all, thin_s, thin_d, thin_Scrit,
            axis_titles=("s", "d", "S_crit"),
            title="3-D KDE of the full posterior in (s, d, S_crit) — THE result",
            filename=f"{prefix}posterior_kde_3d_sd.html",
        )

        # Reference-only view: the SAME samples converted to (log n,
        # log r, S_crit) — not a second independent result, just the
        # original-units interpretation (see the (n,r) reference plots
        # used elsewhere in this module for the same reasoning). Tg is
        # NOT used here (see the module-level note in this function's
        # docstring on why S_crit, not Tg, is analysed).
        _kde_volume_figure(
            np.log10(np.concatenate(n_chains_list)),
            np.log10(np.concatenate(r_chains_list)),
            Scrit_all,
            np.log10(thin_n), np.log10(thin_r), thin_Scrit,
            axis_titles=("log10(n)", "log10(r)", "S_crit"),
            title="Same posterior in (log n, log r, S_crit) — reference only",
            filename=f"{prefix}posterior_kde_3d_nr.html",
        )

    return {
        "acf": {"s": acf_s, "d": acf_d, "S_crit": acf_Scrit, "joint": acf_joint},
        "decorrelation_lag": int(decorrelation_lag),
        "thinned_samples": {
            "s": thin_s, "d": thin_d, "S_crit": thin_Scrit,
            "n": thin_n, "r": thin_r,
        },
        "n_thinned": int(thin_s.size),
    }


def _plot_speed_3d_plotly(
    s_arr: np.ndarray,
    d_arr: np.ndarray,
    Scrit_arr: np.ndarray,
    speed: np.ndarray,
    sample_density: np.ndarray,
    hpd_mask: np.ndarray,
    idx_low: int,
    idx_high: int,
    s_map: float,
    d_map: float,
    Scrit_map: float,
    save_fig_folder: str | None = None,
    species_name: str | None = None,
) -> tuple:
    """Two interactive 3-D Plotly figures over the raw `(s, d, S_crit)`
    posterior samples from `estimate_low_map_high_from_posterior`, meant to
    be viewed side by side to understand WHERE in parameter space the
    colonisation-speed extremes/bump come from (see the discussion this
    follows from: a flat marginal speed histogram can still hide a real
    bump once the joint density is accounted for).

    Both figures use the SAME plain point-scatter style — every raw
    posterior sample plotted at its own `(s, d, S_crit)`, with BOTH marker
    size and color mapped to a per-sample scalar. No surface/isosurface is
    fit or rendered for either — deliberately, so the two figures stay
    directly comparable point-for-point.

    Figure 1 — colored/sized by the analytic speed proxy — shows WHICH
    regions of parameter space produce which speeds. LOW/HIGH/MAP points
    (same picks as the caller's histogram) are marked separately.

    Figure 2 — colored/sized by `sample_density` (the joint `(s, d,
    S_crit)` KDE density evaluated AT each raw sample — the exact same
    values `estimate_low_map_high_from_posterior` already computed to pick
    its HPD region, reused here rather than refit or evaluated on a
    separate grid). Samples inside the caller's HPD region are drawn with
    full opacity; samples outside it are faded, so the region boundary is
    visible directly on the point cloud. Comparing the two figures shows
    whether high-speed regions coincide with high-density (jointly
    plausible) regions or sit off in the low-density tails.

    Returns
    -------
    (fig_scatter, fig_density) : tuple of `plotly.graph_objects.Figure`
        Also written to disk as standalone HTML (if `save_fig_folder` is
        given) or shown inline (if not) — same convention as
        `_kde_3d_plotly` elsewhere in this module.
    """
    prefix = f"{species_name}_" if species_name else ""

    # --- Figure 1: every sample, size + color both mapped to speed -------
    # Faded outside the `ci_mass` HPD region (same `hpd_mask` the caller
    # used to pick LOW/HIGH), split into two traces for the same reason as
    # Figure 2 below: Plotly's `marker.opacity` is per-trace, not per-point.
    speed_range = speed.max() - speed.min() + 1e-12
    sizes = 3.0 + 12.0 * (speed - speed.min()) / speed_range

    scatter_in = go.Scatter3d(
        x=s_arr[hpd_mask], y=d_arr[hpd_mask], z=Scrit_arr[hpd_mask],
        mode="markers",
        marker=dict(
            size=sizes[hpd_mask],
            color=speed[hpd_mask],
            colorscale="Viridis",
            cmin=float(speed.min()), cmax=float(speed.max()),
            colorbar=dict(title="Speed proxy c", x=1.02),
            opacity=0.85,
            line=dict(width=0),
        ),
        name="Inside the HPD region",
    )
    scatter_out = go.Scatter3d(
        x=s_arr[~hpd_mask], y=d_arr[~hpd_mask], z=Scrit_arr[~hpd_mask],
        mode="markers",
        marker=dict(
            size=sizes[~hpd_mask],
            color=speed[~hpd_mask],
            colorscale="Viridis",
            cmin=float(speed.min()), cmax=float(speed.max()),
            showscale=False,
            opacity=0.06,
            line=dict(width=0),
        ),
        name="Outside the HPD region",
    )
    low_pt = go.Scatter3d(
        x=[s_arr[idx_low]], y=[d_arr[idx_low]], z=[Scrit_arr[idx_low]],
        mode="markers+text",
        marker=dict(size=9, color="blue", symbol="diamond",
                    line=dict(width=1, color="black")),
        text=["LOW"], textposition="top center", name="LOW (min speed)",
    )
    high_pt = go.Scatter3d(
        x=[s_arr[idx_high]], y=[d_arr[idx_high]], z=[Scrit_arr[idx_high]],
        mode="markers+text",
        marker=dict(size=9, color="red", symbol="diamond",
                    line=dict(width=1, color="black")),
        text=["HIGH"], textposition="top center", name="HIGH (max speed)",
    )
    map_pt = go.Scatter3d(
        x=[s_map], y=[d_map], z=[Scrit_map],
        mode="markers+text",
        marker=dict(size=9, color="cyan", symbol="x",
                    line=dict(width=1, color="black")),
        text=["MAP"], textposition="top center", name="MAP",
    )
    fig_scatter = go.Figure(data=[scatter_out, scatter_in, low_pt, high_pt, map_pt])
    fig_scatter.update_layout(
        title=f"Colonisation-speed proxy over the (s, d, S_crit) posterior "
              f"— {species_name or ''}",
        scene=dict(xaxis_title="s", yaxis_title="d", zaxis_title="S_crit",
                    aspectmode="cube"),
        legend=dict(title="Legend", x=1.02, y=0.5),
        margin=dict(r=140),
    )

    # --- Figure 2: same raw samples, size + color both mapped to their own
    # KDE density value (`sample_density`, evaluated AT each sample by the
    # caller) — plain point scatter, exactly like Figure 1, no surface fit.
    dens_range = sample_density.max() - sample_density.min() + 1e-12
    dens_sizes = 3.0 + 12.0 * (sample_density - sample_density.min()) / dens_range
    # Fade samples OUTSIDE the HPD region so its boundary reads directly off
    # the point cloud, without hiding them entirely. Plotly's
    # `marker.opacity` only accepts a scalar per-trace, not a per-point
    # array, so this is done by splitting into two traces (in-HPD full
    # opacity, outside-HPD faded) rather than one trace with a fade array.
    density_scatter_in = go.Scatter3d(
        x=s_arr[hpd_mask], y=d_arr[hpd_mask], z=Scrit_arr[hpd_mask],
        mode="markers",
        marker=dict(
            size=dens_sizes[hpd_mask],
            color=sample_density[hpd_mask],
            colorscale="Plasma",
            cmin=float(sample_density.min()), cmax=float(sample_density.max()),
            colorbar=dict(title="Posterior density", x=1.02),
            opacity=0.9,
            line=dict(width=0),
        ),
        name="Inside the HPD region",
    )
    density_scatter_out = go.Scatter3d(
        x=s_arr[~hpd_mask], y=d_arr[~hpd_mask], z=Scrit_arr[~hpd_mask],
        mode="markers",
        marker=dict(
            size=dens_sizes[~hpd_mask],
            color=sample_density[~hpd_mask],
            colorscale="Plasma",
            cmin=float(sample_density.min()), cmax=float(sample_density.max()),
            showscale=False,
            opacity=0.06,
            line=dict(width=0),
        ),
        name="Outside the HPD region",
    )
    map_pt2 = go.Scatter3d(
        x=[s_map], y=[d_map], z=[Scrit_map],
        mode="markers+text",
        marker=dict(size=9, color="cyan", symbol="x",
                    line=dict(width=1, color="black")),
        text=["MAP"], textposition="top center", name="MAP",
    )
    fig_density = go.Figure(data=[density_scatter_out, density_scatter_in, map_pt2])
    fig_density.update_layout(
        title=f"3-D KDE density at each posterior sample over (s, d, S_crit) "
              f"— {species_name or ''}",
        scene=dict(xaxis_title="s", yaxis_title="d", zaxis_title="S_crit",
                    aspectmode="cube"),
        legend=dict(title="Legend", x=1.02, y=0.5),
        margin=dict(r=140),
    )

    if save_fig_folder:
        fig_scatter.write_html(os.path.join(
            save_fig_folder, f"{prefix}low_map_high_speed_3d_scatter.html"))
        fig_density.write_html(os.path.join(
            save_fig_folder, f"{prefix}low_map_high_density_3d_scatter.html"))
    else:
        fig_scatter.show()
        fig_density.show()

    return fig_scatter, fig_density


# ---------------------------------------------------------------------------
# Colonisation-speed ranking of decorrelated posterior samples: low / MAP /
# high bounding parameter sets (see `method="low_MPA_high"` in `batch.py`).
# ---------------------------------------------------------------------------

def estimate_low_map_high_from_posterior(
    npz_path: str,
    hmean: float,
    mdd: float,
    ci_mass: float = 0.68,
    density_frac_of_map: float | None = None,
    map_grid_size: int = 40,
    save_fig_folder: str | None = None,
    species_name: str | None = None,
    plot_3d: bool = True,
) -> dict:
    """Rank ALL raw posterior samples (from a `run_mala` `.npz`) by an
    ANALYTIC colonisation-speed proxy — no simulation at all — and return
    the LOW/MAP/HIGH bounding `(n, r, Tg)` parameter sets for downstream
    (expensive, full-landscape) simulation.

    Replaces the earlier simulation-based ranking entirely (see the
    surrounding discussion: repeatedly running the actual dispersal
    simulation to rank ~100 samples ran into a cascade of practical
    problems — dense-matrix memory limits at large windows, `torch.compile`
    warm-up stalls, MDD-driven window sizing — none of which apply here,
    since this only ever evaluates a closed-form expression per sample).

    THE MATH (see the worked derivation/course from this conversation for
    the full step-by-step reasoning):

    Colonisation is a reaction (logistic growth) + dispersal (diffusion)
    process, so its front — starting from a small, localised colonised
    patch — settles into a travelling wave whose asymptotic speed is given
    by the classic Fisher-KPP minimum-speed selection principle:

        c = 2 * sqrt(rho * D)

    where `rho` is the local growth rate and `D` is the diffusion
    coefficient of one generation's dispersal. Mapping these onto this
    model's own quantities:

    - `rho` <- from `Tg`: the discrete growth step is
      `Un <- Un + (1 - a)*Un*(1 - Un/K)` with `a = 0.05**(1/Tg)`, i.e. a
      discretised logistic step with rate `rho = 1 - a` — this is exactly
      the same `g = (1 - S_crit)/S_crit` growth coefficient already used
      elsewhere in this module (`_Scrit_to_Tg`'s `g`), just reached via
      `Tg` instead of `S_crit` directly.
    - `D` <- from `r`, via `Ew*(r)`, NOT the raw `Ew(r) = C / (hmean**r - C)`
      directly. `Ew(r)` is only the geometric-distribution mean number of
      hops from the CONTINUATION probability `p = Ew/(1+Ew)` alone — it
      ignores that each hop also carries a per-step mortality risk
      `1 - hmean**r` (see `_survival`'s docstring: at each round, either
      stop safely (prob `1-p`) or continue, and if continuing, survive
      that hop with probability `hmean**r` or die with probability
      `1 - hmean**r`). Dying ends the random walk EARLY, so the expected
      number of hops actually completed alive — the quantity that
      genuinely drives spatial DISPLACEMENT variance, since a dead
      disperser contributes no further movement — is smaller than the
      raw `Ew(r)`. Solving `Ew(r) = Ew*/(hmean**r - Ew*(1 - hmean**r))`
      for `Ew*` gives:

          Ew*(r) = Ew(r) * hmean**r / (1 + Ew(r)*(1 - hmean**r))
                 = Ew(r) * hmean**r * S(r)

      where `S(r) = 1/(1 + Ew(r)*(1 - hmean**r))` is exactly `_survival`'s
      own output — so `Ew*` is computed by reusing `_survival` directly,
      not a new formula. Each surviving hop still has natural per-hop
      variance ~= 1 pixel^2, so the variance accumulated per growth
      iteration is `Ew*(r) * 1`, and `D = Var(per unit time)/2` gives
      `D = Ew*(r) / 2`.

    Substituting:

        c(r, Tg) = 2 * sqrt(rho * Ew*(r) / 2) = sqrt(2 * rho * Ew*(r))

    `n` deliberately does not appear — it only reshapes kernel
    SELECTIVITY, not overall reach, and at the huge `n` values typically
    seen in this model's posteriors the kernel is already saturated into
    an extreme-selectivity regime regardless of the exact value (see the
    earlier discussion), so it contributes negligible extra ranking
    information here.

    This is a RANKING proxy, not a literal speed/area prediction: it
    assumes continuous, homogeneous space and the asymptotic (large-`t`)
    front regime, neither of which strictly holds on a small, finite,
    heterogeneous habitat window over a short `n_iter`. Use it to pick
    LOW/MAP/HIGH candidates cheaply; validate with the real simulation
    afterwards if you need actual predicted extents.

    HOW LOW/HIGH ARE ACTUALLY SELECTED (envelope over the 3-D HPD region,
    NOT a percentile of the speed distribution itself — see the
    discussion this implements for why the two are different, and why
    this one is the one to use): first restrict to the `ci_mass` (default
    68%) Highest-Posterior-Density region of the JOINT `(s, d, S_crit)`
    posterior (same empirical recipe as the 68%/95% HPD contours drawn
    elsewhere in this module — evaluate the fitted density AT the samples
    themselves, keep the top `ci_mass` fraction by density), THEN take
    the min/max of the speed proxy WITHIN that retained, statistically
    credible subset — i.e. "the fastest/slowest colonisation speed still
    consistent with the `ci_mass` most probable parameter combinations,"
    not "the middle `ci_mass` of predicted speeds." This is deliberately
    a wider, more conservative bracket than a plain percentile-on-speed
    would give (it's an extremum over a region, not a percentile of a
    1-D distribution), by design — samples entirely outside the credible
    region are excluded up front, but among the ones that remain, the
    true extremes are kept rather than trimmed further.

    Parameters
    ----------
    npz_path:
        Path to the ``*_samples.npz`` file (as saved by `run_mala`/
        `BatchLearner`) for this species.
    hmean, mdd:
        Same meaning as everywhere else in this module — mean habitat
        suitability and the species' target mean dispersal distance, used
        to compute ``Ew(r)`` exactly as `cost_function` does.
    ci_mass:
        Credible mass defining the 3-D HPD region (default 0.68) — see
        above. Applied to the JOINT `(s, d, S_crit)` density, not to the
        speed proxy directly. Ignored if `density_frac_of_map` is given.
    density_frac_of_map:
        Alternative region-selection rule, mutually exclusive with
        `ci_mass` (takes precedence when given). Instead of choosing the
        density threshold to hit a TARGET PROBABILITY MASS (the `ci_mass`
        recipe above), this fixes the threshold directly as a fraction of
        the peak (MAP) density: keep every sample whose joint `(s, d,
        S_crit)` density is `>= density_frac_of_map * density_at_MAP`.
        E.g. `density_frac_of_map=0.5` keeps everything within a factor of
        2 of the MAP's own density — a "relative-density" / likelihood-
        ratio-style support region, sometimes used for peak-width
        summaries, as opposed to a proper Bayesian credible region.

        IMPORTANT CAVEAT: unlike `ci_mass`, this does NOT correspond to a
        fixed, known probability mass — how much mass a given density
        fraction encloses depends entirely on the POSTERIOR'S SHAPE
        (curvature/spread near the mode), and can differ wildly between
        species/runs even at the same `density_frac_of_map` value. A
        sharply peaked posterior encloses very little mass at, say, 50% of
        peak density; a broad/flat one can enclose nearly all of it. The
        actual mass captured (`n_hpd/n_total`) is still reported so you
        can see what you actually got, but it is a CONSEQUENCE of this
        choice, not something you are directly controlling the way
        `ci_mass` lets you. Use this when you want "how far can the
        parameters drift from the MAP while staying within a fixed
        density ratio of it" (a peak-sharpness question); use `ci_mass`
        when you want "the smallest region I am `ci_mass`-confident
        contains the truth" (a genuine credible-region question) — they
        answer different questions and are not interchangeable.
    map_grid_size:
        Grid resolution per axis for the MAP-finding 3-D KDE (cubic
        cost).
    save_fig_folder, species_name:
        If given, the speed-proxy histogram (restricted to the HPD
        subset, with the LOW/MAP/HIGH picks marked) is saved to this
        folder.
    plot_3d:
        If ``True`` (default), also build two interactive Plotly figures
        via :func:`_plot_speed_3d_plotly` — both plain 3-D point scatters
        (no surface/isosurface fit) of every raw sample over `(s, d,
        S_crit)`: one colored/sized by the speed proxy, the other by the
        sample's own KDE density value. Comparing the two shows whether
        high/low-speed regions coincide with jointly plausible
        (high-density) parameter combinations or sit in the
        low-density tails. Saved as standalone HTML next to the histogram
        if `save_fig_folder` is given, else shown inline.

    Returns
    -------
    dict
        ``{"speed": array (n_total,), "ci_mass": float | None,
        "density_frac_of_map": float | None, "region_label": str,
        "n_total": int, "n_hpd": int, "low": {"s":, "d":, "S_crit":, "speed":},
        "high": {"s":, "d":, "S_crit":, "speed":},
        "map": {"s":, "d":, "S_crit":}}`` — `low`/`high`/`map` are each
        ready to unpack directly as the ``(s, d, S_crit)`` triple
        `PopulationSimulator`/`run_simulation` expect — NOT `(n, r, Tg)`.
        ``speed`` covers ALL raw samples (not just the HPD subset) for
        reference.
    """
    from scipy.stats import gaussian_kde

    print(f"\033[96m[low_map_high] Loading ALL raw posterior samples "
          f"({npz_path})\033[0m")
    data = np.load(npz_path)
    s_arr = np.asarray(data["s"]).flatten()
    d_arr = np.asarray(data["d"]).flatten()
    Scrit_arr = np.asarray(data["S_crit"]).flatten()
    n_arr = np.asarray(data["n"]).flatten()
    r_arr = np.asarray(data["r"]).flatten()
    n_total = s_arr.size
    print(f"[low_map_high] {n_total} raw samples loaded (no thinning needed — "
          f"cheap enough to evaluate on every sample).")

    print(f"\033[96m[low_map_high] Computing the analytic colonisation-speed "
          f"proxy c = sqrt(2*g*Ew*(r)) for all {n_total} samples\033[0m")
    C_np = (2.0 * np.exp(-1.11 / mdd)) / (1.0 + np.exp(-2.0 * 1.11 / mdd))
    rmax = (1.0 / np.log(hmean)) * np.log(C_np)

    r_clamped = np.minimum(r_arr, rmax - 1e-6)   # the one hard physical boundary
    denom = hmean ** r_clamped - C_np
    denom = np.where(denom >= 0, np.maximum(denom, 1e-4), np.minimum(denom, -1e-4))
    Ew_arr = np.clip(C_np / denom, 1e-3, 1e4)
    # Ew* — the mortality-corrected expected number of hops actually
    # completed ALIVE (see the docstring: raw Ew ignores per-hop
    # mortality, which ends the random walk early) — reuses `_survival`
    # directly, since Ew* = Ew * hmean**r * S(r) and S(r) IS `_survival`'s
    # own output.
    h_r = hmean ** r_clamped
    S_arr = _survival(r_clamped, hmean, 1.11, mdd)
    Ew_star_arr = Ew_arr * h_r * S_arr
    g_arr = 1.0 / Scrit_arr - 1.0
    speed = np.sqrt(2.0 * g_arr * Ew_star_arr)

    print(f"\033[96m[low_map_high] Fitting the joint (s, d, S_crit) density\033[0m")
    kde3d = gaussian_kde(np.vstack([s_arr, d_arr, Scrit_arr]))
    sample_density = kde3d(np.vstack([s_arr, d_arr, Scrit_arr]))

    # MAP first (via a 3-D KDE grid argmax) — needed BEFORE region
    # selection below when `density_frac_of_map` is used, since that rule
    # thresholds relative to the MAP's own density.
    print(f"\033[96m[low_map_high] Recovering the MAP via a 3-D KDE argmax "
          f"over the (s, d, S_crit) samples\033[0m")
    s_g = np.linspace(s_arr.min(), s_arr.max(), map_grid_size)
    d_g = np.linspace(d_arr.min(), d_arr.max(), map_grid_size)
    Sc_g = np.linspace(Scrit_arr.min(), Scrit_arr.max(), map_grid_size)
    Sg, Dg, Cg = np.meshgrid(s_g, d_g, Sc_g, indexing="ij")
    dens = kde3d(np.vstack([Sg.ravel(), Dg.ravel(), Cg.ravel()])).reshape(Sg.shape)
    i_m, j_m, k_m = np.unravel_index(np.argmax(dens), dens.shape)
    s_map, d_map, Scrit_map = float(s_g[i_m]), float(d_g[j_m]), float(Sc_g[k_m])
    map_density = float(dens.max())
    log_n_map, log_r_map = _sd_to_logn_logr(s_map, d_map)
    n_map, r_map = 10.0 ** log_n_map, 10.0 ** log_r_map
    r_map_clamped = min(r_map, rmax - 1e-6)
    denom_map = hmean ** r_map_clamped - C_np
    denom_map = max(denom_map, 1e-4) if denom_map >= 0 else min(denom_map, -1e-4)
    Ew_map = float(np.clip(C_np / denom_map, 1e-3, 1e4))
    h_r_map = hmean ** r_map_clamped
    S_map = float(_survival(r_map_clamped, hmean, 1.11, mdd))
    Ew_star_map = Ew_map * h_r_map * S_map
    g_map = 1.0 / Scrit_map - 1.0
    speed_map = float(np.sqrt(2.0 * g_map * Ew_star_map))
    print(f"[low_map_high] MAP: s={s_map:.4g}  d={d_map:.4g}  S_crit={Scrit_map:.4g}  "
          f"density={map_density:.4g}  "
          f"(n={n_map:.4g}  r={r_map:.6g}  Ew(r)={Ew_map:.4g}  Ew*(r)={Ew_star_map:.4g}  "
          f"speed={speed_map:.4g})")

    # ── Region selection: either the ci_mass HPD recipe, or a fixed
    # fraction of the MAP's own density (see `density_frac_of_map`'s
    # docstring for why these are NOT interchangeable) ─────────────────
    if density_frac_of_map is not None:
        density_threshold = density_frac_of_map * map_density
        region_label = f"{density_frac_of_map*100:.0f}% of MAP density"
        print(f"\033[96m[low_map_high] Selecting samples with density >= "
              f"{density_frac_of_map*100:.0f}% of the MAP's own density "
              f"({density_threshold:.4g})\033[0m")
    else:
        # HPD threshold: evaluate the density AT the samples themselves,
        # sort, and keep the top `ci_mass` fraction — the standard
        # empirical HPD recipe (Hyndman 1996; same one `_hpd_levels`-style
        # contours use elsewhere in this module), just in 3-D and applied
        # to the samples directly rather than a rendered grid.
        density_threshold = np.percentile(sample_density, (1.0 - ci_mass) * 100)
        region_label = f"{ci_mass*100:.0f}% HPD"
        print(f"\033[96m[low_map_high] Finding the {ci_mass*100:.0f}% HPD "
              f"region\033[0m")

    hpd_mask = sample_density >= density_threshold
    n_hpd = int(hpd_mask.sum())
    print(f"[low_map_high] {n_hpd}/{n_total} samples ({n_hpd/n_total*100:.1f}%) "
          f"inside the {region_label} region.")

    hpd_indices = np.where(hpd_mask)[0]
    speed_hpd = speed[hpd_mask]
    idx_low  = int(hpd_indices[np.argmin(speed_hpd)])
    idx_high = int(hpd_indices[np.argmax(speed_hpd)])
    print(f"[low_map_high] speed proxy within the {region_label} region: "
          f"min={speed[idx_low]:.4g}  max={speed[idx_high]:.4g}  "
          f"(low sample idx={idx_low}, high sample idx={idx_high})")

    print(f"\033[92m[low_map_high] LOW:  s={s_arr[idx_low]:.4g}  "
          f"d={d_arr[idx_low]:.4g}  S_crit={Scrit_arr[idx_low]:.4g}  "
          f"(n={n_arr[idx_low]:.4g}  r={r_arr[idx_low]:.6g}  "
          f"Ew(r)={Ew_arr[idx_low]:.4g}  Ew*(r)={Ew_star_arr[idx_low]:.4g}  "
          f"speed={speed[idx_low]:.4g})\033[0m")
    print(f"\033[92m[low_map_high] HIGH: s={s_arr[idx_high]:.4g}  "
          f"d={d_arr[idx_high]:.4g}  S_crit={Scrit_arr[idx_high]:.4g}  "
          f"(n={n_arr[idx_high]:.4g}  r={r_arr[idx_high]:.6g}  "
          f"Ew(r)={Ew_arr[idx_high]:.4g}  Ew*(r)={Ew_star_arr[idx_high]:.4g}  "
          f"speed={speed[idx_high]:.4g})\033[0m")

    if plot_3d:
        print(f"\033[96m[low_map_high] Building 3-D interactive Plotly figures "
              f"(speed-colored scatter + KDE density scatter)\033[0m")
        _plot_speed_3d_plotly(
            s_arr, d_arr, Scrit_arr, speed, sample_density, hpd_mask,
            idx_low, idx_high, s_map, d_map, Scrit_map,
            save_fig_folder=save_fig_folder, species_name=species_name,
        )

    prefix = f"{species_name}_" if species_name else ""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.hist(speed_hpd, bins=40, density=True, color="steelblue", alpha=0.45,
            edgecolor="white", label=f"samples inside the {region_label} region")
    kde1d = gaussian_kde(speed_hpd)
    xs = np.linspace(speed_hpd.min(), speed_hpd.max(), 400)
    ax.plot(xs, kde1d(xs), color="steelblue", linewidth=2.2, label="KDE fit")
    ax.axvline(speed[idx_low], color="tab:blue", linewidth=2.0, linestyle="--",
               label=f"LOW (s={s_arr[idx_low]:.3g}, d={d_arr[idx_low]:.3g}, S_crit={Scrit_arr[idx_low]:.3g})")
    ax.axvline(speed[idx_high], color="tab:red", linewidth=2.0, linestyle="--",
               label=f"HIGH (s={s_arr[idx_high]:.3g}, d={d_arr[idx_high]:.3g}, S_crit={Scrit_arr[idx_high]:.3g})")
    ax.axvline(speed_map, color="cyan", linewidth=2.0, linestyle="-",
               label=f"MAP (s={s_map:.3g}, d={d_map:.3g}, S_crit={Scrit_map:.3g}, "
                     f"speed={speed_map:.3g})")
    ax.set_xlabel("Analytic colonisation-speed proxy  c = sqrt(2*g*Ew*(r))")
    ax.set_ylabel("Density")
    ax.set_title(f"Colonisation-speed proxy within the {region_label} region "
                 f"(n={n_hpd}/{n_total}) — {species_name or ''}")
    ax.grid(linestyle="--", color="grey", linewidth=0.2, alpha=0.5)
    ax.legend(fontsize=8, framealpha=0.8)
    fig.tight_layout()
    if save_fig_folder:
        plt.savefig(os.path.join(save_fig_folder, f"{prefix}low_map_high_speed_proxy.png"),
                    dpi=150, bbox_inches="tight")
    plt.show()

    return {
        "speed": speed,
        "ci_mass": ci_mass if density_frac_of_map is None else None,
        "density_frac_of_map": density_frac_of_map,
        "region_label": region_label,
        "n_total": n_total,
        "n_hpd": n_hpd,
        # (s, d, S_crit) -- NOT (n, r, Tg) -- ready to unpack directly as
        # the `PopulationSimulator(hs, ewalk, s, d, S_crit, ...)` triple.
        "low": {"s": float(s_arr[idx_low]), "d": float(d_arr[idx_low]),
                "S_crit": float(Scrit_arr[idx_low]), "speed": float(speed[idx_low])},
        "high": {"s": float(s_arr[idx_high]), "d": float(d_arr[idx_high]),
                 "S_crit": float(Scrit_arr[idx_high]), "speed": float(speed[idx_high])},
        "map": {"s": s_map, "d": d_map, "S_crit": Scrit_map, "speed": speed_map},
    }


# ---------------------------------------------------------------------------
# Equilibrium-distribution comparison between two parameter points
# ---------------------------------------------------------------------------

def compare_equilibrium_distributions(
    calibration_sites,
    hmean: float,
    mdd: float,
    carrying_capacity_params: tuple,
    point1: tuple,
    point2: tuple,
    posteriors_and_masks: tuple | None = None,
    label1: str = "Point 1",
    label2: str = "Point 2",
    save_fig_folder: str | None = None,
    species_name: str | None = None,
    max_sites_per_figure: int = 8,
    adaptive: bool = True,
    convergence_ratio_tol: float = 0.0001,
    max_iter: int = 500,
    debug: bool = False,
    debug_site_idx: int | None = None,
) -> None:
    """Compute and visually compare the equilibrium distribution N_inf at
    EVERY calibration site under two different ``(s, d, S_crit)`` points
    (this module's own learning space — see the module-level notes above
    `_hs_local_contrast`/`_Scrit_box` — not ``(n, r, Tg)`` any more), side
    by side (point1 | point2 | difference).

    Useful to sanity-check a point found by the optimiser/grid-scan (e.g.
    one sitting in a suspicious cost valley reachable only via large Tg)
    against a known-reasonable point: if the equilibrium distributions at
    ``point2`` look physically unrealistic (e.g. mass piling up at the
    edges of the 70x70 window) compared to ``point1``, that's direct
    visual evidence the cost reduction is coming from an artifact (e.g.
    boundary effects) rather than a genuinely better fit.

    Parameters
    ----------
    point1, point2:
        ``(s, d, S_crit)`` tuples — the two points to compare, given
        directly in this module's own learning space (not ``(n, r, Tg)``
        any more) — see the module-level notes above
        `_hs_local_contrast`/`_Scrit_box`.
    calibration_sites.breeding_maps, if present (see
    `sample_calibration_sites`'s `breeding_mask` parameter), is now used
    to constrain both points' equilibria exactly like `run_mala`/
    `cost_function` do — growth outside the breeding range is limited to
    the density-dependent mortality term alone (see
    `equilibrium_distribution`'s docstring), not the unconstrained
    logistic growth used everywhere before this was wired in.
    posteriors_and_masks:
        Output of :func:`~paradis.calibration.ratios.compute_all_posteriors`
        (same object passed to `cost_function`/`learn_dispersal_parameters`).
        If given, overlays the exact pixels the likelihood actually uses at
        each site — the same ``taxa_maps>0`` + ``selected_masks`` indexing
        `cost_function` applies — as green circles (used presences, where
        ``obs_maps>0``) and blue triangles (selected absences, where
        ``obs_maps==0``). If ``None``, no markers are drawn.
    label1, label2:
        Legend/title labels for each point (e.g. "SGD endpoint",
        "grid-scan minimum").
    max_sites_per_figure:
        Sites are split across multiple figures of at most this many rows
        each, so the output stays readable/saveable regardless of how many
        calibration sites there are.
    adaptive, convergence_ratio_tol, max_iter:
        Forwarded to :func:`~paradis.core.growth.equilibrium_distribution`.
        Default ``adaptive=True`` here (unlike the training path, which
        defaults to the fixed-``n_iter`` behaviour) — this tool exists
        specifically to check equilibria are genuine, so it computes them
        properly by default rather than reproducing the fixed-iteration
        artifact it's meant to catch.
    debug:
        If ``True``, additionally run
        :func:`~paradis.core.growth.equilibrium_distribution` with
        ``debug=True`` for the site selected by ``debug_site_idx`` (both
        points) — prints the per-iteration min/max/mean trace and shows
        the filmstrip map plots for that one site. Ignored (no-op) if
        ``debug_site_idx`` is ``None``. Restricted to a single site because
        running full debug output across every calibration site would
        produce dozens of figures per point.
    debug_site_idx:
        Index (into ``calibration_sites``) of the single site to run in
        full debug mode when ``debug=True``. Required for ``debug=True``
        to have any effect.
    """
    L, k, x0 = carrying_capacity_params
    n_sites_total = len(calibration_sites)
    size_site = calibration_sites.hs_maps[0].shape[0]

    if posteriors_and_masks is not None:
        _, selected_masks = posteriors_and_masks
        presence_coords, absence_coords = [], []
        for site_idx in range(n_sites_total):
            taxa_mask = calibration_sites.taxa_maps[site_idx] > 0
            rows, cols = np.where(taxa_mask)
            obs_flat = calibration_sites.obs_maps[site_idx][taxa_mask]
            sel = np.asarray(selected_masks[site_idx], dtype=bool)
            presence_used = sel & (obs_flat > 0)
            absence_used  = sel & (obs_flat == 0)
            presence_coords.append((cols[presence_used], rows[presence_used]))
            absence_coords.append((cols[absence_used],  rows[absence_used]))
    else:
        presence_coords = absence_coords = None

    C = (2.0 * np.exp(-1.11 / mdd)) / (1.0 + np.exp(-2.0 * 1.11 / mdd))

    def _ew(r_val: float) -> float:
        denom = hmean ** r_val - C
        denom = denom if abs(denom) > 1e-6 else (1e-6 if denom >= 0 else -1e-6)
        return float(np.clip(C / denom, 1e-3, 1e4))

    have_posts = posteriors_and_masks is not None
    if have_posts:
        posteriors, selected_masks = posteriors_and_masks

    # ONE seed mask per site, drawn once and reused for BOTH point1 and
    # point2 (and for repeated debug=True calls) — without this, each
    # equilibrium_distribution call would fall back to its own fresh
    # random draw (see `random_seed_init`'s default), so any difference
    # between the two computed equilibria could partly reflect a
    # different random seeding pattern rather than purely the (n, r, Tg)
    # difference being compared. Fixing the mask here isolates the
    # parameter effect, exactly like the calibration call sites in this
    # module already do within a single learning run (see
    # `learn_dispersal_parameters`'s `seed_masks_list`).
    site_seed_masks = [
        seed_mask((hs_map.size,), seed_fraction=0.25, device=device)
        for hs_map in calibration_sites.hs_maps
    ]

    # Per-site breeding-range masks, built the same way `run_mala`/
    # `cost_function` do (see `_build_breeding_masks_list`) — without this,
    # `compare` would silently compute both equilibria under fully
    # unconstrained growth even when the underlying MALA run was
    # breeding-range-constrained, making the comparison inconsistent with
    # what was actually optimised. Falls back to a list of `None` when
    # `calibration_sites` carries no breeding_maps, matching the
    # unconstrained-everywhere default elsewhere.
    breeding_masks_list = _build_breeding_masks_list(calibration_sites, n_sites_total)

    def _compute_maps(point: tuple, label: str = "") -> tuple:
        # `point` is (s, d, S_crit) — this function's own learning space —
        # converted here to the actual values `equilibrium_distribution`/
        # `Ew` need. No Tg/a intermediate anywhere: g comes straight from
        # S_crit.
        s_val, d_val, Scrit_val = point
        log_n_val, log_r_val = _sd_to_logn_logr(s_val, d_val)
        n_val, r_val = 10.0 ** log_n_val, 10.0 ** log_r_val
        g_val = 1.0 / Scrit_val - 1.0
        ew_val = _ew(r_val)
        p_val = ew_val / (1.0 + ew_val)
        maps = []
        changes_hist = []
        costs = []
        diagnostics = []  # per-site: max/min N_inf, clamp saturation
        with torch.no_grad():
            n_p  = torch.tensor(float(n_val),  device=device, dtype=torch.float32)
            r_p  = torch.tensor(float(r_val),  device=device, dtype=torch.float32)
            ew_p = torch.tensor(float(ew_val), device=device, dtype=torch.float32)
            g_p = torch.tensor(float(g_val), device=device, dtype=torch.float32)
            site_bar = tqdm(range(n_sites_total), desc=f"  {label} sites", leave=False)
            for site_idx in site_bar:
                hs_map = calibration_sites.hs_maps[site_idx]
                hs_t = torch.tensor(hs_map, dtype=torch.float32, device=device)
                K_is = torch.tensor(
                    _logistic(hs_map.flatten().astype(np.float32), L, k, x0),
                    dtype=torch.float32, device=device,
                )
                adj_mat = adjacency_matrix_torch(hs_t)
                Kd = dispersal_kernel_fast(adj_mat, r=r_p, n=n_p, ewalk=ew_p)

                def _report(epoch: int, ratio: float) -> None:
                    site_bar.set_postfix(site=f"{site_idx+1}/{n_sites_total}",
                                          iter=epoch, ratio=f"{ratio:.4g}")

                site_debug = debug and debug_site_idx is not None and site_idx == debug_site_idx
                if site_debug:
                    print(f"\033[95m[compare] debug=True for site {site_idx} — {label}"
                          f" (n={n_val:.4g}, r={r_val:.4g}, S_crit={Scrit_val:.4g})\033[0m")
                    N_inf, changes, dbg_hist = equilibrium_distribution(
                        K_is, Kd, g_p, plot=False, verbose=False,
                        return_history=True, adaptive=adaptive,
                        convergence_ratio_tol=convergence_ratio_tol, max_iter=max_iter,
                        progress_callback=_report, debug=True,
                        seed_mask_override=site_seed_masks[site_idx],
                        breeding_ground=breeding_masks_list[site_idx],
                    )
                else:
                    N_inf, changes = equilibrium_distribution(
                        K_is, Kd, g_p, plot=False, verbose=False,
                        return_history=True, adaptive=adaptive,
                        convergence_ratio_tol=convergence_ratio_tol, max_iter=max_iter,
                        progress_callback=_report,
                        seed_mask_override=site_seed_masks[site_idx],
                        breeding_ground=breeding_masks_list[site_idx],
                    )
                N_inf_2d = N_inf.reshape(size_site, size_site)
                maps.append(N_inf_2d.cpu().numpy())
                changes_hist.append(changes)

                # Diagnostic: is N_inf saturating against equilibrium_distribution's
                # internal torch.clamp(Un, 0.0, 1.0) — a possible numerical
                # artifact (e.g. from a near-singular (I - p*Wstar) inversion)
                # masquerading as a genuine "no viable population" result.
                n_inf_np = N_inf_2d.cpu().numpy()
                max_val = float(n_inf_np.max())
                min_val = float(n_inf_np.min())
                frac_clamped_0 = float((n_inf_np <= 0.0).mean())
                frac_clamped_1 = float((n_inf_np >= 1.0).mean())
                diagnostics.append({
                    "max": max_val, "min": min_val,
                    "frac_clamped_0": frac_clamped_0, "frac_clamped_1": frac_clamped_1,
                })

                # Per-site cost — computed directly from THIS N_inf (the
                # same one just plotted, possibly adaptively-converged),
                # not via a separate cost_function call, which would
                # silently recompute N_inf with the fixed default n_iter
                # and be inconsistent with what's shown above.
                if have_posts:
                    taxa_mask = torch.tensor(
                        calibration_sites.taxa_maps[site_idx] > 0, device=device,
                    )
                    sel = torch.tensor(
                        np.asarray(selected_masks[site_idx], dtype=bool), device=device,
                    )
                    simulated = N_inf_2d[taxa_mask][sel]
                    n_locs = len(simulated)
                    if n_locs == 0:
                        costs.append(float("nan"))
                    else:
                        site_loss = 0.0
                        for loc_idx, r_sim in enumerate(simulated):
                            dens = extract_density(posteriors, site_idx, loc_idx, r_sim)
                            dens = torch.clamp(dens, min=1e-12)
                            site_loss += (-(1.0 / n_locs) * torch.log(dens)).item()
                        costs.append(site_loss)
                else:
                    costs.append(float("nan"))
        return maps, changes_hist, costs, diagnostics

    # `changes_hist[site][-1]` is the total absolute change between the
    # last two of the fixed 10 iterations — since equilibrium_distribution
    # has NO convergence check, a large last-iteration change (relative to
    # the first) means the run was cut off well before a genuine steady
    # state, and the returned "equilibrium" is not to be trusted (see
    # equilibrium_distribution's docstring — the growth coefficient
    # 1-linear_growth shrinks toward 0 for large Tg, slowing convergence).
    print(f"\033[96m[compare] Computing equilibrium distributions for "
          f"{label1}={point1} ...\033[0m")
    maps1, changes1, costs1, diag1 = _compute_maps(point1, label=label1)
    print(f"\033[96m[compare] Computing equilibrium distributions for "
          f"{label2}={point2} ...\033[0m")
    maps2, changes2, costs2, diag2 = _compute_maps(point2, label=label2)
    costs1, costs2 = np.array(costs1), np.array(costs2)

    if have_posts:
        better2 = int(np.nansum(costs2 < costs1))
        worse2  = int(np.nansum(costs2 > costs1))
        print(f"\033[93m[compare] Per-site cost: {label1} mean="
              f"{np.nanmean(costs1):.5f}   {label2} mean={np.nanmean(costs2):.5f}"
              f"   ({label2} better on {better2}/{n_sites_total} sites, "
              f"worse on {worse2}/{n_sites_total})\033[0m")
    else:
        print("\033[93m[compare] posteriors_and_masks not given — "
              "per-site cost not computed.\033[0m")

    # Numerical-artifact diagnostic: high clamp-saturation fractions mean
    # N_inf hit equilibrium_distribution's internal [0,1] clamp on many
    # pixels, which can flatten a genuinely exploding/degenerate result
    # into a uniform plateau that masks the instability rather than
    # reflecting a real "no viable population" outcome.
    for label, diag in [(label1, diag1), (label2, diag2)]:
        clamp0   = np.array([d["frac_clamped_0"] for d in diag])
        clamp1   = np.array([d["frac_clamped_1"] for d in diag])
        n_flagged = int(np.sum((clamp0 > 0.5) | (clamp1 > 0.5)))
        print(f"\033[93m[compare] {label}: clamp-saturation — "
              f"mean(frac=0)={clamp0.mean():.3f}  "
              f"mean(frac=1)={clamp1.mean():.3f}   "
              f"({n_flagged}/{n_sites_total} sites flagged: "
              f">50% pixels clamped)\033[0m")

    ratio1 = np.array([c[-1] / c[0] if c[0] > 0 else 0.0 for c in changes1])
    ratio2 = np.array([c[-1] / c[0] if c[0] > 0 else 0.0 for c in changes2])
    n_used1 = np.array([len(c) for c in changes1])
    n_used2 = np.array([len(c) for c in changes2])
    print("\033[93m[compare] Convergence check — last-iteration change ratio "
          "(near 0 = converged, near/above 1 = still moving as fast as at "
          "the start) and, with adaptive=True, how many iterations were "
          "actually needed to reach that ratio (more iterations needed = "
          "genuinely slower dynamics, e.g. from a large Tg):\033[0m")
    print(f"\033[93m  {label1}: mean ratio={ratio1.mean():.3g}  "
          f"max ratio={ratio1.max():.3g}  "
          f"iterations used: mean={n_used1.mean():.1f}  max={n_used1.max()}\033[0m")
    print(f"\033[93m  {label2}: mean ratio={ratio2.mean():.3g}  "
          f"max ratio={ratio2.max():.3g}  "
          f"iterations used: mean={n_used2.mean():.1f}  max={n_used2.max()}\033[0m")
    if adaptive and (n_used1.max() >= max_iter or n_used2.max() >= max_iter):
        print(f"\033[91m  Warning: at least one site hit max_iter={max_iter} "
              f"without reaching the target ratio — its equilibrium is "
              f"still not fully trustworthy; consider raising max_iter.\033[0m")

    prefix = f"{species_name}_" if species_name else ""
    n_batches = int(np.ceil(n_sites_total / max_sites_per_figure))
    for b in range(n_batches):
        idxs = list(range(b * max_sites_per_figure,
                           min((b + 1) * max_sites_per_figure, n_sites_total)))
        nrows = len(idxs)
        fig, axes = plt.subplots(nrows, 4, figsize=(15.5, 3.2 * nrows), squeeze=False)
        for row, site_idx in enumerate(idxs):
            m1, m2 = maps1[site_idx], maps2[site_idx]
            vmax = max(float(m1.max()), float(m2.max())) or 1.0
            cost1_str = f"  cost={costs1[site_idx]:.4f}" if have_posts else ""
            cost2_str = f"  cost={costs2[site_idx]:.4f}" if have_posts else ""
            d1, d2 = diag1[site_idx], diag2[site_idx]
            axes[row, 0].imshow(m1, cmap="viridis", vmin=0, vmax=vmax)
            axes[row, 0].set_title(
                f"site {site_idx} — {label1}\n"
                f"ratio={ratio1[site_idx]:.2g}  iters={n_used1[site_idx]}{cost1_str}\n"
                f"N_inf min={d1['min']:.3g} max={d1['max']:.3g}  "
                f"clamp0={d1['frac_clamped_0']:.2f} clamp1={d1['frac_clamped_1']:.2f}",
                fontsize=7)
            axes[row, 1].imshow(m2, cmap="viridis", vmin=0, vmax=vmax)
            axes[row, 1].set_title(
                f"site {site_idx} — {label2}\n"
                f"ratio={ratio2[site_idx]:.2g}  iters={n_used2[site_idx]}{cost2_str}\n"
                f"N_inf min={d2['min']:.3g} max={d2['max']:.3g}  "
                f"clamp0={d2['frac_clamped_0']:.2f} clamp1={d2['frac_clamped_1']:.2f}",
                fontsize=7)
            diff = m2 - m1
            dmax = float(np.abs(diff).max()) or 1.0
            axes[row, 2].imshow(diff, cmap="coolwarm", vmin=-dmax, vmax=dmax)
            diff_cost_str = (f"\nΔcost ({label2}-{label1})={costs2[site_idx]-costs1[site_idx]:+.4f}"
                              if have_posts else "")
            axes[row, 2].set_title(
                f"site {site_idx} — {label2} minus {label1}{diff_cost_str}", fontsize=8)

            # 4th column: breeding range shown as its own panel (not
            # overlaid on the density/diff panels above — an overlay there
            # was tried and reduced readability of the density maps
            # themselves) — plain grayscale, 1=breeding range, 0=outside.
            bm = calibration_sites.breeding_maps[site_idx] if hasattr(
                calibration_sites, "breeding_maps") else None
            if bm is not None:
                bm = np.asarray(bm, dtype=float)
                axes[row, 3].imshow(bm, cmap="Reds", vmin=0, vmax=1)
                axes[row, 3].set_title(f"site {site_idx} — breeding range", fontsize=8)
            else:
                axes[row, 3].text(0.5, 0.5, "no breeding\nrange data",
                                   ha="center", va="center", fontsize=8, color="grey",
                                   transform=axes[row, 3].transAxes)
                axes[row, 3].set_title(f"site {site_idx} — breeding range", fontsize=8)

            # Overlay the exact pixels used by the likelihood: green
            # circles for used presences, blue triangles for selected
            # absences — on both parameter-point panels (properties of the
            # data, not of the point), so any suspicious mismatch between
            # where mass builds up and where real observations sit is
            # directly visible.
            if presence_coords is not None:
                px, py = presence_coords[site_idx]
                ax_, ay = absence_coords[site_idx]
                for ax in (axes[row, 0], axes[row, 1]):
                    ax.scatter(px, py, facecolors="none", edgecolors="lime",
                               marker="o", s=25, linewidths=1.0, label="presence (used)")
                    ax.scatter(ax_, ay, facecolors="none", edgecolors="dodgerblue",
                               marker="^", s=25, linewidths=1.0, label="absence (selected)")

            for ax in axes[row]:
                ax.axis("off")
        if presence_coords is not None:
            axes[0, 0].legend(loc="upper right", fontsize=6, framealpha=0.7)
        fig.suptitle(
            f"{species_name or ''} equilibrium distribution comparison "
            f"({label1} vs {label2}) — batch {b + 1}/{n_batches}"
        )
        fig.tight_layout()
        if save_fig_folder:
            plt.savefig(
                os.path.join(save_fig_folder, f"{prefix}eq_compare_batch{b+1}.png"),
                dpi=150, bbox_inches="tight",
            )
        plt.show()


# ---------------------------------------------------------------------------
# Brute-force cost-surface grid scan (no gradients)
# ---------------------------------------------------------------------------

def _scan_cost_grid(
    mdd: float,
    posteriors_and_masks: tuple,
    calibration_sites,
    carrying_capacity_params: tuple,
    hmean: float,
    adj_mats: list,
    K_is_list: list,
    n_sites_total: int,
    n_range: tuple,
    r_range: tuple,
    tg_range: tuple,
    points_per_axis: int,
    tag: str,
    save_fig_folder: str | None = None,
    species_name: str | None = None,
    seed_masks_list: list | None = None,
    breeding_masks_list: list | None = None,
) -> dict:
    """Full-dataset cost (no gradient) over a `points_per_axis`^3 grid.
    n, r, AND Tg ranges are given in ACTUAL units but ALL sampled
    LOG-spaced (`np.geomspace`) — matching how they're actually
    learned/interpreted everywhere else in this module. Saves the raw
    grid + 2-D heatmap slices (log axes for n/r) + an interactive 3-D
    volume (Plotly, log axis for Tg too) to `save_fig_folder`. Returns a
    dict with the raw
    results plus the grid's minimum location/cost.
    """
    n_grid  = np.geomspace(n_range[0], n_range[1], points_per_axis)
    r_grid  = np.geomspace(r_range[0], r_range[1], points_per_axis)
    tg_grid = np.geomspace(tg_range[0], tg_range[1], points_per_axis)

    results = np.zeros((points_per_axis, points_per_axis, points_per_axis))
    flat_records = []
    all_indices = list(range(n_sites_total))

    pbar = tqdm(total=points_per_axis ** 3, desc=f"  Grid[{tag}]", leave=False)
    with torch.no_grad():
        for i, n_val in enumerate(n_grid):
            for j, r_val in enumerate(r_grid):
                for kk, tg_val in enumerate(tg_grid):
                    n_p  = torch.tensor(float(n_val),  device=device, dtype=torch.float32)
                    r_p  = torch.tensor(float(r_val),  device=device, dtype=torch.float32)
                    # This grid is still defined in Tg-space (a legacy axis
                    # kept for this diagnostic tool's plots/labels) —
                    # converted to S_crit here ONLY to satisfy
                    # `cost_function`'s contract (no Tg/a used internally
                    # anywhere else). g = a's old low-eps relaxation rate;
                    # S_crit = 1/(1+g).
                    a_val = 0.05 ** (1.0 / tg_val)
                    scrit_val = 1.0 / (2.0 - a_val)
                    scrit_p = torch.tensor(float(scrit_val), device=device, dtype=torch.float32)
                    c = cost_function(
                        mdd, posteriors_and_masks, calibration_sites,
                        (n_p, r_p, scrit_p), all_indices,
                        carrying_capacity_params, hmean, adj_mats, K_is_list,
                        plot=False, verbose=False,
                        seed_masks_list=seed_masks_list, breeding_masks_list=breeding_masks_list,
                    )
                    cost_val = c.item()
                    results[i, j, kk] = cost_val
                    flat_records.append((n_val, r_val, tg_val, cost_val))
                    pbar.update(1)
    pbar.close()

    flat_arr = np.array(flat_records)
    prefix = f"{species_name}_" if species_name else ""
    if save_fig_folder:
        np.save(os.path.join(save_fig_folder, f"{prefix}cost_grid_flat_{tag}.npy"), flat_arr)
        np.save(os.path.join(save_fig_folder, f"{prefix}cost_grid_cube_{tag}.npy"), results)

    # 2-D heatmap slices (cost as a function of log10(n), log10(r)) at
    # several fixed Tg values, with 100 white contour lines and a red
    # marker at each slice's own minimum.
    slice_indices = sorted(set([0, points_per_axis // 3, 2 * points_per_axis // 3, points_per_axis - 1]))
    fig, axes = plt.subplots(1, len(slice_indices), figsize=(5 * len(slice_indices), 4.5))
    if len(slice_indices) == 1:
        axes = [axes]
    log_n, log_r = np.log10(n_grid), np.log10(r_grid)
    vmin, vmax = results.min(), results.max()
    for ax, tg_idx in zip(axes, slice_indices):
        im = ax.imshow(
            results[:, :, tg_idx].T,
            origin="lower", aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax,
            extent=[log_n[0], log_n[-1], log_r[0], log_r[-1]],
        )
        ax.contour(
            log_n, log_r, results[:, :, tg_idx].T,
            levels=100, colors="white", linewidths=0.4, alpha=0.6,
        )
        slice_2d = results[:, :, tg_idx]
        i_min, j_min = np.unravel_index(np.argmin(slice_2d), slice_2d.shape)
        ax.plot(log_n[i_min], log_r[j_min], "o", color="red", markersize=7,
                markeredgecolor="white", markeredgewidth=0.8)
        ax.set_xlabel("log10(n)")
        ax.set_ylabel("log10(r)")
        ax.set_title(f"cost(n, r) at Tg={tg_grid[tg_idx]:.3g}")
    fig.colorbar(im, ax=axes, label="full-dataset cost", shrink=0.8)
    if save_fig_folder:
        plt.savefig(os.path.join(save_fig_folder, f"{prefix}cost_grid_slices_{tag}.png"),
                    dpi=150, bbox_inches="tight")
    plt.show()

    # Interactive 3-D volume (Plotly), coloured/opacity-mapped by cost —
    # log axes for n/r/Tg, matching how they're all learned.
    try:
        n_flat, r_flat, tg_flat, cost_flat = flat_arr.T
        fig3d = go.Figure(data=go.Volume(
            x=np.log10(n_flat), y=np.log10(r_flat), z=np.log10(tg_flat), value=cost_flat,
            isomin=float(cost_flat.min()), isomax=float(cost_flat.max()),
            opacity=0.4, opacityscale="min",
            surface_count=21,
            colorscale="Viridis_r",
            colorbar=dict(title="cost"),
            caps=dict(x_show=False, y_show=False, z_show=False),
        ))
        min_idx_flat = np.argmin(cost_flat)
        fig3d.add_trace(go.Scatter3d(
            x=[np.log10(n_flat[min_idx_flat])], y=[np.log10(r_flat[min_idx_flat])], z=[np.log10(tg_flat[min_idx_flat])],
            mode="markers",
            marker=dict(size=6, color="red", symbol="circle", line=dict(color="white", width=1)),
            name="minimum",
        ))
        fig3d.update_layout(
            title=f"[{tag}] {species_name or ''} — full-dataset cost over (log10 n, log10 r, log10 Tg)",
            scene=dict(xaxis_title="log10(n)", yaxis_title="log10(r)", zaxis_title="log10(Tg)",
                       aspectmode="cube"),
            width=900, height=700,
        )
        if save_fig_folder:
            fig3d.write_html(os.path.join(save_fig_folder, f"{prefix}cost_grid_3d_{tag}.html"))
        fig3d.show()
    except Exception as e:
        print(f"[grid:{tag}] Skipped 3-D volume ({e}).")

    min_idx = np.unravel_index(np.argmin(results), results.shape)
    best_tg_val = float(tg_grid[min_idx[2]])
    # Converted to S_crit here so `best`'s 3rd element has the SAME meaning
    # as `_scan_cost_grid_sd`'s own `best` tuple (S_crit, not Tg) —
    # this grid's own axis is still Tg-space (a legacy diagnostic choice),
    # but nothing downstream should have to know that.
    best_a_val = 0.05 ** (1.0 / best_tg_val)
    best_scrit_val = 1.0 / (2.0 - best_a_val)
    best = (float(n_grid[min_idx[0]]), float(r_grid[min_idx[1]]), best_scrit_val)
    best_cost = float(results[min_idx])
    print(f"\033[92m[grid:{tag}] Minimum: cost={best_cost:.5f}  "
          f"n={best[0]:.4g}  r={best[1]:.6g}  S_crit={best[2]:.4g}\033[0m")

    return {
        "results": results, "flat": flat_arr,
        "n_grid": n_grid, "r_grid": r_grid, "tg_grid": tg_grid,
        "best": best, "best_cost": best_cost,
    }


def _plot_cost_slice(ax, slice_2d, x_grid, y_grid, x_label, y_label, title,
                      cmap, vmin, vmax, mark_xy=None, exact_xy=None):
    """Shared heatmap-slice renderer (used by `_scan_cost_grid_sd` and
    `_run_precise_gridscan`): cost colormap (NaN greyed out via `cmap`'s
    "bad" colour), white contour lines, gradient-direction quiver arrows
    (d(cost)/dx, d(cost)/dy — rescaled to ~0.7x one grid cell in each of
    x/y, using each axis's own step size, so arrows stay roughly
    pixel-sized and legible regardless of the raw gradient magnitude), an
    optional marker at `mark_xy` (e.g. this slice's own grid-snapped
    minimum, red circle), and an optional SECOND marker at `exact_xy`
    (the EXACT, continuous — not grid-snapped — point a scan was centred
    on, cyan star, drawn distinctly from `mark_xy`). Returns the imshow
    handle (for a shared colorbar).
    """
    masked = np.ma.masked_invalid(slice_2d.T)
    im = ax.imshow(
        masked,
        origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
        extent=[x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]],
    )
    if np.isfinite(slice_2d).any():
        ax.contour(
            x_grid, y_grid, slice_2d.T,
            levels=100, colors="white", linewidths=0.4, alpha=0.6,
        )
        if mark_xy is not None:
            ax.plot(mark_xy[0], mark_xy[1], "o", color="red", markersize=7,
                    markeredgecolor="white", markeredgewidth=0.8,
                    label="grid minimum (snapped)", zorder=5)
        if exact_xy is not None:
            ax.plot(exact_xy[0], exact_xy[1], "*", color="cyan", markersize=13,
                    markeredgecolor="black", markeredgewidth=0.8,
                    label="exact point", zorder=6)

        dx_step = (x_grid[1] - x_grid[0]) if len(x_grid) > 1 else 1.0
        dy_step = (y_grid[1] - y_grid[0]) if len(y_grid) > 1 else 1.0
        grad_x, grad_y = np.gradient(slice_2d, x_grid, y_grid)
        grad_norm = np.sqrt(grad_x ** 2 + grad_y ** 2)
        safe_norm = np.where(grad_norm > 0, grad_norm, np.nan)
        u = (grad_x / safe_norm) * dx_step * 0.7
        v = (grad_y / safe_norm) * dy_step * 0.7
        arrow_valid = np.isfinite(slice_2d) & np.isfinite(u) & np.isfinite(v)
        X_mesh, Y_mesh = np.meshgrid(x_grid, y_grid, indexing="ij")
        ax.quiver(
            X_mesh[arrow_valid], Y_mesh[arrow_valid],
            u[arrow_valid], v[arrow_valid],
            angles="xy", scale_units="xy", scale=1.0,
            color="black", alpha=0.7, width=0.0025,
            headwidth=4, headlength=5, headaxislength=4.5,
        )
        if mark_xy is not None or exact_xy is not None:
            ax.legend(loc="upper right", fontsize=6, framealpha=0.7)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    return im


def _scan_cost_grid_sd(
    mdd: float,
    posteriors_and_masks: tuple,
    calibration_sites,
    carrying_capacity_params: tuple,
    hmean: float,
    adj_mats: list,
    K_is_list: list,
    n_sites_total: int,
    s_range: tuple,
    d_range: tuple,
    Scrit_range: tuple,
    points_per_axis: int,
    tag: str,
    n_log_range: tuple | None = None,
    r_log_range: tuple | None = None,
    save_fig_folder: str | None = None,
    species_name: str | None = None,
    seed_masks_list: list | None = None,
    breeding_masks_list: list | None = None,
    exact_point: tuple | None = None,
) -> dict:
    """Full-dataset cost (no gradient) over a `points_per_axis`^3 grid,
    sampled directly in the (s, d) reparametrisation of (n, r) instead of
    independently in (log10 n, log10 r) — see the module-level note above
    `_hs_local_contrast` for the derivation. `s = log10(n*r)` is the
    INFORMATIVE axis (controls the dispersal kernel's selectivity
    pattern) and `d = log10(n/r)` is the near-flat one (moves along the
    n<->r trade-off at ~constant selectivity, only perturbing the
    residual HS^r "survival" terms) — sampling a rectangle in (s, d)
    instead of (log10 n, log10 r) concentrates grid points where the cost
    can actually change, instead of wasting many of them along the
    ~45-degree ridge that a (log10 n, log10 r)-aligned box would spread
    evenly (and mostly uninformatively) across.

    `s_range`/`d_range` are given directly in (s, d) units (NOT log10'd
    again — they already are log10-composite quantities), sampled
    LINEARLY (`np.linspace`) since that IS the natural/log scale for this
    axis pair. `Scrit_range` is given directly in `S_crit` units (the
    critical per-dispersal-step survival probability — see the
    module-level growth-timescale note above `_Scrit_box`), sampled
    LINEARLY too (`S_crit`, unlike Tg, is not naturally log-scaled — its
    own box is the fixed universal `(0.5, 1.0)` interval with no reason to
    concentrate points near either end) — this is the actual
    learned/sampled variable everywhere else (batchSGD/refine/MALA), so
    the grid now explores the SAME axis, not `Tg`. `Tg` is only recovered
    internally, per grid point, via `_Scrit_to_Tg(S_crit)` right before
    the `cost_function` call (which still consumes an actual Tg value,
    unchanged).
    Saves the raw grid + 2-D heatmap slices (in (s, d)) + an interactive
    3-D volume (Plotly) to `save_fig_folder`. Returns a dict with the raw
    results plus the grid's minimum location/cost (in both (s, d) and the
    corresponding actual (n, r) values).

    `n_log_range`/`r_log_range`: `n_log_range` is accepted for backward
    compatibility but no longer enforced — `n_min`/`n_max` are a soft
    PRIOR choice (where we expect the informative region to be), not a
    physical impossibility outside of it, and `n` itself can never be
    negative in this log-parametrisation (`n = 10**log_n > 0` always), so
    there is nothing to reject there. Same for the LOWER end of
    `r_log_range` (`r_min`) — also just a prior choice (the point past
    which the model can no longer distinguish `r` from 0 for SURVIVAL,
    not a value `r` is physically forbidden from taking). The ONLY
    physical hard boundary enforced is the UPPER end, `r_log_range[1]`
    (`log10(r_max)`): beyond `r_max`, `Ew`/the dispersal kernel become
    ill-defined (the process can no longer be matched to the target MDD
    at all — see `_survival`'s docstring) — this is a genuine physical
    impossibility, not a prior. Grid points with `r > r_max` are SKIPPED
    (no `cost_function` call — saves compute too) and marked NaN in
    `results`, so callers can grey them out instead of colouring them
    with the cost colormap. If `r_log_range` is left `None` (default), no
    validity check is performed at all (matches the old, unchecked
    behaviour).

    `exact_point`: optional `(s, d, S_crit)` — the EXACT (continuous, not
    grid-snapped) point to also mark on every heatmap slice, e.g. the
    point `_run_precise_gridscan`'s window was centred on (typically
    `refine_from_point`'s converged endpoint). Drawn as a distinct marker
    (cyan star) from the grid's own discovered minimum (red circle,
    which is necessarily grid-snapped and can differ slightly from the
    true continuous optimum) — the two coinciding closely is itself a
    useful visual confirmation that the grid resolution is fine enough
    to have actually found the same point.
    """
    s_grid   = np.linspace(s_range[0], s_range[1], points_per_axis)
    d_grid   = np.linspace(d_range[0], d_range[1], points_per_axis)
    Scrit_grid = np.linspace(Scrit_range[0], Scrit_range[1], points_per_axis)

    results = np.full((points_per_axis, points_per_axis, points_per_axis), np.nan)
    flat_records = []   # (s, d, n, r, S_crit, tg, cost) per grid point — invalid points excluded
    all_indices = list(range(n_sites_total))

    pbar = tqdm(total=points_per_axis ** 3, desc=f"  Grid[{tag}]", leave=False)
    with torch.no_grad():
        for i, s_val in enumerate(s_grid):
            for j, d_val in enumerate(d_grid):
                log_n_val, log_r_val = _sd_to_logn_logr(s_val, d_val)
                n_val = 10.0 ** log_n_val
                r_val = 10.0 ** log_r_val
                # The ONLY hard physical boundary: r > r_max makes Ew/the
                # dispersal kernel ill-defined (MDD can no longer be
                # matched — see `_survival`'s docstring). n_min/n_max and
                # r_min are soft PRIOR choices, not physical impossibilities
                # — n and r themselves can never be negative in this
                # log-parametrisation (n=10**log_n, r=10**log_r are always
                # > 0 by construction), so there is nothing else to reject
                # here. `r_log_range[1]` (== log10(r_max)) is still used
                # for this one genuine boundary; `n_log_range` and the
                # lower end of `r_log_range` are accepted for backward
                # compatibility but no longer enforced.
                if r_log_range is not None and log_r_val > r_log_range[1]:
                    pbar.update(points_per_axis)
                    continue
                n_p = torch.tensor(float(n_val), device=device, dtype=torch.float32)
                r_p = torch.tensor(float(r_val), device=device, dtype=torch.float32)
                for kk, Scrit_val in enumerate(Scrit_grid):
                    Scrit_safe = min(max(Scrit_val, 0.5 + 1e-6), 1.0 - 1e-6)
                    scrit_p = torch.tensor(float(Scrit_safe), device=device, dtype=torch.float32)
                    c = cost_function(
                        mdd, posteriors_and_masks, calibration_sites,
                        (n_p, r_p, scrit_p), all_indices,
                        carrying_capacity_params, hmean, adj_mats, K_is_list,
                        plot=False, verbose=False,
                        seed_masks_list=seed_masks_list, breeding_masks_list=breeding_masks_list,
                    )
                    cost_val = c.item()
                    results[i, j, kk] = cost_val
                    flat_records.append((s_val, d_val, n_val, r_val, Scrit_val, cost_val))
                    pbar.update(1)
    pbar.close()

    flat_arr = np.array(flat_records)
    prefix = f"{species_name}_" if species_name else ""
    if save_fig_folder:
        np.save(os.path.join(save_fig_folder, f"{prefix}cost_grid_sd_flat_{tag}.npy"), flat_arr)
        np.save(os.path.join(save_fig_folder, f"{prefix}cost_grid_sd_cube_{tag}.npy"), results)

    # Global minimum — computed HERE (before any plotting) so its values
    # can be shown as a suptitle on the 2-D heatmap slices below, not just
    # printed to the console after the figures are already drawn.
    min_idx = np.unravel_index(np.nanargmin(results), results.shape)
    best_s, best_d, best_Scrit = float(s_grid[min_idx[0]]), float(d_grid[min_idx[1]]), float(Scrit_grid[min_idx[2]])
    best_Scrit_safe = min(max(best_Scrit, 0.5 + 1e-6), 1.0 - 1e-6)
    best_log_n, best_log_r = _sd_to_logn_logr(best_s, best_d)
    best = (10.0 ** best_log_n, 10.0 ** best_log_r, best_Scrit_safe)
    best_cost = float(results[min_idx])
    minimum_str = (f"[grid:{tag}] Minimum: cost={best_cost:.5f}  "
                    f"s={best_s:.4g}  d={best_d:.4g}  "
                    f"n={best[0]:.4g}  r={best[1]:.6g}  S_crit={best_Scrit:.4g}")
    print(f"\033[92m{minimum_str}\033[0m")

    vmin, vmax = np.nanmin(results), np.nanmax(results)
    # Invalid grid points (NaN — (s,d) maps outside the true (n, r) priors
    # box, see the validity-check note above) are greyed out instead of
    # coloured by the viridis cost colormap, so a spuriously flat/low-cost
    # patch there (a real artifact of the (s,d)-box overshoot, not a
    # genuine good fit) can't be mistaken for one.
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("lightgrey")

    # 2-D heatmap slices (cost as a function of s, d) at several fixed S_crit
    # values (the actual learned/sampled growth-timescale variable — see
    # docstring above), with 100 white contour lines and a red marker at
    # each slice's own minimum.
    slice_indices = sorted(set([0, points_per_axis // 3, 2 * points_per_axis // 3, points_per_axis - 1]))
    fig, axes = plt.subplots(1, len(slice_indices), figsize=(5 * len(slice_indices), 4.5))
    if len(slice_indices) == 1:
        axes = [axes]
    for ax, Scrit_idx in zip(axes, slice_indices):
        slice_2d = results[:, :, Scrit_idx]
        mark_xy = None
        if np.isfinite(slice_2d).any():
            i_min, j_min = np.unravel_index(np.nanargmin(slice_2d), slice_2d.shape)
            mark_xy = (s_grid[i_min], d_grid[j_min])
        im = _plot_cost_slice(
            ax, slice_2d, s_grid, d_grid,
            "s = log10(n*r)  [informative]", "d = log10(n/r)  [near-flat]",
            f"cost(s, d) at S_crit={Scrit_grid[Scrit_idx]:.3g}",
            cmap, vmin, vmax,
            mark_xy=mark_xy,
            exact_xy=(exact_point[0], exact_point[1]) if exact_point is not None else None,
        )
    fig.colorbar(im, ax=axes, label="full-dataset cost", shrink=0.8)
    fig.suptitle(minimum_str)
    fig.tight_layout()
    if save_fig_folder:
        plt.savefig(os.path.join(save_fig_folder, f"{prefix}cost_grid_sd_slices_{tag}.png"),
                    dpi=150, bbox_inches="tight")
    plt.show()

    # TWO additional single-panel slices, BOTH passing exactly through the
    # grid PIXEL CONTAINING `exact_point` if given (nearest grid index to
    # its (s, d, S_crit) coordinates — `exact_point` itself is a
    # CONTINUOUS point that generally does NOT land exactly on a grid
    # node, especially with an even `points_per_axis`, so "the pixel
    # containing it" means the nearest one), or the grid's own discovered
    # minimum (`min_idx`) if `exact_point` is `None` (e.g. plain
    # `_run_grid`/`grid_default` calls, which have no externally-supplied
    # point to target) — unlike the 4 evenly-spaced S_crit slices above,
    # which only show the (s, d) plane at generic S_crit values that may
    # not land exactly on the equilibrium point at all.
    if exact_point is not None:
        i_eq = int(np.argmin(np.abs(s_grid - exact_point[0])))
        j_eq = int(np.argmin(np.abs(d_grid - exact_point[1])))
        k_eq = int(np.argmin(np.abs(Scrit_grid - exact_point[2])))
    else:
        i_eq, j_eq, k_eq = min_idx

    # (1) VERTICAL slice: (s, S_crit) plane at d FIXED to the equilibrium's
    # own d — s on x, S_crit on y. Since the (s,d) "valley" runs mostly
    # along d (see the module-level (s,d) reparametrisation note — d is
    # the near-flat axis), cutting at a single fixed d gives a transversal
    # cross-section THROUGH the valley (perpendicular to its long axis),
    # showing how sharply the cost actually rises away from the
    # equilibrium in the (s, S_crit) directions.
    fig_v, ax_v = plt.subplots(figsize=(6, 5))
    slice_sScrit = results[:, j_eq, :]   # shape (len(s_grid), len(Scrit_grid))
    im_v = _plot_cost_slice(
        ax_v, slice_sScrit, s_grid, Scrit_grid,
        "s = log10(n*r)  [informative]", "S_crit (survival threshold)",
        f"cost(s, S_crit) at d={d_grid[j_eq]:.4g} (through the equilibrium)",
        cmap, vmin, vmax,
        mark_xy=(s_grid[i_eq], Scrit_grid[k_eq]),
        exact_xy=(exact_point[0], exact_point[2]) if exact_point is not None else None,
    )
    fig_v.colorbar(im_v, ax=ax_v, label="full-dataset cost", shrink=0.8)
    fig_v.suptitle(minimum_str)
    fig_v.tight_layout()
    if save_fig_folder:
        plt.savefig(os.path.join(save_fig_folder, f"{prefix}cost_grid_sScrit_slice_{tag}.png"),
                    dpi=150, bbox_inches="tight")
    plt.show()

    # (2) HORIZONTAL slice: (s, d) plane at S_crit FIXED to the
    # equilibrium's own S_crit EXACTLY — same axes as the 4-panel figure
    # above, but guaranteed to pass through the actual minimum (the
    # generic evenly-spaced slices above may only pass near it).
    fig_h, ax_h = plt.subplots(figsize=(6, 5))
    slice_sd_eq = results[:, :, k_eq]
    im_h = _plot_cost_slice(
        ax_h, slice_sd_eq, s_grid, d_grid,
        "s = log10(n*r)  [informative]", "d = log10(n/r)  [near-flat]",
        f"cost(s, d) at S_crit={Scrit_grid[k_eq]:.4g} (through the equilibrium)",
        cmap, vmin, vmax,
        mark_xy=(s_grid[i_eq], d_grid[j_eq]),
        exact_xy=(exact_point[0], exact_point[1]) if exact_point is not None else None,
    )
    fig_h.colorbar(im_h, ax=ax_h, label="full-dataset cost", shrink=0.8)
    fig_h.suptitle(minimum_str)
    fig_h.tight_layout()
    if save_fig_folder:
        plt.savefig(os.path.join(save_fig_folder, f"{prefix}cost_grid_sd_slice_at_eq_{tag}.png"),
                    dpi=150, bbox_inches="tight")
    plt.show()

    # Interactive 3-D volume (Plotly) — (s, d, S_crit) axes. `S_crit` is plotted
    # LINEARLY (unlike the old log10(Tg) axis) since it's already the
    # natural/linear scale for this variable — see the module-level
    # growth-timescale note above `_Scrit_box`.
    try:
        s_flat, d_flat, n_flat, r_flat, Scrit_flat, tg_flat, cost_flat = flat_arr.T
        fig3d = go.Figure(data=go.Volume(
            x=s_flat, y=d_flat, z=Scrit_flat, value=cost_flat,
            isomin=float(cost_flat.min()), isomax=float(cost_flat.max()),
            opacity=0.4, opacityscale="min",
            surface_count=21,
            colorscale="Viridis_r",
            colorbar=dict(title="cost"),
            caps=dict(x_show=False, y_show=False, z_show=False),
        ))
        min_idx_flat = np.argmin(cost_flat)
        fig3d.add_trace(go.Scatter3d(
            x=[s_flat[min_idx_flat]], y=[d_flat[min_idx_flat]], z=[Scrit_flat[min_idx_flat]],
            mode="markers",
            marker=dict(size=6, color="red", symbol="circle", line=dict(color="white", width=1)),
            name="minimum",
        ))
        fig3d.update_layout(
            title=f"[{tag}] {species_name or ''} — full-dataset cost over "
                  f"(s=log10(n*r), d=log10(n/r), S_crit)<br>{minimum_str}",
            scene=dict(
                xaxis_title="s = log10(n*r)",
                yaxis_title="d = log10(n/r)",
                zaxis_title="S_crit",
            ),
            width=900, height=700,
        )
        if save_fig_folder:
            fig3d.write_html(os.path.join(save_fig_folder, f"{prefix}cost_grid_sd_3d_{tag}.html"))
        fig3d.show()
    except Exception as e:
        print(f"[grid:{tag}] Skipped 3-D volume ({e}).")

    return {
        "results": results, "flat": flat_arr,
        "s_grid": s_grid, "d_grid": d_grid, "Scrit_grid": Scrit_grid,
        "best": best, "best_Scrit": best_Scrit, "best_sd": (best_s, best_d), "best_cost": best_cost,
    }


def _run_grid(
    calibration_sites,
    hmean: float,
    mdd: float,
    posteriors_and_masks: tuple,
    carrying_capacity_params: tuple,
    priors: list | None,
    points_per_axis: int,
    grid_n_range: tuple | None,
    grid_r_range: tuple | None,
    grid_tg_range: tuple | None,
    save_fig_folder: str | None = None,
    species_name: str | None = None,
    r_min: float | None = None,
    r_survival_tol: float = 0.005,
    r_survival_alpha: float = 1.11,
    seed_fraction: float = 0.25,
    s_halfwidth: float = 3.0,
    d_max_r_survival_tol: float = 1e-12,
) -> tuple:
    """Grid-scan entry point (self-contained precompute, like
    `refine_from_point`). A single scan over an exact custom (n, r, Tg) box
    (if any `grid_*_range` given — sampled the old way, independently in
    (log10 n, log10 r), since an explicit custom n/r range doesn't
    naturally translate into (s, d) units), or otherwise a scan directly
    in the (s, d) reparametrisation of (n, r) over the full prior box —
    see the module-level note above `_hs_local_contrast`.

    `seed_fraction`: fraction of pixels seeded per site's sparse initial
    density mask, drawn ONCE per site at the start of this run and FIXED
    for every grid point thereafter (not redrawn per evaluation).
    """
    L, k, x0 = carrying_capacity_params

    if priors is None:
        C    = (2.0 * np.exp(-1.11 / mdd)) / (1.0 + np.exp(-2.0 * 1.11 / mdd))
        rmax = (1.0 / np.log(hmean)) * np.log(C)
        if r_min is None:
            r_min = _find_r_min(hmean, r_survival_alpha, mdd, tol=r_survival_tol)
            print(f"[priors] Auto r_min={r_min:.6g} (survival within "
                  f"{r_survival_tol*100:.2g}pp of its r->0 limit below this).")
        log_rmin, log_rmax = np.log10(r_min), np.log10(rmax)
        hs_bar, delta_h_bar = _hs_local_contrast(calibration_sites)
        s_center = _s_center(hs_bar, delta_h_bar)
        smin, smax = _s_range(s_center, s_halfwidth)
        nmin, nmax = _n_box_from_s_and_r(smin, smax, log_rmin, log_rmax)
        priors = [(nmin, nmax), (r_min, rmax), (-2.0, 2.0)]
        print(f"[priors] r range for this species: r_min={r_min:.6g}  "
              f"r_max={rmax:.6g}  (hmean={hmean:.4g}, mdd={mdd:.4g})")

    (nmin, nmax), (rmin, rmax), (_tgmin_unused, _tgmax_unused) = priors
    n_box  = (10.0 ** nmin, 10.0 ** nmax)
    r_box  = (rmin, rmax)
    log_rmin, log_rmax = np.log10(rmin), np.log10(rmax)

    # d_max EXTENSION via a SEPARATE, much smaller r floor (see
    # `learn_dispersal_parameters` for the full derivation).
    r_min_for_dmax = _find_r_min(hmean, r_survival_alpha, mdd, tol=d_max_r_survival_tol)
    log_rmin_for_d = np.log10(r_min_for_dmax)

    # Printed in this exact order (r -> s -> n -> d), see
    # `learn_dispersal_parameters` for why. Computed unconditionally (used
    # by the non-custom scan below, but shown regardless so a `custom`
    # scan's caller still sees how n/d relate to the underlying priors).
    hs_bar, delta_h_bar = _hs_local_contrast(calibration_sites)
    s_center = _s_center(hs_bar, delta_h_bar)
    (smin, smax), (dmin, dmax_conservative) = _sd_box(nmin, nmax, log_rmin, log_rmax, s_center, s_halfwidth)
    # d_max is now recomputed using `log_rmin_for_d` (the SEPARATE,
    # extremely small r floor from `d_max_r_survival_tol`, computed above)
    # instead of the moderate `log_rmin` — extends d_max far out for
    # species whose true optimum needs near-zero dispersal mortality,
    # while `dmin`/`dmax_conservative` (computed with the moderate r
    # floor) are kept around for the default init point's box centre
    # (see below) and for reference/printing. The corner achieving d_max
    # is (log_n=n_max, log_r=log_rmin_for_d): d increases AND s decreases
    # there relative to the conservative corner (s = n_max + log_rmin_for_d
    # < n_max + log_rmin), i.e. the newly reachable region skews toward
    # the UPPER-LEFT (low s, high d) — the opposite of widening n_max,
    # which would skew the same corner toward the upper-RIGHT.
    dmax = nmax - log_rmin_for_d
    print(f"[priors] s prior for this species: hs_bar={hs_bar:.4g}  "
          f"delta_h_bar={delta_h_bar:.4g}  s*=log10(n*r)={s_center:.4g}  "
          f"s box=[{smin:.4g}, {smax:.4g}]  (s_halfwidth={s_halfwidth:.3g})")
    print(f"[priors] n range for this species (data-driven from s box + "
          f"r box): n_min={10**nmin:.4g}  n_max={10**nmax:.4g}  "
          f"(log10: [{nmin:.4g}, {nmax:.4g}])")
    print(f"[priors] d prior for this species (from n box + r box): "
          f"conservative d box=[{dmin:.4g}, {dmax_conservative:.4g}] "
          f"(r_survival_tol={r_survival_tol:.3g}, used for the default init "
          f"point's box centre)  ->  LEARNING d box=[{dmin:.4g}, {dmax:.4g}] "
          f"(d_max extended via d_max_r_survival_tol={d_max_r_survival_tol:.3g})")

    # Tg range for the grid, derived from the SAME `S_crit` box used by
    # batchSGD/refine_from_point/run_mala (see `_Scrit_box` — a fixed
    # universal (0.5, 1.0) box, no species-specific precompute needed) —
    # not the old fixed `priors` Tg slot (`tgmin`/`tgmax`, unused, kept
    # only for the priors tuple's shape) — so a grid scan explores exactly
    # the region the other methods can actually reach, instead of a
    # stale/arbitrary [0.01, 100] range. Both `S_crit=0.5` (g->1, Tg->0)
    # and `S_crit=1.0` (g->0, Tg->inf) are singular endpoints, so both are
    # nudged inward by a small margin before converting to a Tg scan range.
    scrit_min, scrit_max = _Scrit_box()
    scrit_lo = scrit_min + 1e-3 * (scrit_max - scrit_min)
    scrit_hi = scrit_max - 1e-6
    # `_scan_cost_grid` (the legacy Tg-axis grid diagnostic) is still fed a
    # Tg-space range here — Tg is undefined for S_crit <= 0.5, which is
    # now reachable, so clamp the LOWER bound used for this conversion
    # ONLY, well above 0.5, before converting (this is purely this one
    # legacy tool's own fallback default range, not the actual sampling
    # box elsewhere).
    scrit_lo_tg_safe = max(scrit_lo, 0.5 + 1e-3)
    tg_box = (_Scrit_to_Tg(scrit_lo_tg_safe), _Scrit_to_Tg(scrit_hi))
    print(f"[priors] S_crit (survival-threshold) reparametrisation for "
          f"this species: S_crit box=[{scrit_min:.4g}, {scrit_max:.4g}]  "
          f"(interpretation: S_crit = critical per-dispersal-step survival "
          f"probability below which NO growth rate can sustain a local "
          f"population — S_crit=0.5 is the g->1 (fastest possible growth, "
          f"Tg->0) limit, S_crit->1 is the g->0 (Tg->inf) limit)")

    print("[precompute] Building adjacency matrices and K_is ...")
    n_sites_total = len(calibration_sites)
    adj_mats, K_is_list, seed_masks_list = [], [], []
    for hs_map in tqdm(calibration_sites.hs_maps, desc="  Sites", leave=False):
        hs_t = torch.tensor(hs_map, dtype=torch.float32, device=device)
        adj_mats.append(adjacency_matrix_torch(hs_t))
        K_is_flat = torch.tensor(
            _logistic(hs_map.flatten().astype(np.float32), L, k, x0),
            dtype=torch.float32,
        )
        K_is_list.append(K_is_flat)
        seed_masks_list.append(seed_mask(
            K_is_flat.shape, seed_fraction=seed_fraction,
            device=K_is_flat.device, dtype=K_is_flat.dtype,
        ))
    print(f"[precompute] Done — {n_sites_total} sites.")
    breeding_masks_list = _build_breeding_masks_list(calibration_sites, n_sites_total)
    print(f"[precompute] Seeded initial densities: {seed_fraction*100:.0f}% of "
          f"pixels per site, FIXED for the whole run (not redrawn per step).")

    custom = grid_n_range is not None or grid_r_range is not None or grid_tg_range is not None
    scan_kwargs = dict(
        mdd=mdd, posteriors_and_masks=posteriors_and_masks,
        calibration_sites=calibration_sites,
        carrying_capacity_params=carrying_capacity_params, hmean=hmean,
        adj_mats=adj_mats, K_is_list=K_is_list, n_sites_total=n_sites_total,
        points_per_axis=points_per_axis,
        save_fig_folder=save_fig_folder, species_name=species_name,
        seed_masks_list=seed_masks_list, breeding_masks_list=breeding_masks_list,
    )

    if custom:
        info = _scan_cost_grid(
            n_range=grid_n_range or n_box,
            r_range=grid_r_range or r_box,
            tg_range=grid_tg_range or tg_box,
            tag="custom", **scan_kwargs,
        )
    else:
        info = _scan_cost_grid_sd(
            s_range=(smin, smax), d_range=(dmin, dmax), Scrit_range=(scrit_min, scrit_max),
            n_log_range=(nmin, nmax), r_log_range=(log_rmin_for_d, log_rmax),
            tag="single", **scan_kwargs,
        )

    n_best, r_best, Scrit_best = info["best"]
    C = (2.0 * np.exp(-1.11 / mdd)) / (1.0 + np.exp(-2.0 * 1.11 / mdd))
    estimated_ew = float(C / (hmean ** r_best - C))

    return (estimated_ew, n_best, r_best, Scrit_best, [], np.ones(1),
            [np.array([r_best]), np.array([n_best]), np.array([Scrit_best])], info)


def _run_precise_gridscan(
    calibration_sites,
    hmean: float,
    mdd: float,
    posteriors_and_masks: tuple,
    carrying_capacity_params: tuple,
    center: tuple,
    s_window: float = 1.0,
    d_window: float = 1.0,
    Scrit_window: float = 0.1,
    points_per_axis: int = 10,
    save_fig_folder: str | None = None,
    species_name: str | None = None,
    seed_fraction: float = 0.25,
) -> tuple:
    """Local, ZOOMED-IN check of the cost surface around a chosen
    `(s, d, S_crit)` point — typically `refine_from_point`'s converged
    endpoint — to visually inspect its local shape (a clean bowl vs
    something odd) as a complementary, independent sanity check to
    `refine_from_point`'s Hessian-based verdict (see
    `_hessian_at_point_sd`): the Hessian only looks at curvature through
    a fixed finite-difference `eps`, whereas this shows the ACTUAL cost
    landscape over a small but genuinely resolved window around the
    point, so a sharp/narrow feature the Hessian's `eps` might step over
    (or a spurious flat patch) becomes directly visible.

    UNLIKE `_run_grid`/the general grid-scan machinery, this does NOT
    compute a full 3-D `points_per_axis`^3 cube — only the TWO 2-D
    CROSS-SECTIONS through `center` that matter for inspecting it:
    - a HORIZONTAL `(s, d)` slice, at `S_crit` held EXACTLY at `Scrit_c`
      (not grid-snapped — the fixed coordinate is the literal input
      value, so this slice passes through `center` exactly by
      construction);
    - a VERTICAL `(s, S_crit)` slice, at `d` held EXACTLY at `d_c`.
    This costs `2 * points_per_axis**2` `cost_function` evaluations
    instead of `points_per_axis**3` — e.g. 200 instead of 1000 at the
    default `points_per_axis=10` — an order-of-magnitude cheaper, at the
    price of not seeing the full 3-D volume (only these two orthogonal
    cuts through it, which is exactly what's needed to check the point's
    local shape, not a general-purpose exploration).

    Scan window: `s in [s_c - s_window, s_c + s_window]`,
    `d in [d_c - d_window, d_c + d_window]`,
    `S_crit in [Scrit_c - Scrit_window, Scrit_c + Scrit_window]`
    (clamped into `S_crit`'s own `(0.5, 1.0)` box, both endpoints
    singular). Default window sizes (`s_window=1.0`, `d_window=1.0`,
    `Scrit_window=0.1`) combined with the default `points_per_axis=10`
    give each slice a MUCH finer resolution per unit than a full-box scan
    would, since the window itself is far smaller than the full prior box.

    Only the ONE hard physical boundary (`r > r_max`, see `_survival`'s
    docstring — matching the package-wide policy, see the module-level
    note above `_scan_cost_grid_sd`) is checked per point; points beyond
    it are skipped (marked NaN, greyed out) rather than computed.

    Parameters
    ----------
    center:
        `(s_c, d_c, Scrit_c)` — the point to zoom in on, e.g.
        `refine_from_point`'s `(s_final, d_final, Scrit_final)`.
    s_window, d_window, Scrit_window:
        Half-widths of the local scan window around `center`, in each
        axis's own units.
    points_per_axis:
        Resolution per axis, per slice (default 10, i.e. 100 points per
        slice, 200 total).

    Returns
    -------
    Same 8-tuple convention as `_run_grid`:
    ``(Ew, n, r, Tg, [], np.ones(1), [r_vals, n_vals, Tg_vals], info)``.
    """
    L, k, x0 = carrying_capacity_params
    s_c, d_c, Scrit_c = center
    SCRIT_MIN, SCRIT_MAX = _Scrit_box()
    Scrit_c_safe = min(max(Scrit_c, SCRIT_MIN + 1e-6), SCRIT_MAX - 1e-6)
    s_range     = (s_c - s_window, s_c + s_window)
    d_range     = (d_c - d_window, d_c + d_window)
    Scrit_range = (max(SCRIT_MIN + 1e-6, Scrit_c - Scrit_window),
                   min(SCRIT_MAX - 1e-6, Scrit_c + Scrit_window))
    print(f"\033[96m[precise_gridscan] Local 2-slice scan centred at "
          f"s={s_c:.4g}  d={d_c:.4g}  S_crit={Scrit_c:.4g}:\033[0m")
    print(f"  s box=[{s_range[0]:.4g}, {s_range[1]:.4g}]  "
          f"(s_window={s_window:.3g})")
    print(f"  d box=[{d_range[0]:.4g}, {d_range[1]:.4g}]  "
          f"(d_window={d_window:.3g})")
    print(f"  S_crit box=[{Scrit_range[0]:.4g}, {Scrit_range[1]:.4g}]  "
          f"(S_crit_window={Scrit_window:.3g})")
    print(f"  2 x {points_per_axis}^2 = {2 * points_per_axis**2} points "
          f"(horizontal (s,d) @ S_crit={Scrit_c:.4g}, "
          f"vertical (s,S_crit) @ d={d_c:.4g}) — NOT the full "
          f"{points_per_axis}^3 = {points_per_axis**3}-point cube.")

    print("[precompute] Building adjacency matrices and K_is ...")
    n_sites_total = len(calibration_sites)
    adj_mats, K_is_list, seed_masks_list = [], [], []
    for hs_map in tqdm(calibration_sites.hs_maps, desc="  Sites", leave=False):
        hs_t = torch.tensor(hs_map, dtype=torch.float32, device=device)
        adj_mats.append(adjacency_matrix_torch(hs_t))
        K_is_flat = torch.tensor(
            _logistic(hs_map.flatten().astype(np.float32), L, k, x0),
            dtype=torch.float32,
        )
        K_is_list.append(K_is_flat)
        seed_masks_list.append(seed_mask(
            K_is_flat.shape, seed_fraction=seed_fraction,
            device=K_is_flat.device, dtype=K_is_flat.dtype,
        ))
    print(f"[precompute] Done — {n_sites_total} sites.")
    breeding_masks_list = _build_breeding_masks_list(calibration_sites, n_sites_total)

    s_grid     = np.linspace(s_range[0], s_range[1], points_per_axis)
    d_grid     = np.linspace(d_range[0], d_range[1], points_per_axis)
    Scrit_grid = np.linspace(Scrit_range[0], Scrit_range[1], points_per_axis)
    all_indices = list(range(n_sites_total))

    C_const = (2.0 * np.exp(-1.11 / mdd)) / (1.0 + np.exp(-2.0 * 1.11 / mdd))
    rmax = (1.0 / np.log(hmean)) * np.log(C_const)
    log_rmax = np.log10(rmax)

    def _eval(s_val: float, d_val: float, Scrit_val: float):
        """One cost_function evaluation at (s_val, d_val, Scrit_val), or
        NaN if r > r_max (the one hard physical boundary — see the
        module-level note above `_scan_cost_grid_sd`)."""
        log_n_val, log_r_val = _sd_to_logn_logr(s_val, d_val)
        if log_r_val > log_rmax:
            return float("nan")
        n_val = 10.0 ** log_n_val
        r_val = 10.0 ** log_r_val
        Scrit_safe = min(max(Scrit_val, SCRIT_MIN + 1e-6), SCRIT_MAX - 1e-6)
        n_p  = torch.tensor(float(n_val),  device=device, dtype=torch.float32)
        r_p  = torch.tensor(float(r_val),  device=device, dtype=torch.float32)
        scrit_p = torch.tensor(float(Scrit_safe), device=device, dtype=torch.float32)
        with torch.no_grad():
            c = cost_function(
                mdd, posteriors_and_masks, calibration_sites,
                (n_p, r_p, scrit_p), all_indices,
                carrying_capacity_params, hmean, adj_mats, K_is_list,
                plot=False, verbose=False, seed_masks_list=seed_masks_list, breeding_masks_list=breeding_masks_list,
            )
        return float(c.item())

    # HORIZONTAL slice: (s, d) plane at S_crit held EXACTLY at Scrit_c.
    slice_sd = np.full((points_per_axis, points_per_axis), np.nan)
    pbar_sd = tqdm(total=points_per_axis ** 2, desc="  Precise[s,d]", leave=False)
    for i, s_val in enumerate(s_grid):
        for j, d_val in enumerate(d_grid):
            slice_sd[i, j] = _eval(s_val, d_val, Scrit_c)
            pbar_sd.update(1)
    pbar_sd.close()

    # VERTICAL slice: (s, S_crit) plane at d held EXACTLY at d_c.
    slice_sScrit = np.full((points_per_axis, points_per_axis), np.nan)
    pbar_sc = tqdm(total=points_per_axis ** 2, desc="  Precise[s,S_crit]", leave=False)
    for i, s_val in enumerate(s_grid):
        for kk, Scrit_val in enumerate(Scrit_grid):
            slice_sScrit[i, kk] = _eval(s_val, d_c, Scrit_val)
            pbar_sc.update(1)
    pbar_sc.close()

    prefix = f"{species_name}_" if species_name else ""
    if save_fig_folder:
        np.save(os.path.join(save_fig_folder, f"{prefix}precise_slice_sd.npy"), slice_sd)
        np.save(os.path.join(save_fig_folder, f"{prefix}precise_slice_sScrit.npy"), slice_sScrit)

    # Overall minimum across BOTH slices (whichever is lower) — the two
    # slices only share the one grid point at (s_c-ish, d_c/Scrit_c), so
    # comparing their minima directly is meaningful.
    i_sd, j_sd = np.unravel_index(np.nanargmin(slice_sd), slice_sd.shape)
    cost_sd_min = float(slice_sd[i_sd, j_sd])
    i_sc, k_sc = np.unravel_index(np.nanargmin(slice_sScrit), slice_sScrit.shape)
    cost_sc_min = float(slice_sScrit[i_sc, k_sc])
    if cost_sd_min <= cost_sc_min:
        best_s, best_d, best_Scrit, best_cost = (
            float(s_grid[i_sd]), float(d_grid[j_sd]), Scrit_c_safe, cost_sd_min)
    else:
        best_s, best_d, best_Scrit, best_cost = (
            float(s_grid[i_sc]), d_c, float(Scrit_grid[k_sc]), cost_sc_min)
    best_log_n, best_log_r = _sd_to_logn_logr(best_s, best_d)
    n_best, r_best = 10.0 ** best_log_n, 10.0 ** best_log_r
    estimated_ew = float(C_const / (hmean ** r_best - C_const))
    minimum_str = (f"[precise_gridscan] Minimum: cost={best_cost:.5f}  "
                    f"s={best_s:.4g}  d={best_d:.4g}  "
                    f"n={n_best:.4g}  r={r_best:.6g}  S_crit={best_Scrit:.4g}")
    print(f"\033[92m{minimum_str}\033[0m")

    vmin = float(np.nanmin([np.nanmin(slice_sd), np.nanmin(slice_sScrit)]))
    vmax = float(np.nanmax([np.nanmax(slice_sd), np.nanmax(slice_sScrit)]))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("lightgrey")

    fig_h, ax_h = plt.subplots(figsize=(6, 5))
    im_h = _plot_cost_slice(
        ax_h, slice_sd, s_grid, d_grid,
        "s = log10(n*r)  [informative]", "d = log10(n/r)  [near-flat]",
        f"cost(s, d) at S_crit={Scrit_c:.4g} (exact — through the equilibrium)",
        cmap, vmin, vmax,
        mark_xy=(s_grid[i_sd], d_grid[j_sd]), exact_xy=(s_c, d_c),
    )
    fig_h.colorbar(im_h, ax=ax_h, label="full-dataset cost", shrink=0.8)
    fig_h.suptitle(minimum_str)
    fig_h.tight_layout()
    if save_fig_folder:
        plt.savefig(os.path.join(save_fig_folder, f"{prefix}precise_slice_sd.png"),
                    dpi=150, bbox_inches="tight")
    plt.show()

    fig_v, ax_v = plt.subplots(figsize=(6, 5))
    im_v = _plot_cost_slice(
        ax_v, slice_sScrit, s_grid, Scrit_grid,
        "s = log10(n*r)  [informative]", "S_crit (survival threshold)",
        f"cost(s, S_crit) at d={d_c:.4g} (exact — through the equilibrium)",
        cmap, vmin, vmax,
        mark_xy=(s_grid[i_sc], Scrit_grid[k_sc]), exact_xy=(s_c, Scrit_c),
    )
    fig_v.colorbar(im_v, ax=ax_v, label="full-dataset cost", shrink=0.8)
    fig_v.suptitle(minimum_str)
    fig_v.tight_layout()
    if save_fig_folder:
        plt.savefig(os.path.join(save_fig_folder, f"{prefix}precise_slice_sScrit.png"),
                    dpi=150, bbox_inches="tight")
    plt.show()

    info = {
        "slice_sd": slice_sd, "slice_sScrit": slice_sScrit,
        "s_grid": s_grid, "d_grid": d_grid, "Scrit_grid": Scrit_grid,
        "best_cost": best_cost, "best_sd": (best_s, best_d), "best_Scrit": best_Scrit,
    }
    return (estimated_ew, n_best, r_best, best_Scrit, [], np.ones(1),
            [np.array([r_best]), np.array([n_best]), np.array([best_Scrit])], info)


# ---------------------------------------------------------------------------
# 3-D KDE endpoint visualisation (Plotly)
# ---------------------------------------------------------------------------

def _kde_3d_plotly(
    r_ends: np.ndarray,
    n_ends: np.ndarray,
    Scrit_ends: np.ndarray,
    priors: list,
    list_traj: list,
    weights: np.ndarray,
    sizes_scatter,
    learning_runs: list,
    all_together: bool,
    grid_size: int = 30,
):
    """Adaptive 3-D KDE of the endpoint cloud (Plotly Volume + trajectories).

    The z-axis is `S_crit` — the actual learned/sampled critical
    per-dispersal-step survival probability variable (see the
    module-level growth-timescale note) — not `Tg`.
    """
    (nmin, nmax), (rmin, rmax), (_tgmin_unused, _tgmax_unused) = priors
    # n is learned (and its priors' slot given) directly in log10 space —
    # nmin/nmax already ARE log10(n) bounds. rmin/rmax are actual r values
    # though (see learn_dispersal_parameters' priors docstring), so log10
    # them here for the log-scaled axis below. `S_crit`'s own box isn't
    # carried in `priors` (it's data-driven via `_Scrit_box`, not part of
    # the (n, r, Tg) priors list) — its axis range below is instead set
    # from the actual Scrit_ends data.
    log_rmin, log_rmax = np.log10(rmin), np.log10(rmax)

    # Trajectories
    data_lines = []
    for ln, lr, lScrit in zip(list_traj[0], list_traj[1], list_traj[2]):
        data_lines.append(go.Scatter3d(
            x=lr, y=ln, z=lScrit,
            mode="lines", line=dict(color="lightgrey", width=1),
        ))

    # Endpoint scatter
    pts = go.Scatter3d(
        x=r_ends, y=n_ends, z=Scrit_ends,
        mode="markers+text",
        marker=dict(size=sizes_scatter, color="red"),
        text=[str(lr if not all_together else i)
              for i, lr in enumerate(learning_runs)],
        textposition="middle right",
        textfont=dict(color="black", size=8),
        name="Endpoints",
    )

    if not all_together and len(r_ends) > 1:
        # Adaptive KDE — n and r computed in LOG10 space (matching how
        # they're learned/parametrized), but S_crit is learned LINEARLY
        # (tanh-bounded directly to the fixed (0.5, 1.0) box, see
        # `_Scrit_box`), so it's kept linear here too rather than
        # log-transformed — S_crit's own scale is already bounded and
        # roughly uniform in linear units, with no near-0 values to worry
        # about (unlike the old `ell`).
        log_r, log_n = np.log10(r_ends), np.log10(n_ends)
        xyz  = np.vstack([log_r, log_n, Scrit_ends]).T
        sigma = np.std(xyz, axis=0) + 1e-9
        bw    = len(xyz) ** (-1.0 / (3 + 4))
        kde   = KernelDensity(bandwidth=bw, kernel="gaussian")
        kde.fit(xyz / sigma, sample_weight=weights)

        xg = np.linspace(log_r.min(), log_r.max(), grid_size)
        yg = np.linspace(log_n.min(), log_n.max(), grid_size)
        zg = np.linspace(Scrit_ends.min(), Scrit_ends.max(), grid_size)
        X, Y, Z = np.meshgrid(xg, yg, zg)
        gp      = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
        density = np.exp(kde.score_samples(gp / sigma))
        mode    = gp[np.argmax(density)]

        # x/y grid points are in log10 space (matching sigma/kde above);
        # the scene's x/y axes are log-TYPE (see xaxis/yaxis below), which
        # in Plotly expect actual (un-logged) data values, not their logs —
        # so exponentiate back here. z (S_crit) is already linear, both in the
        # KDE grid and in the (linear-type) zaxis below, so it's used as-is.
        vol = go.Volume(
            x=10.0 ** gp[:, 0], y=10.0 ** gp[:, 1], z=gp[:, 2],
            value=density,
            isomin=np.percentile(density, 70),
            isomax=density.max(),
            opacity=0.15,
            surface_count=20,
            caps=dict(x_show=False, y_show=False, z_show=False),
        )
        mode_pt = go.Scatter3d(
            x=[10.0 ** mode[0]], y=[10.0 ** mode[1]], z=[mode[2]],
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
            zaxis_title="S_crit",
            # r and n are shown on LOG axes (matching their log-space
            # learning): for a Plotly log-type axis, `range` is given in
            # log10 units directly — which is exactly what
            # log_rmin/log_rmax/nmin/nmax already are. S_crit is shown on a
            # LINEAR axis instead, ranged to the actual endpoint data (its
            # box, from `_Scrit_box`, is a fixed universal (0.5, 1.0)
            # constant, not carried in `priors`, unlike the old fixed
            # [0, 10] Tg range).
            xaxis=dict(type="log", range=[log_rmin, log_rmax]),
            yaxis=dict(type="log", range=[nmin, nmax]),
            zaxis=dict(type="linear", range=[
                float(Scrit_ends.min()) - 1e-6, float(Scrit_ends.max()) + 1e-6,
            ]),
            aspectmode="cube",
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
