"""Population growth model.

The growth model uses a discrete logistic-type recursion::

    U_{t+1} = U_t + (1 - a) * U_t * (1 - U_t / K)

where ``a = 0.05^{1/Tg}`` is the linear growth coefficient and ``K`` is
the per-pixel carrying capacity inferred from the habitat-suitability map.

Key functions
-------------
:func:`growth_coefficient`
    Compute the scalar growth coefficient ``a`` from the characteristic
    growth time ``Tg``.
:func:`carrying_capacity_from_hs`
    Compute per-pixel carrying capacities using a fitted logistic
    relationship between HS and relative abundance.
:func:`equilibrium_distribution`
    Iterate the growth–dispersal loop to approximate the stationary
    distribution.
:func:`growth_step`
    Apply a single growth step to a 2-D abundance map.

Classes
-------
:class:`GrowthModel`
    Object-oriented wrapper around the growth functions.
"""

from __future__ import annotations

import numpy as np
import torch
import matplotlib.pyplot as plt

from paradis._device import device


def growth_coefficient(tgrowth: float) -> float:
    """Return the linear growth coefficient ``a = 0.05^{1/Tg}``.

    Parameters
    ----------
    tgrowth:
        Characteristic growth time. ``a`` is DEFINED so that, if the
        relative distance to carrying capacity ``eps_t = 1 - U_t/K``
        decayed by a constant factor ``a`` every step (``eps_{t+1} =
        a*eps_t``), it would reach ``eps_Tg = a^Tg = 0.05`` (5 % short of
        ``K``, i.e. 95 % of ``K``) after ``Tg`` steps — hence
        ``a = 0.05**(1/Tg)``.

        This is NOT literally "the time for a population starting at
        near-zero density to reach 95 % of K": the EXACT recursion for
        ``eps_t`` under this model (substitute ``U_t = K*(1-eps_t)`` into
        ``U_{t+1} = U_t + (1-a)*U_t*(1-U_t/K)`` and simplify) is

            eps_{t+1} = a*eps_t + (1-a)*eps_t**2

        which is QUADRATIC in ``eps_t``, not linear — the simple
        ``eps_t = a**t`` relation used to define ``a`` above is only the
        exact behaviour in the small-``eps`` limit (i.e. NEAR ``K``,
        where the quadratic term is negligible), not for ``eps`` near 1
        (i.e. ``U`` near 0). In fact ``U = 0`` is itself a fixed point of
        the exact recursion (``eps_{t+1} = a*1 + (1-a)*1 = 1`` when
        ``eps_t = 1``) — a population starting EXACTLY at zero density
        never grows at all under this deterministic map, regardless of
        ``Tg`` (only dispersal bringing in mass from elsewhere can start
        it moving — see `equilibrium_distribution`'s seeding machinery).

        So ``Tg`` is better understood as the model's characteristic
        RELAXATION time near the equilibrium ``K`` (the timescale that
        genuinely governs ``eps_{t+1} approx a*eps_t`` once a trajectory
        is already close to ``K``), not a literal near-zero-to-95%
        travel time — the "starting near zero" framing is a convenient
        but not strictly accurate way to motivate the ``0.05`` constant.

    Returns
    -------
    float
    """
    return 0.05 ** (1.0 / tgrowth)


def _logistic(x: np.ndarray | torch.Tensor, L: float, k: float, x0: float):
    """Logistic curve shifted so that ``f(0) = 0``.

    ``f(x) = L / (1 + exp(k*(x - x0))) - L / (1 + exp(k*(0 - x0)))``

    Parameters
    ----------
    x:
        Input values (HS scores in ``[0, 1]``).
    L, k, x0:
        Shape, slope, and inflection-point parameters.

    Returns
    -------
    numpy.ndarray or torch.Tensor
        Carrying-capacity values with the same type as *x*.
    """
    def g(t):
        if isinstance(t, torch.Tensor):
            z = torch.clamp(k * (t - x0), -700, 700)
            return L / (1.0 + torch.exp(z))
        else:
            z = np.clip(k * (t - x0), -700, 700)
            return L / (1.0 + np.exp(z))

    return g(x) - g(0)


def carrying_capacity_from_hs(
    hs: np.ndarray | torch.Tensor,
    tgrowth: float,
    L: float,
    k: float,
    x0: float,
) -> tuple:
    """Compute growth parameters from a habitat-suitability map.

    Parameters
    ----------
    hs:
        2-D or flat array of HS values in ``[0, 1]``.
    tgrowth:
        Characteristic growth time.
    L, k, x0:
        Logistic parameters for the K(HS) relationship.

    Returns
    -------
    a : float
        Linear growth coefficient.
    K_is : torch.Tensor
        Per-pixel carrying capacities (flat).
    vecb : torch.Tensor
        Per-pixel growth intercepts ``K_i * (1 - a)``.
    """
    if not isinstance(hs, torch.Tensor):
        hs = torch.tensor(hs, dtype=torch.float32, device="cpu")
    hs_flat = hs.flatten().float()
    K_is = _logistic(hs_flat, L, k, x0)
    K_is = torch.tensor(np.array(K_is), dtype=torch.float32) if isinstance(K_is, np.ndarray) else K_is.float()
    a = growth_coefficient(tgrowth)
    vecb = K_is * (1.0 - a)
    return a, K_is, vecb


def seed_mask(shape, seed_fraction: float = 0.25, device=None, dtype=None) -> torch.Tensor:
    """Draw a fresh Bernoulli seeding mask, ``True`` on a random fraction of pixels.

    Shared by :func:`equilibrium_distribution` (to build the sparse starting
    density, see its docstring) and by the calibration-grid visualisation in
    ``paradis.visualization.plots`` (to illustrate what that seeding looks
    like on a real HS map) — factored out here so both call sites draw the
    mask identically instead of duplicating the ``torch.rand(...) <
    seed_fraction`` logic.

    Parameters
    ----------
    shape:
        Shape of the mask to draw (e.g. ``init_distrib.shape``).
    seed_fraction:
        Fraction of pixels set to ``True`` (default 0.25 = 25 %).
    device, dtype:
        Passed to ``torch.rand`` — should match the tensor the mask will be
        multiplied against.

    Returns
    -------
    torch.Tensor
        Boolean mask, freshly and independently drawn on every call (no
        fixed seed, no caching) — callers that need reproducibility should
        seed the global RNG themselves before calling.
    """
    return torch.rand(shape, device=device, dtype=dtype) < seed_fraction


def equilibrium_distribution(
    init_distrib: torch.Tensor,
    kernel: torch.Tensor,
    linear_growth: float | torch.Tensor,
    n_iter: int = 10,
    plot: bool = False,
    verbose: bool = False,
    return_history: bool = False,
    adaptive: bool = False,
    convergence_ratio_tol: float = 0.0001,
    max_iter: int = 500,
    progress_callback=None,
    debug: bool = False,
    random_seed_init: bool = True,
    seed_fraction: float = 0.25,
    seed_mask_override: torch.Tensor | None = None,
    breeding_ground: torch.Tensor | None = None,
) -> torch.Tensor:
    """Approximate the stationary distribution by iterating dispersal + growth.

    By default (``adaptive=False``) this runs a FIXED number of iterations
    (`n_iter`, default 10) with no convergence check/early stop. The
    per-iteration growth coefficient is `1 - linear_growth` — since
    `linear_growth = 0.05**(1/Tg)` (see cost_function), this coefficient
    shrinks toward 0 as Tg grows large (e.g. ~0.26 at Tg=10 vs ~0.03 at
    Tg=100), meaning the dynamics move increasingly slowly per iteration.
    For large enough Tg, `n_iter=10` steps may stop well before the true
    steady state is reached — the returned "equilibrium" can then be
    dominated by `init_distrib` itself rather than genuine
    dispersal-driven dynamics (confirmed empirically: ~0.3-0.4%
    last/first-iteration change ratio at reasonable Tg vs ~4-5% at large
    Tg — an order of magnitude further from convergence).

    Set ``adaptive=True`` to instead iterate (at least `n_iter` times,
    up to `max_iter`) until the change ratio (last iteration's total
    absolute change, divided by the first iteration's) drops below
    `convergence_ratio_tol` — i.e. stop once actually converged, instead
    of after a fixed step count that may or may not be enough depending
    on Tg. This is the surer fix for the artifact above, but changes the
    per-call cost (variable, possibly much larger than `n_iter` for slow
    dynamics) — left opt-in rather than the default to avoid silently
    changing the speed/behaviour of existing training/inference call sites.

    Parameters
    ----------
    init_distrib:
        Flat initial distribution (length ``N``), used also as carrying
        capacity (``K = init_distrib``).
    kernel:
        ``(N, N)`` dispersal transition matrix.
    linear_growth:
        Scalar growth coefficient ``a``.
    n_iter:
        Number of iterations (default 10). With ``adaptive=True``, this is
        instead the MINIMUM number of iterations run before the
        convergence-ratio check is applied.
    plot:
        If ``True``, plot the distribution at each iteration.
    verbose:
        If ``True``, print intermediate statistics.
    return_history:
        If ``True``, return ``(Un, list_changes)`` instead of just ``Un``
        — ``list_changes[i]`` is the total absolute change between
        iteration ``i-1`` and ``i`` (``list_changes[0]`` vs the initial
        distribution), letting you check convergence directly (e.g. is
        ``list_changes[-1]`` still large relative to earlier iterations).
    adaptive:
        If ``True``, keep iterating past `n_iter` (up to `max_iter`) until
        the change ratio drops below `convergence_ratio_tol`.
    convergence_ratio_tol:
        ``adaptive=True``-only. Stop once
        ``list_changes[-1] / list_changes[0] < convergence_ratio_tol``
        (default 0.0001 = 0.01%).
    max_iter:
        ``adaptive=True``-only. Hard cap on iterations, in case the
        dynamics never reach the target ratio (e.g. truly pathological
        parameters) — prevents an unbounded/hanging loop.
    progress_callback:
        Optional ``callable(epoch: int, ratio: float)``, invoked once per
        iteration (``adaptive=True``-only — ``ratio`` is only meaningful
        once at least 2 iterations have run, and is ``nan`` before that).
        Lets a caller live-update a progress bar/postfix with the current
        convergence ratio, without needing the final `list_changes` after
        the whole call returns.
    debug:
        If ``True``, record min/max/mean of every intermediate quantity
        computed each iteration (``growth`` and the final clamped
        ``Un``), plus the dispersal-only mortality of the WHOLE
        population passing through the kernel this iteration
        (``sent`` via ``Un @ kernel`` vs ``arrived``), print a
        one-line summary per iteration, and — at the end — plot their
        trajectories across iterations. Meant for diagnosing *why*
        the equilibrium collapses (or explodes) past some parameter
        threshold: e.g. does ``growth`` go strongly negative everywhere at
        once (density-dependent die-off) or only in a shrinking subset of
        cells (a spreading collapse front)? Also stores the full 2-D
        ``Un`` and ``growth`` maps at every iteration and, at the end,
        displays them as filmstrip grids (one thumbnail per iteration,
        shared colour scale) so the spatial pattern of a collapse/blow-up
        can be inspected directly — independent of ``plot`` (which only
        shows the raw density map one iteration at a time, live, and the
        final total-change curve). If ``random_seed_init`` is also on, the
        realised seed fraction and seeded-pixel count are appended to the
        per-iteration debug print (iteration 0 only) since they directly
        affect how to interpret the rest of the trace.
    random_seed_init:
        If ``True`` (the default), the starting density ``Un(t=0)`` is NOT
        simply the full carrying-capacity map ``init_distrib`` — instead,
        a fresh random ``seed_fraction`` of pixels start at their own local
        ``K_i`` and the remaining pixels start at exactly ``0``. This
        matters because at ``Un=0`` the logistic growth term
        ``(1 - a) * Un * (1 - Un / K)`` is identically ``0`` — a pixel that
        starts empty can ONLY become populated through dispersal from an
        occupied neighbour, never spontaneously via growth. Starting
        instead at ``Un = K`` everywhere (the old, and still available via
        ``random_seed_init=False``, behaviour) makes the growth term
        exactly zero EVERYWHERE at the first iteration too (since
        ``Un/K = 1``), and for fast growth (small `Tg`, `linear_growth`
        near 0) any dispersal-driven deviation from `K` gets "healed" by
        growth almost immediately on later iterations — so the fitted
        equilibrium ends up sitting extremely close to the raw habitat map
        `K` itself, with the loss barely sensitive to the dispersal
        parameters (`n`, `r`) at all. This exactly reproduces a previously
        observed pathological grid-scan result where the optimum collapsed
        to the fastest-growth edge of the `Tg` box with an oddly flat,
        dispersal-insensitive cost landscape. Sparse seeding forces genuine
        dispersal-driven colonisation of the empty 75 % of pixels, so the
        resulting equilibrium — and hence the loss — actually depends on
        how far and how effectively the kernel spreads mass, giving a much
        better-conditioned training signal. The carrying capacity `K` used
        throughout (`carrying_cap`, below) is always the FULL, unchanged
        `init_distrib` regardless of this setting — only the starting
        density is sparse, not the ceiling that growth targets. The mask is
        redrawn fresh (see :func:`seed_mask`) on every single call — no
        fixed seed, no caching across calls — UNLESS `seed_mask_override`
        is supplied (see below), in which case that mask is used verbatim
        instead and this fresh-draw path is skipped entirely. Calibration
        call sites (`paradis.learning.optimizer`) now draw one mask per
        site at the start of the run and pass it in via
        `seed_mask_override` on every subsequent call for that site, so in
        practice this fresh-every-call path is a fallback that is no
        longer exercised by training — kept available for ad-hoc/manual
        use of this function outside the calibration machinery. The
        motivation for fixing the mask per site (rather than redrawing it
        every call, as originally done) is to remove a confound from the
        calibration loss surface: with a fresh draw every call, two
        evaluations of the SAME dispersal parameters at the SAME site
        could land on different masks and hence different equilibria,
        contaminating the loss's parameter-dependence with pure sampling
        noise (noisier gradients, jittery grid scans). Fixing one mask per
        site means every parameter value tried during a given calibration
        run for that site is evaluated against the exact same sparse
        starting pattern, so differences in cost across parameter values
        reflect the parameters, not a different random draw — while still
        forcing genuine dispersal-driven colonisation, since large
        contiguous chunks of the site start at exactly ``0`` and stay
        unreachable by growth alone (see the paragraph above on why
        ``Un=0`` pixels can't grow without dispersal — that reasoning is
        unaffected by whether the mask is fixed or fresh). Different sites
        still get their own independently drawn masks — only fixed WITHIN
        a site's run, not shared ACROSS sites.
    seed_fraction:
        ``random_seed_init=True``-only. Fraction of pixels seeded at their
        own `K_i` (default 0.25 = 25 %). Chosen deliberately not-too-sparse:
        sparse enough that large empty stretches exist and dispersal is
        unavoidable (the model "can't cheat" by starting dense
        everywhere), but dense enough that (a) convergence to a genuine
        equilibrium is still reasonably fast and (b) call-to-call
        variability in the resulting equilibrium (from the fresh random
        draw) stays small — a much sparser seed would converge slower and
        be noisier run-to-run.
    seed_mask_override:
        If given (not ``None``), a precomputed boolean mask used directly
        as the seeding pattern (``Un(t=0) = carrying_cap * seed_mask_override``),
        completely bypassing the internal fresh-draw logic described above
        under `random_seed_init` — `seed_fraction` and the RNG are not
        consulted at all in this case (the caller already baked its choice
        of fraction into the mask it is passing in). This is what lets a
        caller draw a mask ONCE per calibration site and reuse the exact
        same mask across every gradient step / grid point evaluated for
        that site during a run, rather than paying for (and being
        confounded by) a fresh draw on every call — see the extended
        discussion under `random_seed_init` above. When ``None`` (the
        default), behaviour falls back to `random_seed_init` /
        `seed_fraction` as already described.
    breeding_ground:
        Optional mask (same flat shape as `init_distrib`, values in
        ``[0, 1]``) constraining WHERE the population can actually grow.
        Applied only to the POSITIVE part of the per-iteration logistic
        growth term — a pixel outside the breeding ground (mask value 0)
        can still DECLINE (the negative-growth case, e.g. density above
        local carrying capacity relaxing back down, or simply density
        that arrived via dispersal without enough local recruitment to
        sustain it) but cannot GAIN density from local reproduction; a
        pixel can only gain density there via dispersal FROM elsewhere,
        never via its own local growth term. This deliberately does NOT
        zero out the growth term entirely outside the breeding ground
        (that would also block decline, effectively freezing density
        there) — only its positive part is masked:
        ``growth = where(growth > 0, growth * breeding_ground, growth)``.
        ``None`` (default) leaves growth completely unconstrained
        everywhere, matching every existing call site's behaviour before
        this parameter was added.

    Returns
    -------
    torch.Tensor
        Approximate stationary distribution (length ``N``), or
        ``(Un, list_changes)`` if ``return_history=True``.
    """
    carrying_cap = init_distrib.to(device)
    bg_t = None
    if breeding_ground is not None:
        bg_t = breeding_ground.to(device) if isinstance(breeding_ground, torch.Tensor) \
            else torch.tensor(breeding_ground, dtype=torch.float32, device=device)
    mask = None
    if seed_mask_override is not None:
        # Caller-supplied fixed mask (e.g. drawn once per calibration site
        # and reused across every step of that site's run) — bypasses the
        # fresh-draw logic entirely.
        mask = seed_mask_override.to(device)
        Un = carrying_cap * mask
    elif random_seed_init:
        # Sparse start: fresh Bernoulli mask, seeded pixels get their own
        # local K_i, the rest start at exactly 0 (see docstring above for
        # why this matters). Drawn fresh every call — no fixed seed.
        mask = seed_mask(
            init_distrib.shape, seed_fraction=seed_fraction,
            device=carrying_cap.device, dtype=carrying_cap.dtype,
        )
        Un = carrying_cap * mask
    else:
        Un = init_distrib.to(device)
    list_changes = []
    debug_history = [] if debug else None

    if debug:
        # A TRUE snapshot of the starting density, BEFORE any dispersal or
        # growth has run — distinct from `debug_history[1]` (formerly
        # index 0), which is already the state AFTER one full
        # dispersal+growth pass. With a wide dispersal kernel relative to
        # the site window (e.g. MDD comparable to or larger than the
        # window size), a single dispersal step can already spread a
        # sparse seed mask's mass across most of the window, making the
        # first POST-step frame look deceptively continuous — this frame
        # is the only place the actual raw seeded pattern (e.g. the
        # ~25%-of-pixels-occupied checkerboard from `random_seed_init`)
        # can be inspected directly, unaffected by any dynamics.
        size0 = int(Un.shape[0] ** 0.5)
        seed_stats = {
            "growth": (0.0, 0.0, 0.0),
            "un_post_growth": (
                float(Un.min().item()), float(Un.max().item()), float(Un.mean().item())
            ),
            "un_final": (
                float(Un.min().item()), float(Un.max().item()), float(Un.mean().item())
            ),
            "change": 0.0,
            "ratio": float("nan"),
            "un_final_map": Un.detach().cpu().numpy().reshape(size0, size0).copy(),
            "growth_map": np.zeros((size0, size0), dtype=np.float32),
            "dispersal_sent": 0.0, "dispersal_arrived": 0.0,
            "dispersal_mortality": 0.0, "dispersal_mortality_frac": 0.0,
            "is_seed_snapshot": True,
        }
        debug_history.append(seed_stats)
        if mask is not None:
            realized_frac = float(mask.float().mean().item())
            source = "seed_mask_override" if seed_mask_override is not None else "random_seed_init"
            print(f"  [debug] seed snapshot (before any dispersal/growth) — "
                  f"{source}: seed_fraction={seed_fraction:.3g} "
                  f"realized={realized_frac:.3g}  "
                  f"Un(seed) [{seed_stats['un_final'][0]:.4g}, {seed_stats['un_final'][1]:.4g}] "
                  f"mean={seed_stats['un_final'][2]:.4g}")

    n_steps = max_iter if adaptive else n_iter
    for epoch in range(n_steps):
        prev = Un.clone().detach()

        if plot:
            size = int(Un.shape[0] ** 0.5)
            plt.figure()
            plt.imshow(Un.detach().cpu().numpy().reshape(size, size), cmap="viridis")
            plt.title(f"Distribution – iteration {epoch}")
            plt.colorbar()
            plt.show()

        # Simplified "whole-population" model (deliberate choice, not the
        # only-new-individuals-disperse model tried earlier): EVERY pixel's
        # entire density disperses every iteration, not just the growth
        # increment. This treats `Un` as an abstraction of presence/density
        # rather than tracking individuals — it doesn't distinguish
        # established residents from newcomers, so a small, isolated patch
        # surrounded by poor habitat keeps bleeding density into the matrix
        # every single iteration (a real "small isolated patch may not be
        # viable" effect), whereas the only-dispersers model let a patch,
        # once colonised, get "protected" from further dispersal loss as
        # soon as growth stopped producing a surplus. The trade-off: real
        # site-fidelity/home-range behaviour (established individuals
        # generally do NOT keep dispersing every generation) isn't captured
        # — this is an explicit simplification, not a claim of higher
        # biological realism in every respect.
        sent = Un.detach().sum().item() if debug else None
        Un = Un @ kernel
        arrived = Un.detach().sum().item() if debug else None
        dispersal_mortality = (sent - arrived) if debug else None

        # Logistic growth, applied to the WHOLE post-dispersal density
        # (no distinction between residents and newcomers — there is no
        # such distinction left to make).
        # Clamp carrying_cap away from 0 to prevent Un/0 = inf.
        # nan_to_num only fixes NaN; -inf from positive/0 passes through and
        # causes NaN gradients in the backward pass.  The clamp value (1e-7) is
        # far below any physically meaningful carrying capacity.
        safe_cap = carrying_cap.clamp(min=1e-7)
        growth = (1.0 - linear_growth) * Un * (1.0 - Un / safe_cap)
        growth = torch.nan_to_num(growth, nan=0.0, posinf=0.0, neginf=0.0)
        if bg_t is not None:
            # Only the POSITIVE part of growth is breeding-ground-gated —
            # decline (e.g. relaxation back toward a lower carrying
            # capacity, or unsustained density that arrived via dispersal)
            # is allowed everywhere; only local reproduction is restricted
            # to the breeding ground. See this function's docstring.
            growth = torch.where(growth > 0, growth * bg_t, growth)
        un_post_growth = (Un + growth).detach() if debug else None
        Un = Un + growth
        Un = torch.clamp(Un, 0.0, 1.0)

        change = torch.nansum(torch.abs(Un - prev)).item()
        list_changes.append(change)
        if verbose:
            print(f"  iter {epoch}: change = {change:.6f}")

        ratio = (change / list_changes[0]) if list_changes[0] > 0 else 0.0
        if progress_callback is not None:
            progress_callback(epoch, ratio)

        if debug:
            def _stats(t: torch.Tensor) -> tuple:
                t = t.detach()
                return float(t.min().item()), float(t.max().item()), float(t.mean().item())

            size = int(Un.shape[0] ** 0.5)
            stats = {
                "growth": _stats(growth),
                "un_post_growth": _stats(un_post_growth),
                "un_final": _stats(Un),
                "change": change,
                "ratio": ratio,
                "un_final_map": Un.detach().cpu().numpy().reshape(size, size).copy(),
                "growth_map": growth.detach().cpu().numpy().reshape(size, size).copy(),
                # Dispersal-only mortality: mass lost purely to the kernel's
                # sub-stochastic row sums (kernel.sum(dim=1) < 1, i.e.
                # `survival` in dispersal_kernel), separate from growth's
                # own logistic die-off. Under this whole-population model,
                # this is EVERY pixel's density passing through the kernel
                # this iteration — not just newly-produced individuals.
                "dispersal_sent": sent, "dispersal_arrived": arrived,
                "dispersal_mortality": dispersal_mortality,
                "dispersal_mortality_frac": (
                    dispersal_mortality / sent if sent > 0 else 0.0
                ),
            }
            debug_history.append(stats)
            if epoch == 0 and mask is not None:
                realized_frac = float(mask.float().mean().item())
                source = "seed_mask_override" if seed_mask_override is not None else "random_seed_init"
                print(
                    f"  [debug] {source}: seed_fraction={seed_fraction:.3g} "
                    f"realized={realized_frac:.3g} "
                    f"({int(mask.sum().item())}/{mask.numel()} pixels seeded)"
                )
            print(
                f"  [debug] iter {epoch:3d}: "
                f"growth [{stats['growth'][0]:.4g}, {stats['growth'][1]:.4g}]  "
                f"Un(post-growth) [{stats['un_post_growth'][0]:.4g}, {stats['un_post_growth'][1]:.4g}]  "
                f"Un(final) [{stats['un_final'][0]:.4g}, {stats['un_final'][1]:.4g}] mean={stats['un_final'][2]:.4g}  "
                f"ratio={ratio:.4g}\n"
                f"           dispersal mortality (whole population, this iteration): "
                f"sent={sent:.4g} arrived={arrived:.4g} lost={dispersal_mortality:.4g} "
                f"({stats['dispersal_mortality_frac']*100:.2f}%)"
            )

        if adaptive and epoch + 1 >= n_iter:
            if ratio < convergence_ratio_tol:
                if verbose:
                    print(f"  Converged at iter {epoch}: ratio={ratio:.4g} "
                          f"< {convergence_ratio_tol}")
                break

    if plot:
        plt.figure()
        plt.plot(list_changes)
        plt.xlabel("Iteration")
        plt.ylabel("Total change")
        plt.title("Convergence to equilibrium")
        plt.show()

    if debug and debug_history:
        keys = ["growth", "un_post_growth", "un_final"]
        titles = ["Growth", "U_n (post-growth)", "U_n (final, clamped)"]
        fig, axes = plt.subplots(len(keys) + 2, 1, figsize=(9, 3 * (len(keys) + 2)), sharex=True)
        iters = list(range(len(debug_history)))
        for ax, key, title in zip(axes[:-2], keys, titles):
            mins = [d[key][0] for d in debug_history]
            maxs = [d[key][1] for d in debug_history]
            means = [d[key][2] for d in debug_history]
            ax.plot(iters, maxs, label="max", color="crimson")
            ax.plot(iters, means, label="mean", color="black", linestyle="--")
            ax.plot(iters, mins, label="min", color="steelblue")
            ax.axhline(0.0, color="gray", linewidth=0.5)
            ax.set_ylabel(title, fontsize=8)
            ax.legend(fontsize=7, loc="upper right")
        axes[-2].plot(iters, [d["dispersal_mortality_frac"] * 100 for d in debug_history],
                      color="purple")
        axes[-2].set_ylabel("Dispersal mortality\n(% of population sent)", fontsize=8)
        axes[-1].plot(iters, [d["ratio"] for d in debug_history], color="darkorange")
        axes[-1].set_yscale("log")
        axes[-1].set_ylabel("Change ratio\n(log scale)", fontsize=8)
        axes[-1].set_xlabel("Iteration")
        fig.suptitle("equilibrium_distribution — per-iteration debug trace")
        fig.tight_layout()
        plt.show()

        # Filmstrip of the spatial maps themselves, one thumbnail per
        # iteration (subsampled to at most `max_frames` if there are more
        # iterations than that, always including the first and last), so a
        # spreading collapse front is visually distinguishable from a
        # uniform, all-at-once collapse.
        max_frames = 30
        n_frames = len(debug_history)
        if n_frames > max_frames:
            frame_idxs = sorted(set(
                np.linspace(0, n_frames - 1, max_frames).round().astype(int).tolist()
            ))
        else:
            frame_idxs = list(range(n_frames))

        for map_key, cmap, title in [
            ("un_final_map", "viridis", "U_n (final, clamped) — per iteration"),
            ("growth_map", "coolwarm", "Growth — per iteration"),
        ]:
            ncols = min(6, len(frame_idxs))
            nrows = int(np.ceil(len(frame_idxs) / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(2.2 * ncols, 2.2 * nrows))
            axes = np.atleast_1d(axes).flatten()
            if map_key == "growth_map":
                vabs = max(abs(debug_history[i][map_key]).max() for i in frame_idxs) or 1.0
                vmin, vmax = -vabs, vabs
            else:
                vmin, vmax = 0.0, max(debug_history[i][map_key].max() for i in frame_idxs) or 1.0
            for ax, i in zip(axes, frame_idxs):
                im = ax.imshow(debug_history[i][map_key], cmap=cmap, vmin=vmin, vmax=vmax)
                # debug_history[0] is the pre-dispersal/growth seed snapshot
                # (see its insertion above) — label it "seed", not "iter 0",
                # and shift the iteration numbers of every later frame down
                # by 1 so they still read as 0-indexed simulation steps.
                frame_label = "seed" if debug_history[i].get("is_seed_snapshot") else f"iter {i-1}"
                ax.set_title(frame_label, fontsize=7)
                ax.set_xticks([])
                ax.set_yticks([])
            for ax in axes[len(frame_idxs):]:
                ax.axis("off")
            fig.suptitle(title)
            fig.colorbar(im, ax=axes.tolist(), shrink=0.6)
            plt.show()

    if return_history:
        if debug:
            return Un, list_changes, debug_history
        return Un, list_changes
    if debug:
        return Un, debug_history
    return Un


def growth_step(
    distrib: torch.Tensor,
    linear_growth: float | torch.Tensor,
    K_is: torch.Tensor,
    tested_xs: list,
    tested_ys: list,
    window_half_size: int,
    threshold_abundance: float,
    blacklisted_points: list | None = None,
    breeding_ground: torch.Tensor | None = None,
    plot: bool = False,
) -> tuple[torch.Tensor, list]:
    """Apply one growth step to a full 2-D abundance map.

    Parameters
    ----------
    distrib:
        2-D abundance map ``(H, W)``.
    linear_growth:
        Scalar growth coefficient.
    K_is:
        Flat carrying-capacity vector (length ``H * W``).
    tested_xs, tested_ys:
        Coordinates of recently evaluated dispersal centres.
    window_half_size:
        Half-width of the dispersal evaluation window (``mw / 2``).
    threshold_abundance:
        Relative-abundance value corresponding to species presence.
    blacklisted_points:
        List of ``(x, y)`` tuples already near carrying capacity.
    breeding_ground:
        Optional 2-D binary mask – growth only where this is 1.
    plot:
        If ``True``, visualise the growth contribution.

    Returns
    -------
    new_distrib : torch.Tensor
        Updated 2-D abundance map.
    blacklisted_points : list
        Updated blacklist.
    """
    if blacklisted_points is None:
        blacklisted_points = []

    m, n = distrib.shape
    K_raster = K_is.reshape(m, n)
    un = distrib.flatten().clone().detach().to(device)

    if breeding_ground is None:
        bg = torch.ones((m, n), device=device, dtype=torch.float32).flatten()
    else:
        if not isinstance(breeding_ground, torch.Tensor):
            breeding_ground = torch.tensor(breeding_ground, dtype=torch.float32, device=device)
        bg = breeding_ground.flatten().clone().detach().to(device)

    growth = (1.0 - linear_growth) * un * (1.0 - un / K_is.to(device))
    growth = torch.nan_to_num(growth, nan=0.0)
    # Only the POSITIVE part is breeding-ground-gated — decline is allowed
    # everywhere, only local reproduction is restricted to the breeding
    # ground (see `equilibrium_distribution`'s docstring for the same
    # convention). Previously this multiplied the WHOLE growth term by
    # `bg`, which also zeroed out decline outside the breeding ground —
    # freezing density there instead of letting it relax down.
    growth = torch.where(growth > 0, growth * bg, growth)

    new_un = un + growth
    new_un = torch.clamp(new_un, 0.0, 1.0)
    new_un = torch.nan_to_num(new_un, nan=0.0)
    new_distrib = new_un.reshape(m, n)

    if plot:
        plt.figure()
        ratio = (growth / (new_un + 1e-10)).reshape(m, n)
        plt.imshow(ratio.cpu().numpy(), cmap="plasma", vmin=-1, vmax=1)
        plt.colorbar(label="Growth / (Growth + U_t)")
        plt.title("Growth contribution")
        plt.show()

    return new_distrib, blacklisted_points


# ---------------------------------------------------------------------------
# Object-oriented wrapper
# ---------------------------------------------------------------------------

class GrowthModel:
    """Encapsulates the logistic growth model for a species.

    Parameters
    ----------
    tgrowth:
        Characteristic growth time (years / time steps).
    L, k, x0:
        Logistic parameters describing ``K(HS)``.

    Examples
    --------
    >>> import numpy as np
    >>> from paradis.core.growth import GrowthModel
    >>> gm = GrowthModel(tgrowth=7.5, L=0.05, k=5.0, x0=0.5)
    >>> hs = np.random.rand(50, 50).astype("float32")
    >>> a, K_is, vecb = gm.carrying_capacity(hs)
    """

    def __init__(
        self,
        tgrowth: float,
        L: float,
        k: float,
        x0: float,
    ) -> None:
        self.tgrowth = tgrowth
        self.L = L
        self.k = k
        self.x0 = x0
        self._a = growth_coefficient(tgrowth)

    @property
    def a(self) -> float:
        """Scalar linear growth coefficient."""
        return self._a

    def carrying_capacity(self, hs: np.ndarray | torch.Tensor) -> tuple:
        """Compute carrying-capacity parameters for *hs*.

        Returns
        -------
        a, K_is, vecb
            See :func:`carrying_capacity_from_hs`.
        """
        return carrying_capacity_from_hs(hs, self.tgrowth, self.L, self.k, self.x0)

    def __repr__(self) -> str:
        return (
            f"GrowthModel(tgrowth={self.tgrowth}, "
            f"L={self.L}, k={self.k}, x0={self.x0})"
        )
