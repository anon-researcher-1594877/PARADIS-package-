"""Batch processing: learn parameters for many species over folder structures.

:func:`align_folders`
    Match files across multiple directories using a naming pattern.
:func:`learn_over_folder`
    Full pipeline for an entire folder of species.

Classes
-------
:class:`BatchLearner`
    Object-oriented wrapper around :func:`learn_over_folder`.
"""

from __future__ import annotations

import os
import random
import traceback

import numpy as np
import pandas as pd
import torch
from PIL import Image

from paradis.calibration.sites import sample_calibration_sites
from paradis.calibration.presence import calibrate_presence_threshold
from paradis.calibration.carrying_capacity import estimate_carrying_capacity, plot_carrying_capacity, logistic
from paradis.calibration.ratios import compute_all_posteriors
from paradis.learning.optimizer import (
    learn_dispersal_parameters,
    refine_from_point,
    compare_equilibrium_distributions,
    run_mala,
    estimate_low_map_high_from_posterior,
)


# ---------------------------------------------------------------------------
# Folder alignment
# ---------------------------------------------------------------------------

def align_folders(
    folders: list[str],
    name_formats: list[str],
) -> tuple[list[list[int]], list[list[str]]]:
    """Match files across directories by their variable name component.

    The variable part is marked with the placeholder ``XxX`` in each format
    string.  The function strips the constant prefix/suffix and matches files
    with the same variable part (spaces and underscores are treated as
    equivalent).

    Parameters
    ----------
    folders:
        Directories to align (one per data type).
    name_formats:
        Name templates, e.g. ``["XxX_HS.tif", "XxX_Obs.tif"]``.

    Returns
    -------
    ordered_indices:
        ``ordered_indices[folder_idx][match_idx]`` → file index in that folder.
    variable_parts:
        ``variable_parts[folder_idx][match_idx]`` → species name string.
    """
    file_lists = [sorted(os.listdir(f)) for f in folders]
    var_parts_per_folder = []

    for fmt, files in zip(name_formats, file_lists):
        pre_len = fmt.index("XxX")
        suffix = fmt[pre_len + 3:]
        suf_len = len(suffix)
        var_parts_per_folder.append(
            [fn[pre_len: -suf_len if suf_len else None].replace(" ", "_")
             for fn in files]
        )

    ordered = [[] for _ in folders]
    for i0, vpart in enumerate(var_parts_per_folder[0]):
        indices = [i0]
        try:
            for fi in range(1, len(folders)):
                indices.append(var_parts_per_folder[fi].index(vpart))
        except ValueError:
            print(f"No match for '{vpart}' in all folders – skipped.")
            continue
        if len(indices) == len(folders):
            for fi, idx in enumerate(indices):
                ordered[fi].append(idx)

    missing = max(len(vp) for vp in var_parts_per_folder) - len(ordered[0])
    if missing > 0:
        print(f"{missing} files without a match across all folders.")
    else:
        print("All files aligned successfully.")

    return ordered, var_parts_per_folder, file_lists


# ---------------------------------------------------------------------------
# Sampling effort helpers
# ---------------------------------------------------------------------------

def _build_breeding_range_map(folder_breeding_range: str, name_format: str) -> dict[str, str]:
    """Scan *folder_breeding_range* and return a dict mapping normalised
    species name -> full file path, using the SAME ``XxX`` name-format
    convention as :func:`align_folders` (spaces/underscores both mapped to
    underscore, matching `sp_name`'s own convention exactly). Unlike
    `align_folders`, this is a SINGLE optional folder, not aligned against
    the mandatory hs/obs/range folders — a species with no match here
    simply gets ``None`` (unconstrained growth), never an error or a
    dropped species.
    """
    pre_len = name_format.index("XxX")
    suffix = name_format[pre_len + 3:]
    suf_len = len(suffix)
    breeding_map: dict[str, str] = {}
    for fname in sorted(os.listdir(folder_breeding_range)):
        vpart = fname[pre_len: -suf_len if suf_len else None].replace(" ", "_")
        breeding_map[vpart] = os.path.join(folder_breeding_range, fname)
    return breeding_map


def _build_effort_map(folder_sampling_effort: str) -> dict[str, str]:
    """Scan *folder_sampling_effort* and return a dict mapping normalised group
    names to their full file paths.

    Files must be named ``sampling_effort_<group>.tif`` (case-insensitive).
    """
    effort_map: dict[str, str] = {}
    for fname in os.listdir(folder_sampling_effort):
        lower = fname.lower()
        if lower.startswith("sampling_effort_") and lower.endswith(".tif"):
            group_key = lower[len("sampling_effort_"):-len(".tif")].replace("_", " ").strip()
            effort_map[group_key] = os.path.join(folder_sampling_effort, fname)
    return effort_map


# ---------------------------------------------------------------------------
# Batch learning pipeline
# ---------------------------------------------------------------------------

def learn_over_folder(
    folder_hs: str,
    folder_obs: str,
    folder_range: str,
    path_mdd_table: str,
    output_folder: str,
    path_taxa_ref: str | None = None,
    path_sampling_effort_groups: str | None = None,
    folder_sampling_effort: str | None = None,
    sampling_effort_species_col: str = "SpeciesName",
    sampling_effort_group_col: str = "SamplingEffortGroup",
    folder_breeding_range: str | None = None,
    breeding_range_name_format: str = "XxX.tif",
    save_fig_folder: str | None = None,
    name_formats: list[str] | None = None,
    mdd_col: str = "Dispersal_km",
    species_col: str = "scientificName",
    map_window: int = 70,
    max_iter: int = 50,
    all_together: bool = True,
    n_learning_sites: int = 5,
    n_random_sites: int = 3,
    n_calibration_samples: int = 800,
    time_budget: float = 30.0,
    test_species: str | list[str] | None = None,
    seed: int | None = None,
    gpu_memory_fraction: float | None = 0.8,
    r_min: float | None = None,
    r_survival_tol: float = 0.005,
    r_survival_alpha: float = 1.11,
    d_max_r_survival_tol: float = 1e-12,
    s_halfwidth: float = 3.0,
    seed_fraction: float = 0.25,
    compute_hessian: bool = False,
    hessian_eps: tuple = (0.05, 0.05, 0.01),
    method: str = "batchSGD",
    init_point: tuple | None = None,
    precise_gridscan_center: tuple | None = None,
    precise_gridscan_s_window: float = 1.0,
    precise_gridscan_d_window: float = 1.0,
    precise_gridscan_Scrit_window: float = 0.1,
    refine_max_iter: int = 300,
    refine_lr: float = 0.05,
    refine_chunk_size: int = 3,
    refine_check_every: int = 10,
    points_per_axis: int = 10,
    grid_n_range: tuple | None = None,
    grid_r_range: tuple | None = None,
    grid_tg_range: tuple | None = None,
    point1: tuple | None = None,
    point2: tuple | None = None,
    compare_label1: str = "Point 1",
    compare_label2: str = "Point 2",
    compare_max_sites_per_figure: int = 8,
    compare_debug: bool = False,
    compare_debug_site_idx: int | None = None,
    init_params: dict | None = None,
    robust_gradient: bool = False,
    batch_chunk_size: int | None = None,
    mala_num_samples: int = 500,
    mala_warmup_steps: int = 200,
    mala_num_chains: int = 1,
    mala_step_size: float | None = None,
    mala_target_accept_prob: float = 0.574,
    mala_chunk_size: int = 3,
    mala_disperse_chains: bool = True,
    mala_pre_optimize_steps: int = 50,
    low_map_high_npz_folder: str | None = None,
    low_map_high_ci_mass: float = 0.68,
    low_map_high_map_grid_size: int = 25,
) -> pd.DataFrame:
    """Run the full PARADIS calibration pipeline for every species in *folder_hs*.

    Previously computed species are skipped (resume-friendly).

    Sampling effort can be provided in two ways (mutually exclusive):

    * **Single raster** – pass ``path_taxa_ref``: one raster used for all
      species (original behaviour).
    * **Per-species raster** – pass both ``path_sampling_effort_groups`` and
      ``folder_sampling_effort``: the CSV maps each species to a group name,
      and the folder must contain files named
      ``sampling_effort_<group>.tif`` for each group.

    Parameters
    ----------
    folder_hs:
        Directory of habitat-suitability rasters (GeoTIFF or PIL-readable).
    folder_obs:
        Directory of observation count rasters.
    folder_range:
        Directory of current-range rasters.
    path_mdd_table:
        CSV with columns *species_col* and *mdd_col*.
    output_folder:
        Directory for the output CSV.
    path_taxa_ref:
        Path to a single sampling-effort raster used for all species.
    path_sampling_effort_groups:
        CSV mapping species names to sampling effort groups.
    folder_sampling_effort:
        Folder with ``sampling_effort_<group>.tif`` files.
    sampling_effort_species_col:
        Column name for species names in the sampling effort groups CSV.
    sampling_effort_group_col:
        Column name for group names in the sampling effort groups CSV.
    folder_breeding_range:
        Optional folder of per-species breeding-range rasters — when a
        species has a matching file here (matched by species name via
        `breeding_range_name_format`, SEPARATELY from the mandatory hs/
        obs/range alignment — a species without a match is NOT skipped or
        treated as an error, it just gets unconstrained growth, identical
        to the behaviour before this parameter existed), reproduction is
        constrained to that mask for every simulation/calibration
        involving that species — INCLUDING while estimating parameters
        (threaded through as `breeding_masks_list` into every
        `cost_function`/`equilibrium_distribution` call for that
        species' calibration sites, not just the final production
        simulation). Only POSITIVE growth is restricted — decline is
        still allowed everywhere (see `equilibrium_distribution`'s
        `breeding_ground` docstring for the exact math/rationale). A
        small bar-chart summary ("N species with a breeding range" vs.
        "N without") is shown once per batch run, mirroring
        `align_folders`'s own aligned/unaligned reporting for the
        mandatory folders.
    breeding_range_name_format:
        ``XxX``-placeholder file-name pattern for `folder_breeding_range`
        (default ``"XxX.tif"``), same convention as `name_formats`.
    save_fig_folder:
        Optional directory for diagnostic figures.
    name_formats:
        File-name patterns with ``XxX`` placeholder.
        Default: ``["XxX.tif", "XxX.tif", "XxX.tif"]``.
    mdd_col, species_col:
        Column names in the MDD table.
    map_window:
        Calibration window size (pixels).
    max_iter:
        Gradient-descent iterations per site.
    all_together, n_learning_sites, n_random_sites:
        ``method="batchSGD"``-only. Forwarded to
        :func:`~paradis.learning.optimizer.learn_dispersal_parameters`.
        ``all_together=True`` (default): one stochastic run per species,
        drawing random ``n_random_sites``-site mini-batches at every step.
        ``all_together=False``: instead runs ``n_learning_sites`` separate
        full SGD runs, each trained on a single calibration site only —
        useful to see each site's own trajectory in (n, r, Tg) and the
        density of their individual endpoints (see `_kde_3d_plotly`),
        e.g. to spot a site that disagrees with the rest.
    n_calibration_samples:
        Random candidate centres per iteration.
    time_budget:
        Time budget for site selection (seconds).
    seed:
        If given, seeds ``numpy``/``random``/``torch``(+CUDA) once at the
        start, before any species' calibration-site sampling or SGD run —
        for a reproducible folder-wide run (same calibration sites, same
        Adam trajectory every time).
    gpu_memory_fraction:
        Forwarded to
        :func:`~paradis.learning.optimizer.learn_dispersal_parameters` for
        each species — caps PyTorch's CUDA allocator at this fraction of
        total GPU memory (default 0.8 = 80%) to avoid the process growing
        into and exhausting the whole card. Set to ``None`` for PyTorch's
        default (effectively 100%).
    r_min, r_survival_tol, r_survival_alpha:
        Forwarded to
        :func:`~paradis.learning.optimizer.learn_dispersal_parameters` for
        each species — r (like n) is now learned in log10 space; r_min is
        its auto-computed (or explicit) lower bound, the largest r below
        which mean survival probability changes by less than
        ``r_survival_tol`` (default ``0.005`` = 0.5 percentage points —
        the MODERATE tolerance, used for ``n_min``/``d_min`` and the
        default init point's box centre) as r decreases further, i.e. no
        longer distinguishable from r=0 by the model.
    d_max_r_survival_tol:
        A SEPARATE, much smaller tolerance (default ``1e-12``) used ONLY
        to extend ``d_max`` far out — species whose true optimum needs
        near-zero dispersal mortality (confirmed for e.g. wolf/wild boar)
        can still reach it during learning, without shifting ``d_min`` or
        the default init point's (conservative, ``r_survival_tol``-based)
        box centre. See
        :func:`~paradis.learning.optimizer.learn_dispersal_parameters`'s
        module-level note above ``_n_box_from_s_and_r``/``_sd_box`` for
        the full derivation.
    compute_hessian:
        ``method="batchSGD"``: forwarded to
        :func:`~paradis.learning.optimizer.learn_dispersal_parameters`.
        Default False (currently disabled — see that function's docstring).
        ``method="refine"``: forwarded to
        :func:`~paradis.learning.optimizer.refine_from_point` instead —
        computes the Hessian at the FINAL converged ``(s, d, S_crit)``
        point and prints whether it's a genuine local minimum (a true
        equilibrium) or a saddle point. See that function's docstring.
    hessian_eps:
        ``method="refine"``-only. Forwarded to
        :func:`~paradis.learning.optimizer.refine_from_point` as
        ``hessian_eps`` — the `(eps_s, eps_d, eps_Scrit)` finite-difference
        step sizes used when ``compute_hessian=True``.
    method:
        ``"batchSGD"`` (default) — the usual stochastic mini-batch Adam run
        via :func:`~paradis.learning.optimizer.learn_dispersal_parameters`.
        ``"refine"`` — instead calls
        :func:`~paradis.learning.optimizer.refine_from_point`: a purely
        deterministic full-dataset Adam run (no mini-batching/noise)
        starting from the exact ``init_point`` you give, to directly check
        whether the optimizer keeps moving in a sensible direction from a
        chosen point (e.g. a stochastic run's endpoint), without noise
        muddying the picture. Requires ``init_point`` and is best paired
        with ``test_species`` restricted to a single species, since
        ``init_point`` is one fixed ``(s, d, S_crit)`` shared across
        whichever species are processed.
        ``"grid"`` — forwards to
        :func:`~paradis.learning.optimizer.learn_dispersal_parameters`
        with ``method="grid"``: a brute-force full-dataset cost scan (no
        gradients) over (n, r, Tg), saved as heatmap slices + an
        interactive 3-D volume to ``save_fig_folder``, for directly
        visualising the cost surface's shape rather than optimising it.
        Best paired with ``test_species`` restricted to one species, since
        each grid point costs ~1s (~15-20min per 1000-point stage).
        ``"compare"`` — instead calls
        :func:`~paradis.learning.optimizer.compare_equilibrium_distributions`:
        computes and plots the equilibrium distribution at EVERY
        calibration site under two explicit ``(s, d, S_crit)`` points
        (``point1``/``point2``), side by side. No gradient/optimisation
        involved — a pure diagnostic to visually check whether a
        suspicious point (e.g. one reachable only via unusually large Tg,
        or sitting in a narrow cost valley) produces physically sensible
        equilibrium distributions or is exploiting an artifact (e.g. mass
        piling up at the calibration window's edges). Requires ``point1``
        and ``point2`` and is best paired with ``test_species`` restricted
        to a single species.
        ``"precise_gridscan"`` — forwards to
        :func:`~paradis.learning.optimizer.learn_dispersal_parameters`
        with ``method="precise_gridscan"``: a small, ZOOMED-IN grid scan
        (default 10x10x10) around a chosen ``(s, d, S_crit)`` point —
        e.g. `method="refine"`'s converged endpoint — to visually verify
        the cost surface's local shape as an independent check on
        whether that point is a genuine local minimum. See the
        ``precise_gridscan_*`` parameters below. Requires
        ``precise_gridscan_center`` and is best paired with
        ``test_species`` restricted to a single species.
    precise_gridscan_center, precise_gridscan_s_window,
    precise_gridscan_d_window, precise_gridscan_Scrit_window:
        ``method="precise_gridscan"``-only. Forwarded to
        :func:`~paradis.learning.optimizer.learn_dispersal_parameters`.
        `precise_gridscan_center` (required, ``(s, d, S_crit)``) is the
        point to zoom in on. The `_window` parameters (defaults ``1.0``,
        ``1.0``, ``0.1``) are the half-widths of the local scan window
        around it. Resolution uses the SAME `points_per_axis` as
        ``method="grid"`` (see below).
    init_point:
        ``method="refine"``-only. The exact ``(s, d, S_crit)`` starting
        point (this package's own learning space — not ``(n, r, Tg)`` any
        more). Default ``None`` — starts exactly at the box centre (see
        :func:`~paradis.learning.optimizer.refine_from_point`).
    refine_max_iter, refine_lr, refine_chunk_size, refine_check_every:
        ``method="refine"``-only. Forwarded to
        :func:`~paradis.learning.optimizer.refine_from_point` as
        ``max_iter``, ``lr``, ``chunk_size``, ``check_every``.
    points_per_axis, grid_n_range, grid_r_range, grid_tg_range:
        ``method="grid"`` or ``"precise_gridscan"``-only. Forwarded to
        :func:`~paradis.learning.optimizer.learn_dispersal_parameters` —
        see its docstring. The SAME `points_per_axis` (default ``10``)
        is used for both methods — one parameter, not two separate ones.
    point1, point2:
        ``method="compare"``-only. The two exact ``(s, d, S_crit)`` points
        to compare (this package's own learning space — not
        ``(n, r, Tg)`` any more). Required when ``method="compare"``.
    compare_label1, compare_label2, compare_max_sites_per_figure:
        ``method="compare"``-only. Forwarded to
        :func:`~paradis.learning.optimizer.compare_equilibrium_distributions`
        as ``label1``, ``label2``, ``max_sites_per_figure``.
    compare_debug, compare_debug_site_idx:
        ``method="compare"``-only. Forwarded to
        :func:`~paradis.learning.optimizer.compare_equilibrium_distributions`
        as ``debug``, ``debug_site_idx`` — set ``compare_debug=True`` and
        pick a ``compare_debug_site_idx`` to get the full per-iteration
        min/max/mean trace and filmstrip map plots
        (``equilibrium_distribution(..., debug=True)``) for that one site,
        at both ``point1`` and ``point2``.
    init_params:
        ``method="batchSGD"``-only. Forwarded to
        :func:`~paradis.learning.optimizer.learn_dispersal_parameters` —
        optional custom starting point
        ``{"s": ..., "d": ..., "S_crit": ...}`` (this package's own
        learning space — not ``(n, r, Tg)`` any more) instead of the
        default of starting at each box's centre.
    robust_gradient:
        ``method="batchSGD"`` or ``method="refine"``. Forwarded to
        :func:`~paradis.learning.optimizer.learn_dispersal_parameters` or
        :func:`~paradis.learning.optimizer.refine_from_point` respectively
        — combines per-site gradients via the (Byzantine-robust) geometric
        median instead of their mean, at every step, so a site whose
        gradient strongly disagrees with the rest doesn't dominate the
        update. For ``method="refine"`` this uses ALL sites' gradients
        (exact, deterministic) every step, not a mini-batch — the most
        statistically meaningful use of this option. Default False
        (unchanged plain-average behaviour).
    batch_chunk_size:
        ``method="batchSGD"``-only (and only when ``robust_gradient=False``
        — that path already computes one site at a time). Forwarded to
        :func:`~paradis.learning.optimizer.learn_dispersal_parameters`.
        The default mini-batch path holds every site's dense matrix-inverse
        graph in memory simultaneously for a single `.backward()` call —
        fine for small `n_random_sites`, but can exhaust GPU memory if
        `n_random_sites` is set close to (or above) the total number of
        calibration sites, e.g. to train on all of them every step. Set
        this to bound memory to at most `batch_chunk_size` sites' graphs
        at a time (gradients accumulated across chunks, same technique as
        `refine_from_point`'s `chunk_size`). Default ``None`` (no
        chunking, original behaviour).
    mala_num_samples, mala_warmup_steps, mala_num_chains, mala_step_size,
    mala_target_accept_prob, mala_chunk_size, mala_disperse_chains,
    mala_pre_optimize_steps:
        ``method="MALA"``-only. Forwarded to
        :func:`~paradis.learning.optimizer.run_mala` (as ``num_samples``,
        ``warmup_steps``, ``num_chains``, ``step_size``,
        ``target_accept_prob``, ``chunk_size``, ``disperse_chains``
        respectively) — full joint posterior over ``(s, d, Tg)`` via a
        hand-rolled MALA sampler (no external MCMC library needed, unlike
        the removed NUTS/Pyro path): gradient-informed Metropolis-Hastings,
        one gradient per proposal (cheaper per step than NUTS's several
        leapfrogs, better mixing than a plain symmetric random walk). With
        the default ``mala_disperse_chains=True`` and ``mala_num_chains >
        1``, each chain starts at its OWN random point in the valid
        ``(s, d, S_crit)`` region (then a short
        ``mala_pre_optimize_steps``-step gradient ascent, default 50,
        slides it off flat/high-cost terrain onto the nearest ridge/valley
        floor before real warmup — set to 0 to use the raw random draw
        directly) rather than all starting at the identical conservative
        box centre — needed for split-R-hat to be able to detect a chain
        stuck in a different plateau/valley than the others; see
        `run_mala`'s docstring for the full rationale. As
        with `method="NUTS"` before it, the recorded `n`/`r`/`Tg`/`Ew` in
        `learned_parameters.csv` are the POSTERIOR MEANS — use the saved
        samples for anything requiring the joint posterior.
    low_map_high_npz_folder, low_map_high_ci_mass, low_map_high_map_grid_size:
        ``method="low_MPA_high"``-only. Requires a ``{species}_samples.npz``
        already saved by a prior ``method="MALA"`` run — searched for in
        ``low_map_high_npz_folder`` (default ``None`` -> falls back to
        ``output_folder``, i.e. wherever `method="MALA"` itself saves
        them); a species without one is skipped with a warning, not a
        hard failure. Forwarded to
        :func:`~paradis.learning.optimizer.estimate_low_map_high_from_posterior`
        (as ``ci_mass``, ``map_grid_size``) — ranks ALL raw posterior
        samples by a closed-form, ANALYTIC colonisation-speed proxy
        derived from Fisher-KPP travelling-front theory
        (``c = sqrt(2*g*Ew(r))``, no simulation at all — see that
        function's docstring for the full derivation), takes the
        ``ci_mass`` equal-tailed CI of that proxy, and returns the LOW/
        MAP/HIGH bounding ``(n, r, Tg)`` parameter sets. Diagnostic-only,
        same as ``method="compare"``: does not touch
        `learned_parameters.csv`.

    Returns
    -------
    pandas.DataFrame
        Table with learned parameters (``Ew``, ``n``, ``r``, ``Tg``).
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    if name_formats is None:
        name_formats = ["XxX.tif", "XxX.tif", "XxX.tif"]

    # Validate sampling-effort arguments
    use_per_species_effort = path_sampling_effort_groups is not None
    if use_per_species_effort and folder_sampling_effort is None:
        raise ValueError(
            "folder_sampling_effort must be provided together with "
            "path_sampling_effort_groups."
        )
    if not use_per_species_effort and path_taxa_ref is None:
        raise ValueError(
            "Provide either path_taxa_ref (single raster) or both "
            "path_sampling_effort_groups and folder_sampling_effort."
        )

    os.makedirs(output_folder, exist_ok=True)
    if save_fig_folder is None:
        save_fig_folder = os.path.join(output_folder, "figs")
    os.makedirs(save_fig_folder, exist_ok=True)

    ordered, var_parts, file_lists = align_folders(
        [folder_hs, folder_obs, folder_range], name_formats
    )
    mdd_table = pd.read_csv(path_mdd_table)

    # --- Sampling effort setup ---
    if use_per_species_effort:
        effort_groups_df = pd.read_csv(path_sampling_effort_groups)
        sp_to_group: dict[str, str] = {
            str(row[sampling_effort_species_col]).replace("_", " ").strip().lower():
            str(row[sampling_effort_group_col]).strip()
            for _, row in effort_groups_df.iterrows()
        }
        effort_file_map = _build_effort_map(folder_sampling_effort)
        _effort_cache: dict[str, np.ndarray] = {}
        taxa_ref_global = None
        print(f"[batch] Per-species sampling effort mode — "
              f"{len(sp_to_group)} species mapped, "
              f"{len(effort_file_map)} group rasters found: "
              f"{list(effort_file_map.keys())}")
    else:
        taxa_ref_global = np.array(Image.open(path_taxa_ref))
        sp_to_group = {}
        effort_file_map = {}
        _effort_cache = {}
        print("[batch] Single sampling-effort raster mode.")

    output_csv = os.path.join(output_folder, "learned_parameters.csv")
    records = []

    indices = list(range(len(ordered[0])))
    if test_species is not None:
        names = [test_species] if isinstance(test_species, str) else test_species
        test_keys = {n.replace(" ", "_").lower() for n in names}
        indices = [
            i for i in indices
            if var_parts[0][ordered[0][i]].replace(" ", "_").lower() in test_keys
        ]
        if not indices:
            available = [var_parts[0][ordered[0][i]] for i in range(len(ordered[0]))]
            raise ValueError(
                f"test_species={names!r} not found. "
                f"Available: {available}"
            )
        print(f"[batch] test_species mode — running only: {names}")

    # --- Breeding-range setup (optional, per-species, NOT aligned via
    # align_folders — see `folder_breeding_range`'s docstring for why) ---
    if folder_breeding_range is not None:
        breeding_range_map = _build_breeding_range_map(
            folder_breeding_range, breeding_range_name_format,
        )
        species_keys = [var_parts[0][ordered[0][i]] for i in indices]
        n_with = sum(1 for k in species_keys if k in breeding_range_map)
        n_without = len(species_keys) - n_with
        print(f"[batch] Breeding-range mode — {len(breeding_range_map)} rasters "
              f"found in '{folder_breeding_range}'.")
        print(f"[batch] species with breeding range: {n_with}  "
              f"species without: {n_without}")
    else:
        breeding_range_map = {}

    for i in indices:
        sp_name = var_parts[0][ordered[0][i]]
        record: dict = {"scientific_name": sp_name, "ew": None, "n": None,
                        "r": None, "tg": None, "Err": None}

        # Resume: skip already computed
        if os.path.exists(output_csv):
            df_existing = pd.read_csv(output_csv)
            if sp_name in df_existing["scientific_name"].values:
                print(f"Skipping {sp_name} (already in output).")
                continue

        mdd_match = mdd_table[mdd_table[species_col] == sp_name.replace("_", " ")]
        if mdd_match.empty:
            print(f"MDD not found for {sp_name}.")
            record["Err"] = "MDD_not_found"
            records.append(record)
            _append_csv(output_csv, record)
            continue

        try:
            hs = np.array(Image.open(
                os.path.join(folder_hs,    file_lists[0][ordered[0][i]])
            )).astype(float)
            obs = np.array(Image.open(
                os.path.join(folder_obs,   file_lists[1][ordered[1][i]])
            )).astype(float)
            cr = np.array(Image.open(
                os.path.join(folder_range, file_lists[2][ordered[2][i]])
            )).astype(float)

            # Normalise
            hs[hs == np.nanmax(hs)] = 0.0
            hs = hs / np.nanmax(hs)
            cr[cr == np.nanmin(cr)] = 0.0
            cr = cr / np.nanmax(cr)
            obs[obs < 0] = 0.0

            mdd_val = float(mdd_match[mdd_col].values[0])
            n_obs_total = int(np.nansum(obs))
            sp_display = sp_name.replace("_", " ")
            fn_hs  = file_lists[0][ordered[0][i]]
            fn_obs = file_lists[1][ordered[1][i]]
            fn_cr  = file_lists[2][ordered[2][i]]
            if use_per_species_effort:
                sp_key_disp = sp_name.replace("_", " ").strip().lower()
                group_disp  = sp_to_group.get(sp_key_disp, "unknown")
            else:
                group_disp = "global"
            print(
                f"\033[92m[{sp_display}]\033[0m  "
                f"\033[33mMDD={mdd_val} km\033[0m  "
                f"\033[94m[{group_disp}]\033[0m  "
                f"obs={n_obs_total}  "
                f"\033[93mHS={fn_hs}  Obs={fn_obs}  CR={fn_cr}\033[0m"
            )
            hmean = float(hs[cr > 0].mean())

            # Resolve breeding-range raster for this species (optional —
            # no match just means unconstrained growth, see
            # `folder_breeding_range`'s docstring).
            breeding_range_path = breeding_range_map.get(sp_name)
            if breeding_range_path is not None:
                breeding_range = np.array(Image.open(breeding_range_path)).astype(float)
                breeding_range = np.nan_to_num(breeding_range, nan=0.0)
                # Normalise to [0, 1] — same convention as `cr` above —
                # a mask should already be 0/1, but this is defensive
                # against e.g. a raster stored as 0/255.
                max_val = np.nanmax(breeding_range)
                if max_val > 0:
                    breeding_range = np.clip(breeding_range / max_val, 0.0, 1.0)
                print(f"\033[93m  [{sp_name}] Breeding range: "
                      f"'{os.path.basename(breeding_range_path)}' "
                      f"({(breeding_range > 0).sum()} px >0).\033[0m")
            else:
                breeding_range = None

            # Resolve sampling-effort raster for this species
            if use_per_species_effort:
                sp_key = sp_name.replace("_", " ").strip().lower()
                group = sp_to_group.get(sp_key)
                if group is None:
                    print(f"  [{sp_name}] No sampling effort group found — skipping.")
                    record["Err"] = "no_sampling_effort_group"
                    records.append(record)
                    _append_csv(output_csv, record)
                    continue
                group_key = group.replace("_", " ").strip().lower()
                if group_key not in effort_file_map:
                    print(f"  [{sp_name}] Raster for group '{group}' not found — skipping.")
                    record["Err"] = f"missing_raster_group_{group}"
                    records.append(record)
                    _append_csv(output_csv, record)
                    continue
                if group_key not in _effort_cache:
                    _effort_cache[group_key] = np.array(Image.open(effort_file_map[group_key]))
                taxa_ref = _effort_cache[group_key]
            else:
                taxa_ref = taxa_ref_global

            pres_thresh = calibrate_presence_threshold(
                cr, obs, taxa_ref, plot=save_fig_folder is not None,
                save_path=(
                    os.path.join(save_fig_folder, f"{sp_name}_pres_thresh.png")
                    if save_fig_folder else None
                ),
                species_name=sp_name,
            )

            # Run the heavy HS-bin computation without plotting yet, so we can
            # correct the presence threshold before the figure is generated.
            L, k, x0, bin_data = estimate_carrying_capacity(
                hs, obs, taxa_ref, cr,
                plot=False,
                save_path=None,
                species_name=sp_name,
                return_bins=True,
            )

            # Kmax is the true maximum of the fitted logistic curve over the
            # real HS domain [0, 1] — NOT the asymptote parameter L.
            # logistic(x) = g(x) - g(0) is shifted so f(0)=0, so its actual
            # max on [0,1] is g(1)-g(0), which is only ≈ L when x0 is close
            # to 0. For species where x0 sits further from 0, using L instead
            # of the true curve max underestimates the threshold-correction
            # trigger and lets an unreliable pres_thresh slip through uncorrected.
            Kmax = float(logistic(np.array([1.0]), L, k, x0)[0])
            if pres_thresh > Kmax:
                print(
                    f"\033[93m  [{sp_name}] presence_threshold ({pres_thresh:.4f}) "
                    f"> Kmax ({Kmax:.4f}) — correcting to Kmax/2 = {Kmax/2:.4f}\033[0m"
                )
                pres_thresh = Kmax / 2

            # Now plot with the (possibly corrected) threshold.
            if save_fig_folder is not None:
                plot_carrying_capacity(
                    bin_data, L, k, x0,
                    presence_threshold=pres_thresh,
                    species_name=sp_name,
                    save_path=os.path.join(save_fig_folder, f"{sp_name}_carrying_cap.png"),
                    show=True,
                )

            calib_sites = sample_calibration_sites(
                hs, obs, taxa_ref,
                n_samples=n_calibration_samples,
                window_size=map_window,
                distmin=80,
                time_budget=time_budget,
                plot=True,
                verbose=False,
                save_path=os.path.join(save_fig_folder, f"{sp_name}_calib_sites_map.png"),
                species_name=sp_name,
                breeding_range=breeding_range,
            )

            # If the time budget expired without finding any site that passes
            # the quality criteria, the learning cannot proceed. Raise an
            # explicit error rather than letting a ZeroDivisionError occur
            # deep inside the optimiser.
            if len(calib_sites) == 0:
                raise ValueError(
                    "Cannot find reliable calibration sites for this species: "
                    "no window passed the quality criteria within the time budget."
                )

            posts_masks = compute_all_posteriors(
                calib_sites, verbose=False,
                save_path=(
                    os.path.join(save_fig_folder, f"{sp_name}_resampl.png")
                    if save_fig_folder else None
                ),
                species_name=sp_name,
            )

            if method == "compare":
                if point1 is None or point2 is None:
                    raise ValueError(
                        "method='compare' requires point1=(s, d, S_crit) and "
                        "point2=(s, d, S_crit)."
                    )
                compare_equilibrium_distributions(
                    calib_sites, hmean, mdd_val, (L, k, x0),
                    point1=point1, point2=point2,
                    posteriors_and_masks=posts_masks,
                    label1=compare_label1, label2=compare_label2,
                    save_fig_folder=save_fig_folder, species_name=sp_name,
                    max_sites_per_figure=compare_max_sites_per_figure,
                    debug=compare_debug, debug_site_idx=compare_debug_site_idx,
                )
                # Diagnostic-only: no (n, r, Tg) is "learned" here, so don't
                # touch learned_parameters.csv at all for this species (see
                # the `if method == "compare": continue` below) — errors
                # during exploratory comparisons must stay independent of
                # the resume/skip bookkeeping used for real optimisation runs.
                continue
            elif method == "low_MPA_high":
                npz_folder = low_map_high_npz_folder or output_folder
                npz_path = os.path.join(npz_folder, f"{sp_name}_samples.npz")
                if not os.path.exists(npz_path):
                    print(f"\033[91m  [{sp_name}] No '{npz_path}' found — run "
                          f"method='MALA' for this species first (or point "
                          f"low_map_high_npz_folder at wherever its "
                          f"*_samples.npz already lives) — skipping.\033[0m")
                    continue
                # Diagnostic-only, same as method="compare" — doesn't touch
                # learned_parameters.csv, and a per-species failure here
                # shouldn't affect the resume/skip bookkeeping of real
                # optimisation runs.
                estimate_low_map_high_from_posterior(
                    npz_path, hmean, mdd_val,
                    ci_mass=low_map_high_ci_mass,
                    map_grid_size=low_map_high_map_grid_size,
                    save_fig_folder=save_fig_folder,
                    species_name=sp_name,
                )
                continue
            elif method == "refine":
                # init_point=None is allowed — refine_from_point itself
                # then starts exactly at the (s, d, S_crit) box centre.
                Ew, n, r, Tg, *_ = refine_from_point(
                    calib_sites, hmean, mdd_val, posts_masks, (L, k, x0),
                    init_point=init_point,
                    max_iter=refine_max_iter,
                    lr=refine_lr,
                    chunk_size=refine_chunk_size,
                    check_every=refine_check_every,
                    plot_summary=True,
                    save_fig_folder=save_fig_folder,
                    species_name=sp_name,
                    r_min=r_min,
                    r_survival_tol=r_survival_tol,
                    d_max_r_survival_tol=d_max_r_survival_tol,
                    r_survival_alpha=r_survival_alpha,
                    s_halfwidth=s_halfwidth,
                    seed_fraction=seed_fraction,
                    gpu_memory_fraction=gpu_memory_fraction,
                    robust_gradient=robust_gradient,
                    compute_hessian=compute_hessian,
                    hessian_eps=hessian_eps,
                )
            elif method == "MALA":
                mala_result = run_mala(
                    calib_sites, hmean, mdd_val, posts_masks, (L, k, x0),
                    init_point=init_point,
                    num_samples=mala_num_samples,
                    warmup_steps=mala_warmup_steps,
                    num_chains=mala_num_chains,
                    step_size=mala_step_size,
                    target_accept_prob=mala_target_accept_prob,
                    chunk_size=mala_chunk_size,
                    disperse_chains=mala_disperse_chains,
                    pre_optimize_steps=mala_pre_optimize_steps,
                    r_min=r_min,
                    r_survival_tol=r_survival_tol,
                    d_max_r_survival_tol=d_max_r_survival_tol,
                    r_survival_alpha=r_survival_alpha,
                    s_halfwidth=s_halfwidth,
                    seed_fraction=seed_fraction,
                    gpu_memory_fraction=gpu_memory_fraction,
                    plot_summary=True,
                    save_fig_folder=save_fig_folder,
                    species_name=sp_name,
                    seed=seed,
                )
                samples = mala_result["samples"]
                ci_95 = mala_result["ci_95"]
                # The joint posterior itself (not just its mean) is the
                # actual point of this method — save the raw samples,
                # their 95% credible intervals, and the acceptance rate to
                # disk, since `learned_parameters.csv` below only has room
                # for a single point per species (the posterior means).
                # Saved next to `learned_parameters.csv` itself (in
                # `output_folder`, not `save_fig_folder`/figs), compressed
                # (`savez_compressed`) since posterior sample arrays are
                # pure floating-point data that compresses well and this
                # avoids depending on any extra library (h5py, pyarrow...)
                # beyond numpy, which is already a hard dependency here.
                np.savez_compressed(
                    os.path.join(output_folder, f"{sp_name}_samples.npz"),
                    **samples,
                    **{f"{p}_ci95": np.array(ci_95[p]) for p in ci_95},
                    accept_rate=np.array(mala_result["accept_rate"]),
                )
                C_mala = (2.0 * np.exp(-1.11 / mdd_val)) / (1.0 + np.exp(-2.0 * 1.11 / mdd_val))
                Ew = float(np.mean(C_mala / (hmean ** samples["r"] - C_mala)))
                n, r, Tg = (float(np.mean(samples[k])) for k in ("n", "r", "Tg"))
            else:
                Ew, n, r, Tg, *_ = learn_dispersal_parameters(
                    calib_sites, hmean, mdd_val, posts_masks, (L, k, x0),
                    max_iter=max_iter,
                    all_together=all_together,
                    n_learning_sites=n_learning_sites,
                    n_random_sites=n_random_sites,
                    average_method="size",
                    plot=False,
                    plot_summary=True,
                    verbose=False,
                    save_fig_folder=save_fig_folder,
                    species_name=sp_name,
                    seed=seed,
                    gpu_memory_fraction=gpu_memory_fraction,
                    r_min=r_min,
                    r_survival_tol=r_survival_tol,
                    d_max_r_survival_tol=d_max_r_survival_tol,
                    r_survival_alpha=r_survival_alpha,
                    s_halfwidth=s_halfwidth,
                    seed_fraction=seed_fraction,
                    compute_hessian=compute_hessian,
                    method=method,
                    points_per_axis=points_per_axis,
                    grid_n_range=grid_n_range,
                    grid_r_range=grid_r_range,
                    grid_tg_range=grid_tg_range,
                    init_params=init_params,
                    robust_gradient=robust_gradient,
                    batch_chunk_size=batch_chunk_size,
                    precise_gridscan_center=precise_gridscan_center,
                    precise_gridscan_s_window=precise_gridscan_s_window,
                    precise_gridscan_d_window=precise_gridscan_d_window,
                    precise_gridscan_Scrit_window=precise_gridscan_Scrit_window,
                )

            record.update({"ew": Ew, "n": n, "r": r, "tg": Tg})
            print(f"  {sp_name}: Ew={Ew:.1f}  n={n:.1f}  r={r:.5f}  Tg={Tg:.2f}")

        except Exception as e:
            # Save only the error message (not the full traceback) in the CSV,
            # so the output stays readable. Print the full traceback to the
            # console for debugging.
            record["Err"] = str(e)
            print(f"Error processing {sp_name}:\n{traceback.format_exc(limit=1)}")

        records.append(record)
        _append_csv(output_csv, record)

    return pd.DataFrame(records)


def _append_csv(path: str, record: dict) -> None:
    row = pd.DataFrame([record])
    if os.path.exists(path):
        existing = pd.read_csv(path)
        row = pd.concat([existing, row], ignore_index=True)
    row.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Object-oriented wrapper
# ---------------------------------------------------------------------------

class BatchLearner:
    """Batch processor for multi-species dispersal parameter estimation.

    Parameters
    ----------
    folder_hs:
        Directory with HS rasters.
    folder_obs:
        Directory with observation rasters.
    folder_range:
        Directory with current-range rasters.
    path_taxa_ref:
        Path to the sampling-effort raster.
    path_mdd_table:
        CSV with MDD values.
    output_folder:
        Output directory.

    Examples
    --------
    >>> from paradis.learning import BatchLearner
    >>> bl = BatchLearner("HS/", "Obs/", "CR/", "taxa.tif", "mdd.csv", "output/")
    >>> df = bl.run()
    """

    def __init__(
        self,
        folder_hs: str,
        folder_obs: str,
        folder_range: str,
        path_mdd_table: str,
        output_folder: str,
        path_taxa_ref: str | None = None,
        path_sampling_effort_groups: str | None = None,
        folder_sampling_effort: str | None = None,
        folder_breeding_range: str | None = None,
        **kwargs,
    ) -> None:
        self.folder_hs = folder_hs
        self.folder_obs = folder_obs
        self.folder_range = folder_range
        self.path_mdd_table = path_mdd_table
        self.output_folder = output_folder
        self.path_taxa_ref = path_taxa_ref
        self.path_sampling_effort_groups = path_sampling_effort_groups
        self.folder_sampling_effort = folder_sampling_effort
        self.folder_breeding_range = folder_breeding_range
        self._kwargs = kwargs

    def run(self) -> pd.DataFrame:
        """Execute the batch learning pipeline."""
        return learn_over_folder(
            self.folder_hs,
            self.folder_obs,
            self.folder_range,
            self.path_mdd_table,
            self.output_folder,
            path_taxa_ref=self.path_taxa_ref,
            path_sampling_effort_groups=self.path_sampling_effort_groups,
            folder_sampling_effort=self.folder_sampling_effort,
            folder_breeding_range=self.folder_breeding_range,
            **self._kwargs,
        )

    def __repr__(self) -> str:
        return f"BatchLearner(output='{self.output_folder}')"
