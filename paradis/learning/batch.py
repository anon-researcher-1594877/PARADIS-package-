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
import traceback

import numpy as np
import pandas as pd
from PIL import Image

from paradis.calibration.sites import sample_calibration_sites
from paradis.calibration.presence import calibrate_presence_threshold
from paradis.calibration.carrying_capacity import estimate_carrying_capacity, plot_carrying_capacity
from paradis.calibration.ratios import compute_all_posteriors
from paradis.learning.optimizer import learn_dispersal_parameters


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
    save_fig_folder: str | None = None,
    name_formats: list[str] | None = None,
    mdd_col: str = "Dispersal_km",
    species_col: str = "scientificName",
    map_window: int = 70,
    max_iter: int = 50,
    n_calibration_samples: int = 800,
    time_budget: float = 30.0,
    test_species: str | list[str] | None = None,
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
    n_calibration_samples:
        Random candidate centres per iteration.
    time_budget:
        Time budget for site selection (seconds).

    Returns
    -------
    pandas.DataFrame
        Table with learned parameters (``Ew``, ``n``, ``r``, ``Tg``).
    """
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

            # Kmax is the asymptote L of the fitted logistic K=f(HS).
            # If the presence threshold exceeds Kmax the threshold is
            # unreliable (it would classify every pixel as absent), so we
            # fall back to Kmax/2 and warn the user.
            Kmax = L
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
                plot=False,
                verbose=False,
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

            Ew, n, r, Tg, *_ = learn_dispersal_parameters(
                calib_sites, hmean, mdd_val, posts_masks, (L, k, x0),
                max_iter=max_iter,
                all_together=True,
                n_random_sites=3,
                average_method="size",
                plot=False,
                plot_summary=True,
                verbose=False,
                save_fig_folder=save_fig_folder,
                species_name=sp_name,
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
            **self._kwargs,
        )

    def __repr__(self) -> str:
        return f"BatchLearner(output='{self.output_folder}')"
