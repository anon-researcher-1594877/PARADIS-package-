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
from paradis.calibration.carrying_capacity import estimate_carrying_capacity
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
    file_lists = [os.listdir(f) for f in folders]
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

    return ordered, var_parts_per_folder


# ---------------------------------------------------------------------------
# Batch learning pipeline
# ---------------------------------------------------------------------------

def learn_over_folder(
    folder_hs: str,
    folder_obs: str,
    folder_range: str,
    path_taxa_ref: str,
    path_mdd_table: str,
    output_folder: str,
    save_fig_folder: str | None = None,
    name_formats: list[str] | None = None,
    mdd_col: str = "Dispersal_km",
    species_col: str = "scientificName",
    map_window: int = 70,
    max_iter: int = 50,
    n_calibration_samples: int = 800,
    time_budget: float = 30.0,
) -> pd.DataFrame:
    """Run the full PARADIS calibration pipeline for every species in *folder_hs*.

    Previously computed species are skipped (resume-friendly).

    Parameters
    ----------
    folder_hs:
        Directory of habitat-suitability rasters (GeoTIFF or PIL-readable).
    folder_obs:
        Directory of observation count rasters.
    folder_range:
        Directory of current-range rasters.
    path_taxa_ref:
        Path to the reference-taxa sampling-effort raster.
    path_mdd_table:
        CSV with columns *species_col* and *mdd_col*.
    output_folder:
        Directory for the output CSV.
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

    os.makedirs(output_folder, exist_ok=True)
    if save_fig_folder:
        os.makedirs(save_fig_folder, exist_ok=True)

    ordered, var_parts = align_folders(
        [folder_hs, folder_obs, folder_range], name_formats
    )
    mdd_table = pd.read_csv(path_mdd_table)
    taxa_ref = np.array(Image.open(path_taxa_ref))

    output_csv = os.path.join(output_folder, "learned_parameters.csv")
    records = []

    for i in range(len(ordered[0])):
        sp_name = var_parts[0][i]
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
                os.path.join(folder_hs,  os.listdir(folder_hs) [ordered[0][i]])
            )).astype(float)
            obs = np.array(Image.open(
                os.path.join(folder_obs, os.listdir(folder_obs)[ordered[1][i]])
            )).astype(float)
            cr = np.array(Image.open(
                os.path.join(folder_range, os.listdir(folder_range)[ordered[2][i]])
            )).astype(float)

            # Normalise
            hs[hs == np.nanmax(hs)] = 0.0
            hs = hs / np.nanmax(hs)
            cr[cr == np.nanmin(cr)] = 0.0
            cr = cr / np.nanmax(cr)

            mdd_val = float(mdd_match[mdd_col].values[0])
            hmean = float(hs[cr > 0].mean())

            pres_thresh = calibrate_presence_threshold(
                cr, obs, taxa_ref, plot=False,
                save_path=(
                    os.path.join(save_fig_folder, f"{sp_name}_pres_thresh.png")
                    if save_fig_folder else None
                ),
                species_name=sp_name,
            )

            L, k, x0 = estimate_carrying_capacity(
                hs, obs, taxa_ref, cr, plot=False,
                presence_threshold=pres_thresh,
                save_path=(
                    os.path.join(save_fig_folder, f"{sp_name}_carrying_cap.png")
                    if save_fig_folder else None
                ),
                species_name=sp_name,
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
                verbose=False,
            )

            record.update({"ew": Ew, "n": n, "r": r, "tg": Tg})
            print(f"  {sp_name}: Ew={Ew:.1f}  n={n:.1f}  r={r:.5f}  Tg={Tg:.2f}")

        except Exception:
            record["Err"] = traceback.format_exc(limit=1)
            print(f"Error processing {sp_name}:\n{record['Err']}")

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
        path_taxa_ref: str,
        path_mdd_table: str,
        output_folder: str,
        **kwargs,
    ) -> None:
        self.folder_hs = folder_hs
        self.folder_obs = folder_obs
        self.folder_range = folder_range
        self.path_taxa_ref = path_taxa_ref
        self.path_mdd_table = path_mdd_table
        self.output_folder = output_folder
        self._kwargs = kwargs

    def run(self) -> pd.DataFrame:
        """Execute the batch learning pipeline."""
        return learn_over_folder(
            self.folder_hs,
            self.folder_obs,
            self.folder_range,
            self.path_taxa_ref,
            self.path_mdd_table,
            self.output_folder,
            **self._kwargs,
        )

    def __repr__(self) -> str:
        return f"BatchLearner(output='{self.output_folder}')"
