"""Quick-start batch learning example – parameter calibration for a folder of species.

Demonstrates the :class:`~paradis.learning.BatchLearner` pipeline, which runs
the full PARADIS calibration (steps 1-7) over an entire folder of species
rasters instead of a single species at a time. This is the batch equivalent
of ``quick_start_learning_params.py``.

The example restricts the run to two species (*Canis lupus* and
*Castor fiber*) via the ``test_species`` parameter, so it can be used as a
smoke test without waiting for the whole folder to be processed.

Expected folder layout
-----------------------
::

    <data>/
        HS_mammals/                       one HS raster per species
        Obs_mammals/                      one observation-count raster per species
        Current_Range_mammals/            one binary current-range raster per species
        Dispersal_mammals.csv             MDD table (species_col, mdd_col)
        sampling_effort_groups.csv        species -> sampling-effort group
        sampling_effort_<group>.tif       one raster per sampling-effort group

This example ships with a self-contained two-species subset of this layout
under ``examples/data_batch/`` (Canis lupus + Castor fiber only), so it runs
out of the box without any external data.

Usage
-----
From the package root::

    python examples/quick_start_batch_learning.py
"""

import random
import sys
import pathlib

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from paradis.learning import BatchLearner

# ── Reproducibility seed ─────────────────────────────────────────────────────
_SEED = 42
random.seed(_SEED)
np.random.seed(_SEED)
torch.manual_seed(_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(_SEED)

# ---------------------------------------------------------------------------
# Self-contained two-species dataset bundled with the package
# (examples/data_batch/): Canis lupus and Castor fiber only. To run the batch
# learner on your own full folder of species, point _DATA elsewhere and drop
# the test_species restriction below.
# ---------------------------------------------------------------------------
_DATA = pathlib.Path(__file__).parent / "data_batch"
_OUT  = pathlib.Path(__file__).parent / "quick_start_batch_output"

# ---------------------------------------------------------------------------
# Build the batch learner
# ---------------------------------------------------------------------------

bl = BatchLearner(
    folder_hs     = str(_DATA / "HS_mammals"),
    folder_obs    = str(_DATA / "Obs_mammals"),
    folder_range  = str(_DATA / "Current_Range_mammals"),
    path_mdd_table = str(_DATA / "Dispersal_mammals.csv"),
    output_folder  = str(_OUT),

    # File-name patterns: "XxX" is replaced by the species name in each folder.
    # Here HS/Obs use "<species>.tif" and current-range uses "<species>binary_50.tif".
    name_formats = ["XxX.tif", "XxX.tif", "XxXbinary_50.tif"],

    # Per-species sampling effort: species are mapped to a group (e.g.
    # "Chiroptera", "large_mammalia") via the CSV, and each group has its own
    # sampling-effort raster in folder_sampling_effort. Omit both and pass
    # path_taxa_ref instead to use a single raster for every species.
    path_sampling_effort_groups = str(_DATA / "sampling_effort_groups.csv"),
    folder_sampling_effort      = str(_DATA),

    # Restrict the run to two species for this quick-start example.
    # Pass a single string, a list of strings, or omit entirely to run the
    # whole folder.
    test_species = ["Castor_fiber", "Canis_lupus"],

    # Adam steps per species — reduced here for speed; use ~2500 in practice.
    max_iter = 250,
)

# ---------------------------------------------------------------------------
# Run — this writes learned_parameters.csv (resumable) and, per species,
# diagnostic figures (presence-threshold, carrying-capacity, calibration-site
# map, resampling grid, cost curve) under <output_folder>/figs/.
# ---------------------------------------------------------------------------

df = bl.run()
print(df)
