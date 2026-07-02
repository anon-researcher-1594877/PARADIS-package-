"""Quick-start learning example – parameter calibration for Elanus caeruleus.

Demonstrates the full PARADIS calibration pipeline using real data for the
Black-winged Kite (*Elanus caeruleus*) over Europe.  The entire calibration
(steps 1–7) is executed by a single call to :func:`paradis.calibrate`.

Known reference parameters (full-dataset calibration)
------------------------------------------------------
    Ew = 1137,  n = 2094,  r = 0.0159,  Tg = 2.35

Usage
-----
From the package root::

    python examples/quick_start_learning.py
"""

import random
import sys
import pathlib

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
import paradis
from paradis.io import load_hs, load_obs, load_mask

# ── Reproducibility seed ─────────────────────────────────────────────────────
_SEED = 42
random.seed(_SEED)
np.random.seed(_SEED)
torch.manual_seed(_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(_SEED)


_DATA    = pathlib.Path(__file__).parent / "data"
_OUT_PNG = pathlib.Path(__file__).parent / "quick_start_learning_result.png"

# ---------------------------------------------------------------------------
# Load rasters — one line each, all NODATA / scaling handled automatically
# ---------------------------------------------------------------------------

hs            = load_hs  (_DATA / "HS_Elanus_caeruleus_Europe.tif")
obs           = load_obs (_DATA / "Obs_Elanus_caeruleus_Europe.tif")
taxa_ref      = load_obs (_DATA / "Taxa_Aves_Europe.tif")
current_range = load_mask(_DATA / "Range_Elanus_caeruleus_Europe.tif")

# ---------------------------------------------------------------------------
# Run the full calibration pipeline in one call
# ---------------------------------------------------------------------------

# for speed, the number of learning step is reduced here to 10 (however it is usually of the order of 2500)

results = paradis.calibrate(
    hs, obs, taxa_ref,
    mdd           = 21.0,
    current_range = current_range,
    hs_bin_width  = 0.025,
    window_size   = 70,   # side length of each calibration site in pixels (= km here)
    distmin       = 70.0, # minimum distance between site centres (pixels)
    min_obs       = 5,    # minimum species observations required inside a site
    max_iter      = 15,   # change until convergence, default is 2500
    plot          = True, # set to True to plot the calibration summary figure
    save_path     = str(_OUT_PNG),
    species_name  = "Elanus caeruleus",
)

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

print(
    f"Ew={results['Ew']:.4f}  n={results['n']:.4f}  "
    f"r={results['r']:.8f}  Tg={results['Tg']:.6f}  "
    f"L={results['L']:.6f}  k={results['k']:.6f}  x0={results['x0']:.6f}"
)

# ---------------------------------------------------------------------------
# Re-plot the summary figure without re-running calibration
# ---------------------------------------------------------------------------
# The results dict contains everything needed to regenerate the figure.
# Call this any time after calibrate() to adjust save path or display.

#paradis.plot_calibration_summary(
#    results,
#    species_name = "Elanus caeruleus",
#    save_path    = str(_OUT_PNG),
#    show         = True,
#)
