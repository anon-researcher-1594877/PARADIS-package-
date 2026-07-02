"""Quick-start example – Elanus caeruleus dispersal simulation over France.

Runs a complete dispersal simulation for the Black-winged Kite
(*Elanus caeruleus*) over France at 1 km resolution on a 200 × 200 km grid.

All rasters are loaded with :func:`paradis.io.load_hs`,
:func:`paradis.io.load_obs`, and :func:`paradis.io.load_mask` — NODATA,
scaling, and normalisation are handled automatically.

The initial distribution is a 10 × 10 km founder patch (value 0.017) in
southern France, stored in ``data/Init_distrib_Elanus_caeruleus_200km_sim.tif``.

Usage
-----
From the package root directory::

    python examples/quick_start.py

Expected output
---------------
* A year-by-year expansion grid via show_expansion().
* A three-panel summary figure via show_final().

Example output
--------------
.. image:: quickstart_example_run.png
"""

import sys
import pathlib
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from paradis.simulation import PopulationSimulator, show_expansion, show_final
from paradis.io import load_hs, load_obs, load_mask

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_DATA = pathlib.Path(__file__).parent / "data"

HS_PATH   = _DATA / "HS_Elanus_caeruleus_200km.tif"
MASK_PATH = _DATA / "depFrance_200km.tif"
INIT_PATH = _DATA / "Init_distrib_Elanus_caeruleus_200km_sim.tif"

# ---------------------------------------------------------------------------
# 1.  Load rasters — NODATA / scaling handled automatically
# ---------------------------------------------------------------------------

hs           = load_hs  (HS_PATH)
france_mask  = load_mask(MASK_PATH)
init_distrib = load_obs (INIT_PATH)   # continuous values — no binarisation

hs *= france_mask   # zero out pixels outside France

print(f"Map shape   : {hs.shape}")
print(f"HS range    : {hs[hs > 0].min():.4f} – {hs.max():.4f}")
print(f"Init patch  : {(init_distrib > 0).sum()} px  (value {init_distrib.max():.4f})")

# ---------------------------------------------------------------------------
# 2.  Set up the simulator
# ---------------------------------------------------------------------------

# parameters are estimated in the quick_start_learning example, and are hard-coded here for convenience
PRESENCE_THRESHOLD = 0.017

sim = PopulationSimulator(
    hs                       = hs,
    ewalk                    = 924.5,
    n                        = 1980.5,
    r                        = 0.000918,
    tgrowth                  = 2.509,
    carrying_capacity_params = (0.02422, -7.2439, 0.5344),
    solver                   = "sparse",
    presence_threshold       = PRESENCE_THRESHOLD,
)

print("\nSimulator ready:", sim)
print(f"Device        : {'GPU' if torch.cuda.is_available() else 'CPU'}")

# ---------------------------------------------------------------------------
# 3.  Run the simulation
# ---------------------------------------------------------------------------

N_STEPS = 15
print(f"\nRunning {N_STEPS} simulation steps ...")

sim.run(
    n_steps      = N_STEPS,
    init_distrib = init_distrib,
    window_size  = 60,
    sub_window   = 30,
)

# ---------------------------------------------------------------------------
# 4.  Visualise
# ---------------------------------------------------------------------------

show_expansion(sim)
show_final(sim)
