# PARADIS – Species Population Dispersal

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

**PARADIS** is a Python package for modelling habitat-driven species dispersal from occurrence
data and habitat-suitability (HS) maps.  It implements:

- A **random-walk dispersal kernel** where transition probabilities are derived from a
  weighted adjacency matrix of the habitat raster (GPU-accelerated via PyTorch).
- A **calibration pipeline** that selects geographically spread calibration sites,
  estimates per-pixel relative-abundance posteriors, and infers a logistic
  carrying-capacity relationship.
- **Gradient-based parameter learning** (via PyTorch autograd) that fits three
  dispersal parameters (risk scaling *r*, risk avoidance *n*, characteristic growth
  time *Tg*) to the observed spatial pattern of occurrence ratios.
- A **population simulation engine** that alternates sliding-window dispersal with
  logistic growth over an arbitrary number of time steps.

---

## Installation

### From source (development)

```bash
git clone https://github.com/your-org/paradis.git
cd paradis
pip install -e ".[dev,geo]"
```

### Core dependencies only

```bash
pip install -e .
```

Geospatial extras (`rasterio`, `geopandas`) are required only when loading
GeoTIFF files or rasterising shapefiles:

```bash
pip install -e ".[geo]"
```

> **GPU support** – Install `torch` with CUDA support before installing PARADIS if
> you want GPU acceleration.  See [pytorch.org/get-started](https://pytorch.org/get-started/locally/).

---

## Quick start

A self-contained example with synthetic data is provided in
[`examples/quick_start.py`](examples/quick_start.py).

```python
import numpy as np
from paradis.simulation import PopulationSimulator

# Create a synthetic 100 × 100 habitat suitability map
hs = np.random.rand(100, 100).astype("float32")
hs[hs < 0.1] = 0.0  # impassable cells (e.g. ocean)

# Set up the simulator with known parameters
sim = PopulationSimulator(
    hs=hs,
    ewalk=50.0,          # expected steps per dispersal event
    tolerance=500.0,     # risk-avoidance exponent
    alpha=0.05,          # risk-scaling exponent
    tgrowth=7.5,         # years to reach 95 % of carrying capacity
    carrying_capacity_params=(0.08, 5.0, 0.5),  # logistic K(HS) params
    presence_threshold=0.02,
)

# Run 20 time steps from a small seed population at the centre
result = sim.run(n_steps=20, plot=True)
```

---

## Package structure

```
paradis/
├── core/
│   ├── adjacency.py          # Adjacency matrix construction
│   ├── dispersal.py          # Dispersal kernel (DispersalKernel class)
│   └── growth.py             # Population growth (GrowthModel class)
├── calibration/
│   ├── sites.py              # Site selection (CalibrationSites class)
│   ├── presence.py           # Presence threshold calibration
│   ├── carrying_capacity.py  # Logistic K(HS) fitting
│   └── ratios.py             # Observation-ratio posteriors
├── learning/
│   ├── optimizer.py          # Gradient descent (ParameterLearner class)
│   └── batch.py              # Multi-species batch runner (BatchLearner class)
├── simulation/
│   ├── engine.py             # Sliding-window simulation (PopulationSimulator class)
│   └── stochastic.py        # Stochastic & long-distance dispersal
├── io/
│   └── raster.py             # GeoTIFF loading and shapefile rasterisation
├── visualization/
│   └── plots.py              # Plotting utilities
└── _device.py                # Centralised GPU/CPU device selection
```

---

## Full calibration + learning workflow

```python
import numpy as np
from paradis.io import load_tif
from paradis.calibration import (
    calibrate_presence_threshold,
    estimate_carrying_capacity,
    sample_calibration_sites,
    compute_all_posteriors,
)
from paradis.learning import ParameterLearner

# Load data
hs   = load_tif("path/to/hs.tif")
obs  = load_tif("path/to/obs.tif", normalise=False)
taxa = load_tif("path/to/taxa_ref.tif", normalise=False)
cr   = load_tif("path/to/current_range.tif", normalise=False)

# 1. Calibrate presence threshold
t_pres = calibrate_presence_threshold(cr, obs, taxa, plot=True)

# 2. Fit carrying-capacity logistic curve
L, k, x0 = estimate_carrying_capacity(hs, obs, taxa, cr, plot=True)

# 3. Select calibration sites
sites = sample_calibration_sites(hs, obs, taxa, window_size=70, plot=True)

# 4. Compute observation-ratio posteriors
posts, masks = compute_all_posteriors(sites)

# 5. Learn dispersal parameters
hmean = float(hs[cr > 0].mean())
mdd   = 39.0  # prior mean dispersal distance (km → pixels depending on resolution)

learner = ParameterLearner(mdd=mdd, max_iter=300)
Ew, n, r, Tg, costs, weights, endpoints = learner.fit(
    sites, hmean, (posts, masks), (L, k, x0),
    all_together=True,
    n_random_sites=3,
    average_method="size",
)
print(f"Learned: Ew={Ew:.1f}  n={n:.1f}  r={r:.5f}  Tg={Tg:.2f}")

# 6. Run simulation
from paradis.simulation import PopulationSimulator
init = np.zeros_like(hs); init[hs > 0.5] = t_pres
sim = PopulationSimulator(hs, Ew, n, r, Tg, (L, k, x0), t_pres)
final = sim.run(n_steps=35, plot=True)
```

---

## Batch processing (multiple species)

```python
from paradis.learning import BatchLearner

bl = BatchLearner(
    folder_hs="data/HS/",
    folder_obs="data/Obs/",
    folder_range="data/CurrentRange/",
    path_taxa_ref="data/sampling_effort.tif",
    path_mdd_table="data/dispersal_distances.csv",
    output_folder="output/",
)
df = bl.run()
```

---

## Citation

If you use PARADIS in published work, please cite:

> *[Author et al. (in prep.)]* PARADIS: a gradient-based framework for inferring
> habitat-driven dispersal parameters from occurrence data.

---

## Contributing

Contributions are welcome.  Please open an issue before submitting a pull
request.  Code should pass `ruff check .` and `pytest` before review.

---

## License

MIT – see [LICENSE](LICENSE).
