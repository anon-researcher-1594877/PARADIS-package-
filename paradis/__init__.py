"""PARADIS – Species Population Dispersal model.

A Python package for simulating and calibrating habitat-driven dispersal of
animal populations from occurrence and habitat-suitability data.

Quick start
-----------
Run the full calibration pipeline in one call::

    import paradis
    results = paradis.calibrate(hs, obs, taxa_ref, mdd=21.0,
                                species_name="Elanus caeruleus", plot=True)
    print(results["Ew"], results["n"], results["r"], results["Tg"])

See ``examples/quick_start_learning.py`` for a complete worked example.

Subpackages
-----------
paradis.pipeline
    One-call calibration entry point (:func:`paradis.calibrate`).
paradis.core
    Adjacency matrix construction, dispersal kernels, and population growth.
paradis.calibration
    Site selection, presence threshold, carrying capacity, and posteriors.
paradis.learning
    Gradient-based dispersal parameter estimation.
paradis.simulation
    Sliding-window dispersal simulation engine (deterministic + stochastic).
paradis.io
    Raster loading and shapefile rasterisation.
paradis.visualization
    Plotting utilities.
"""

__version__ = "0.1.0"
__author__ = "PARADIS contributors"

# ── CUDA availability check ──────────────────────────────────────────────────
import torch as _torch

if _torch.cuda.is_available():
    _dev = _torch.cuda.get_device_name(0)
    print(f"[PARADIS] CUDA available — using GPU: {_dev}")
else:
    print(
        "\n"
        "╔══════════════════════════════════════════════════════════════════╗\n"
        "║  PARADIS — CUDA not detected                                     ║\n"
        "╠══════════════════════════════════════════════════════════════════╣\n"
        "║  This package relies heavily on GPU acceleration via CUDA.       ║\n"
        "║  Running on CPU is possible but will be significantly slower.    ║\n"
        "║                                                                  ║\n"
        "║  To install CUDA support:                                        ║\n"
        "║                                                                  ║\n"
        "║  1. Check your GPU is CUDA-capable:                              ║\n"
        "║     https://developer.nvidia.com/cuda-gpus                       ║\n"
        "║                                                                  ║\n"
        "║  2. Install the CUDA toolkit (≥ 11.8 recommended):               ║\n"
        "║     https://developer.nvidia.com/cuda-downloads                  ║\n"
        "║                                                                  ║\n"
        "║  3. Install PyTorch with CUDA support:                           ║\n"
        "║     pip install torch torchvision --index-url                    ║\n"
        "║       https://download.pytorch.org/whl/cu121                    ║\n"
        "║     (replace cu121 with your CUDA version, e.g. cu118)          ║\n"
        "║                                                                  ║\n"
        "║  4. Verify with:  python -c \"import torch;                       ║\n"
        "║                   print(torch.cuda.is_available())\"              ║\n"
        "╚══════════════════════════════════════════════════════════════════╝\n"
    )
del _torch

from paradis._device import device
from paradis.pipeline import calibrate, plot_calibration_summary

# Convenience re-exports so ``from paradis import X`` works for the most common
# classes and functions.
from paradis.core import (
    DispersalKernel,
    GrowthModel,
    adjacency_matrix_torch,
    dispersal_kernel,
    distance_matrix_from_raster,
)
from paradis.calibration import (
    CalibrationSites,
    sample_calibration_sites,
    calibrate_presence_threshold,
    estimate_carrying_capacity,
    compute_all_posteriors,
)
from paradis.learning import (
    ParameterLearner,
    learn_dispersal_parameters,
    BatchLearner,
)
from paradis.simulation import (
    PopulationSimulator,
    run_simulation,
)

__all__ = [
    "__version__",
    "device",
    # pipeline
    "calibrate",
    "plot_calibration_summary",
    # core
    "DispersalKernel",
    "GrowthModel",
    "adjacency_matrix_torch",
    "dispersal_kernel",
    "distance_matrix_from_raster",
    # calibration
    "CalibrationSites",
    "sample_calibration_sites",
    "calibrate_presence_threshold",
    "estimate_carrying_capacity",
    "compute_all_posteriors",
    # learning
    "ParameterLearner",
    "learn_dispersal_parameters",
    "BatchLearner",
    # simulation
    "PopulationSimulator",
    "run_simulation",
]
