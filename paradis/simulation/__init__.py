"""Dispersal simulation engine (deterministic and stochastic)."""

from paradis.simulation.engine import (
    PopulationSimulator,
    crop_to_mask,
    dispersal_step,
    plot_density_map,
    run_simulation,
    show_expansion,
    show_final,
)
from paradis.simulation.kernel_cache import DiskKernelCache, MemoryKernelCache
from paradis.simulation.stochastic import (
    apply_kernel,
    lognormal_2d_kernel,
    stochastic_dispersal_step,
)
from paradis.simulation.tuning import suggest_window_params

__all__ = [
    "PopulationSimulator",
    "crop_to_mask",
    "plot_density_map",
    "show_expansion",
    "show_final",
    "DiskKernelCache",
    "MemoryKernelCache",
    "dispersal_step",
    "run_simulation",
    "apply_kernel",
    "lognormal_2d_kernel",
    "stochastic_dispersal_step",
    "suggest_window_params",
]
