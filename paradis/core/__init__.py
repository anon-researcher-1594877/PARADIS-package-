"""Core mathematical components of the PARADIS dispersal model."""

from paradis.core.adjacency import adjacency_matrix, adjacency_matrix_torch
from paradis.core.dispersal import (
    DispersalKernel,
    core_dispersal,
    dispersal_kernel,
    dispersal_kernel_fast,
    dispersal_rules,
    distance_matrix_from_raster,
    row_normalise,
    single_window_dispersal,
    weighted_adjacency,
)
from paradis.core.growth import (
    GrowthModel,
    carrying_capacity_from_hs,
    equilibrium_distribution,
    growth_coefficient,
    growth_step,
)

__all__ = [
    # adjacency
    "adjacency_matrix",
    "adjacency_matrix_torch",
    # dispersal
    "DispersalKernel",
    "core_dispersal",
    "dispersal_kernel",
    "dispersal_kernel_fast",
    "dispersal_rules",
    "distance_matrix_from_raster",
    "row_normalise",
    "single_window_dispersal",
    "weighted_adjacency",
    # growth
    "GrowthModel",
    "carrying_capacity_from_hs",
    "equilibrium_distribution",
    "growth_coefficient",
    "growth_step",
]
