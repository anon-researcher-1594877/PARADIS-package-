"""Tests for paradis.core – adjacency, dispersal kernel, and growth model."""

import numpy as np
import pytest
import torch

from paradis.core.adjacency import adjacency_matrix, adjacency_matrix_torch
from paradis.core.dispersal import (
    DispersalKernel,
    dispersal_kernel_fast,
    row_normalise,
    weighted_adjacency,
)
from paradis.core.growth import GrowthModel, carrying_capacity_from_hs, growth_coefficient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_hs_np():
    """5 × 5 float32 HS map with known values."""
    rng = np.random.default_rng(0)
    return rng.random((5, 5)).astype(np.float32)


@pytest.fixture
def small_hs_torch(small_hs_np):
    return torch.tensor(small_hs_np)


# ---------------------------------------------------------------------------
# Adjacency matrix
# ---------------------------------------------------------------------------

class TestAdjacencyMatrix:
    def test_shape_numpy(self, small_hs_np):
        adj = adjacency_matrix(small_hs_np)
        n = 5 * 5
        assert adj.shape == (n, n)

    def test_shape_torch(self, small_hs_torch):
        adj = adjacency_matrix_torch(small_hs_torch)
        n = 5 * 5
        assert adj.shape == (n, n)

    def test_symmetry_numpy(self, small_hs_np):
        adj = adjacency_matrix(small_hs_np)
        np.testing.assert_allclose(adj, adj.T, atol=1e-6)

    def test_symmetry_torch(self, small_hs_torch):
        adj = adjacency_matrix_torch(small_hs_torch)
        diff = (adj - adj.T).abs().max().item()
        assert diff < 1e-5

    def test_nonnegative(self, small_hs_np):
        adj = adjacency_matrix(small_hs_np)
        assert np.all(adj >= 0)

    def test_nan_treated_as_zero(self):
        hs = np.ones((3, 3), dtype=np.float32)
        hs[1, 1] = np.nan
        adj = adjacency_matrix(hs)
        assert np.all(np.isfinite(adj))

    def test_numpy_torch_agree(self, small_hs_np, small_hs_torch):
        adj_np = adjacency_matrix(small_hs_np)
        adj_t = adjacency_matrix_torch(small_hs_torch).numpy()
        np.testing.assert_allclose(adj_np, adj_t, atol=1e-5)

    def test_edge_weights_are_averages(self):
        hs = np.array([[0.4, 0.6], [0.8, 1.0]], dtype=np.float32)
        adj = adjacency_matrix(hs)
        # Pixel (0,0) ↔ (0,1): weight = (0.4 + 0.6) / 2 = 0.5
        assert abs(adj[0, 1] - 0.5) < 1e-6
        # Pixel (0,0) ↔ (1,0): weight = (0.4 + 0.8) / 2 = 0.6
        assert abs(adj[0, 2] - 0.6) < 1e-6

    def test_uniform_map_weights_equal_to_value(self):
        v = 0.7
        hs = np.full((4, 4), v, dtype=np.float32)
        adj = adjacency_matrix(hs)
        # All non-zero entries should equal v
        nonzero = adj[adj > 0]
        np.testing.assert_allclose(nonzero, v, atol=1e-6)


# ---------------------------------------------------------------------------
# Row-normalise and weighted adjacency
# ---------------------------------------------------------------------------

class TestWeightedAdjacency:
    def test_row_normalise_shape(self, small_hs_torch):
        adj = adjacency_matrix_torch(small_hs_torch)
        rn = row_normalise(adj)
        assert rn.shape == adj.shape

    def test_weighted_adjacency_row_sum(self, small_hs_torch):
        adj = adjacency_matrix_torch(small_hs_torch)
        # Zero-out any rows that are all-zero (impassable pixels)
        Wstar = weighted_adjacency(adj, tolerance=1.0)
        row_sums = Wstar.sum(dim=1)
        # Rows with non-zero original adjacency should sum to ≤ 1
        non_isolated = adj.sum(dim=1) > 0
        assert (row_sums[non_isolated] <= 1.0 + 1e-5).all()

    def test_weighted_adjacency_nonnegative(self, small_hs_torch):
        adj = adjacency_matrix_torch(small_hs_torch)
        Wstar = weighted_adjacency(adj, tolerance=2.0)
        assert (Wstar >= 0).all()

    def test_higher_tolerance_sharpens_selection(self):
        """Higher tolerance → more weight on edges with high HS."""
        hs = torch.tensor([[0.1, 0.9], [0.5, 0.5]], dtype=torch.float32)
        adj = adjacency_matrix_torch(hs)
        Wl = weighted_adjacency(adj, tolerance=1.0)
        Wh = weighted_adjacency(adj, tolerance=100.0)
        # The high-HS edge should have more relative weight with high tolerance
        # (just check the matrices differ meaningfully)
        diff = (Wh - Wl).abs().max().item()
        assert diff > 1e-3


# ---------------------------------------------------------------------------
# Dispersal kernel
# ---------------------------------------------------------------------------

class TestDispersalKernelFast:
    def test_output_shape(self, small_hs_torch):
        adj = adjacency_matrix_torch(small_hs_torch)
        kernel = dispersal_kernel_fast(adj, alpha=0.05, tolerance=100.0, ewalk=50.0)
        n = 5 * 5
        assert kernel.shape == (n, n)

    def test_output_nonnegative(self, small_hs_torch):
        adj = adjacency_matrix_torch(small_hs_torch)
        kernel = dispersal_kernel_fast(adj, alpha=0.05, tolerance=100.0, ewalk=50.0)
        assert (kernel >= -1e-6).all()

    def test_row_sums_at_most_one(self, small_hs_torch):
        adj = adjacency_matrix_torch(small_hs_torch)
        kernel = dispersal_kernel_fast(adj, alpha=0.05, tolerance=100.0, ewalk=50.0)
        row_sums = kernel.sum(dim=1)
        assert (row_sums <= 1.0 + 1e-4).all()


class TestDispersalKernelClass:
    def test_repr(self):
        dk = DispersalKernel(alpha=0.05, tolerance=500.0, ewalk=100.0)
        assert "DispersalKernel" in repr(dk)

    def test_compute_returns_three_tensors(self, small_hs_torch):
        adj = adjacency_matrix_torch(small_hs_torch)
        dk = DispersalKernel(alpha=0.05, tolerance=500.0, ewalk=100.0)
        kernel, mean_dist, survival = dk.compute(adj)
        assert kernel.ndim == 2
        assert mean_dist.ndim == 0 or mean_dist.ndim == 1
        assert survival.ndim == 0 or survival.ndim == 1

    def test_survival_in_zero_one(self, small_hs_torch):
        adj = adjacency_matrix_torch(small_hs_torch)
        dk = DispersalKernel(alpha=0.05, tolerance=500.0, ewalk=50.0)
        _, _, survival = dk.compute(adj)
        s = float(survival)
        assert 0.0 <= s <= 1.0 + 1e-4


# ---------------------------------------------------------------------------
# Growth model
# ---------------------------------------------------------------------------

class TestGrowthCoefficient:
    def test_tgrowth_1(self):
        assert abs(growth_coefficient(1.0) - 0.05) < 1e-9

    def test_tgrowth_large_approaches_1(self):
        a = growth_coefficient(1000.0)
        assert 0.995 < a < 1.0

    def test_tgrowth_increases_a(self):
        a5 = growth_coefficient(5.0)
        a10 = growth_coefficient(10.0)
        assert a10 > a5


class TestCarryingCapacityFromHS:
    def test_shape(self):
        hs = np.random.rand(10, 10).astype(np.float32)
        a, K_is, vecb = carrying_capacity_from_hs(hs, tgrowth=7.5,
                                                   L=0.1, k=5.0, x0=0.5)
        assert K_is.shape[0] == 100
        assert vecb.shape[0] == 100

    def test_K_nonnegative(self):
        # k < 0 gives an increasing logistic → K(HS) >= 0 for HS in [0, 1]
        hs = np.random.rand(8, 8).astype(np.float32)
        _, K_is, _ = carrying_capacity_from_hs(hs, 7.5, 0.1, -5.0, 0.5)
        assert (K_is >= 0).all()

    def test_zero_hs_gives_zero_K(self):
        hs = np.zeros((4, 4), dtype=np.float32)
        _, K_is, _ = carrying_capacity_from_hs(hs, 7.5, 0.1, 5.0, 0.5)
        # K(0) = logistic(0) = 0 by the shifted definition
        np.testing.assert_allclose(K_is.numpy(), 0.0, atol=1e-6)

    def test_a_coefficient_matches_formula(self):
        tg = 5.0
        a, _, _ = carrying_capacity_from_hs(
            np.ones((3, 3), dtype=np.float32) * 0.5, tg, 0.1, 5.0, 0.5
        )
        expected = 0.05 ** (1.0 / tg)
        assert abs(a - expected) < 1e-9


class TestGrowthModelClass:
    def test_repr(self):
        gm = GrowthModel(tgrowth=7.5, L=0.1, k=5.0, x0=0.5)
        assert "GrowthModel" in repr(gm)

    def test_a_property(self):
        gm = GrowthModel(tgrowth=10.0, L=0.1, k=5.0, x0=0.5)
        expected = 0.05 ** (1.0 / 10.0)
        assert abs(gm.a - expected) < 1e-9

    def test_carrying_capacity_returns_tuple(self):
        gm = GrowthModel(tgrowth=7.5, L=0.1, k=5.0, x0=0.5)
        hs = np.random.rand(6, 6).astype(np.float32)
        result = gm.carrying_capacity(hs)
        assert len(result) == 3
