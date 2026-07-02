"""Smoke tests for paradis.simulation – PopulationSimulator and run_simulation.

These tests use a small synthetic habitat map and very few time steps so that
the full dispersal-growth loop runs quickly in a CI environment without GPU.
"""

import numpy as np
import pytest

from paradis.simulation import PopulationSimulator, run_simulation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_simulator(size: int = 50) -> PopulationSimulator:
    """Build a PopulationSimulator on a tiny synthetic map."""
    rng = np.random.default_rng(99)
    hs = rng.random((size, size)).astype(np.float32)
    hs[hs < 0.05] = 0.0  # impassable cells

    return PopulationSimulator(
        hs=hs,
        ewalk=10.0,
        tolerance=50.0,
        alpha=0.1,
        tgrowth=5.0,
        carrying_capacity_params=(0.08, 5.0, 0.5),
        presence_threshold=0.02,
    )


def _make_init(hs: np.ndarray, threshold: float) -> np.ndarray:
    h, w = hs.shape
    init = np.zeros_like(hs)
    cx, cy = h // 2, w // 2
    init[cx - 3: cx + 3, cy - 3: cy + 3] = threshold * 1.5
    return init


# ---------------------------------------------------------------------------
# PopulationSimulator construction
# ---------------------------------------------------------------------------

class TestPopulationSimulatorInit:
    def test_repr(self):
        sim = _make_simulator()
        assert "PopulationSimulator" in repr(sim)
        assert "Ew=" in repr(sim)

    def test_attributes_stored(self):
        sim = _make_simulator()
        assert sim.ewalk == 10.0
        assert sim.tolerance == 50.0
        assert sim.alpha == 0.1
        assert sim.tgrowth == 5.0
        assert sim.presence_threshold == 0.02


# ---------------------------------------------------------------------------
# PopulationSimulator.run – output shape and value properties
# ---------------------------------------------------------------------------

class TestPopulationSimulatorRun:
    """Smoke tests: verify output shape, dtype, and basic invariants."""

    @pytest.fixture(scope="class")
    def sim_and_result(self):
        sim = _make_simulator(size=50)
        init = _make_init(sim.hs, sim.presence_threshold)
        # Use small windows so the test runs in seconds
        result = sim.run(
            n_steps=2,
            init_distrib=init,
            window_size=20,
            sub_window=9,
            plot=False,
        )
        return sim, result

    def test_output_shape(self, sim_and_result):
        sim, result = sim_and_result
        assert result.shape == sim.hs.shape

    def test_output_is_numpy(self, sim_and_result):
        _, result = sim_and_result
        assert isinstance(result, np.ndarray)

    def test_output_nonnegative(self, sim_and_result):
        _, result = sim_and_result
        assert np.all(result >= -1e-6)

    def test_output_bounded(self, sim_and_result):
        _, result = sim_and_result
        assert np.nanmax(result) <= 1.0 + 1e-4

    def test_population_spread(self, sim_and_result):
        sim, result = sim_and_result
        n_present = int((result > sim.presence_threshold).sum())
        assert n_present > 0

    def test_default_init_distrib(self):
        """Passing no init_distrib should not raise."""
        sim = _make_simulator(size=50)
        result = sim.run(n_steps=1, window_size=20, sub_window=9, plot=False)
        assert result.shape == sim.hs.shape


# ---------------------------------------------------------------------------
# run_simulation functional API
# ---------------------------------------------------------------------------

class TestRunSimulation:
    def test_returns_ndarray(self):
        rng = np.random.default_rng(5)
        hs = rng.random((50, 50)).astype(np.float32)
        init = np.zeros_like(hs)
        init[22:28, 22:28] = 0.03
        result = run_simulation(
            hs=hs,
            init_distrib=init,
            ewalk=10.0,
            tolerance=50.0,
            alpha=0.1,
            tgrowth=5.0,
            carrying_capacity_params=(0.08, 5.0, 0.5),
            presence_threshold=0.02,
            n_steps=1,
            window_size=20,
            sub_window=9,
            plot=False,
        )
        assert isinstance(result, np.ndarray)
        assert result.shape == hs.shape

    def test_zero_init_stays_zero(self):
        """A map with no initial population should produce zero output."""
        rng = np.random.default_rng(3)
        hs = rng.random((50, 50)).astype(np.float32)
        init = np.zeros_like(hs)
        result = run_simulation(
            hs=hs,
            init_distrib=init,
            ewalk=10.0,
            tolerance=50.0,
            alpha=0.1,
            tgrowth=5.0,
            carrying_capacity_params=(0.08, 5.0, 0.5),
            presence_threshold=0.02,
            n_steps=2,
            window_size=20,
            sub_window=9,
            plot=False,
        )
        assert float(result.max()) < 1e-9
