"""Tests for paradis.calibration – logistic helpers and spatial thinning."""

import numpy as np
import pytest

from paradis.calibration.carrying_capacity import fit_logistic, logistic
from paradis.calibration.sites import spatial_min_distance_filter


# ---------------------------------------------------------------------------
# Logistic function
# ---------------------------------------------------------------------------

class TestLogistic:
    def test_zero_at_origin(self):
        """Shifted logistic must satisfy f(0) = 0."""
        for L, k, x0 in [(0.1, 5.0, 0.5), (0.2, 3.0, 0.3), (1.0, 10.0, 0.8)]:
            assert abs(logistic(0.0, L, k, x0)) < 1e-9, f"f(0) != 0 for L={L}"

    def test_monotone_decreasing_for_positive_k(self):
        """Positive k → g(x)=L/(1+exp(k(x-x0))) is decreasing → shifted form is negative/decreasing."""
        x = np.linspace(0, 1, 50)
        y = logistic(x, L=0.2, k=5.0, x0=0.5)
        assert np.all(np.diff(y) <= 1e-9)

    def test_monotone_increasing_for_negative_k(self):
        """Negative k → g(x) is increasing → shifted form is non-negative and increasing."""
        x = np.linspace(0, 1, 50)
        y = logistic(x, L=0.2, k=-5.0, x0=0.5)
        assert np.all(np.diff(y) >= -1e-9)

    def test_output_bounded_by_L(self):
        # With negative k the function is increasing and bounded by L
        x = np.linspace(0, 1, 100)
        for L in [0.05, 0.1, 0.5]:
            y = logistic(x, L=L, k=-8.0, x0=0.4)
            assert np.all(y <= L + 1e-9)

    def test_scalar_input(self):
        val = logistic(0.5, L=0.1, k=5.0, x0=0.5)
        assert np.isscalar(val) or val.ndim == 0

    def test_array_output_shape(self):
        x = np.linspace(0, 1, 30)
        y = logistic(x, L=0.1, k=5.0, x0=0.5)
        assert y.shape == (30,)


# ---------------------------------------------------------------------------
# Logistic fitting
# ---------------------------------------------------------------------------

class TestFitLogistic:
    def _make_noisy_data(self, L=0.1, k=-6.0, x0=0.5, noise=0.005, n=20, seed=42):
        """Generate synthetic data from an increasing logistic (k < 0)."""
        rng = np.random.default_rng(seed)
        x = np.linspace(0.05, 0.95, n)
        y = logistic(x, L, k, x0) + rng.normal(0, noise, n)
        y = np.clip(y, 0, None)
        sigma = np.full(n, noise)
        return x, y, sigma

    def test_returns_dict_with_correct_keys(self):
        x, y, sigma = self._make_noisy_data()
        result = fit_logistic(x, y, sigma)
        assert set(result.keys()) == {"L", "k", "x0"}

    def test_fitted_L_positive(self):
        x, y, sigma = self._make_noisy_data()
        result = fit_logistic(x, y, sigma)
        assert result["L"] > 0

    def test_recovers_parameters_approximately(self):
        """Fit should recover L, x0 within 30 % relative error on clean data."""
        L_true, k_true, x0_true = 0.1, -6.0, 0.5
        x, y, sigma = self._make_noisy_data(L=L_true, k=k_true, x0=x0_true, noise=1e-4)
        result = fit_logistic(x, y, sigma)
        assert abs(result["L"] - L_true) / L_true < 0.3
        assert abs(result["x0"] - x0_true) / x0_true < 0.3

    def test_unit_interval_constraint(self):
        x, y, sigma = self._make_noisy_data()
        result = fit_logistic(x, y, sigma, unit_interval=True)
        assert 0 <= result["L"] <= 1
        assert 0 <= result["x0"] <= 1

    def test_fitted_curve_passes_through_data(self):
        """Residuals should be small relative to the data range."""
        x, y, sigma = self._make_noisy_data(noise=1e-6)
        result = fit_logistic(x, y, sigma)
        y_pred = logistic(x, result["L"], result["k"], result["x0"])
        rms = np.sqrt(np.mean((y - y_pred) ** 2))
        assert rms < 0.01  # within 1 % of range [0, 1]


# ---------------------------------------------------------------------------
# Spatial minimum-distance filter
# ---------------------------------------------------------------------------

class TestSpatialMinDistanceFilter:
    def test_single_point_retained(self):
        xs, ys = spatial_min_distance_filter(
            np.array([5.0]), np.array([5.0]), distmin=10.0
        )
        assert len(xs) == 1

    def test_identical_points_reduced_to_one(self):
        xs_in = np.full(20, 5.0)
        ys_in = np.full(20, 5.0)
        xs, ys = spatial_min_distance_filter(xs_in, ys_in, distmin=1.0)
        assert len(xs) == 1

    def test_minimum_spacing_enforced(self):
        rng = np.random.default_rng(7)
        xs_in = rng.uniform(0, 100, 50)
        ys_in = rng.uniform(0, 100, 50)
        distmin = 15.0
        xs_out, ys_out = spatial_min_distance_filter(xs_in, ys_in, distmin)
        if len(xs_out) > 1:
            from scipy.spatial.distance import cdist
            pts = np.column_stack([xs_out, ys_out])
            D = cdist(pts, pts)
            np.fill_diagonal(D, np.inf)
            assert D.min() >= distmin - 1e-9

    def test_output_is_subset_of_input(self):
        xs_in = np.array([0.0, 5.0, 10.0, 15.0, 20.0])
        ys_in = np.array([0.0, 0.0,  0.0,  0.0,  0.0])
        xs_out, ys_out = spatial_min_distance_filter(xs_in, ys_in, distmin=4.0)
        for xo, yo in zip(xs_out, ys_out):
            found = any(abs(xo - xi) < 1e-9 and abs(yo - yi) < 1e-9
                        for xi, yi in zip(xs_in, ys_in))
            assert found

    def test_distmin_zero_keeps_all(self):
        xs_in = np.array([0.0, 1.0, 2.0, 3.0])
        ys_in = np.zeros(4)
        xs_out, _ = spatial_min_distance_filter(xs_in, ys_in, distmin=0.0)
        assert len(xs_out) == 4

    def test_large_distmin_keeps_one(self):
        rng = np.random.default_rng(1)
        xs_in = rng.uniform(0, 10, 30)
        ys_in = rng.uniform(0, 10, 30)
        xs_out, _ = spatial_min_distance_filter(xs_in, ys_in, distmin=1000.0)
        assert len(xs_out) == 1

    def test_returns_lists(self):
        xs_out, ys_out = spatial_min_distance_filter(
            np.array([0.0, 5.0]), np.array([0.0, 5.0]), distmin=1.0
        )
        assert isinstance(xs_out, list)
        assert isinstance(ys_out, list)
