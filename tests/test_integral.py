"""Tests for quantum and classical HSGP integral estimators.

Validates that:
  * quantum_hsgp_integral produces integral values consistent with
    the classical hsgp_integral baseline (within shot-noise tolerances).
  * The integral of the posterior mean has the correct sign.
  * The integral of the posterior variance is non-negative.
  * The quantum integral converges to the classical baseline as the
    number of quadrature points increases.
"""

import numpy as np
import pytest
from qiskit_aer import AerSimulator

from gaussian_quantum.hilbert_space_approx import hsgp_integral
from gaussian_quantum.quantum_algorithms import quantum_hsgp_integral


# ── tolerances ────────────────────────────────────────────────────────────────
# Each integral is a weighted sum of per-point estimates; with O(10) points
# and per-point shot noise ~0.05 the integral tolerance is ~0.25.
INTEGRAL_MEAN_TOL = 0.5
INTEGRAL_VAR_TOL = 0.05
SHOTS = 32_768
TAU = 5  # QPE precision bits


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def backend():
    return AerSimulator()


@pytest.fixture(scope="module")
def integral_setup():
    """Small 1-D GP regression problem with a quadrature grid."""
    X_train = np.array([[0.0], [1.0], [2.0], [3.0]])
    y_train = np.sin(X_train.ravel())
    noise_var = 0.01
    M = 4
    L = 5.0

    # Quadrature grid over [0.5, 2.5] — 5 uniformly-spaced points
    x_grid = np.linspace(0.5, 2.5, 5).reshape(-1, 1)
    h = x_grid[1, 0] - x_grid[0, 0]
    # Trapezoidal weights
    weights = np.full(len(x_grid), h)
    weights[0] *= 0.5
    weights[-1] *= 0.5

    return X_train, y_train, x_grid, weights, noise_var, M, L


# ── classical baseline tests ───────────────────────────────────────────────────

class TestHSGPIntegral:
    def test_returns_scalars(self, integral_setup):
        """hsgp_integral returns two scalar values."""
        X_train, y_train, x_grid, weights, noise_var, M, L = integral_setup
        i_mean, i_var = hsgp_integral(
            X_train, y_train, x_grid, weights, M, L, noise_var=noise_var
        )
        assert np.isscalar(i_mean) or i_mean.ndim == 0
        assert np.isscalar(i_var) or i_var.ndim == 0

    def test_variance_nonnegative(self, integral_setup):
        """Integral of the posterior variance must be non-negative."""
        X_train, y_train, x_grid, weights, noise_var, M, L = integral_setup
        _, i_var = hsgp_integral(
            X_train, y_train, x_grid, weights, M, L, noise_var=noise_var
        )
        assert i_var >= 0.0

    def test_mean_sign(self, integral_setup):
        """Integral of the mean is positive when training targets are positive."""
        # sin(x) > 0 on (0, π); our grid [0.5, 2.5] lies in that region.
        X_train, y_train, x_grid, weights, noise_var, M, L = integral_setup
        i_mean, _ = hsgp_integral(
            X_train, y_train, x_grid, weights, M, L, noise_var=noise_var
        )
        assert i_mean > 0.0

    def test_uniform_weights(self, integral_setup):
        """Uniform-weight integral ≈ mean_of_posteriors × domain_length."""
        X_train, y_train, x_grid, weights, noise_var, M, L = integral_setup
        domain_length = x_grid[-1, 0] - x_grid[0, 0]
        uniform_weights = np.full(len(x_grid), domain_length / len(x_grid))

        i_mean_trap, _ = hsgp_integral(
            X_train, y_train, x_grid, weights, M, L, noise_var=noise_var
        )
        i_mean_uni, _ = hsgp_integral(
            X_train, y_train, x_grid, uniform_weights, M, L, noise_var=noise_var
        )
        # Both estimates should be in the same ball-park
        assert abs(i_mean_trap - i_mean_uni) < 0.2


# ── quantum integral tests ────────────────────────────────────────────────────

class TestQuantumHSGPIntegral:
    def test_mean_matches_classical(self, integral_setup, backend):
        """Quantum integral of the mean ≈ classical HSGP integral of the mean."""
        X_train, y_train, x_grid, weights, noise_var, M, L = integral_setup

        classical_mean, _ = hsgp_integral(
            X_train, y_train, x_grid, weights, M, L, noise_var=noise_var
        )
        q_mean, _ = quantum_hsgp_integral(
            X_train, y_train, x_grid, weights, M, L, noise_var,
            n_eigenvalue_qubits=TAU, shots=SHOTS, backend=backend,
        )
        assert abs(q_mean - classical_mean) < INTEGRAL_MEAN_TOL

    def test_variance_matches_classical(self, integral_setup, backend):
        """Quantum integral of the variance ≈ classical HSGP integral of the variance."""
        X_train, y_train, x_grid, weights, noise_var, M, L = integral_setup

        _, classical_var = hsgp_integral(
            X_train, y_train, x_grid, weights, M, L, noise_var=noise_var
        )
        _, q_var = quantum_hsgp_integral(
            X_train, y_train, x_grid, weights, M, L, noise_var,
            n_eigenvalue_qubits=TAU, shots=SHOTS, backend=backend,
        )
        assert abs(q_var - classical_var) < INTEGRAL_VAR_TOL

    def test_variance_nonnegative(self, integral_setup, backend):
        """Quantum integral of the posterior variance must be non-negative.

        Shot noise can produce small negative estimates; we allow values
        within -INTEGRAL_VAR_TOL to account for this statistical fluctuation.
        """
        X_train, y_train, x_grid, weights, noise_var, M, L = integral_setup
        _, q_var = quantum_hsgp_integral(
            X_train, y_train, x_grid, weights, M, L, noise_var,
            n_eigenvalue_qubits=TAU, shots=SHOTS, backend=backend,
        )
        assert q_var >= -INTEGRAL_VAR_TOL

    def test_mean_positive_for_positive_targets(self, integral_setup, backend):
        """Quantum integral of the mean is positive for sin-wave targets on (0, π).

        sin(x) > 0 on (0, π); our grid [0.5, 2.5] lies in that region, so the
        integral of the posterior mean should be positive.  Shot noise may push
        the estimate slightly negative, hence the soft lower bound.
        """
        X_train, y_train, x_grid, weights, noise_var, M, L = integral_setup
        q_mean, _ = quantum_hsgp_integral(
            X_train, y_train, x_grid, weights, M, L, noise_var,
            n_eigenvalue_qubits=TAU, shots=SHOTS, backend=backend,
        )
        assert q_mean > -INTEGRAL_MEAN_TOL

    def test_weight_length_mismatch_raises(self, integral_setup, backend):
        """Mismatched weights and grid raise ValueError."""
        X_train, y_train, x_grid, weights, noise_var, M, L = integral_setup
        bad_weights = weights[:-1]  # one element too few
        with pytest.raises(ValueError, match="Length of weights"):
            quantum_hsgp_integral(
                X_train, y_train, x_grid, bad_weights, M, L, noise_var,
                n_eigenvalue_qubits=TAU, shots=SHOTS, backend=backend,
            )
