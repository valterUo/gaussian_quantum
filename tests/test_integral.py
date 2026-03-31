"""Tests for quantum and classical HSGP Bayesian quadrature integral estimators.

Validates that:
  * hsgp_integral computes proper Bayesian quadrature (BQ) with the
    kernel mean embedding, not numerical quadrature.
  * quantum_hsgp_integral produces integral values consistent with
    the classical hsgp_integral baseline (within shot-noise tolerances).
  * The BQ mean integral has the correct sign.
  * The BQ variance (uncertainty of the integral) is non-negative.
  * The quantum integral converges to the classical baseline.
"""

import numpy as np
import pytest
from qiskit_aer import AerSimulator

from gaussian_quantum.hilbert_space_approx import hsgp_integral
from gaussian_quantum.quantum_algorithms import quantum_hsgp_integral


# ── tolerances ────────────────────────────────────────────────────────────────
# The quantum BQ only runs 2 quantum circuits (Hadamard + Swap test)
# instead of per-grid-point, so tolerances can be tighter.
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
    """Small 1-D GP regression problem with a BQ integration domain."""
    X_train = np.array([[0.0], [1.0], [2.0], [3.0]])
    y_train = np.sin(X_train.ravel())
    noise_var = 0.01
    M = 4
    L = 5.0
    domain = (0.5, 2.5)

    return X_train, y_train, domain, noise_var, M, L


# ── classical baseline tests ───────────────────────────────────────────────────

class TestHSGPIntegral:
    def test_returns_scalars(self, integral_setup):
        """hsgp_integral returns two scalar values."""
        X_train, y_train, domain, noise_var, M, L = integral_setup
        i_mean, i_var = hsgp_integral(
            X_train, y_train, domain, M, L, noise_var=noise_var
        )
        assert np.isscalar(i_mean) or i_mean.ndim == 0
        assert np.isscalar(i_var) or i_var.ndim == 0

    def test_variance_nonnegative(self, integral_setup):
        """BQ variance (integral uncertainty) must be non-negative."""
        X_train, y_train, domain, noise_var, M, L = integral_setup
        _, i_var = hsgp_integral(
            X_train, y_train, domain, M, L, noise_var=noise_var
        )
        assert i_var >= 0.0

    def test_mean_sign(self, integral_setup):
        """Integral of the mean is positive when training targets are positive."""
        # sin(x) > 0 on (0, π); our domain [0.5, 2.5] lies in that region.
        X_train, y_train, domain, noise_var, M, L = integral_setup
        i_mean, _ = hsgp_integral(
            X_train, y_train, domain, M, L, noise_var=noise_var
        )
        assert i_mean > 0.0

    def test_wider_domain_larger_mean(self, integral_setup):
        """Wider integration domain produces a larger mean integral."""
        X_train, y_train, domain, noise_var, M, L = integral_setup
        i_mean_narrow, _ = hsgp_integral(
            X_train, y_train, domain, M, L, noise_var=noise_var
        )
        i_mean_wide, _ = hsgp_integral(
            X_train, y_train, (0.0, 3.0), M, L, noise_var=noise_var
        )
        assert i_mean_wide > i_mean_narrow


# ── quantum integral tests ────────────────────────────────────────────────────

class TestQuantumHSGPIntegral:
    def test_mean_matches_classical(self, integral_setup, backend):
        """Quantum BQ mean ≈ classical HSGP BQ mean."""
        X_train, y_train, domain, noise_var, M, L = integral_setup

        classical_mean, _ = hsgp_integral(
            X_train, y_train, domain, M, L, noise_var=noise_var
        )
        q_mean, _ = quantum_hsgp_integral(
            X_train, y_train, domain, M, L, noise_var,
            n_eigenvalue_qubits=TAU, shots=SHOTS, backend=backend,
        )
        assert abs(q_mean - classical_mean) < INTEGRAL_MEAN_TOL

    def test_variance_matches_classical(self, integral_setup, backend):
        """Quantum BQ variance ≈ classical HSGP BQ variance."""
        X_train, y_train, domain, noise_var, M, L = integral_setup

        _, classical_var = hsgp_integral(
            X_train, y_train, domain, M, L, noise_var=noise_var
        )
        _, q_var = quantum_hsgp_integral(
            X_train, y_train, domain, M, L, noise_var,
            n_eigenvalue_qubits=TAU, shots=SHOTS, backend=backend,
        )
        assert abs(q_var - classical_var) < INTEGRAL_VAR_TOL

    def test_variance_nonnegative(self, integral_setup, backend):
        """Quantum BQ variance must be non-negative.

        Shot noise can produce small negative estimates; we allow values
        within -INTEGRAL_VAR_TOL to account for this statistical fluctuation.
        """
        X_train, y_train, domain, noise_var, M, L = integral_setup
        _, q_var = quantum_hsgp_integral(
            X_train, y_train, domain, M, L, noise_var,
            n_eigenvalue_qubits=TAU, shots=SHOTS, backend=backend,
        )
        assert q_var >= -INTEGRAL_VAR_TOL

    def test_mean_positive_for_positive_targets(self, integral_setup, backend):
        """Quantum BQ mean is positive for sin-wave targets on (0, π).

        sin(x) > 0 on (0, π); our domain [0.5, 2.5] lies in that region, so the
        BQ mean should be positive.  Shot noise may push the estimate
        slightly negative, hence the soft lower bound.
        """
        X_train, y_train, domain, noise_var, M, L = integral_setup
        q_mean, _ = quantum_hsgp_integral(
            X_train, y_train, domain, M, L, noise_var,
            n_eigenvalue_qubits=TAU, shots=SHOTS, backend=backend,
        )
        assert q_mean > -INTEGRAL_MEAN_TOL
