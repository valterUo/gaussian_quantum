"""Integration tests for the full quantum HSGP pipeline (Stages 1+2+3).

Validates that quantum_hsgp_mean and quantum_hsgp_variance produce results
that are consistent with the classical HSGP posterior, within tolerances
accounted for by QPE discretisation and shot noise.
"""

import numpy as np
import pytest
from qiskit_aer import AerSimulator

from gaussian_quantum.classical import gp_posterior
from gaussian_quantum.hilbert_space_approx import hs_gp_posterior
from gaussian_quantum.quantum_algorithms import (
    quantum_hsgp_mean,
    quantum_hsgp_variance,
)


# ── tolerances ────────────────────────────────────────────────────────────────
# With finite QPE (tau=5) the eigenvalue error is O(1/2^tau) ≈ 0.03.
# Combined with shot noise at 32K shots, we allow ±0.25 absolute tolerance
# on the reconstructed mean and ±0.01 on the variance (which is smaller).
MEAN_TOL = 0.25
VAR_TOL = 0.01
SHOTS = 32_768
TAU = 5  # QPE precision bits


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def backend():
    return AerSimulator()


@pytest.fixture(scope="module")
def gp_setup():
    """Small 1-D GP regression problem."""
    X_train = np.array([[0.0], [1.0], [2.0], [3.0]])
    y_train = np.sin(X_train.ravel())
    X_test = np.array([[1.5]])
    noise_var = 0.01
    M = 4
    L = 5.0
    return X_train, y_train, X_test, noise_var, M, L


# ── full pipeline tests ──────────────────────────────────────────────────────

class TestQuantumHSGPPipeline:
    def test_mean_matches_hsgp(self, gp_setup, backend):
        """Quantum HSGP mean ≈ classical HSGP mean."""
        X_train, y_train, X_test, noise_var, M, L = gp_setup
        hsgp_mean, _ = hs_gp_posterior(
            X_train, y_train, X_test, M, L, noise_var=noise_var
        )
        q_mean = quantum_hsgp_mean(
            X_train, y_train, X_test, M, L, noise_var,
            n_eigenvalue_qubits=TAU, shots=SHOTS, backend=backend,
        )
        assert abs(q_mean - hsgp_mean[0]) < MEAN_TOL

    def test_variance_matches_hsgp(self, gp_setup, backend):
        """Quantum HSGP variance ≈ classical HSGP variance."""
        X_train, y_train, X_test, noise_var, M, L = gp_setup
        _, hsgp_var = hs_gp_posterior(
            X_train, y_train, X_test, M, L, noise_var=noise_var
        )
        q_var = quantum_hsgp_variance(
            X_train, X_test, M, L, noise_var,
            n_eigenvalue_qubits=TAU, shots=SHOTS, backend=backend,
        )
        assert abs(q_var - hsgp_var[0]) < VAR_TOL

    def test_variance_nonnegative(self, gp_setup, backend):
        """Quantum HSGP variance must be non-negative."""
        X_train, y_train, X_test, noise_var, M, L = gp_setup
        q_var = quantum_hsgp_variance(
            X_train, X_test, M, L, noise_var,
            n_eigenvalue_qubits=TAU, shots=SHOTS, backend=backend,
        )
        assert q_var >= 0

    def test_mean_sign_consistent(self, gp_setup, backend):
        """Quantum mean has the same sign as the classical mean."""
        X_train, y_train, X_test, noise_var, M, L = gp_setup
        hsgp_mean, _ = hs_gp_posterior(
            X_train, y_train, X_test, M, L, noise_var=noise_var
        )
        q_mean = quantum_hsgp_mean(
            X_train, y_train, X_test, M, L, noise_var,
            n_eigenvalue_qubits=TAU, shots=SHOTS, backend=backend,
        )
        # Both should be positive for sin(1.5) ≈ 0.997
        assert q_mean > 0
        assert hsgp_mean[0] > 0

    def test_pipeline_with_different_test_point(self, backend):
        """Pipeline works for a test point at the training boundary."""
        X_train = np.array([[0.0], [1.0], [2.0], [3.0]])
        y_train = np.sin(X_train.ravel())
        x_test = np.array([[0.0]])  # at a training point
        noise_var = 0.01
        M, L = 4, 5.0

        q_mean = quantum_hsgp_mean(
            X_train, y_train, x_test, M, L, noise_var,
            n_eigenvalue_qubits=TAU, shots=SHOTS, backend=backend,
        )
        # sin(0) = 0, mean should be close to 0
        assert abs(q_mean) < 0.5
