"""Tests for classical and quantum GP mean/variance algorithms.

Validates that the quantum Hadamard-test (mean) and Swap-test (variance)
implementations match the classical GP posterior values within shot-noise
tolerances.
"""

import numpy as np
import pytest
from qiskit_aer import AerSimulator

from gaussian_quantum.classical import rbf_kernel, gp_posterior
from gaussian_quantum.quantum_algorithms import (
    hadamard_test,
    quantum_gp_mean,
    quantum_gp_variance,
    swap_test,
)

# ── tolerances ────────────────────────────────────────────────────────────────
# With 32 768 shots the standard error of P(0) is ~0.003, giving a 2σ bound of
# ~0.012 on the estimated inner product.  We use 0.05 as a comfortable margin.
ABS_TOL = 0.05
SHOTS = 32_768


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
    return X_train, y_train, X_test, noise_var


# ── Hadamard test unit tests ───────────────────────────────────────────────────

class TestHadamardTest:
    def test_identical_vectors(self, backend):
        """Re⟨v̂|v̂⟩ = 1."""
        v = np.array([1.0, 0.5, -0.3, 0.7])
        result = hadamard_test(v, v, shots=SHOTS, backend=backend)
        assert abs(result - 1.0) < ABS_TOL

    def test_orthogonal_vectors(self, backend):
        """Re⟨v̂₁|v̂₂⟩ = 0 for orthogonal unit vectors."""
        v1 = np.array([1.0, 0.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0, 0.0])
        result = hadamard_test(v1, v2, shots=SHOTS, backend=backend)
        assert abs(result) < ABS_TOL

    def test_known_inner_product(self, backend):
        """Re⟨v̂₁|v̂₂⟩ matches classical normalised dot product."""
        v1 = np.array([3.0, 4.0])
        v2 = np.array([4.0, 3.0])
        expected = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        result = hadamard_test(v1, v2, shots=SHOTS, backend=backend)
        assert abs(result - expected) < ABS_TOL

    def test_antiparallel_vectors(self, backend):
        """Re⟨v̂|−v̂⟩ = −1."""
        v = np.array([1.0, 2.0, 3.0, 4.0])
        result = hadamard_test(v, -v, shots=SHOTS, backend=backend)
        assert abs(result - (-1.0)) < ABS_TOL


# ── Swap test unit tests ───────────────────────────────────────────────────────

class TestSwapTest:
    def test_identical_vectors(self, backend):
        """|⟨v̂|v̂⟩|² = 1."""
        v = np.array([1.0, 0.5, -0.3, 0.7])
        result = swap_test(v, v, shots=SHOTS, backend=backend)
        assert abs(result - 1.0) < ABS_TOL

    def test_orthogonal_vectors(self, backend):
        """|⟨v̂₁|v̂₂⟩|² = 0 for orthogonal vectors."""
        v1 = np.array([1.0, 0.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0, 0.0])
        result = swap_test(v1, v2, shots=SHOTS, backend=backend)
        assert abs(result) < ABS_TOL

    def test_known_squared_inner_product(self, backend):
        """|⟨v̂₁|v̂₂⟩|² matches classical value."""
        v1 = np.array([3.0, 4.0])
        v2 = np.array([4.0, 3.0])
        expected = (
            np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        ) ** 2
        result = swap_test(v1, v2, shots=SHOTS, backend=backend)
        assert abs(result - expected) < ABS_TOL

    def test_nonnegative(self, backend):
        """Swap-test result is always non-negative (antiparallel vectors)."""
        v = np.array([1.0, 2.0, 3.0, 4.0])
        result = swap_test(v, -v, shots=SHOTS, backend=backend)
        assert result >= 0.0
        assert abs(result - 1.0) < ABS_TOL  # |<v|-v>|² = |-1|² = 1


# ── Quantum vs Classical GP ────────────────────────────────────────────────────

class TestQuantumVsClassicalGP:
    def test_quantum_mean_matches_classical(self, gp_setup, backend):
        """Quantum GP mean (Hadamard test) ≈ classical GP posterior mean."""
        X_train, y_train, X_test, noise_var = gp_setup
        K_train = rbf_kernel(X_train, X_train)
        k_star = rbf_kernel(X_train, X_test).ravel()

        classical_mean, _ = gp_posterior(
            X_train, y_train, X_test, noise_var=noise_var
        )

        K_noisy = K_train + noise_var * np.eye(len(y_train))
        alpha = np.linalg.solve(K_noisy, y_train)
        q_mean = quantum_gp_mean(alpha, k_star, shots=SHOTS, backend=backend)

        assert abs(q_mean - classical_mean[0]) < ABS_TOL

    def test_quantum_variance_matches_classical(self, gp_setup, backend):
        """Quantum GP variance (Swap test) ≈ classical GP posterior variance."""
        X_train, y_train, X_test, noise_var = gp_setup
        K_train = rbf_kernel(X_train, X_train)
        k_star = rbf_kernel(X_train, X_test).ravel()
        k_star_star = rbf_kernel(X_test, X_test)[0, 0]

        _, classical_var = gp_posterior(
            X_train, y_train, X_test, noise_var=noise_var
        )

        q_var = quantum_gp_variance(
            k_star_star, k_star, K_train, noise_var,
            shots=SHOTS, backend=backend,
        )

        assert abs(q_var - classical_var[0]) < ABS_TOL

    def test_quantum_variance_nonnegative(self, gp_setup, backend):
        """GP posterior variance must be non-negative."""
        X_train, y_train, X_test, noise_var = gp_setup
        K_train = rbf_kernel(X_train, X_train)
        k_star = rbf_kernel(X_train, X_test).ravel()
        k_star_star = rbf_kernel(X_test, X_test)[0, 0]

        q_var = quantum_gp_variance(
            k_star_star, k_star, K_train, noise_var,
            shots=SHOTS, backend=backend,
        )

        assert q_var >= -ABS_TOL
