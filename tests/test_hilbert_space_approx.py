"""Tests for the Hilbert-space kernel approximation (Stage 1).

Validates eigenfunction/eigenvalue computation, spectral density, feature
matrix construction, and convergence of the HSGP posterior to the exact GP.
"""

import numpy as np
import pytest

from gaussian_quantum.hilbert_space_approx import (
    hilbert_space_features,
    hs_gp_posterior,
    laplace_eigenfunctions,
    laplace_eigenvalues,
    spectral_density_rbf,
)
from gaussian_quantum.classical import gp_posterior


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def gp_setup():
    """Small 1-D GP regression problem."""
    X_train = np.array([[0.0], [1.0], [2.0], [3.0]])
    y_train = np.sin(X_train.ravel())
    X_test = np.array([[1.5]])
    noise_var = 0.01
    return X_train, y_train, X_test, noise_var


# ── Laplace eigenvalues ──────────────────────────────────────────────────────

class TestLaplaceEigenvalues:
    def test_1d_eigenvalues(self):
        """λ_j = (jπ / (2L))² for 1-D."""
        M, L = 5, 4.0
        indices, eigenvalues = laplace_eigenvalues(M, L, d=1)
        expected = (np.arange(1, M + 1) * np.pi / (2 * L)) ** 2
        np.testing.assert_allclose(eigenvalues, expected)

    def test_indices_shape(self):
        """Indices array has correct shape for d dimensions."""
        M, L, d = 3, 5.0, 2
        indices, eigenvalues = laplace_eigenvalues(M, L, d)
        assert indices.shape == (M ** d, d)
        assert eigenvalues.shape == (M ** d,)

    def test_eigenvalues_positive(self):
        """All eigenvalues are strictly positive."""
        indices, eigenvalues = laplace_eigenvalues(4, 5.0, d=1)
        assert np.all(eigenvalues > 0)


# ── Laplace eigenfunctions ───────────────────────────────────────────────────

class TestLaplaceEigenfunctions:
    def test_output_shape(self):
        """Phi matrix has shape (N, M)."""
        X = np.array([[0.0], [1.0], [2.0]])
        indices, _ = laplace_eigenvalues(4, 5.0, d=1)
        Phi = laplace_eigenfunctions(X, indices, L=5.0)
        assert Phi.shape == (3, 4)

    def test_orthogonality_approximate(self):
        """Eigenfunctions are approximately orthogonal on dense grid."""
        N = 500
        L = 5.0
        X = np.linspace(-L, L, N).reshape(-1, 1)
        M = 4
        indices, _ = laplace_eigenvalues(M, L, d=1)
        Phi = laplace_eigenfunctions(X, indices, L)

        # Approximate ∫ φ_i φ_j dx ≈ (2L/N) Σ φ_i(x_k) φ_j(x_k)
        dx = 2 * L / N
        gram = dx * Phi.T @ Phi
        np.testing.assert_allclose(gram, np.eye(M), atol=0.05)


# ── Spectral density ─────────────────────────────────────────────────────────

class TestSpectralDensity:
    def test_positive(self):
        """Spectral density is positive for all frequencies."""
        omega_sq = np.array([0.0, 0.5, 1.0, 5.0])
        S = spectral_density_rbf(omega_sq)
        assert np.all(S > 0)

    def test_decreasing(self):
        """Spectral density decreases with increasing frequency."""
        omega_sq = np.linspace(0, 5, 20)
        S = spectral_density_rbf(omega_sq)
        assert np.all(np.diff(S) < 0)

    def test_peak_at_zero(self):
        """Maximum spectral density is at ω = 0."""
        omega_sq = np.linspace(0, 5, 20)
        S = spectral_density_rbf(omega_sq)
        assert np.argmax(S) == 0


# ── Feature matrix ───────────────────────────────────────────────────────────

class TestHilbertSpaceFeatures:
    def test_output_shapes(self):
        """Feature matrix and weights have correct shapes."""
        X = np.array([[0.0], [1.0], [2.0]])
        X_feat, sqrt_S, indices, evals = hilbert_space_features(X, M=4, L=5.0)
        assert X_feat.shape == (3, 4)
        assert sqrt_S.shape == (4,)
        assert indices.shape == (4, 1)
        assert evals.shape == (4,)

    def test_kernel_approximation(self):
        """X X^T approximates the RBF kernel matrix for large M."""
        X = np.array([[0.0], [1.0], [2.0], [3.0]])
        from gaussian_quantum.classical import rbf_kernel
        K_exact = rbf_kernel(X, X)

        X_feat, _, _, _ = hilbert_space_features(X, M=32, L=5.0)
        K_approx = X_feat @ X_feat.T
        np.testing.assert_allclose(K_approx, K_exact, atol=0.05)


# ── HSGP posterior ───────────────────────────────────────────────────────────

class TestHSGPPosterior:
    def test_converges_to_exact_gp(self, gp_setup):
        """HSGP mean/var converge to exact GP as M increases."""
        X_train, y_train, X_test, noise_var = gp_setup
        exact_mean, exact_var = gp_posterior(
            X_train, y_train, X_test, noise_var=noise_var
        )

        hsgp_mean, hsgp_var = hs_gp_posterior(
            X_train, y_train, X_test, M=32, L=5.0, noise_var=noise_var
        )
        np.testing.assert_allclose(hsgp_mean, exact_mean, atol=0.01)
        np.testing.assert_allclose(hsgp_var, exact_var, atol=0.01)

    def test_variance_nonnegative(self, gp_setup):
        """HSGP variance is non-negative."""
        X_train, y_train, X_test, noise_var = gp_setup
        _, hsgp_var = hs_gp_posterior(
            X_train, y_train, X_test, M=8, L=5.0, noise_var=noise_var
        )
        assert np.all(hsgp_var >= 0)

    def test_mean_shape(self, gp_setup):
        """HSGP posterior returns correct shapes."""
        X_train, y_train, X_test, noise_var = gp_setup
        mean, var = hs_gp_posterior(
            X_train, y_train, X_test, M=4, L=5.0, noise_var=noise_var
        )
        assert mean.shape == (1,)
        assert var.shape == (1,)
