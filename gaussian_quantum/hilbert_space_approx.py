"""Hilbert-space approximate Gaussian process regression (HSGP).

Implements the reduced-rank kernel approximation from Solin & Särkkä (2020),
which is the classical preprocessing stage (Stage 1) of the quantum-assisted
GP algorithm in arXiv:2402.00544.

The stationary kernel k(x, x') is approximated as

    k(x, x') ≈ Σ_{j=1}^{M} S(√λ_j) φ_j(x) φ_j(x')

where φ_j are Laplace eigenfunctions on a bounded domain [-L, L]^d with
Dirichlet boundary conditions, λ_j are the corresponding eigenvalues, and
S is the spectral density of the kernel.

This yields a feature matrix  X = Φ diag(√s_1, …, √s_M)  such that
X X^T ≈ K, and the posterior quantities become:

    E[f*] = X*^T (X^T X + σ²I)^{-1} X^T y
    V[f*] = σ² X*^T (X^T X + σ²I)^{-1} X*

where the M×M matrix  X^T X + σ²I  replaces the N×N system in exact GP.
"""

import numpy as np
from itertools import product as cart_product


# ---------------------------------------------------------------------------
# Laplace eigenbasis on [-L, L]^d
# ---------------------------------------------------------------------------

def laplace_eigenvalues(M, L, d=1):
    """Laplace eigenvalues on [-L, L]^d with Dirichlet boundary conditions.

    For a single dimension the j-th eigenvalue is  λ_j = (jπ / (2L))².
    In d dimensions the multi-index eigenvalue is  λ_j = Σ_k (j_k π / (2L_k))².

    Args:
        M: Number of basis functions per dimension.
        L: Domain boundary. Scalar for isotropic, or array of length d.
        d: Number of input dimensions.

    Returns:
        indices: (M_total, d) array of multi-indices (1-based).
        eigenvalues: (M_total,) array of eigenvalues.
    """
    L = np.atleast_1d(np.asarray(L, dtype=float))
    if L.size == 1:
        L = np.repeat(L, d)

    per_dim = [np.arange(1, M + 1) for _ in range(d)]
    indices = np.array(list(cart_product(*per_dim)))  # (M^d, d)

    eigenvalues = np.sum((indices * np.pi / (2.0 * L[None, :])) ** 2, axis=1)
    return indices, eigenvalues


def laplace_eigenfunctions(X, indices, L):
    """Evaluate Laplace eigenfunctions at input locations.

    φ_j(x) = Π_k  √(1/L_k) sin(j_k π (x_k + L_k) / (2 L_k))

    Args:
        X: (N, d) input locations.
        indices: (M_total, d) multi-indices (from :func:`laplace_eigenvalues`).
        L: Domain boundary (scalar or length-d array).

    Returns:
        Phi: (N, M_total) matrix of eigenfunction values.
    """
    X = np.atleast_2d(X)
    N, d = X.shape
    L = np.atleast_1d(np.asarray(L, dtype=float))
    if L.size == 1:
        L = np.repeat(L, d)

    Phi = np.ones((N, len(indices)))
    for k in range(d):
        Phi *= np.sqrt(1.0 / L[k]) * np.sin(
            indices[:, k][None, :] * np.pi * (X[:, k : k + 1] + L[k]) / (2.0 * L[k])
        )
    return Phi


# ---------------------------------------------------------------------------
# Spectral density of the RBF (squared-exponential) kernel
# ---------------------------------------------------------------------------

def spectral_density_rbf(omega_sq, length_scale=1.0, amplitude=1.0, d=1):
    """Spectral density of the RBF kernel evaluated at ‖ω‖² = omega_sq.

    S(ω) = amplitude² · (2π)^{d/2} · l^d · exp(−l² ‖ω‖² / 2)

    Args:
        omega_sq: Squared frequency (scalar or array), equal to eigenvalue λ.
        length_scale: Kernel length-scale l.
        amplitude: Kernel signal amplitude σ_f.
        d: Input dimensionality.

    Returns:
        Spectral density value(s).
    """
    l = length_scale
    return (
        amplitude ** 2
        * (2.0 * np.pi * l ** 2) ** (d / 2.0)
        * np.exp(-0.5 * l ** 2 * omega_sq)
    )


# ---------------------------------------------------------------------------
# Feature matrix construction
# ---------------------------------------------------------------------------

def hilbert_space_features(X, M, L, length_scale=1.0, amplitude=1.0):
    """Compute HSGP feature matrix  X_feat = Φ diag(√s_1, …, √s_M).

    Args:
        X: (N, d) input locations.
        M: Number of basis functions per dimension.
        L: Domain boundary (scalar or length-d array).
        length_scale: RBF kernel length scale.
        amplitude: RBF kernel signal amplitude.

    Returns:
        X_feat: (N, M_total) feature matrix.
        diag_sqrt_S: (M_total,) square-root spectral weights.
        indices: (M_total, d) multi-indices.
        eigenvalues: (M_total,) Laplace eigenvalues.
    """
    X = np.atleast_2d(X)
    d = X.shape[1]

    indices, eigenvalues = laplace_eigenvalues(M, L, d)
    Phi = laplace_eigenfunctions(X, indices, L)

    S_vals = spectral_density_rbf(eigenvalues, length_scale, amplitude, d)
    diag_sqrt_S = np.sqrt(np.maximum(S_vals, 0.0))

    X_feat = Phi * diag_sqrt_S[None, :]
    return X_feat, diag_sqrt_S, indices, eigenvalues


# ---------------------------------------------------------------------------
# Classical HSGP posterior (reference for quantum algorithm)
# ---------------------------------------------------------------------------

def hs_gp_posterior(X_train, y_train, X_test, M, L, noise_var=1e-6,
                    length_scale=1.0, amplitude=1.0):
    """HSGP posterior mean and variance using the reduced-rank approximation.

    E[f*] = X*^T (X^T X + σ²I)^{-1} X^T y
    V[f*] = σ² X*^T (X^T X + σ²I)^{-1} X*

    where  X = Φ √Λ_S  is the (N, M_total) HSGP feature matrix.

    Args:
        X_train: (N, d) training inputs.
        y_train: (N,) training targets.
        X_test:  (m, d) test inputs.
        M: Number of basis functions per dimension.
        L: Domain boundary.
        noise_var: Observation noise variance σ².
        length_scale: RBF kernel length scale.
        amplitude: RBF kernel signal amplitude.

    Returns:
        mean: (m,) posterior mean at each test point.
        var:  (m,) posterior marginal variance at each test point.
    """
    X_train = np.atleast_2d(X_train)
    X_test = np.atleast_2d(X_test)

    X_feat, _, _, _ = hilbert_space_features(
        X_train, M, L, length_scale, amplitude
    )
    X_star_feat, _, _, _ = hilbert_space_features(
        X_test, M, L, length_scale, amplitude
    )

    M_total = X_feat.shape[1]
    XtX = X_feat.T @ X_feat                         # (M_total, M_total)
    A = XtX + noise_var * np.eye(M_total)            # (M_total, M_total)

    # Mean:  X*^T A^{-1} X^T y
    Xty = X_feat.T @ y_train                         # (M_total,)
    A_inv_Xty = np.linalg.solve(A, Xty)              # (M_total,)
    mean = X_star_feat @ A_inv_Xty                    # (m,)

    # Variance:  σ² X*^T A^{-1} X*
    A_inv_Xstar = np.linalg.solve(A, X_star_feat.T)  # (M_total, m)
    var = noise_var * np.sum(X_star_feat.T * A_inv_Xstar, axis=0)  # (m,)
    return mean, var


def hsgp_integral(X_train, y_train, X_test_grid, weights, M, L,
                  noise_var=1e-6, length_scale=1.0, amplitude=1.0):
    """Numerically integrate the HSGP posterior mean and variance over a grid.

    Computes the quadrature estimates

        I_mean = Σ_i  w_i · E[f*(x_i)]
        I_var  = Σ_i  w_i · V[f*(x_i)]

    using the classical HSGP posterior at the supplied test points.  This
    serves as the reference baseline for the quantum integral estimators.

    Args:
        X_train: (N, d) training inputs.
        y_train: (N,) training targets.
        X_test_grid: (m, d) quadrature test-point locations.
        weights: (m,) quadrature weights (e.g. trapezoidal or uniform).
        M: Number of HSGP basis functions per dimension.
        L: Domain boundary (scalar or length-d array).
        noise_var: Observation noise variance σ².
        length_scale: RBF kernel length scale.
        amplitude: RBF kernel signal amplitude.

    Returns:
        integral_mean: Scalar ≈ ∫ E[f*(x)] dx.
        integral_var:  Scalar ≈ ∫ V[f*(x)] dx.
    """
    weights = np.asarray(weights, dtype=float)
    mean, var = hs_gp_posterior(
        X_train, y_train, X_test_grid, M, L,
        noise_var=noise_var,
        length_scale=length_scale,
        amplitude=amplitude,
    )
    return float(weights @ mean), float(weights @ var)
