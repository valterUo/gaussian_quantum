"""Classical GP regression algorithms for comparison with quantum implementations.

Implements GP posterior mean and variance using standard NumPy linear algebra,
serving as the ground-truth reference for the quantum algorithms in
arXiv:2402.00544 (Quantum-Assisted Hilbert-Space Gaussian Process Regression).

Also provides classical Bayesian quadrature (GPQ) using the closed-form kernel
mean embedding of the RBF kernel, following the formulation in
Gaussian_process_quadrature from the Quantum_HSGPQ reference implementation.
"""

import numpy as np
from scipy.special import erf


def rbf_kernel(X1, X2, length_scale=1.0, amplitude=1.0):
    """Radial basis function (squared-exponential) kernel.

    k(x, x') = amplitude² · exp(−‖x − x'‖² / (2 · length_scale²))

    Args:
        X1: (n1, d) array of first inputs.
        X2: (n2, d) array of second inputs.
        length_scale: Characteristic length scale l.
        amplitude: Signal amplitude σ_f.

    Returns:
        (n1, n2) kernel matrix.
    """
    X1 = np.atleast_2d(X1)
    X2 = np.atleast_2d(X2)
    sq_dists = (
        np.sum(X1 ** 2, axis=1, keepdims=True)
        + np.sum(X2 ** 2, axis=1)
        - 2.0 * X1 @ X2.T
    )
    return amplitude ** 2 * np.exp(-0.5 / length_scale ** 2 * sq_dists)


def gp_posterior(X_train, y_train, X_test, noise_var=1e-6, **kernel_kwargs):
    """Compute GP posterior mean and variance at test points.

    For a zero-mean GP prior with RBF kernel and Gaussian noise:

        μ* = K(X*, X) [K(X, X) + σ²I]⁻¹ y
        σ²* = K(X*, X*) − K(X*, X) [K(X, X) + σ²I]⁻¹ K(X, X*)

    Args:
        X_train:    (n, d) training inputs.
        y_train:    (n,) training targets.
        X_test:     (m, d) test inputs.
        noise_var:  Observation noise variance σ².
        **kernel_kwargs: Forwarded to :func:`rbf_kernel`.

    Returns:
        mean: (m,) posterior mean at each test point.
        var:  (m,) posterior marginal variance at each test point.
    """
    n = len(y_train)
    K = rbf_kernel(X_train, X_train, **kernel_kwargs) + noise_var * np.eye(n)
    K_s = rbf_kernel(X_train, X_test, **kernel_kwargs)   # (n, m)
    K_ss = rbf_kernel(X_test, X_test, **kernel_kwargs)   # (m, m)

    alpha = np.linalg.solve(K, y_train)                  # (K + σ²I)⁻¹ y
    mean = K_s.T @ alpha
    var = np.diag(K_ss - K_s.T @ np.linalg.solve(K, K_s))
    return mean, var


# ---------------------------------------------------------------------------
# Bayesian quadrature with the full RBF kernel
# ---------------------------------------------------------------------------

def rbf_kernel_mean_embedding(X, domain, length_scale=1.0, amplitude=1.0):
    """Kernel mean embedding k_μ(x) = ∫_a^b k(x, x') dx' for the RBF kernel.

    Closed form (1-D, uniform measure):

        k_μ(x) = σ_f² √(π/2) ℓ [erf((x−a)/(√2 ℓ)) − erf((x−b)/(√2 ℓ))]

    Args:
        X: (n, 1) array of input locations.
        domain: (a, b) integration bounds.
        length_scale: RBF length scale ℓ.
        amplitude: RBF signal amplitude σ_f.

    Returns:
        k_mu: (n,) kernel mean embedding at each input.
    """
    X = np.atleast_2d(X).ravel()
    a, b = float(domain[0]), float(domain[1])
    ell = float(length_scale)
    sf2 = float(amplitude) ** 2
    sqrt2_ell = np.sqrt(2.0) * ell
    k_mu = sf2 * np.sqrt(np.pi / 2.0) * ell * (
        erf((X - a) / sqrt2_ell) - erf((X - b) / sqrt2_ell)
    )
    return k_mu


def rbf_kernel_double_integral(domain, length_scale=1.0, amplitude=1.0):
    """Double integral μ(k_μ) = ∫∫_[a,b]² k(x, x') dx dx' for the RBF kernel.

    Closed form (1-D, uniform measure):

        μ(k_μ) = 2 σ_f² ℓ [ℓ (exp(−(a−b)²/(2ℓ²)) − 1)
                            + √(π/2)(a−b) erf((a−b)/(√2 ℓ))]

    Args:
        domain: (a, b) integration bounds.
        length_scale: RBF length scale ℓ.
        amplitude: RBF signal amplitude σ_f.

    Returns:
        Scalar double integral value.
    """
    a, b = float(domain[0]), float(domain[1])
    ell = float(length_scale)
    sf2 = float(amplitude) ** 2
    diff = a - b
    sqrt2_ell = np.sqrt(2.0) * ell
    return float(
        2.0 * sf2 * ell * (
            ell * (np.exp(-0.5 * diff ** 2 / ell ** 2) - 1.0)
            + np.sqrt(np.pi / 2.0) * diff * erf(diff / sqrt2_ell)
        )
    )


def gpq_integral(X_train, y_train, domain, noise_var=1e-6,
                 length_scale=1.0, amplitude=1.0):
    """Bayesian quadrature with the full RBF kernel (classical GPQ).

    Computes:
        Q_BQ = k_μᵀ (K + σ²I)⁻¹ y
        V_BQ = μ(k_μ) − k_μᵀ (K + σ²I)⁻¹ k_μ

    Args:
        X_train: (n, d) training inputs.
        y_train: (n,) training targets.
        domain: Integration domain (a, b).
        noise_var: Observation noise variance σ².
        length_scale: RBF kernel length scale.
        amplitude: RBF kernel signal amplitude.

    Returns:
        integral_mean: Scalar BQ posterior mean of the integral.
        integral_var:  Scalar BQ posterior variance of the integral.
    """
    X_train = np.atleast_2d(X_train)
    n = len(y_train)
    kw = dict(length_scale=length_scale, amplitude=amplitude)

    K = rbf_kernel(X_train, X_train, **kw) + noise_var * np.eye(n)
    k_mu = rbf_kernel_mean_embedding(X_train, domain, **kw)
    mu_k_mu = rbf_kernel_double_integral(domain, **kw)

    K_inv_y = np.linalg.solve(K, y_train)
    K_inv_kmu = np.linalg.solve(K, k_mu)

    integral_mean = float(k_mu @ K_inv_y)
    integral_var = float(mu_k_mu - k_mu @ K_inv_kmu)
    return integral_mean, integral_var
