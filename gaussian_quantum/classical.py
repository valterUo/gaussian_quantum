"""Classical GP regression algorithms for comparison with quantum implementations.

Implements GP posterior mean and variance using standard NumPy linear algebra,
serving as the ground-truth reference for the quantum algorithms in
arXiv:2402.00544 (Quantum-Assisted Hilbert-Space Gaussian Process Regression).
"""

import numpy as np


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
