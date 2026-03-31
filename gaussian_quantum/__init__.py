"""Gaussian-quantum: classical and quantum GP regression algorithms.

Implements the full three-stage algorithm from arXiv:2402.00544:
    Stage 1 — Hilbert-space kernel approximation (hilbert_space_approx)
    Stage 2 — Quantum PCA with QPE and conditional rotations (qpca)
    Stage 3 — Hadamard / Swap tests for inner-product estimation (quantum_algorithms)

Also provides:
    - Classical GP regression and Bayesian quadrature (classical)
    - Insurance claim distributions and payoff functions (insurance)
"""

from gaussian_quantum.classical import (          # noqa: F401
    rbf_kernel,
    gp_posterior,
    rbf_kernel_mean_embedding,
    rbf_kernel_double_integral,
    gpq_integral,
)

from gaussian_quantum.hilbert_space_approx import (  # noqa: F401
    hilbert_space_features,
    hs_gp_posterior,
    kernel_mean_features,
    hsgp_integral,
)

from gaussian_quantum.quantum_algorithms import (    # noqa: F401
    hadamard_test,
    swap_test,
    quantum_gp_mean,
    quantum_gp_variance,
    quantum_hsgp_mean,
    quantum_hsgp_variance,
    quantum_hsgp_integral,
)

from gaussian_quantum.insurance import (             # noqa: F401
    make_integrand,
    exact_integral,
    DISTRIBUTIONS,
    PAYOFFS,
    PAYOFF_DEFAULTS,
    ordinary_deductible,
    franchise_deductible,
    policy_limit,
    deductible_with_limit,
    stop_loss,
)
