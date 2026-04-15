"""Quantum algorithms for GP posterior mean and variance estimation.

Implements the full three-stage algorithm from arXiv:2402.00544
(Quantum-Assisted Hilbert-Space Gaussian Process Regression):

  Stage 1 — Hilbert-space kernel approximation (see hilbert_space_approx.py):
      k(x, x') ≈ Σ_j S(√λ_j) φ_j(x)φ_j(x'),  yielding  X = Φ√Λ.

  Stage 2 — Quantum PCA + conditional rotations (see qpca.py):
      QPE on ρ = X^T X / ‖X‖_F² extracts eigenvalues; conditional
      rotations encode (σ_r² + σ²)^{-1} (mean) or σ_r/√(σ_r²+σ²) (variance).

  Stage 3 — Hadamard / Swap tests (this module):
      • Hadamard test  →  Re⟨ψ₁|ψ₂⟩  (posterior mean)
      • Swap test      →  |⟨ψ'₁|ψ'₂⟩|² (posterior variance)

The low-level Hadamard and Swap tests also support direct inner-product
estimation on arbitrary amplitude-encoded vectors (legacy API).
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import StatePreparation
from qiskit_aer import AerSimulator

from gaussian_quantum.hilbert_space_approx import (
    hilbert_space_features,
    kernel_mean_features,
)
from gaussian_quantum.qpca import (
    prepare_mean_states,
    prepare_variance_states,
    prepare_mean_states_analytical,
    prepare_variance_states_analytical,
)

__all__ = [
    "hadamard_test",
    "swap_test",
    "quantum_gp_mean",
    "quantum_gp_variance",
    "quantum_hsgp_mean",
    "quantum_hsgp_variance",
    "quantum_hsgp_integral",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _n_qubits_for(n: int) -> int:
    """Minimum qubits to represent a vector of length *n*."""
    return max(1, int(np.ceil(np.log2(n))))


def _pad_normalize(v: np.ndarray, n_qubits: int) -> np.ndarray:
    """Zero-pad *v* to 2**n_qubits and return the unit vector."""
    dim = 2 ** n_qubits
    v_pad = np.zeros(dim, dtype=complex)
    v_pad[: len(v)] = v
    norm = np.linalg.norm(v_pad)
    if norm < 1e-12:
        raise ValueError("Cannot prepare a zero vector as a quantum state.")
    return v_pad / norm


# ---------------------------------------------------------------------------
# Hadamard test
# ---------------------------------------------------------------------------

def hadamard_test(v1, v2, shots: int = 8192, backend=None, seed=None) -> float:
    """Estimate Re⟨v̂₁|v̂₂⟩ using the Hadamard test.

    Circuit overview (ancilla qubit 0, system qubits 1…n_q):

        1. Prepare |v₁⟩ on system.
        2. H on ancilla  →  (|0⟩+|1⟩)/√2 ⊗ |v₁⟩.
        3. Controlled (A₂A₁†) when ancilla=1
              →  (|0⟩|v₁⟩ + |1⟩|v₂⟩)/√2.
        4. H on ancilla, measure.
              P(ancilla=0) = (1 + Re⟨v̂₁|v̂₂⟩) / 2.

    Args:
        v1, v2: Real 1-D arrays of identical length.
        shots:  Measurement shots.
        backend: Qiskit backend (default: AerSimulator).

    Returns:
        Estimate of Re⟨v̂₁|v̂₂⟩ ∈ [−1, 1].
    """
    if backend is None:
        backend = AerSimulator(seed_simulator=seed)

    v1 = np.asarray(v1, dtype=complex)
    v2 = np.asarray(v2, dtype=complex)
    if v1.shape != v2.shape:
        raise ValueError("v1 and v2 must have the same shape.")

    n_q = _n_qubits_for(len(v1))
    v1_n = _pad_normalize(v1, n_q)
    v2_n = _pad_normalize(v2, n_q)

    sp1 = StatePreparation(v1_n)
    sp2 = StatePreparation(v2_n)

    # Unitary that maps |v₁⟩ → |v₂⟩: U = A₂ A₁†
    transform = QuantumCircuit(n_q)
    transform.compose(sp1.inverse(), inplace=True)
    transform.compose(sp2, inplace=True)

    qc = QuantumCircuit(1 + n_q, 1)
    qc.compose(sp1, qubits=range(1, 1 + n_q), inplace=True)   # prepare |v₁⟩
    qc.h(0)                                                     # H on ancilla
    qc.compose(transform.control(1),
               qubits=list(range(0, 1 + n_q)), inplace=True)   # controlled U
    qc.h(0)
    qc.measure(0, 0)

    tqc = transpile(qc, backend)
    counts = backend.run(tqc, shots=shots).result().get_counts()
    p0 = counts.get("0", 0) / shots
    return 2.0 * p0 - 1.0


# ---------------------------------------------------------------------------
# Swap test
# ---------------------------------------------------------------------------

def swap_test(v1, v2, shots: int = 8192, backend=None, seed=None) -> float:
    """Estimate |⟨v̂₁|v̂₂⟩|² using the Swap test.

    Circuit overview (ancilla qubit 0, first register 1…n_q,
    second register n_q+1…2n_q):

        1. Prepare |v₁⟩ and |v₂⟩ on separate registers.
        2. H on ancilla.
        3. Controlled-SWAP (Fredkin) for each qubit pair.
        4. H on ancilla, measure.
              P(ancilla=0) = (1 + |⟨v̂₁|v̂₂⟩|²) / 2.

    Args:
        v1, v2: Real 1-D arrays of identical length.
        shots:  Measurement shots.
        backend: Qiskit backend (default: AerSimulator).

    Returns:
        Estimate of |⟨v̂₁|v̂₂⟩|² ∈ [0, 1].
    """
    if backend is None:
        backend = AerSimulator(seed_simulator=seed)

    v1 = np.asarray(v1, dtype=complex)
    v2 = np.asarray(v2, dtype=complex)
    if v1.shape != v2.shape:
        raise ValueError("v1 and v2 must have the same shape.")

    n_q = _n_qubits_for(len(v1))
    v1_n = _pad_normalize(v1, n_q)
    v2_n = _pad_normalize(v2, n_q)

    qc = QuantumCircuit(1 + 2 * n_q, 1)
    qc.compose(StatePreparation(v1_n),
               qubits=range(1, 1 + n_q), inplace=True)
    qc.compose(StatePreparation(v2_n),
               qubits=range(1 + n_q, 1 + 2 * n_q), inplace=True)
    qc.h(0)
    for i in range(n_q):
        qc.cswap(0, 1 + i, 1 + n_q + i)
    qc.h(0)
    qc.measure(0, 0)

    tqc = transpile(qc, backend)
    counts = backend.run(tqc, shots=shots).result().get_counts()
    p0 = counts.get("0", 0) / shots
    return max(0.0, 2.0 * p0 - 1.0)


# ---------------------------------------------------------------------------
# GP mean and variance via quantum tests
# ---------------------------------------------------------------------------

def quantum_gp_mean(alpha, k_star, shots: int = 8192, backend=None) -> float:
    """Estimate GP posterior mean at a test point via the Hadamard test.

    Computes  μ* = k*ᵀ α = ‖k*‖ · ‖α‖ · Re⟨k̂*|α̂⟩.

    Args:
        alpha:   1-D array, α = (K + σ²I)⁻¹ y.
        k_star:  1-D array, cross-kernel vector k(X_train, x*).
        shots:   Measurement shots.
        backend: Qiskit backend.

    Returns:
        Scalar estimate of μ* = k*ᵀ α.
    """
    norm_k = float(np.linalg.norm(k_star))
    norm_a = float(np.linalg.norm(alpha))
    if norm_k < 1e-12 or norm_a < 1e-12:
        return 0.0
    inner = hadamard_test(k_star, alpha, shots=shots, backend=backend)
    return norm_k * norm_a * inner


def quantum_gp_variance(
    k_star_star, k_star, K_train, noise_var, shots: int = 8192, backend=None
) -> float:
    """Estimate GP posterior variance at a test point via the Swap test.

    Computes:
        σ²* = k** − k*ᵀ (K + σ²I)⁻¹ k*
             = k** − ‖k*‖ · ‖β‖ · |⟨k̂*|β̂⟩|

    where β = (K+σ²I)⁻¹k*. Since k* and β are real and K+σ²I is positive
    definite, ⟨k̂*|β̂⟩ > 0, so |⟨k̂*|β̂⟩| = √(|⟨k̂*|β̂⟩|²) (Swap-test output).

    Args:
        k_star_star: Scalar k(x*, x*), prior variance at the test point.
        k_star:      1-D array, cross-kernel vector k(X_train, x*).
        K_train:     (n, n) training kernel matrix K(X_train, X_train).
        noise_var:   Observation noise variance σ².
        shots:       Measurement shots.
        backend:     Qiskit backend.

    Returns:
        Scalar estimate of σ²*.
    """
    n = K_train.shape[0]
    K_noisy = K_train + noise_var * np.eye(n)
    beta = np.linalg.solve(K_noisy, k_star)   # β = (K+σ²I)⁻¹k*

    norm_k = float(np.linalg.norm(k_star))
    norm_b = float(np.linalg.norm(beta))
    if norm_k < 1e-12 or norm_b < 1e-12:
        return float(k_star_star)

    inner_sq = swap_test(k_star, beta, shots=shots, backend=backend)
    quadratic_form = norm_k * norm_b * np.sqrt(inner_sq)
    return float(k_star_star) - quadratic_form


# ---------------------------------------------------------------------------
# Full quantum HSGP pipeline (Stages 1 + 2 + 3)
# ---------------------------------------------------------------------------

def quantum_hsgp_mean(
    X_train, y_train, x_test, M, L, noise_var,
    length_scale=1.0, amplitude=1.0,
    n_eigenvalue_qubits=6, shots=8192, backend=None,
    analytical=True,
):
    """Full quantum-assisted HSGP posterior mean (arXiv:2402.00544).

    Executes all three stages of the algorithm:

        Stage 1  Hilbert-space kernel approximation → feature matrix X, X*.
        Stage 2  qPCA + conditional rotations → states |ψ₁⟩, |ψ₂⟩.
        Stage 3  Hadamard test → Re⟨ψ₁|ψ₂⟩ → μ*.

    Args:
        X_train: (N, d) training inputs.
        y_train: (N,) training targets.
        x_test:  (d,) or (1, d) single test input.
        M: Number of HSGP basis functions per dimension.
        L: Domain boundary for HSGP.
        noise_var: Observation noise σ².
        length_scale: RBF kernel length scale.
        amplitude: RBF kernel signal amplitude.
        n_eigenvalue_qubits: QPE precision bits τ (circuit mode only).
        shots: Measurement shots for the Hadamard test (circuit mode only).
        backend: Qiskit backend.
        analytical: If True (default), use exact eigendecomposition.

    Returns:
        Scalar estimate of the GP posterior mean μ*.
    """
    X_train = np.atleast_2d(X_train)
    x_test = np.atleast_2d(x_test)

    # ── Stage 1: HSGP features ──────────────────────────────────────────
    X_feat, _, _, _ = hilbert_space_features(
        X_train, M, L, length_scale, amplitude
    )
    X_star_feat, _, _, _ = hilbert_space_features(
        x_test, M, L, length_scale, amplitude
    )
    x_star_feat = X_star_feat.ravel()

    # ── Stage 2: qPCA state preparation ─────────────────────────────────
    if analytical:
        psi1, psi2, norm1, norm2, c_mean, sprob = prepare_mean_states_analytical(
            X_feat, y_train, x_star_feat, noise_var,
        )
    else:
        sv_backend = AerSimulator(method="statevector")
        psi1, psi2, norm1, norm2, c_mean, sprob = prepare_mean_states(
            X_feat, y_train, x_star_feat, noise_var,
            n_eigenvalue_qubits=n_eigenvalue_qubits,
            backend=sv_backend,
        )

    if norm1 < 1e-12 or norm2 < 1e-12 or sprob < 1e-15:
        return 0.0

    # ── Stage 3: Hadamard test on qPCA-prepared states ──────────────────
    if analytical:
        inner = float(np.real(np.vdot(psi1, psi2)))
    else:
        inner = hadamard_test(psi1, psi2, shots=shots, backend=backend)

    # Reconstruct mean: μ* = (1/c_mean) · norm1 · norm2 · Re⟨ψ₁|ψ₂⟩
    return float((1.0 / c_mean) * norm1 * norm2 * inner)


def quantum_hsgp_variance(
    X_train, x_test, M, L, noise_var,
    length_scale=1.0, amplitude=1.0,
    n_eigenvalue_qubits=6, shots=8192, backend=None,
    analytical=True,
):
    """Full quantum-assisted HSGP posterior variance (arXiv:2402.00544).

    Executes all three stages of the algorithm:

        Stage 1  Hilbert-space kernel approximation → feature matrix X, X*.
        Stage 2  qPCA + conditional rotations → states |ψ'₁⟩ ∝ A⁻¹X*,
                 |ψ'₂⟩ = X*/‖X*‖.
        Stage 3  Swap test → |⟨ψ'₁|ψ'₂⟩|² → σ²*.

    The variance is recovered as

        V[f*] = σ² · ‖X*‖² · √p / c · √|⟨ψ'₁|ψ'₂⟩|²

    where  p  is the qPCA success probability and  c  is the conditional
    rotation normalisation constant.

    Args:
        X_train: (N, d) training inputs.
        x_test:  (d,) or (1, d) single test input.
        M: Number of HSGP basis functions per dimension.
        L: Domain boundary for HSGP.
        noise_var: σ².
        length_scale: RBF kernel length scale.
        amplitude: RBF kernel signal amplitude.
        n_eigenvalue_qubits: QPE precision bits τ (circuit mode only).
        shots: Measurement shots for the Swap test (circuit mode only).
        backend: Qiskit backend.
        analytical: If True (default), use exact eigendecomposition.

    Returns:
        Scalar estimate of the GP posterior variance σ²*.
    """
    X_train = np.atleast_2d(X_train)
    x_test = np.atleast_2d(x_test)

    # ── Stage 1: HSGP features ──────────────────────────────────────────
    X_feat, _, _, _ = hilbert_space_features(
        X_train, M, L, length_scale, amplitude
    )
    X_star_feat, _, _, _ = hilbert_space_features(
        x_test, M, L, length_scale, amplitude
    )
    x_star_feat = X_star_feat.ravel()

    # ── Stage 2: qPCA state preparation (mean rotation for A⁻¹X*) ──────
    if analytical:
        psi1, psi2, norm_xstar, c_mean, sprob = prepare_variance_states_analytical(
            X_feat, x_star_feat, noise_var,
        )
    else:
        sv_backend = AerSimulator(method="statevector")
        psi1, psi2, norm_xstar, c_mean, sprob = prepare_variance_states(
            X_feat, x_star_feat, noise_var,
            n_eigenvalue_qubits=n_eigenvalue_qubits,
            backend=sv_backend,
        )

    if norm_xstar < 1e-12 or sprob < 1e-15:
        return float(noise_var)

    # ── Stage 3: Swap test on qPCA-prepared states ──────────────────────
    if analytical:
        inner_sq = float(np.real(np.vdot(psi1, psi2))) ** 2
    else:
        inner_sq = swap_test(psi1, psi2, shots=shots, backend=backend)

    # V[f*] = σ² · ‖X*‖² · √p / c · √|⟨ψ'₁|ψ'₂⟩|²
    return float(
        noise_var * norm_xstar ** 2 * np.sqrt(sprob) / c_mean
        * np.sqrt(inner_sq)
    )


# ---------------------------------------------------------------------------
# Quantum numerical integral of the GP posterior
# ---------------------------------------------------------------------------

def quantum_hsgp_integral(
    X_train, y_train, domain, M, L, noise_var,
    length_scale=1.0, amplitude=1.0,
    n_eigenvalue_qubits=6, shots=8192, backend=None,
    analytical=False, seed=None,
):
    """Bayesian quadrature integral via the quantum HSGP algorithm.

    Computes the Bayesian quadrature estimates (with uniform measure μ=1):

        Q_BQ = z_μᵀ (XᵀX + σ²I)⁻¹ Xᵀy
        V_BQ = σ² z_μᵀ (XᵀX + σ²I)⁻¹ z_μ

    where z_μ = √Λ · Φ_μ is the kernel mean embedding feature vector.

    The computation re-uses the full three-stage quantum pipeline.  Instead
    of evaluating the GP posterior at individual test points, the kernel mean
    embedding z_μ is passed as the "test feature" so that the Hadamard /
    Swap tests directly estimate the BQ mean and variance in a single call
    each.

    Args:
        X_train: (N, d) training inputs.
        y_train: (N,) training targets.
        domain: Integration domain (a, b) for 1-D, or list of (a_k, b_k).
        M: Number of HSGP basis functions per dimension.
        L: Domain boundary for HSGP.
        noise_var: Observation noise σ².
        length_scale: RBF kernel length scale.
        amplitude: RBF kernel signal amplitude.
        n_eigenvalue_qubits: QPE precision bits τ (circuit mode only).
        shots: Measurement shots (circuit mode only).
        backend: Qiskit backend (default: AerSimulator).
        analytical: If True, use exact eigendecomposition instead
            of QPE circuits.  This gives the ideal quantum result without
            finite-precision QPE error.

    Returns:
        integral_mean: Scalar, BQ posterior mean of the integral.
        integral_var:  Scalar, BQ posterior variance of the integral.
    """
    X_train = np.atleast_2d(X_train)

    # ── Stage 1: HSGP features + kernel mean embedding ──────────────────
    X_feat, _, _, _ = hilbert_space_features(
        X_train, M, L, length_scale, amplitude
    )
    z_mu, _, _, _ = kernel_mean_features(
        domain, M, L, length_scale, amplitude
    )

    # ── Stage 2 + 3 (mean): qPCA with z_μ as "test feature" ────────────
    if analytical:
        psi1, psi2, norm1, norm2, c_mean, sprob = prepare_mean_states_analytical(
            X_feat, y_train, z_mu, noise_var,
        )
    else:
        sv_backend = AerSimulator(method="statevector", seed_simulator=seed)
        psi1, psi2, norm1, norm2, c_mean, sprob = prepare_mean_states(
            X_feat, y_train, z_mu, noise_var,
            n_eigenvalue_qubits=n_eigenvalue_qubits,
            backend=sv_backend,
        )

    if norm1 < 1e-12 or norm2 < 1e-12 or sprob < 1e-15:
        integral_mean = 0.0
    else:
        if analytical:
            inner = float(np.real(np.vdot(psi1, psi2)))
        else:
            inner = hadamard_test(psi1, psi2, shots=shots, backend=backend, seed=seed)
        integral_mean = float((1.0 / c_mean) * norm1 * norm2 * inner)

    # ── Stage 2 + 3 (variance): qPCA + Swap test with z_μ ──────────────
    if analytical:
        psi1_v, psi2_v, norm_z, c_mean_v, sprob_v = prepare_variance_states_analytical(
            X_feat, z_mu, noise_var,
        )
    else:
        sv_backend = AerSimulator(method="statevector", seed_simulator=seed)
        psi1_v, psi2_v, norm_z, c_mean_v, sprob_v = prepare_variance_states(
            X_feat, z_mu, noise_var,
            n_eigenvalue_qubits=n_eigenvalue_qubits,
            backend=sv_backend,
        )

    if norm_z < 1e-12 or sprob_v < 1e-15:
        integral_var = 0.0
    else:
        if analytical:
            inner_sq = float(np.real(np.vdot(psi1_v, psi2_v))) ** 2
        else:
            inner_sq = swap_test(psi1_v, psi2_v, shots=shots, backend=backend, seed=seed)
        integral_var = float(
            noise_var * norm_z ** 2 * np.sqrt(sprob_v) / c_mean_v
            * np.sqrt(inner_sq)
        )

    return integral_mean, integral_var
