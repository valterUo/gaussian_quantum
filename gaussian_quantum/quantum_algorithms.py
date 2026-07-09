"""Quantum algorithms for GP quadrature and regression (Stage 3).

Implements the readout stage of the algorithms in Farooq et al.
(PRA 109, 052410, 2024) and Galvis-Florez et al. (arXiv:2502.14467):

  Stage 1 — Hilbert-space kernel approximation (hilbert_space_approx.py):
      k(x, x') ≈ Σ_j S(√λ_j) φ_j(x)φ_j(x'),  yielding  X = Φ√Λ.

  Stage 2 — Data encoding + qPCA + conditional rotations (qpca.py):
      |ψ_X⟩ = Σ_{n,m} x_nm|m⟩|n⟩/‖X‖_F;  QPE with U = exp(2πi XᵀX/δ) on the
      |m⟩ register; conditional rotations encode 1/(s_r²+σ²) (mean) or
      1/(s_r√(s_r²+σ²)) (variance) into an ancilla.

  Stage 3 — Hadamard / SWAP tests (this module):
      • Mean (arXiv Eqs. 40–41): Hadamard test between |ψ₁⟩ and
        |ψ₂⟩ = |X̂_μ⟩|ŷ⟩|0⟩|1⟩ gives  p₀ = (1 + Re⟨ψ₁|ψ₂⟩)/2  and
            Q = ‖X‖_F ‖X_μ‖ ‖y‖ · (2p̂₀ − 1) / c₁.
      • Variance (arXiv Eqs. 43–44): SWAP test between the |m⟩ register and
        |X̂_μ⟩, measuring both ancillas, gives
            V = σ² ‖X_μ‖² (‖X‖_F² / c₂²) · (p̂(a=1) − 2 p̂₁₁),
        where p(a=1) estimates the spectral sum Σ_r 1/(s_r²+σ²) and p₁₁ the
        overlap term — no classical eigenvalue knowledge is needed.

Simulation note: the Stage-2 preparation circuits are executed gate-by-gate
on the Aer statevector simulator; the Stage-3 measurement outcomes are then
sampled from the exact outcome distribution of the test circuit (binomial /
multinomial in the shot count), which reproduces the statistics of running
the full test circuit on a noiseless simulator.

The low-level :func:`hadamard_test` and :func:`swap_test` build and execute
explicit test circuits for arbitrary amplitude-encoded vectors (legacy API,
used for the standalone GP regression helpers below).
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
    prepare_mean_state_circuit,
    prepare_variance_state_circuit,
    mean_overlap,
    variance_probabilities,
    qbq_mean_analytical,
    qbq_variance_analytical,
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


def _estimate_qbq_mean(X_feat, y_train, x_mu, noise_var,
                       n_eigenvalue_qubits, shots, seed=None,
                       delta_margin=0.05, rank=None, window=2, backend=None):
    """Run the mean circuit and reconstruct Q from Hadamard-test statistics."""
    norm_mu = float(np.linalg.norm(x_mu))
    norm_y = float(np.linalg.norm(y_train))
    if norm_mu < 1e-12 or norm_y < 1e-12:
        return 0.0

    sv, c1, info = prepare_mean_state_circuit(
        X_feat, noise_var, n_eigenvalue_qubits,
        delta_margin=delta_margin, rank=rank, window=window, backend=backend,
    )
    re_exact = mean_overlap(sv, info, x_mu, y_train)

    if shots is None:
        re_hat = re_exact
    else:
        p0 = float(np.clip((1.0 + re_exact) / 2.0, 0.0, 1.0))
        rng = np.random.default_rng(seed)
        re_hat = 2.0 * rng.binomial(int(shots), p0) / int(shots) - 1.0

    return float(info["frob"] * norm_mu * norm_y * re_hat / c1)


def _estimate_qbq_variance(X_feat, x_mu, noise_var,
                           n_eigenvalue_qubits, shots, seed=None,
                           delta_margin=0.05, rank=None, window=2,
                           backend=None):
    """Run the variance circuit and reconstruct V from SWAP-test statistics."""
    norm_mu = float(np.linalg.norm(x_mu))
    if norm_mu < 1e-12:
        return 0.0

    sv, c2, info = prepare_variance_state_circuit(
        X_feat, noise_var, n_eigenvalue_qubits,
        delta_margin=delta_margin, rank=rank, window=window, backend=backend,
    )
    probs = variance_probabilities(sv, info, x_mu)   # [p00, p01, p10, p11]

    if shots is None:
        p1_hat = probs[2] + probs[3]
        p11_hat = probs[3]
    else:
        rng = np.random.default_rng(seed)
        counts = rng.multinomial(int(shots), probs)
        p1_hat = (counts[2] + counts[3]) / int(shots)
        p11_hat = counts[3] / int(shots)

    return float(
        noise_var * norm_mu ** 2 * info["frob"] ** 2 / c2 ** 2
        * (p1_hat - 2.0 * p11_hat)
    )


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
# GP mean and variance via quantum tests (legacy full-kernel API)
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
    n_eigenvalue_qubits=8, shots=8192, backend=None,
    analytical=False, seed=None, rank=None, window=2, delta_margin=0.05,
):
    """Quantum-assisted HSGP posterior mean (PRA 109, 052410).

    Executes all three stages of the algorithm with the test-point feature
    vector X* in the role of X_μ:

        Stage 1  Hilbert-space kernel approximation → feature matrices.
        Stage 2  |ψ_X⟩ encoding + QPE + mean conditional rotation → |ψ₁⟩.
        Stage 3  Hadamard test against |X̂*⟩|ŷ⟩|0⟩|1⟩ → μ*.

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
        shots: Measurement shots; None for the exact outcome distribution.
        backend: Qiskit statevector backend for the preparation circuit.
        analytical: If True, use the exact SVD sum instead of circuits.
        seed: RNG seed for measurement sampling.
        rank: Optional low-rank truncation R (both modes).
        window: Half-width of the rotation windows around the dominant
            eigenphases (circuit mode); None rotates on every register value.
        delta_margin: Relative margin of the QPE phase scaling δ above λ_max.

    Returns:
        Scalar estimate of the GP posterior mean μ*.
    """
    X_train = np.atleast_2d(X_train)
    x_test = np.atleast_2d(x_test)

    X_feat, _, _, _ = hilbert_space_features(
        X_train, M, L, length_scale, amplitude
    )
    X_star_feat, _, _, _ = hilbert_space_features(
        x_test, M, L, length_scale, amplitude
    )
    x_star_feat = X_star_feat.ravel()

    if analytical:
        return qbq_mean_analytical(X_feat, y_train, x_star_feat, noise_var,
                                   rank=rank)
    return _estimate_qbq_mean(
        X_feat, y_train, x_star_feat, noise_var,
        n_eigenvalue_qubits, shots, seed=seed,
        delta_margin=delta_margin, rank=rank, window=window, backend=backend,
    )


def quantum_hsgp_variance(
    X_train, x_test, M, L, noise_var,
    length_scale=1.0, amplitude=1.0,
    n_eigenvalue_qubits=8, shots=8192, backend=None,
    analytical=False, seed=None, rank=None, window=2, delta_margin=0.05,
):
    """Quantum-assisted HSGP posterior variance (PRA 109, 052410).

    Executes all three stages with X* in the role of X_μ:

        Stage 1  Hilbert-space kernel approximation → feature matrices.
        Stage 2  |ψ_X⟩ encoding + QPE + variance conditional rotation.
        Stage 3  SWAP test between the |m⟩ register and |X̂*⟩, measuring
                 both ancillas → σ²* via  V = σ²‖X*‖²(F²/c₂²)(p̂₁ − 2p̂₁₁).

    Args: see :func:`quantum_hsgp_mean`.

    Returns:
        Scalar estimate of the GP posterior variance σ²*.
    """
    X_train = np.atleast_2d(X_train)
    x_test = np.atleast_2d(x_test)

    X_feat, _, _, _ = hilbert_space_features(
        X_train, M, L, length_scale, amplitude
    )
    X_star_feat, _, _, _ = hilbert_space_features(
        x_test, M, L, length_scale, amplitude
    )
    x_star_feat = X_star_feat.ravel()

    if analytical:
        return qbq_variance_analytical(X_feat, x_star_feat, noise_var,
                                       rank=rank)
    return _estimate_qbq_variance(
        X_feat, x_star_feat, noise_var,
        n_eigenvalue_qubits, shots, seed=seed,
        delta_margin=delta_margin, rank=rank, window=window, backend=backend,
    )


# ---------------------------------------------------------------------------
# Quantum Bayesian quadrature integral (arXiv:2502.14467)
# ---------------------------------------------------------------------------

def quantum_hsgp_integral(
    X_train, y_train, domain, M, L, noise_var,
    length_scale=1.0, amplitude=1.0,
    n_eigenvalue_qubits=8, shots=8192, backend=None,
    analytical=False, seed=None, rank=None, window=2, delta_margin=0.05,
):
    """Bayesian quadrature integral via the quantum HSGP algorithm.

    Computes the Bayesian quadrature estimates (with uniform measure μ=1):

        Q_BQ = z_μᵀ (XᵀX + σ²I)⁻¹ Xᵀy
        V_BQ = σ² z_μᵀ (XᵀX + σ²I)⁻¹ z_μ

    where z_μ = √Λ · Φ_μ is the kernel mean embedding feature vector, used
    in the role of X_μ in the papers' circuits.

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
        shots: Measurement shots; None for the exact outcome distribution.
        backend: Qiskit statevector backend for the preparation circuits.
        analytical: If True, use the exact SVD sums (papers' Eqs. 14–15)
            instead of circuits — the ideal quantum result without QPE
            discretisation or shot noise.
        seed: RNG seed for measurement sampling.
        rank: Optional low-rank truncation R (both modes).
        window: Half-width of the rotation windows around the dominant
            eigenphases (circuit mode); None rotates on every register value.
        delta_margin: Relative margin of the QPE phase scaling δ above λ_max.

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

    if analytical:
        integral_mean = qbq_mean_analytical(X_feat, y_train, z_mu, noise_var,
                                            rank=rank)
        integral_var = qbq_variance_analytical(X_feat, z_mu, noise_var,
                                               rank=rank)
        return integral_mean, integral_var

    # ── Stages 2 + 3 ─────────────────────────────────────────────────────
    integral_mean = _estimate_qbq_mean(
        X_feat, y_train, z_mu, noise_var,
        n_eigenvalue_qubits, shots, seed=seed,
        delta_margin=delta_margin, rank=rank, window=window, backend=backend,
    )
    integral_var = _estimate_qbq_variance(
        X_feat, z_mu, noise_var,
        n_eigenvalue_qubits, shots, seed=seed,
        delta_margin=delta_margin, rank=rank, window=window, backend=backend,
    )
    return integral_mean, integral_var
