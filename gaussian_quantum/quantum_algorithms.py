"""Quantum algorithms for GP posterior mean and variance estimation.

Implements the two core quantum primitives from arXiv:2402.00544
(Quantum-Assisted Hilbert-Space Gaussian Process Regression):

  • Hadamard test  →  Re⟨v̂₁|v̂₂⟩  (used for posterior mean μ*)
  • Swap test      →  |⟨v̂₁|v̂₂⟩|² (used for posterior variance σ²*)

Both tests operate on amplitude-encoded quantum states: a classical
real vector v ∈ ℝⁿ is encoded as the unit state |v̂⟩ = |v⟩/‖v‖.

The GP quantities are then recovered by re-scaling with the classical norms:

    μ*  = ‖k*‖ · ‖α‖ · Re⟨k̂*|α̂⟩                (Hadamard test)
    σ²* = k** − ‖k*‖ · ‖β‖ · |⟨k̂*|β̂⟩|           (Swap test + √)

where α = (K+σ²I)⁻¹y and β = (K+σ²I)⁻¹k*.
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import StatePreparation
from qiskit_aer import AerSimulator


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

def hadamard_test(v1, v2, shots: int = 8192, backend=None) -> float:
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
        backend = AerSimulator()

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

def swap_test(v1, v2, shots: int = 8192, backend=None) -> float:
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
        backend = AerSimulator()

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
