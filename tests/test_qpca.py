"""Tests for the quantum PCA module (Stage 2).

Validates density-matrix unitary construction, QPE circuit, conditional
rotations, and the full qPCA state-preparation pipeline.
"""

import numpy as np
import pytest
from qiskit_aer import AerSimulator

from gaussian_quantum.hilbert_space_approx import hilbert_space_features
from gaussian_quantum.qpca import (
    build_density_matrix_unitary,
    conditional_rotation_mean,
    extract_postselected_state,
    inverse_qft_circuit,
    prepare_mean_states,
    prepare_qpca_state,
    prepare_variance_states,
    qpe_circuit,
    run_qpca_statevector,
)


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def hsgp_setup():
    """Feature matrices for a small 1-D GP problem."""
    X_train = np.array([[0.0], [1.0], [2.0], [3.0]])
    y_train = np.sin(X_train.ravel())
    x_test = np.array([[1.5]])
    noise_var = 0.01
    M, L = 4, 5.0

    X_feat, _, _, _ = hilbert_space_features(X_train, M, L)
    X_star_feat, _, _, _ = hilbert_space_features(x_test, M, L)
    x_star = X_star_feat.ravel()
    return X_feat, y_train, x_star, noise_var


@pytest.fixture(scope="module")
def sv_backend():
    return AerSimulator(method="statevector")


# ── density-matrix unitary ───────────────────────────────────────────────────

class TestDensityMatrixUnitary:
    def test_unitary_property(self, hsgp_setup):
        """U is a valid unitary matrix (U U† = I)."""
        X_feat, _, _, _ = hsgp_setup
        XtX = X_feat.T @ X_feat
        F_sq = float(np.trace(XtX))
        U, _, _, _ = build_density_matrix_unitary(XtX, F_sq)
        np.testing.assert_allclose(
            U @ U.conj().T, np.eye(U.shape[0]), atol=1e-10
        )

    def test_eigenvalues_nonnegative(self, hsgp_setup):
        """Eigenvalues of X^T X are non-negative."""
        X_feat, _, _, _ = hsgp_setup
        XtX = X_feat.T @ X_feat
        F_sq = float(np.trace(XtX))
        _, eigenvalues, _, _ = build_density_matrix_unitary(XtX, F_sq)
        assert np.all(eigenvalues >= 0)

    def test_n_state_qubits(self, hsgp_setup):
        """State register size matches ceil(log2(M))."""
        X_feat, _, _, _ = hsgp_setup
        M = X_feat.shape[1]
        XtX = X_feat.T @ X_feat
        F_sq = float(np.trace(XtX))
        _, _, _, n_sq = build_density_matrix_unitary(XtX, F_sq)
        assert 2 ** n_sq >= M


# ── inverse QFT ──────────────────────────────────────────────────────────────

class TestInverseQFT:
    def test_circuit_size(self):
        """IQFT circuit has the correct number of qubits."""
        qc = inverse_qft_circuit(3)
        assert qc.num_qubits == 3


# ── QPE circuit ──────────────────────────────────────────────────────────────

class TestQPECircuit:
    def test_circuit_size(self, hsgp_setup):
        """QPE circuit has τ + n_state qubits."""
        X_feat, _, _, _ = hsgp_setup
        XtX = X_feat.T @ X_feat
        F_sq = float(np.trace(XtX))
        U, _, _, n_sq = build_density_matrix_unitary(XtX, F_sq)
        tau = 3
        qc = qpe_circuit(U, tau, n_sq)
        assert qc.num_qubits == tau + n_sq

    def test_eigenvector_input(self, hsgp_setup, sv_backend):
        """QPE on an eigenvector of ρ recovers the correct eigenphase."""
        X_feat, _, _, _ = hsgp_setup
        XtX = X_feat.T @ X_feat
        F_sq = float(np.trace(XtX))
        U, eigenvalues, eigenvectors, n_sq = build_density_matrix_unitary(
            XtX, F_sq
        )
        dim = 2 ** n_sq
        tau = 6  # enough precision for this test

        # Use the largest eigenvector (most distinct phase)
        idx = np.argmax(eigenvalues)
        v = np.zeros(dim, dtype=complex)
        v[: eigenvectors.shape[0]] = eigenvectors[:, idx]
        v /= np.linalg.norm(v)

        from gaussian_quantum.qpca import _bit_reverse
        from qiskit import QuantumCircuit, transpile
        from qiskit.circuit.library import StatePreparation

        qc = QuantumCircuit(tau + n_sq)
        qc.compose(StatePreparation(v),
                   qubits=range(tau, tau + n_sq), inplace=True)
        qc.compose(qpe_circuit(U, tau, n_sq), inplace=True)
        qc.save_statevector()
        tqc = transpile(qc, sv_backend)
        result = sv_backend.run(tqc, shots=1).result()
        sv = np.asarray(result.get_statevector(qc))

        # Find the dominant eigenvalue-register value
        probs = np.zeros(2 ** tau)
        for j in range(2 ** tau):
            for s in range(dim):
                idx_sv = j + (s << tau)
                probs[j] += abs(sv[idx_sv]) ** 2
        j_max = np.argmax(probs)
        # QPE without bit-reversal swaps: phase = bit_reverse(j) / 2^tau
        theta_est = _bit_reverse(j_max, tau) / (2 ** tau)

        expected_theta = eigenvalues[idx] / F_sq
        assert abs(theta_est - expected_theta) < 1.0 / (2 ** tau) + 0.01


# ── conditional rotations ────────────────────────────────────────────────────

class TestConditionalRotation:
    def test_mean_rotation_qubit_count(self):
        """Mean conditional rotation circuit has τ + 1 qubits."""
        qc, c = conditional_rotation_mean(3, 0.01, 3.0)
        assert qc.num_qubits == 4  # 3 eigenvalue + 1 ancilla

    def test_mean_rotation_c_positive(self):
        """Normalisation constant c is positive."""
        _, c = conditional_rotation_mean(3, 0.01, 3.0)
        assert c > 0

    def test_mean_rotation_c_equals_noise_var(self):
        """For j=0 (zero eigenvalue), max target is 1/σ², so c = σ²."""
        noise_var = 0.05
        _, c = conditional_rotation_mean(3, noise_var, 3.0)
        np.testing.assert_allclose(c, noise_var, rtol=1e-10)


# ── full qPCA state preparation ─────────────────────────────────────────────

class TestQPCAStatePreparation:
    def test_psi1_unit_norm(self, hsgp_setup, sv_backend):
        """Post-selected mean state |ψ₁⟩ is unit-normalised."""
        X_feat, y_train, x_star, noise_var = hsgp_setup
        psi1, _, _, _, _, _ = prepare_mean_states(
            X_feat, y_train, x_star, noise_var,
            n_eigenvalue_qubits=3, backend=sv_backend,
        )
        np.testing.assert_allclose(np.linalg.norm(psi1), 1.0, atol=1e-10)

    def test_psi2_unit_norm(self, hsgp_setup, sv_backend):
        """Amplitude-encoded X^T y state |ψ₂⟩ is unit-normalised."""
        X_feat, y_train, x_star, noise_var = hsgp_setup
        _, psi2, _, _, _, _ = prepare_mean_states(
            X_feat, y_train, x_star, noise_var,
            n_eigenvalue_qubits=3, backend=sv_backend,
        )
        np.testing.assert_allclose(np.linalg.norm(psi2), 1.0, atol=1e-10)

    def test_success_prob_bounded(self, hsgp_setup, sv_backend):
        """Success probability is in (0, 1]."""
        X_feat, y_train, x_star, noise_var = hsgp_setup
        _, _, _, _, _, sprob = prepare_mean_states(
            X_feat, y_train, x_star, noise_var,
            n_eigenvalue_qubits=3, backend=sv_backend,
        )
        assert 0 < sprob <= 1

    def test_variance_states_unit_norm(self, hsgp_setup, sv_backend):
        """Variance states are unit-normalised."""
        X_feat, _, x_star, noise_var = hsgp_setup
        psi1, psi2, _, _, _ = prepare_variance_states(
            X_feat, x_star, noise_var,
            n_eigenvalue_qubits=3, backend=sv_backend,
        )
        np.testing.assert_allclose(np.linalg.norm(psi1), 1.0, atol=1e-10)
        np.testing.assert_allclose(np.linalg.norm(psi2), 1.0, atol=1e-10)

    def test_mean_inner_product_sign(self, hsgp_setup, sv_backend):
        """Re⟨ψ₁|ψ₂⟩ is positive (mean is positive for sin data)."""
        X_feat, y_train, x_star, noise_var = hsgp_setup
        psi1, psi2, _, _, _, _ = prepare_mean_states(
            X_feat, y_train, x_star, noise_var,
            n_eigenvalue_qubits=4, backend=sv_backend,
        )
        inner = float(np.real(np.vdot(psi1, psi2)))
        assert inner > 0
