"""Tests for the quantum PCA module (Stage 2).

Validates density-matrix unitary construction, QPE circuit, conditional
rotations, data-matrix encoding, and the paper-faithful state-preparation
circuits together with their analytical (SVD) references.
"""

import numpy as np
import pytest
from qiskit_aer import AerSimulator

from gaussian_quantum.hilbert_space_approx import (
    hilbert_space_features,
    hs_gp_posterior,
)
from gaussian_quantum.qpca import (
    build_density_matrix_unitary,
    conditional_rotation_mean,
    conditional_rotation_variance,
    eigenphase_window_bins,
    encode_data_matrix,
    inverse_qft_circuit,
    mean_overlap,
    prepare_mean_state_circuit,
    prepare_variance_state_circuit,
    qbq_mean_analytical,
    qbq_variance_analytical,
    qpe_circuit,
    spectral_scale,
    variance_probabilities,
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

    def test_mean_rotation_c_excludes_zero_bin(self):
        """Normalisation constant c excludes the j=0 bin (θ=0) and is
        determined by the smallest non-zero QPE phase instead."""
        noise_var = 0.05
        F_sq = 3.0
        tau = 3
        _, c = conditional_rotation_mean(tau, noise_var, F_sq)
        # c should be larger than σ² because the j=0 bin is excluded
        assert c > noise_var
        # c = 1 / max(nonzero targets)
        # smallest non-zero phase = 1/2^tau (bit_reverse(1,tau)/2^tau can
        # vary, but the minimum non-zero theta is 1/2^tau)
        min_nonzero_sigma_sq = (1.0 / 2**tau) * F_sq
        expected_c = min_nonzero_sigma_sq + noise_var
        np.testing.assert_allclose(c, expected_c, rtol=1e-10)


# ── phase scaling and rotation windows ──────────────────────────────────────

class TestSpectralScale:
    def test_above_lambda_max(self):
        """δ exceeds the largest eigenvalue so all phases are < 1."""
        eigs = np.array([0.1, 0.5, 2.0])
        delta = spectral_scale(eigs)
        assert delta > 2.0
        assert np.all(eigs / delta < 1.0)

    def test_degenerate_spectrum(self):
        """All-zero spectrum falls back to a positive scale."""
        assert spectral_scale(np.zeros(4)) > 0


class TestEigenphaseWindows:
    def test_windows_cover_eigenphases(self):
        """Each eigenphase's central bin is in the active set."""
        from gaussian_quantum.qpca import _bit_reverse
        eigs = np.array([0.2, 1.0])
        delta = spectral_scale(eigs)
        tau = 5
        bins = eigenphase_window_bins(eigs, delta, tau, window=1)
        for lam in eigs:
            centre = int(np.round(lam / delta * 2 ** tau))
            assert _bit_reverse(centre, tau) in bins

    def test_rank_truncation_shrinks_windows(self):
        """rank=1 keeps at most one window's worth of bins."""
        eigs = np.array([0.1, 1.0])
        delta = spectral_scale(eigs)
        all_bins = eigenphase_window_bins(eigs, delta, 5, window=1)
        top1 = eigenphase_window_bins(eigs, delta, 5, rank=1, window=1)
        assert top1 <= all_bins
        assert len(top1) <= 3


# ── data-matrix encoding ─────────────────────────────────────────────────────

class TestEncodeDataMatrix:
    def test_unit_norm(self, hsgp_setup):
        """|ψ_X⟩ is unit-normalised."""
        X_feat, _, _, _ = hsgp_setup
        psi_x, _, _, _ = encode_data_matrix(X_feat)
        np.testing.assert_allclose(np.linalg.norm(psi_x), 1.0, atol=1e-12)

    def test_layout(self, hsgp_setup):
        """Amplitude index n·2^{n_m} + m holds X[n, m]/‖X‖_F."""
        X_feat, _, _, _ = hsgp_setup
        psi_x, frob, n_m, _ = encode_data_matrix(X_feat)
        N, M = X_feat.shape
        for n in (0, N - 1):
            for m in (0, M - 1):
                np.testing.assert_allclose(
                    psi_x[(n << n_m) + m], X_feat[n, m] / frob, atol=1e-12
                )

    def test_partial_trace_is_gram_matrix(self, hsgp_setup):
        """Tracing out |n⟩ gives ρ = XᵀX/‖X‖_F² on the |m⟩ register."""
        X_feat, _, _, _ = hsgp_setup
        psi_x, frob, n_m, n_n = encode_data_matrix(X_feat)
        M = X_feat.shape[1]
        block = psi_x.reshape(2 ** n_n, 2 ** n_m)
        rho = block.conj().T @ block
        np.testing.assert_allclose(
            rho[:M, :M], X_feat.T @ X_feat / frob ** 2, atol=1e-12
        )


# ── analytical SVD references (papers' Eqs. 14–15) ──────────────────────────

class TestAnalyticalReference:
    def test_mean_matches_classical_posterior(self, hsgp_setup):
        """Full-rank SVD sum equals X*ᵀ(XᵀX+σ²I)⁻¹Xᵀy exactly."""
        X_feat, y_train, x_star, noise_var = hsgp_setup
        expected = x_star @ np.linalg.solve(
            X_feat.T @ X_feat + noise_var * np.eye(X_feat.shape[1]),
            X_feat.T @ y_train,
        )
        got = qbq_mean_analytical(X_feat, y_train, x_star, noise_var)
        np.testing.assert_allclose(got, expected, rtol=1e-10)

    def test_variance_matches_classical_posterior(self, hsgp_setup):
        """Full-rank SVD sum equals σ²X*ᵀ(XᵀX+σ²I)⁻¹X* exactly."""
        X_feat, _, x_star, noise_var = hsgp_setup
        expected = noise_var * x_star @ np.linalg.solve(
            X_feat.T @ X_feat + noise_var * np.eye(X_feat.shape[1]), x_star,
        )
        got = qbq_variance_analytical(X_feat, x_star, noise_var)
        np.testing.assert_allclose(got, expected, rtol=1e-10)

    def test_rank_truncation_monotone(self, hsgp_setup):
        """The rank-R mean approaches the full-rank mean as R grows."""
        X_feat, y_train, x_star, noise_var = hsgp_setup
        full = qbq_mean_analytical(X_feat, y_train, x_star, noise_var)
        errs = [abs(qbq_mean_analytical(X_feat, y_train, x_star, noise_var,
                                        rank=R) - full)
                for R in (1, 2, 4)]
        assert errs[-1] <= errs[0] + 1e-12


# ── paper-faithful state-preparation circuits ───────────────────────────────

class TestStatePreparationCircuits:
    def test_mean_state_norm_and_reconstruction(self, hsgp_setup, sv_backend):
        """The mean circuit statevector is normalised and its Hadamard-test
        readout reconstructs the classical HSGP posterior mean."""
        X_feat, y_train, x_star, noise_var = hsgp_setup
        sv, c1, info = prepare_mean_state_circuit(
            X_feat, noise_var, n_eigenvalue_qubits=6, backend=sv_backend,
        )
        np.testing.assert_allclose(np.linalg.norm(sv), 1.0, atol=1e-10)

        re = mean_overlap(sv, info, x_star, y_train)
        q_mean = (info["frob"] * np.linalg.norm(x_star)
                  * np.linalg.norm(y_train) * re / c1)
        expected = qbq_mean_analytical(X_feat, y_train, x_star, noise_var)
        assert abs(q_mean - expected) < 0.05 * max(abs(expected), 0.1)

    def test_variance_probabilities_valid(self, hsgp_setup, sv_backend):
        """Variance-circuit outcome probabilities form a distribution."""
        X_feat, _, x_star, noise_var = hsgp_setup
        sv, c2, info = prepare_variance_state_circuit(
            X_feat, noise_var, n_eigenvalue_qubits=6, backend=sv_backend,
        )
        probs = variance_probabilities(sv, info, x_star)
        assert np.all(probs >= 0)
        np.testing.assert_allclose(np.sum(probs), 1.0, atol=1e-10)

    def test_variance_reconstruction(self, hsgp_setup, sv_backend):
        """SWAP-test statistics reconstruct the analytical variance."""
        X_feat, _, x_star, noise_var = hsgp_setup
        sv, c2, info = prepare_variance_state_circuit(
            X_feat, noise_var, n_eigenvalue_qubits=6, backend=sv_backend,
        )
        probs = variance_probabilities(sv, info, x_star)
        p1 = probs[2] + probs[3]
        p11 = probs[3]
        q_var = (noise_var * np.linalg.norm(x_star) ** 2
                 * info["frob"] ** 2 / c2 ** 2 * (p1 - 2.0 * p11))
        expected = qbq_variance_analytical(X_feat, x_star, noise_var)
        assert abs(q_var - expected) < 0.15 * max(expected, 1e-6)

    def test_variance_rotation_constant_positive(self):
        """Variance rotation returns a positive normalisation constant."""
        _, c2 = conditional_rotation_variance(4, 0.01, 1.0)
        assert c2 > 0
