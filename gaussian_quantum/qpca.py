"""Quantum PCA state preparation for GP quadrature / regression (Stage 2).

Implements the quantum pipeline of Farooq et al. (PRA 109, 052410, 2024) and
Galvis-Florez et al. (arXiv:2502.14467), following the papers' circuit
architecture:

    1. Amplitude-encode the data matrix,  |ψ_X⟩ = Σ_{n,m} x_nm |m⟩|n⟩ / ‖X‖_F,
       over log2(N) + log2(M) qubits.  Its Schmidt decomposition is
       |ψ_X⟩ = Σ_r λ̂_r |v_r⟩_m |u_r⟩_n  with λ̂_r = s_r / ‖X‖_F, where s_r,
       u_r, v_r are the singular values/vectors of X.
    2. QPE with  U = exp(2πi X^T X / δ)  acting on the |m⟩ register writes the
       eigenphase θ_r = s_r² / δ into a τ-qubit register.  Following the
       papers (and Cleve et al. 1998), δ is chosen slightly *above* the
       largest eigenvalue of X^T X so the phases spread over [0, 1).
    3. A conditional R_y rotation keyed on the eigenvalue register encodes
           c₁ / (s_r² + σ²)             (mean,     arXiv Eq. 38)
           c₂ / (s_r √(s_r² + σ²))      (variance, arXiv Eq. 42)
       into the |1⟩ amplitude of an ancilla qubit.
    4. Mean: inverse QPE uncomputes the eigenvalue register (arXiv Eq. 39);
       the mean is then read out with a Hadamard test against
       |ψ₂⟩ = |X_μ⟩|y⟩|0⟩|1⟩ (arXiv Eqs. 40–41).
       Variance: the eigenvalue register is *not* uncomputed; the variance is
       read out with a SWAP test between the |m⟩ register and |X_μ⟩ from the
       joint statistics p(a=1) and p(a=1, b=1) (arXiv Eqs. 43–44).

Simulation note: the state-preparation circuits (steps 1–4) are built and
executed gate-by-gate on the Aer statevector simulator.  The final Hadamard /
SWAP test measurement outcomes are then sampled from the exact outcome
distribution of the corresponding test circuit (see quantum_algorithms.py),
which is statistically identical to executing the test circuit on a noiseless
simulator, but avoids synthesising a controlled version of the entire
preparation circuit.
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import StatePreparation, UnitaryGate
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# Inverse QFT
# ---------------------------------------------------------------------------

def _bit_reverse(j, n_bits):
    """Reverse the bit order of integer *j* with *n_bits* bits."""
    result = 0
    for _ in range(n_bits):
        result = (result << 1) | (j & 1)
        j >>= 1
    return result


def inverse_qft_circuit(n_qubits):
    """Build an inverse Quantum Fourier Transform circuit on *n_qubits*.

    Uses the standard QPE-compatible convention **without** bit-reversal
    swaps, so that the eigenvalue register value *j* directly encodes the
    phase via  θ = bit_reverse(j) / 2^n_qubits.

    Returns:
        QuantumCircuit on *n_qubits* qubits implementing QFT†.
    """
    qc = QuantumCircuit(n_qubits, name="IQFT")
    for j in reversed(range(n_qubits)):
        for k in reversed(range(j + 1, n_qubits)):
            qc.cp(-np.pi / 2 ** (k - j), j, k)
        qc.h(j)
    # No bit-reversal swaps — see _bit_reverse() for phase interpretation.
    return qc


# ---------------------------------------------------------------------------
# Phase scaling and density-matrix unitary construction
# ---------------------------------------------------------------------------

def spectral_scale(eigenvalues, margin=0.05, n_bins=None):
    """Phase-scaling constant δ for  U = exp(2πi X^T X / δ).

    Following the papers (δ slightly greater than the largest eigenvalue of
    X^T X, cf. Cleve et al. 1998), returns  δ ≈ (1 + margin) · λ_max  so the
    largest eigenphase sits near 1/(1+margin) < 1 and the spectrum fills the
    QPE window instead of being compressed by the trace normalisation.

    When *n_bins* (= 2^τ) is given, δ is additionally snapped so that the
    dominant eigenphase λ_max/δ falls exactly on a QPE bin.  δ is a free
    parameter of the algorithm, and aligning it with the grid removes the
    QPE leakage of the dominant spectral component entirely — the same
    δ-tuning freedom the papers exploit by taking medians over several δ.

    Args:
        eigenvalues: Eigenvalues of X^T X (any array-like).
        margin: Relative safety margin above λ_max.
        n_bins: Number of QPE bins 2^τ for grid snapping (None = no snap).

    Returns:
        Scalar δ > 0.
    """
    lam_max = float(np.max(eigenvalues)) if len(np.atleast_1d(eigenvalues)) else 0.0
    if lam_max <= 0.0:
        return 1.0
    delta = (1.0 + margin) * lam_max
    if n_bins:
        k = max(1, int(np.floor(n_bins * lam_max / delta)))
        delta = lam_max * n_bins / k
    return delta


def build_density_matrix_unitary(XtX, scale):
    """Construct the unitary  U = exp(2πi X^T X / scale).

    The eigendecomposition of X^T X gives phases θ_r = s_r² / scale, and
    U = V diag(e^{2πi θ_1}, …, e^{2πi θ_M}) V^T.  Use
    :func:`spectral_scale` for the papers' δ-scaling.

    Args:
        XtX: (M, M) Gram matrix  X^T X.
        scale: Phase scaling δ (must exceed the largest eigenvalue).

    Returns:
        U: (M_pad, M_pad) unitary matrix (padded to power-of-2 dimension).
        eigenvalues: (M,) eigenvalues of X^T X (s_r²).
        eigenvectors: (M, M) columns are eigenvectors of X^T X.
        n_state_qubits: Number of qubits for the |m⟩ register.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(XtX)
    eigenvalues = np.maximum(eigenvalues, 0.0)  # numerical safety

    M = len(eigenvalues)
    n_state_qubits = max(1, int(np.ceil(np.log2(M))))
    dim = 2 ** n_state_qubits

    # Phases: θ_r = s_r² / δ
    phases = eigenvalues / scale

    # Pad eigenvectors to dim × dim
    V_pad = np.eye(dim, dtype=complex)
    V_pad[:M, :M] = eigenvectors

    # Diagonal of eigenphases (padded entries get phase 0 → identity)
    D = np.diag(np.exp(2j * np.pi * np.concatenate([phases, np.zeros(dim - M)])))

    U = V_pad @ D @ V_pad.conj().T
    return U, eigenvalues, eigenvectors, n_state_qubits


# ---------------------------------------------------------------------------
# Data-matrix amplitude encoding
# ---------------------------------------------------------------------------

def encode_data_matrix(X_feat):
    """Amplitude-encode the data matrix as |ψ_X⟩ = Σ_{n,m} x_nm |m⟩|n⟩ / ‖X‖_F.

    The |m⟩ register (basis-function index, dim ≥ M) occupies the low-order
    qubits and the |n⟩ register (data index, dim ≥ N) the high-order qubits,
    i.e. amplitude index  i = n · 2^{n_m} + m.  Tracing out |n⟩ then yields
    the density matrix  ρ = X^T X / ‖X‖_F²  on the |m⟩ register.

    Args:
        X_feat: (N, M) real feature matrix.

    Returns:
        psi_x: (2^{n_m + n_n},) unit-norm complex amplitudes.
        frob: Frobenius norm ‖X‖_F.
        n_m: Qubits in the |m⟩ register.
        n_n: Qubits in the |n⟩ register.
    """
    X_feat = np.atleast_2d(np.asarray(X_feat, dtype=float))
    N, M = X_feat.shape
    frob = float(np.linalg.norm(X_feat))
    if frob < 1e-300:
        raise ValueError("Cannot encode an all-zero data matrix.")

    n_m = max(1, int(np.ceil(np.log2(M))))
    n_n = max(1, int(np.ceil(np.log2(N))))
    psi_x = np.zeros(2 ** (n_m + n_n), dtype=complex)
    for n in range(N):
        base = n << n_m
        psi_x[base:base + M] = X_feat[n, :] / frob
    return psi_x, frob, n_m, n_n


# ---------------------------------------------------------------------------
# Conditional rotations
# ---------------------------------------------------------------------------

def eigenphase_window_bins(eigenvalues, scale, tau, rank=None, window=4,
                           min_eigenvalue=None):
    """Eigenvalue-register values around the dominant eigenphases.

    Implements the papers' rank-R information extraction: conditional
    rotations are applied only on register values within ±*window* bins of
    the eigenphases θ_r = λ_r / scale of the *rank* largest eigenvalues.
    On hardware these dominant bins are located by sampling the eigenvalue
    register after QPE (the qPCA step of Lloyd et al. 2014); in simulation
    they are placed at the classically known eigenphases.

    Restricting the rotations to these windows (a) realises the low-rank
    truncation of the papers ("up to a desired rank"), (b) prevents QPE
    leakage into near-zero bins from being amplified by the large rotation
    weights there, and (c) lets the normalisation constants c₁, c₂ be set by
    the smallest *active* bin instead of the smallest representable bin,
    which greatly improves the shot-noise budget of the tests.

    Args:
        eigenvalues: Eigenvalues of X^T X.
        scale: Phase scaling δ.
        tau: Number of eigenvalue-register qubits.
        rank: Keep the R largest eigenvalues (None = all resolvable ones).
        window: Half-width of the bin window around each eigenphase.
        min_eigenvalue: Discard eigenvalues below this floor.  Pass the QPE
            resolution δ/2^τ: eigenvalues below one bin cannot be resolved
            by the register anyway, and windows placed at the near-zero bins
            would catch leakage from the large eigenvalues with maximal
            rotation weight, biasing the estimates upward.  The floor is
            therefore the rank truncation the τ-qubit register supports,
            and it vanishes as τ grows.

    Returns:
        Set of eigenvalue-register values |j⟩ to rotate on.
    """
    lams = np.sort(np.maximum(np.asarray(eigenvalues, dtype=float), 0.0))[::-1]
    floor = lams[0] * 1e-12 if lams.size and lams[0] > 0 else 0.0
    if min_eigenvalue is not None:
        floor = max(floor, float(min_eigenvalue))
    lams = lams[lams > floor]
    if rank is not None:
        lams = lams[:rank]

    n_vals = 2 ** tau
    bins = set()
    for lam in lams:
        centre = int(np.round(lam / scale * n_vals))
        for k in range(centre - window, centre + window + 1):
            if 0 <= k < n_vals:
                # Register value j encodes phase bit_reverse(j)/2^τ, so the
                # phase-index k corresponds to register value bit_reverse(k).
                bins.add(_bit_reverse(k, tau))
    return bins


def _apply_binwise_rotations(qc, targets, tau):
    """Append a multi-controlled R_y per eigenvalue-register value |j⟩."""
    for j in range(len(targets)):
        amp = np.clip(targets[j], -1.0, 1.0)
        angle = 2.0 * np.arcsin(amp)
        if abs(angle) < 1e-14:
            continue

        bits = format(j, f"0{tau}b")
        for b_idx, bit in enumerate(reversed(bits)):
            if bit == "0":
                qc.x(b_idx)

        qc.mcry(angle, list(range(tau)), tau)

        for b_idx, bit in enumerate(reversed(bits)):
            if bit == "0":
                qc.x(b_idx)


def conditional_rotation_mean(n_eigenvalue_qubits, noise_var, scale,
                              active_bins=None):
    """Conditional R_y rotation for the mean estimator (arXiv Eq. 38).

    After QPE (without bit-reversal swaps), register value |j⟩ encodes the
    eigenphase  θ_j = bit_reverse(j) / 2^τ, i.e. the eigenvalue estimate
    ŝ² = θ_j · scale.  The rotation encodes

        sin(angle/2) = c₁ / (ŝ² + σ²)

    so the ancilla |1⟩ amplitude carries the spectral factor of the mean.

    The normalisation constant c₁ is set by the largest target over the
    active non-zero bins, i.e. c₁ = ŝ²_min-active + σ².  With
    active_bins=None every bin rotates and c₁ = scale/2^τ + σ²; the j = 0
    bin (θ = 0) would then need amplitude c₁/σ² > 1 and is clipped to 1,
    under-weighting eigenvalues below the QPE resolution (their mean
    contribution is O(s_r) and vanishes as τ grows).  With the windowed
    bins from :func:`eigenphase_window_bins`, c₁ is set by the smallest
    *significant* eigenvalue instead, which improves the Hadamard-test
    shot-noise budget by orders of magnitude.

    Args:
        n_eigenvalue_qubits: τ — number of eigenvalue qubits.
        noise_var: Observation noise variance σ².
        scale: Phase scaling δ used in the QPE unitary.
        active_bins: Register values to rotate on (None = all).

    Returns:
        (QuantumCircuit on τ+1 qubits, normalisation constant c₁).
    """
    tau = n_eigenvalue_qubits
    n_vals = 2 ** tau
    qc = QuantumCircuit(tau + 1, name="CondRot_mean")

    if active_bins is None:
        active_bins = range(n_vals)
    active_bins = set(active_bins)

    targets = np.zeros(n_vals)
    for j in active_bins:
        theta_j = _bit_reverse(j, tau) / n_vals
        s_sq = theta_j * scale
        targets[j] = 1.0 / (s_sq + noise_var)

    nonzero = [targets[j] for j in active_bins if _bit_reverse(j, tau) > 0]
    if nonzero:
        c = 1.0 / max(nonzero)
    else:
        c = 1.0 / np.max(targets) if np.max(targets) > 0 else 1.0
    targets = np.minimum(targets * c, 1.0)   # clip the j=0 bin

    _apply_binwise_rotations(qc, targets, tau)
    return qc, c


def conditional_rotation_variance(n_eigenvalue_qubits, noise_var, scale,
                                  active_bins=None):
    """Conditional R_y rotation for the variance estimator (arXiv Eq. 42).

    Encodes  sin(angle/2) = c₂ / (ŝ √(ŝ² + σ²))  with ŝ = √(θ_j · scale).
    Combined with the Schmidt amplitude s_r/‖X‖_F of |ψ_X⟩ this yields
    ancilla-|1⟩ amplitudes  c₂ / (‖X‖_F √(s_r² + σ²)), so the s_r cancels
    and small-but-non-zero eigenvalues keep their full 1/(s_r²+σ²) weight.

    The j = 0 bin (ŝ = 0) has no amplitude to rotate — components with
    s_r = 0 are absent from |ψ_X⟩ — and is set to zero; eigenvalues below
    the QPE resolution are therefore truncated, which is exactly the
    low-rank (R-truncation) semantics of the papers.  Windowed active bins
    (see :func:`eigenphase_window_bins`) additionally stop QPE leakage from
    being amplified by the 1/(ŝ√(ŝ²+σ²)) weights of near-zero bins, and let
    c₂ be set by the smallest active bin.

    Args:
        n_eigenvalue_qubits: τ — number of eigenvalue qubits.
        noise_var: σ².
        scale: Phase scaling δ used in the QPE unitary.
        active_bins: Register values to rotate on (None = all).

    Returns:
        (QuantumCircuit on τ+1 qubits, normalisation constant c₂).
    """
    tau = n_eigenvalue_qubits
    n_vals = 2 ** tau
    qc = QuantumCircuit(tau + 1, name="CondRot_var")

    if active_bins is None:
        active_bins = range(n_vals)
    active_bins = set(active_bins)

    targets = np.zeros(n_vals)
    for j in active_bins:
        theta_j = _bit_reverse(j, tau) / n_vals
        s_sq = theta_j * scale
        if s_sq > 0.0:
            targets[j] = 1.0 / (np.sqrt(s_sq) * np.sqrt(s_sq + noise_var))

    c = 1.0 / np.max(targets) if np.max(targets) > 1e-300 else 1.0
    targets = np.minimum(targets * c, 1.0)

    _apply_binwise_rotations(qc, targets, tau)
    return qc, c


# ---------------------------------------------------------------------------
# Full state-preparation circuits (papers' Figs. 3 and 4)
# ---------------------------------------------------------------------------

def _prepare_qbq_state(X_feat, noise_var, n_eigenvalue_qubits,
                       kind, delta_margin=0.05, rank=None, window=4,
                       backend=None):
    """Build and simulate the |ψ₁⟩ preparation circuit on the full register.

    Qubit layout (little-endian):
        [0 … τ-1]                       eigenvalue register
        [τ … τ+n_m-1]                   |m⟩ register (basis functions)
        [τ+n_m … τ+n_m+n_n-1]           |n⟩ register (data points)
        [τ+n_m+n_n]                     rotation ancilla a

    Steps: StatePreparation(|ψ_X⟩) → QPE on |m⟩ → conditional rotation on
    the rank-R eigenphase windows (see :func:`eigenphase_window_bins`) →
    inverse QPE (mean only; the variance circuit keeps the eigenvalue
    register, which is traced out at readout, cf. arXiv Fig. 4).

    Rotation-bin policy: rotations are restricted to windows of ±*window*
    bins around the eigenphases of the resolvable spectrum (eigenvalues
    above the register resolution δ/2^τ), realising the papers' rank-R
    information extraction.  Rotating on every register value — the formal
    reading of Eqs. 38/42 — is numerically fragile: the rotation targets are
    maximal at the near-zero bins, so the O(1/d²) QPE leakage of the large
    eigenvalues into those bins is amplified by up to (λ_max+σ²)/σ² and
    biases both estimators upward.  This is precisely the δ-sensitivity the
    papers report and median away; the windows remove it at its source.
    The phase scaling δ is snapped so the dominant eigenphase lies exactly
    on a QPE bin (see :func:`spectral_scale`), which removes its leakage
    altogether.

    Args:
        rank: Low-rank truncation R for the rotation windows (None = all
            resolvable eigenvalues).
        window: Half-width of the rotation window around each eigenphase.
            The QPE leakage mass outside a ±w window is O(0.2/w), which
            bounds the truncation bias of the windowed rotations.

    Returns:
        sv: Full statevector, shape (2^{τ+n_m+n_n+1},).
        c: Rotation normalisation constant (c₁ or c₂).
        info: Dict with keys delta, frob, n_m, n_n, tau.
    """
    XtX = X_feat.T @ X_feat
    eigvals = np.maximum(np.linalg.eigvalsh(XtX), 0.0)
    tau = n_eigenvalue_qubits
    delta = spectral_scale(eigvals, margin=delta_margin, n_bins=2 ** tau)

    U, _, _, n_m_check = build_density_matrix_unitary(XtX, delta)
    psi_x, frob, n_m, n_n = encode_data_matrix(X_feat)
    assert n_m == n_m_check

    active_bins = eigenphase_window_bins(
        eigvals, delta, tau, rank=rank, window=window,
        min_eigenvalue=delta / 2 ** tau,
    )
    if kind == "mean":
        cond_rot, c = conditional_rotation_mean(tau, noise_var, delta,
                                                active_bins=active_bins)
    elif kind == "variance":
        cond_rot, c = conditional_rotation_variance(tau, noise_var, delta,
                                                    active_bins=active_bins)
    else:
        raise ValueError(f"Unknown kind '{kind}'.")

    total = tau + n_m + n_n + 1
    ancilla = tau + n_m + n_n
    qc = QuantumCircuit(total, name=f"QBQ_{kind}")

    # 1. Amplitude-encode |ψ_X⟩ on the (m, n) registers
    qc.compose(StatePreparation(psi_x),
               qubits=range(tau, tau + n_m + n_n), inplace=True)

    # 2. QPE on the eigenvalue + |m⟩ registers
    qpe = qpe_circuit(U, tau, n_m)
    qc.compose(qpe, qubits=range(tau + n_m), inplace=True)

    # 3. Conditional rotation on the ancilla
    qc.compose(cond_rot, qubits=list(range(tau)) + [ancilla], inplace=True)

    # 4. Inverse QPE (mean only, arXiv Eq. 39)
    if kind == "mean":
        qc.compose(inverse_qpe_circuit(U, tau, n_m),
                   qubits=range(tau + n_m), inplace=True)

    sv = run_qpca_statevector(qc, backend)
    info = {"delta": delta, "frob": frob, "n_m": n_m, "n_n": n_n, "tau": tau}
    return sv, c, info


def prepare_mean_state_circuit(X_feat, noise_var, n_eigenvalue_qubits,
                               delta_margin=0.05, rank=None, window=4,
                               backend=None):
    """|ψ₁⟩ for the quadrature mean (arXiv Fig. 3 / Eq. 39)."""
    return _prepare_qbq_state(X_feat, noise_var, n_eigenvalue_qubits,
                              "mean", delta_margin, rank, window, backend)


def prepare_variance_state_circuit(X_feat, noise_var, n_eigenvalue_qubits,
                                   delta_margin=0.05, rank=None, window=4,
                                   backend=None):
    """|ξ₃⟩ for the quadrature variance (arXiv Fig. 4 / Eq. 42)."""
    return _prepare_qbq_state(X_feat, noise_var, n_eigenvalue_qubits,
                              "variance", delta_margin, rank, window, backend)


# ---------------------------------------------------------------------------
# Exact readout statistics of the Hadamard / SWAP test circuits
# ---------------------------------------------------------------------------

def mean_overlap(sv, info, x_mu, y_train):
    """Re⟨ψ₁|ψ₂⟩ with |ψ₂⟩ = |X̂_μ⟩_m |ŷ⟩_n |0⟩_τ |1⟩_a (arXiv Eq. 40).

    Args:
        sv: Statevector from :func:`prepare_mean_state_circuit`.
        info: Metadata dict from the same call.
        x_mu: (M,) kernel-mean (or test-point) feature vector, unnormalised.
        y_train: (N,) training targets, unnormalised.

    Returns:
        Re⟨ψ₁|ψ₂⟩ using the *normalised* x_mu and y.
    """
    tau, n_m, n_n = info["tau"], info["n_m"], info["n_n"]
    z_hat = np.zeros(2 ** n_m)
    z_hat[: len(x_mu)] = x_mu / np.linalg.norm(x_mu)
    y_hat = np.zeros(2 ** n_n)
    y_hat[: len(y_train)] = y_train / np.linalg.norm(y_train)

    block = np.asarray(sv).reshape(2, 2 ** n_n, 2 ** n_m, 2 ** tau)
    # a = 1, eigenvalue register = |0⟩; contract m with ẑ and n with ŷ
    psi1_slice = block[1, :, :, 0]                     # (2^{n_n}, 2^{n_m})
    return float(np.real(y_hat @ psi1_slice @ z_hat))


def variance_probabilities(sv, info, x_mu):
    """Joint outcome probabilities of the variance circuit (arXiv Eq. 43).

    Both ancillas of the papers' Fig. 4 are measured: the rotation ancilla a
    and the SWAP-test ancilla b (SWAP test between the |m⟩ register and a
    fresh register prepared in |X̂_μ⟩).  The eigenvalue and |n⟩ registers act
    as traced-out environment, which is what makes the SWAP test measure the
    mixed-state overlap ⟨X̂_μ| ρ'_m |X̂_μ⟩.

    Args:
        sv: Statevector from :func:`prepare_variance_state_circuit`.
        info: Metadata dict from the same call.
        x_mu: (M,) kernel-mean (or test-point) feature vector, unnormalised.

    Returns:
        probs: Length-4 array [p00, p01, p10, p11] for outcomes (a, b).
    """
    tau, n_m, n_n = info["tau"], info["n_m"], info["n_n"]
    z_hat = np.zeros(2 ** n_m)
    z_hat[: len(x_mu)] = x_mu / np.linalg.norm(x_mu)

    block = np.asarray(sv).reshape(2, 2 ** n_n, 2 ** n_m, 2 ** tau)
    probs = np.zeros(4)
    for a in (0, 1):
        p_a = float(np.sum(np.abs(block[a]) ** 2))
        if p_a < 1e-300:
            probs[2 * a + 0] = 0.0
            probs[2 * a + 1] = 0.0
            continue
        # Overlap of the reduced |m⟩ state with |ẑ⟩:  tr(ρ'_m |ẑ⟩⟨ẑ|)
        amp = np.tensordot(block[a], z_hat, axes=([1], [0]))  # (2^{n_n}, 2^{τ})
        overlap = float(np.sum(np.abs(amp) ** 2)) / p_a
        probs[2 * a + 0] = p_a * (1.0 + overlap) / 2.0   # b = 0
        probs[2 * a + 1] = p_a * (1.0 - overlap) / 2.0   # b = 1
    # Guard against tiny negative rounding before sampling
    probs = np.maximum(probs, 0.0)
    total = probs.sum()
    if total > 0:
        probs /= total
    return probs


# ---------------------------------------------------------------------------
# Statevector execution helper
# ---------------------------------------------------------------------------

def run_qpca_statevector(circuit, backend=None):
    """Execute a preparation circuit on the statevector simulator.

    Args:
        circuit: QuantumCircuit (no measurements).
        backend: AerSimulator configured for statevector (default created).

    Returns:
        statevector as a NumPy array.
    """
    if backend is None:
        backend = AerSimulator(method="statevector")
    qc = circuit.copy()
    qc.save_statevector()
    tqc = transpile(qc, backend, optimization_level=0)
    result = backend.run(tqc, shots=1).result()
    return np.asarray(result.get_statevector(qc))


# ---------------------------------------------------------------------------
# QPE circuit
# ---------------------------------------------------------------------------

def qpe_circuit(unitary_matrix, n_eigenvalue_qubits, n_state_qubits):
    """Build a Quantum Phase Estimation circuit.

    The eigenvalue register (qubits 0 … τ-1) is initialised to |0⟩^τ and,
    after the circuit, encodes the phase θ such that  U|v⟩ = e^{2πiθ}|v⟩.

    Args:
        unitary_matrix: (2^n_state, 2^n_state) unitary.
        n_eigenvalue_qubits: τ — precision qubits.
        n_state_qubits: Qubits in the state register.

    Returns:
        QuantumCircuit on (τ + n_state) qubits.
    """
    tau = n_eigenvalue_qubits
    total = tau + n_state_qubits
    qc = QuantumCircuit(total, name="QPE")

    # Hadamard on eigenvalue register
    for q in range(tau):
        qc.h(q)

    u_gate = UnitaryGate(unitary_matrix, label="U")

    for k in range(tau):
        power = 2 ** k
        cu = u_gate.power(power).control(1)
        qc.append(cu, [k] + list(range(tau, total)))

    # Inverse QFT on eigenvalue register
    iqft = inverse_qft_circuit(tau)
    qc.compose(iqft, qubits=range(tau), inplace=True)

    return qc


def inverse_qpe_circuit(unitary_matrix, n_eigenvalue_qubits, n_state_qubits):
    """Build the inverse (adjoint) of the QPE circuit.

    Used to uncompute the eigenvalue register after conditional rotations.
    """
    return qpe_circuit(
        unitary_matrix, n_eigenvalue_qubits, n_state_qubits
    ).inverse()


# ---------------------------------------------------------------------------
# Analytical (infinite-precision) reference — papers' Eqs. (14)–(15)
# ---------------------------------------------------------------------------

def _svd_components(X_feat, rank=None):
    """SVD of X restricted to (numerically) non-zero singular values."""
    U, s, Vt = np.linalg.svd(np.atleast_2d(X_feat), full_matrices=False)
    keep = s > s[0] * 1e-12 if s.size and s[0] > 0 else np.zeros_like(s, bool)
    U, s, Vt = U[:, keep], s[keep], Vt[keep, :]
    if rank is not None:
        U, s, Vt = U[:, :rank], s[:rank], Vt[:rank, :]
    return U, s, Vt


def qbq_mean_analytical(X_feat, y_train, x_mu, noise_var, rank=None):
    """Quadrature mean via the exact SVD sum (arXiv Eq. 14).

        Q = Σ_r  s_r / (s_r² + σ²) · (x_μ^T v_r)(u_r^T y)

    This is the τ → ∞, shots → ∞ limit of the quantum circuit.  With
    rank=None it equals the classical HSGP-BQ mean; a finite *rank* gives
    the papers' R-truncated low-rank quadrature.

    Returns:
        Scalar mean estimate.
    """
    U, s, Vt = _svd_components(X_feat, rank)
    if s.size == 0:
        return 0.0
    weights = s / (s ** 2 + noise_var)
    return float(np.sum(weights * (Vt @ x_mu) * (U.T @ y_train)))


def qbq_variance_analytical(X_feat, x_mu, noise_var, rank=None):
    """Quadrature variance via the exact SVD sum (arXiv Eq. 15).

        V = σ² Σ_r  (x_μ^T v_r)² / (s_r² + σ²)

    The sum runs over the non-zero singular values only (the papers'
    low-rank semantics): components of x_μ in the null space of X are not
    represented in |ψ_X⟩ and are therefore excluded, unlike the full
    classical formula σ² x_μ^T (X^T X + σ²I)^{-1} x_μ which weights them
    by 1/σ².  For a full-rank X^T X the two coincide.

    Returns:
        Scalar variance estimate.
    """
    U, s, Vt = _svd_components(X_feat, rank)
    if s.size == 0:
        return 0.0
    proj = Vt @ x_mu
    return float(noise_var * np.sum(proj ** 2 / (s ** 2 + noise_var)))
