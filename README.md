# gaussian_quantum

Qiskit implementation of quantum-assisted Hilbert-space Gaussian process
regression and Bayesian quadrature, following

- Farooq, Galvis-Florez & Särkkä, *Quantum-assisted Hilbert-space Gaussian
  process regression*, [PRA 109, 052410 (2024)](https://doi.org/10.1103/PhysRevA.109.052410)
  ([arXiv:2402.00544](https://arxiv.org/abs/2402.00544)),
- Galvis-Florez, Farooq & Särkkä, *Provable Quantum Algorithm Advantage for
  Gaussian Process Quadrature*,
  [arXiv:2502.14467](https://arxiv.org/abs/2502.14467),

applied to insurance payoff expectations `E[Π(Z)] = ∫ Π(z) f_Z(z) dz` over
claim-severity distributions (Pareto, lognormal, gamma, Weibull, Poisson) and
contract payoffs (deductibles, policy limits, stop-loss).

## Algorithm overview

The algorithm combines a classical kernel approximation with quantum
subroutines for matrix inversion and inner-product estimation:

| Stage | Component | Module |
|-------|-----------|--------|
| **1** | Hilbert-space kernel approximation (Solin & Särkkä) | `hilbert_space_approx.py` |
| **2** | Data encoding + qPCA — QPE + conditional rotations | `qpca.py` |
| **3** | Hadamard / Swap tests for the quadrature mean/variance | `quantum_algorithms.py` |

### Stage 1 — Hilbert-space kernel approximation

The stationary kernel is replaced by a reduced-rank approximation using
Laplace eigenfunctions on a bounded domain:

```
k(x, x') ≈ Σ_j S(√λ_j) φ_j(x) φ_j(x')
```

This yields a feature matrix `X = Φ √Λ` of shape `(N, M)` where `M ≪ N`,
so the BQ/GP posterior reduces to the `M × M` system:

```
Q_BQ = z_μᵀ (XᵀX + σ²I)⁻¹ Xᵀy         (mean;  z_μ = √Λ Φ_μ kernel mean embedding)
V_BQ = σ² z_μᵀ (XᵀX + σ²I)⁻¹ z_μ       (variance)
```

### Stage 2 — Data encoding + quantum PCA (papers' Figs. 3–4)

1. **Amplitude encoding** of the data matrix,
   `|ψ_X⟩ = Σ_{n,m} x_nm |m⟩|n⟩ / ‖X‖_F`, over `log₂N + log₂M` qubits.
2. **Density-matrix unitary** `U = exp(2πi XᵀX/δ)` acting on the `|m⟩`
   register, with the phase scaling `δ` slightly above the largest
   eigenvalue of `XᵀX` (Cleve et al.), snapped so the dominant eigenphase
   sits exactly on a QPE bin.
3. **QPE** writes the eigenphases `s_r²/δ` into a τ-qubit register.
4. **Conditional rotations** keyed on the eigenvalue register encode
   `c₁/(s_r²+σ²)` (mean) or `c₂/(s_r√(s_r²+σ²))` (variance) into an ancilla.
   Rotations are restricted to windows around the resolvable eigenphases —
   the papers' rank-R information extraction.
5. **Inverse QPE** uncomputes the eigenvalue register (mean circuit only).

### Stage 3 — Hadamard & Swap tests

| Primitive | Statistics | Reconstruction |
|-----------|------------|----------------|
| **Hadamard test** vs `\|X̂_μ⟩\|ŷ⟩\|0⟩\|1⟩` | p₀ | `Q = ‖X‖_F ‖X_μ‖ ‖y‖ (2p₀−1)/c₁` |
| **Swap test** vs `\|X̂_μ⟩` (both ancillas) | p(a=1), p₁₁ | `V = σ²‖X_μ‖² (‖X‖_F²/c₂²)(p(a=1) − 2p₁₁)` |

**Simulation note.** The Stage-2 preparation circuits are built and executed
gate-by-gate on the Aer statevector simulator; the Stage-3 measurement
outcomes are then sampled from the exact outcome distribution of the test
circuit (binomial/multinomial in the shot count), which is statistically
identical to running the test circuit on a noiseless simulator.  The
eigendecomposition used to *construct* `U` in simulation is classical — the
simulation validates the circuits and their measurement statistics, not the
asymptotic speedup, which is established analytically in the papers.

The low-level Hadamard and Swap tests also support direct inner-product
estimation on arbitrary amplitude-encoded vectors (legacy API).

## Structure

```
gaussian_quantum/
├── classical.py              # Classical GP: RBF kernel, posterior mean/variance
├── hilbert_space_approx.py   # Stage 1: HSGP features, classical HSGP posterior
├── qpca.py                   # Stage 2: QPE, conditional rotations, state preparation
└── quantum_algorithms.py     # Stage 3: Hadamard/Swap tests + full pipeline

tests/
├── test_algorithms.py        # Hadamard/Swap test + legacy quantum GP tests
├── test_hilbert_space_approx.py  # Stage 1 unit tests
├── test_qpca.py              # Stage 2 unit tests (QPE, rotations, state prep)
└── test_full_pipeline.py     # Integration tests: quantum HSGP vs classical GP
```

## Usage

### Full quantum HSGP pipeline (all three stages)

```python
import numpy as np
from gaussian_quantum.quantum_algorithms import quantum_hsgp_mean, quantum_hsgp_variance
from gaussian_quantum.hilbert_space_approx import hs_gp_posterior

X_train = np.array([[0.], [1.], [2.], [3.]])
y_train = np.sin(X_train.ravel())
X_test  = np.array([[1.5]])
noise_var = 0.01
M = 8    # number of HSGP basis functions
L = 5.0  # domain boundary

# Classical HSGP reference
hsgp_mean, hsgp_var = hs_gp_posterior(
    X_train, y_train, X_test, M, L, noise_var=noise_var
)

# Full quantum pipeline (Stages 1 + 2 + 3)
q_mean = quantum_hsgp_mean(
    X_train, y_train, X_test, M, L, noise_var,
    n_eigenvalue_qubits=5, shots=32768,
)
q_var = quantum_hsgp_variance(
    X_train, X_test, M, L, noise_var,
    n_eigenvalue_qubits=5, shots=32768,
)

print(f"Mean:     HSGP={hsgp_mean[0]:.4f}  quantum={q_mean:.4f}")
print(f"Variance: HSGP={hsgp_var[0]:.4f}  quantum={q_var:.4f}")
```

### Low-level Hadamard/Swap tests (Stage 3 only)

```python
from gaussian_quantum.quantum_algorithms import hadamard_test, swap_test

v1 = np.array([3.0, 4.0])
v2 = np.array([4.0, 3.0])
print(f"Re<v1|v2> ≈ {hadamard_test(v1, v2):.4f}")
print(f"|<v1|v2>|² ≈ {swap_test(v1, v2):.4f}")
```

## Requirements

```
pip install -r requirements.txt
```