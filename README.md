# gaussian_quantum

Qiskit implementation of the full three-stage quantum-assisted GP regression
algorithm from [arXiv:2402.00544](https://arxiv.org/abs/2402.00544) —
*Quantum-Assisted Hilbert-Space Gaussian Process Regression*.

## Algorithm overview

The algorithm combines a classical kernel approximation with quantum
subroutines for matrix inversion and inner-product estimation:

| Stage | Component | Module |
|-------|-----------|--------|
| **1** | Hilbert-space kernel approximation (Solin & Särkkä) | `hilbert_space_approx.py` |
| **2** | Quantum PCA — QPE + conditional rotations | `qpca.py` |
| **3** | Hadamard / Swap tests for inner products | `quantum_algorithms.py` |

### Stage 1 — Hilbert-space kernel approximation

The stationary kernel is replaced by a reduced-rank approximation using
Laplace eigenfunctions on a bounded domain:

```
k(x, x') ≈ Σ_j S(√λ_j) φ_j(x) φ_j(x')
```

This yields a feature matrix `X = Φ √Λ` of shape `(N, M)` where `M ≪ N`,
so the GP posterior reduces to the `M × M` system:

```
E[f*] = X*ᵀ (XᵀX + σ²I)⁻¹ Xᵀy
V[f*] = σ² X*ᵀ (XᵀX + σ²I)⁻¹ X*
```

### Stage 2 — Quantum PCA (qPCA)

The quantum speedup comes from encoding the spectral decomposition of
`ρ = XᵀX / ‖X‖²_F` into a quantum register via:

1. **Density-matrix unitary** `U = exp(2πi ρ)`
2. **Quantum Phase Estimation (QPE)** extracts eigenvalues `σ_r²/‖X‖²_F`
   into a τ-qubit register
3. **Conditional rotations** encode `1/(σ_r² + σ²)` (mean) into an ancilla
   amplitude, keyed on the eigenvalue register
4. **Inverse QPE** uncomputes the eigenvalue register

### Stage 3 — Hadamard & Swap tests

The qPCA-prepared states are fed to quantum inner-product estimation:

| Primitive | Computes | Used for |
|-----------|----------|----------|
| **Hadamard test** | Re⟨ψ₁&#x7C;ψ₂⟩ | Posterior mean μ* |
| **Swap test** | &#x7C;⟨ψ'₁&#x7C;ψ'₂⟩&#x7C;² | Posterior variance σ²* |

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

## Running Tests

```
pytest tests/
```
