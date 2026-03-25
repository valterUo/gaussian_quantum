# gaussian_quantum

Minimal Qiskit implementation of the mean and variance algorithms from
[arXiv:2402.00544](https://arxiv.org/abs/2402.00544) —
*Quantum-Assisted Hilbert-Space Gaussian Process Regression*.

## Algorithms

Two core quantum primitives are used to compute GP posterior statistics:

| Primitive | Computes | Used for |
|-----------|----------|----------|
| **Hadamard test** | Re⟨v̂₁&#x7C;v̂₂⟩ | Posterior mean μ* |
| **Swap test** | &#x7C;⟨v̂₁&#x7C;v̂₂⟩&#x7C;² | Posterior variance σ²* |

Both tests operate on amplitude-encoded states: a real vector **v** is
encoded as the unit state |v̂⟩ = |v⟩/‖v‖. The GP values are recovered by
re-scaling with classical norms:

```
μ*  = ‖k*‖ · ‖α‖ · Re⟨k̂*|α̂⟩          (Hadamard test)
σ²* = k** − ‖k*‖ · ‖β‖ · |⟨k̂*|β̂⟩|    (Swap test)
```

where `α = (K+σ²I)⁻¹ y` and `β = (K+σ²I)⁻¹ k*`.

## Structure

```
gaussian_quantum/
├── classical.py           # Classical GP: RBF kernel, posterior mean/variance
└── quantum_algorithms.py  # Hadamard test, Swap test, quantum_gp_mean/variance

tests/
└── test_algorithms.py     # Validates quantum results against classical GP
```

## Usage

```python
import numpy as np
from gaussian_quantum.classical import rbf_kernel, gp_posterior
from gaussian_quantum.quantum_algorithms import quantum_gp_mean, quantum_gp_variance

X_train = np.array([[0.], [1.], [2.], [3.]])
y_train = np.sin(X_train.ravel())
X_test  = np.array([[1.5]])
noise_var = 0.01

# Classical reference
K           = rbf_kernel(X_train, X_train)
k_star      = rbf_kernel(X_train, X_test).ravel()
k_star_star = rbf_kernel(X_test, X_test)[0, 0]
classical_mean, classical_var = gp_posterior(X_train, y_train, X_test, noise_var=noise_var)

# Quantum
K_noisy = K + noise_var * np.eye(len(y_train))
alpha   = np.linalg.solve(K_noisy, y_train)

q_mean = quantum_gp_mean(alpha, k_star)
q_var  = quantum_gp_variance(k_star_star, k_star, K, noise_var)

print(f"Mean:     classical={classical_mean[0]:.4f}  quantum={q_mean:.4f}")
print(f"Variance: classical={classical_var[0]:.4f}  quantum={q_var:.4f}")
```

## Requirements

```
pip install -r requirements.txt
```

## Running Tests

```
pytest tests/
```
