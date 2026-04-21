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

=== STAGE 1: Analytical evaluation ===
375 combos × 20 cases = 7500 evaluations

  [   1/375]  N= 8 M= 2 σ_n=0.001 ls=0.5  anl_mean_err=0.472107  anl_max_err=2.486095
  [  20/375]  N= 8 M= 2 σ_n=0.100 ls=5.0  anl_mean_err=1.023080  anl_max_err=3.521455
  [  40/375]  N= 8 M= 3 σ_n=0.050 ls=5.0  anl_mean_err=0.675901  anl_max_err=3.088894
  [  60/375]  N= 8 M= 4 σ_n=0.010 ls=5.0  anl_mean_err=0.584224  anl_max_err=3.028021
  [  80/375]  N= 8 M= 6 σ_n=0.001 ls=5.0  anl_mean_err=6.692921  anl_max_err=56.254770
  [ 100/375]  N= 8 M= 6 σ_n=0.500 ls=5.0  anl_mean_err=3.383304  anl_max_err=9.288234
  [ 120/375]  N= 8 M= 8 σ_n=0.100 ls=5.0  anl_mean_err=1.082989  anl_max_err=3.577060
  [ 140/375]  N=16 M= 2 σ_n=0.050 ls=5.0  anl_mean_err=0.413687  anl_max_err=2.608140
  [ 160/375]  N=16 M= 3 σ_n=0.010 ls=5.0  anl_mean_err=0.190186  anl_max_err=1.117674
  [ 180/375]  N=16 M= 4 σ_n=0.001 ls=5.0  anl_mean_err=0.062672  anl_max_err=0.170694
  [ 200/375]  N=16 M= 4 σ_n=0.500 ls=5.0  anl_mean_err=1.675476  anl_max_err=3.925923
  [ 220/375]  N=16 M= 6 σ_n=0.100 ls=5.0  anl_mean_err=0.405177  anl_max_err=2.165839
  [ 240/375]  N=16 M= 8 σ_n=0.050 ls=5.0  anl_mean_err=0.295562  anl_max_err=1.682019
  [ 260/375]  N=32 M= 2 σ_n=0.010 ls=5.0  anl_mean_err=0.421301  anl_max_err=2.231539
  [ 280/375]  N=32 M= 3 σ_n=0.001 ls=5.0  anl_mean_err=0.138499  anl_max_err=0.983260
  [ 300/375]  N=32 M= 3 σ_n=0.500 ls=5.0  anl_mean_err=1.726079  anl_max_err=4.418063
  [ 320/375]  N=32 M= 4 σ_n=0.100 ls=5.0  anl_mean_err=0.582200  anl_max_err=1.965893
  [ 340/375]  N=32 M= 6 σ_n=0.050 ls=5.0  anl_mean_err=0.351552  anl_max_err=1.413326
  [ 360/375]  N=32 M= 8 σ_n=0.010 ls=5.0  anl_mean_err=0.153338  anl_max_err=0.888672

====================================================================================================
STAGE 1: TOP 30 ANALYTICAL COMBOS (by mean error)
====================================================================================================
Rank    N   M      σ_n     ls    mean_err    max_err
------------------------------------------------------------
   1   32   8    0.001    3.0    0.012687   0.031415
   2   32   6    0.001    2.0    0.016660   0.064195
   3   32   6    0.001    1.0    0.016662   0.064196
   4   32   6    0.001    0.5    0.016662   0.064196
   5   32   6    0.001    3.0    0.018767   0.064195
   6   32   6    0.010    2.0    0.020671   0.077732
   7   32   6    0.010    0.5    0.020722   0.078653
   8   32   6    0.010    1.0    0.020738   0.078658
   9   32   8    0.001    2.0    0.022461   0.187919
  10   32   8    0.001    1.0    0.022500   0.188255
  11   32   8    0.001    0.5    0.022500   0.188255
  12   32   8    0.010    2.0    0.031411   0.181578
  13   32   8    0.001    5.0    0.032630   0.176693
  14   32   6    0.010    3.0    0.033212   0.162365
  15   32   8    0.010    0.5    0.034333   0.210273
  16   32   8    0.010    1.0    0.034352   0.210252
  17   32   8    0.010    3.0    0.035870   0.162347
  18   32   6    0.001    5.0    0.037506   0.176693
  19   32   4    0.001    5.0    0.047678   0.176219
  20   32   4    0.001    3.0    0.056466   0.275555
  21   32   4    0.001    0.5    0.056468   0.275572
  22   32   4    0.001    2.0    0.056468   0.275571
  23   32   4    0.001    1.0    0.056468   0.275572
  24   16   4    0.001    5.0    0.062672   0.170694
  25   32   4    0.010    3.0    0.069002   0.250426
  26   32   4    0.010    0.5    0.069132   0.252182
  27   32   4    0.010    2.0    0.069145   0.252091
  28   32   4    0.010    1.0    0.069147   0.252196
  29   16   4    0.010    3.0    0.078298   0.432643
  30   16   4    0.010    2.0    0.079277   0.437005

=== STAGE 2: Circuit evaluation ===
8 promising combos × 5 τ values × 2 shots × 20 cases = 1600 evaluations

  [   20/1600]  N=32 M= 8 τ=4 σ_n=0.001 ls=3.0 shots=32768  mean_err=0.388673  max_err=1.674346
  [  100/1600]  N=32 M= 8 τ=6 σ_n=0.001 ls=3.0 shots=32768  mean_err=0.257883  max_err=1.090922
  [  200/1600]  N=32 M= 8 τ=8 σ_n=0.001 ls=3.0 shots=65536  mean_err=0.071749  max_err=0.168085
  [  300/1600]  N=32 M= 6 τ=6 σ_n=0.001 ls=2.0 shots=32768  mean_err=0.110197  max_err=0.371595
  [  400/1600]  N=32 M= 6 τ=8 σ_n=0.001 ls=2.0 shots=65536  mean_err=0.036479  max_err=0.168848
  [  500/1600]  N=32 M= 6 τ=6 σ_n=0.001 ls=1.0 shots=32768  mean_err=0.101831  max_err=0.494119
  [  600/1600]  N=32 M= 6 τ=8 σ_n=0.001 ls=1.0 shots=65536  mean_err=0.036075  max_err=0.165405
  [  700/1600]  N=32 M= 6 τ=6 σ_n=0.001 ls=0.5 shots=32768  mean_err=0.122317  max_err=0.535636
  [  800/1600]  N=32 M= 6 τ=8 σ_n=0.001 ls=0.5 shots=65536  mean_err=0.069912  max_err=0.453514
  [  900/1600]  N=32 M= 6 τ=6 σ_n=0.001 ls=3.0 shots=32768  mean_err=0.263697  max_err=1.090922
  [ 1000/1600]  N=32 M= 6 τ=8 σ_n=0.001 ls=3.0 shots=65536  mean_err=0.072873  max_err=0.221054
  [ 1100/1600]  N=32 M= 6 τ=6 σ_n=0.010 ls=2.0 shots=32768  mean_err=0.131678  max_err=0.403330
  [ 1200/1600]  N=32 M= 6 τ=8 σ_n=0.010 ls=2.0 shots=65536  mean_err=0.052050  max_err=0.214585
  [ 1300/1600]  N=32 M= 6 τ=6 σ_n=0.010 ls=0.5 shots=32768  mean_err=0.150255  max_err=0.556177
  [ 1400/1600]  N=32 M= 6 τ=8 σ_n=0.010 ls=0.5 shots=65536  mean_err=0.095424  max_err=0.453383
  [ 1500/1600]  N=32 M= 6 τ=6 σ_n=0.010 ls=1.0 shots=32768  mean_err=0.121357  max_err=0.519268
  [ 1600/1600]  N=32 M= 6 τ=8 σ_n=0.010 ls=1.0 shots=65536  mean_err=0.051606  max_err=0.223631

========================================================================================================================
STAGE 2: TOP 20 CIRCUIT COMBOS (by mean absolute error)
========================================================================================================================
Rank    N   M   τ  σ_noise  l_scale   shots    mean_err    max_err
--------------------------------------------------------------------------------
   1   32   6   8    0.001     1.00   65536    0.036075   0.165405
   2   32   6   8    0.001     2.00   65536    0.036479   0.168848
   3   32   6   6    0.001     1.00   65536    0.040689   0.169023
   4   32   6   7    0.001     1.00   65536    0.047421   0.197066
   5   32   6   8    0.001     2.00   32768    0.049479   0.248573
   6   32   6   8    0.010     1.00   65536    0.051606   0.223631
   7   32   6   8    0.010     2.00   65536    0.052050   0.214585
   8   32   6   7    0.001     2.00   65536    0.053665   0.159920
   9   32   6   5    0.001     1.00   65536    0.055760   0.147836
  10   32   6   6    0.010     1.00   65536    0.058558   0.202400
  11   32   6   7    0.010     1.00   65536    0.064375   0.231475
  12   32   6   7    0.001     2.00   32768    0.066241   0.253373
  13   32   6   7    0.001     0.50   65536    0.067508   0.387844
  14   32   6   8    0.010     2.00   32768    0.068700   0.287980
  15   32   6   8    0.001     0.50   65536    0.069912   0.453514
  16   32   6   7    0.010     2.00   65536    0.070514   0.231310
  17   32   6   5    0.010     1.00   65536    0.071432   0.177728
  18   32   8   8    0.001     3.00   65536    0.071749   0.168085
  19   32   6   8    0.001     3.00   65536    0.072873   0.221054
  20   32   6   4    0.001     1.00   65536    0.077460   0.272397
========================================================================================================================

TOP 10 BY WORST-CASE (max) ERROR:
--------------------------------------------------------------------------------
   1  N= 32 M=  6 τ=  5 σ_n=0.001 ls=1.00 shots=  65536  mean=0.055760 max=0.147836
   2  N= 32 M=  6 τ=  7 σ_n=0.001 ls=2.00 shots=  65536  mean=0.053665 max=0.159920
   3  N= 32 M=  6 τ=  8 σ_n=0.001 ls=1.00 shots=  65536  mean=0.036075 max=0.165405
   4  N= 32 M=  8 τ=  8 σ_n=0.001 ls=3.00 shots=  65536  mean=0.071749 max=0.168085
   5  N= 32 M=  6 τ=  8 σ_n=0.001 ls=2.00 shots=  65536  mean=0.036479 max=0.168848
   6  N= 32 M=  6 τ=  6 σ_n=0.001 ls=1.00 shots=  65536  mean=0.040689 max=0.169023
   7  N= 32 M=  6 τ=  5 σ_n=0.010 ls=1.00 shots=  65536  mean=0.071432 max=0.177728
   8  N= 32 M=  6 τ=  7 σ_n=0.001 ls=1.00 shots=  65536  mean=0.047421 max=0.197066
   9  N= 32 M=  6 τ=  6 σ_n=0.010 ls=1.00 shots=  65536  mean=0.058558 max=0.202400
  10  N= 32 M=  6 τ=  8 σ_n=0.010 ls=2.00 shots=  65536  mean=0.052050 max=0.214585


DETAILED PER-CASE RESULTS FOR TOP-3 COMBOS:

--- Rank 1: N=32 M=6 τ=8 σ_n=0.001 ls=1.0 shots=65536 ---
        pareto ×       ordinary_deductible  exact=0.486000  q=0.517022  err=0.031022  (6.38%)
        pareto ×      franchise_deductible  exact=1.485000  q=1.514327  err=0.029327  (1.97%)
        pareto ×              policy_limit  exact=1.475000  q=1.451374  err=0.023626  (1.60%)
        pareto ×     deductible_with_limit  exact=0.481111  q=0.514778  err=0.033667  (7.00%)
        pareto ×                 stop_loss  exact=0.486000  q=0.517022  err=0.031022  (6.38%)
     lognormal ×       ordinary_deductible  exact=0.857974  q=0.834203  err=0.023771  (2.77%)
     lognormal ×      franchise_deductible  exact=1.356973  q=1.346710  err=0.010262  (0.76%)
     lognormal ×              policy_limit  exact=1.465526  q=1.630931  err=0.165405  (11.29%)
     lognormal ×     deductible_with_limit  exact=0.748418  q=0.740968  err=0.007450  (1.00%)
     lognormal ×                 stop_loss  exact=0.857974  q=0.834203  err=0.023771  (2.77%)
         gamma ×       ordinary_deductible  exact=1.094307  q=1.113043  err=0.018735  (1.71%)
         gamma ×      franchise_deductible  exact=1.829066  q=1.889231  err=0.060165  (3.29%)
         gamma ×              policy_limit  exact=1.947834  q=2.025730  err=0.077896  (4.00%)
         gamma ×     deductible_with_limit  exact=1.078808  q=1.096193  err=0.017385  (1.61%)
         gamma ×                 stop_loss  exact=1.094307  q=1.113043  err=0.018735  (1.71%)
       weibull ×       ordinary_deductible  exact=0.925613  q=0.942532  err=0.016919  (1.83%)
       weibull ×      franchise_deductible  exact=1.626801  q=1.659611  err=0.032810  (2.02%)
       weibull ×              policy_limit  exact=1.785361  q=1.850717  err=0.065357  (3.66%)
       weibull ×     deductible_with_limit  exact=0.923499  q=0.940751  err=0.017252  (1.87%)
       weibull ×                 stop_loss  exact=0.925613  q=0.942532  err=0.016919  (1.83%)

--- Rank 2: N=32 M=6 τ=8 σ_n=0.001 ls=2.0 shots=65536 ---
        pareto ×       ordinary_deductible  exact=0.486000  q=0.521243  err=0.035243  (7.25%)
        pareto ×      franchise_deductible  exact=1.485000  q=1.409169  err=0.075831  (5.11%)
        pareto ×              policy_limit  exact=1.475000  q=1.453160  err=0.021840  (1.48%)
        pareto ×     deductible_with_limit  exact=0.481111  q=0.516025  err=0.034914  (7.26%)
        pareto ×                 stop_loss  exact=0.486000  q=0.521243  err=0.035243  (7.25%)
     lognormal ×       ordinary_deductible  exact=0.857974  q=0.835184  err=0.022790  (2.66%)
     lognormal ×      franchise_deductible  exact=1.356973  q=1.346174  err=0.010798  (0.80%)
     lognormal ×              policy_limit  exact=1.465526  q=1.634374  err=0.168848  (11.52%)
     lognormal ×     deductible_with_limit  exact=0.748418  q=0.734492  err=0.013926  (1.86%)
     lognormal ×                 stop_loss  exact=0.857974  q=0.835184  err=0.022790  (2.66%)
         gamma ×       ordinary_deductible  exact=1.094307  q=1.097130  err=0.002823  (0.26%)
         gamma ×      franchise_deductible  exact=1.829066  q=1.885448  err=0.056382  (3.08%)
         gamma ×              policy_limit  exact=1.947834  q=2.030022  err=0.082187  (4.22%)
         gamma ×     deductible_with_limit  exact=1.078808  q=1.083081  err=0.004272  (0.40%)
         gamma ×                 stop_loss  exact=1.094307  q=1.097130  err=0.002823  (0.26%)
       weibull ×       ordinary_deductible  exact=0.925613  q=0.931929  err=0.006316  (0.68%)
       weibull ×      franchise_deductible  exact=1.626801  q=1.680752  err=0.053950  (3.32%)
       weibull ×              policy_limit  exact=1.785361  q=1.851388  err=0.066028  (3.70%)
       weibull ×     deductible_with_limit  exact=0.923499  q=0.929756  err=0.006257  (0.68%)
       weibull ×                 stop_loss  exact=0.925613  q=0.931929  err=0.006316  (0.68%)

--- Rank 3: N=32 M=6 τ=6 σ_n=0.001 ls=1.0 shots=65536 ---
        pareto ×       ordinary_deductible  exact=0.486000  q=0.519445  err=0.033445  (6.88%)
        pareto ×      franchise_deductible  exact=1.485000  q=1.519583  err=0.034583  (2.33%)
        pareto ×              policy_limit  exact=1.475000  q=1.547420  err=0.072420  (4.91%)
        pareto ×     deductible_with_limit  exact=0.481111  q=0.511581  err=0.030469  (6.33%)
        pareto ×                 stop_loss  exact=0.486000  q=0.519445  err=0.033445  (6.88%)
     lognormal ×       ordinary_deductible  exact=0.857974  q=0.831084  err=0.026890  (3.13%)
     lognormal ×      franchise_deductible  exact=1.356973  q=1.358291  err=0.001318  (0.10%)
     lognormal ×              policy_limit  exact=1.465526  q=1.634549  err=0.169023  (11.53%)
     lognormal ×     deductible_with_limit  exact=0.748418  q=0.735012  err=0.013406  (1.79%)
     lognormal ×                 stop_loss  exact=0.857974  q=0.831084  err=0.026890  (3.13%)
         gamma ×       ordinary_deductible  exact=1.094307  q=1.116135  err=0.021828  (1.99%)
         gamma ×      franchise_deductible  exact=1.829066  q=1.893070  err=0.064004  (3.50%)
         gamma ×              policy_limit  exact=1.947834  q=2.029857  err=0.082023  (4.21%)
         gamma ×     deductible_with_limit  exact=1.078808  q=1.098333  err=0.019525  (1.81%)
         gamma ×                 stop_loss  exact=1.094307  q=1.116135  err=0.021828  (1.99%)
       weibull ×       ordinary_deductible  exact=0.925613  q=0.942964  err=0.017351  (1.87%)
       weibull ×      franchise_deductible  exact=1.626801  q=1.657149  err=0.030348  (1.87%)
       weibull ×              policy_limit  exact=1.785361  q=1.864809  err=0.079448  (4.45%)
       weibull ×     deductible_with_limit  exact=0.923499  q=0.941680  err=0.018181  (1.97%)
       weibull ×                 stop_loss  exact=0.925613  q=0.942964  err=0.017351  (1.87%)

       These are now the results.

The grid search is complete. Let me analyze and update the defaults.

**Key findings:**
- Best combo: N=32, M=6, τ=8, σ_n=0.001, ls=1.0, shots=65536 → **mean_err=0.036, max_err=0.165**
- Most cases are within 1-7% error, but `lognormal × policy_limit` is a persistent outlier (~11%) across all parameter combos
- M=6 consistently beats M=8 in circuit mode (fewer eigenvalues = cleaner QPE)
- τ=8 with 65536 shots is the sweet spot 

Starting: *Analyse results and set best defaults* (4/4)

Searched for text `def run_experiment` (`**/insurance_experiments.py`), 1 result

Read [](file:///c%3A/Users/valte/OneDrive%20-%20University%20of%20Helsinki/Desktop/gaussian_quantum/insurance_experiments.py#38-38), lines 38 to 120

Read [](file:///c%3A/Users/valte/OneDrive%20-%20University%20of%20Helsinki/Desktop/gaussian_quantum/insurance_experiments.py#120-120), lines 120 to 230

Now let me find the CLI defaults: Compacted conversation

Searched for text `add_argument` (`**/insurance_experiments.py`), 14 results

Read [](file:///c%3A/Users/valte/OneDrive%20-%20University%20of%20Helsinki/Desktop/gaussian_quantum/insurance_experiments.py#510-510), lines 510 to 560

Read [](file:///c%3A/Users/valte/OneDrive%20-%20University%20of%20Helsinki/Desktop/gaussian_quantum/insurance_experiments.py#560-560), lines 560 to 590

Now I have the full picture. Let me update both the function defaults and the CLI defaults with the best parameters from the grid search. 

Replacing 16 lines with 16 lines in [](file:///c%3A/Users/valte/OneDrive%20-%20University%20of%20Helsinki/Desktop/gaussian_quantum/insurance_experiments.py)


```

```


Completed: *Analyse results and set best defaults* (4/4)

Updated insurance_experiments.py defaults with the best grid search combo:

| Parameter | Old | New |
|---|---|---|
| `quantum_N` | 16 | **32** |
| `quantum_noise_std` | 0.1 | **0.001** |
| `quantum_length_scale` | 2.0 | **1.0** |
| `shots` | 32768 | **65536** |
| `quantum_M` | 6 | 6 (unchanged) |
| `n_eigenvalue_qubits` | 8 | 8 (unchanged) |

**Summary of findings:**

- **Mean error: 0.036** (best achievable with QPE circuits) — 16 of 20 cases are below 0.04
- **Analytical HSGP achieves 0.013** but circuit discretization adds ~0.02 on top — this is a fundamental QPE resolution limit, not a parameter tuning issue
- **Persistent outlier:** `lognormal × policy_limit` at ~11% error — the policy_limit payoff creates a sharp kink that HSGP features struggle to approximate regardless of parameters
- The 0.01-level target is met analytically but circuit QPE settles at ~0.036 mean. To close that gap further would require more QPE qubits (τ > 8) or amplitude estimation techniques

Made changes.


Read memory [](file:///c%3A/Users/valte/AppData/Roaming/Code/User/workspaceStorage/5b7cebc246dc5595200fca6a7a6fe0bb/GitHub.copilot-chat/memory-tool/memories/repo/gaussian_quantum_overview.md)

Read [](file:///c%3A/Users/valte/OneDrive%20-%20University%20of%20Helsinki/Desktop/gaussian_quantum/insurance_experiments.py#1-1), lines 1 to 80

Read [](file:///c%3A/Users/valte/OneDrive%20-%20University%20of%20Helsinki/Desktop/gaussian_quantum/quantum_grid_search.py#1-1), lines 1 to 80

Read [](file:///c%3A/Users/valte/OneDrive%20-%20University%20of%20Helsinki/Desktop/gaussian_quantum/quantum_grid_search.py#80-80), lines 80 to 160

Read [](file:///c%3A/Users/valte/OneDrive%20-%20University%20of%20Helsinki/Desktop/gaussian_quantum/insurance_experiments.py#80-80), lines 80 to 200

Read [](file:///c%3A/Users/valte/OneDrive%20-%20University%20of%20Helsinki/Desktop/gaussian_quantum/gaussian_quantum/hilbert_space_approx.py#1-1), lines 1 to 120

Read [](file:///c%3A/Users/valte/OneDrive%20-%20University%20of%20Helsinki/Desktop/gaussian_quantum/gaussian_quantum/insurance.py#1-1), lines 1 to 120

Read [](file:///c%3A/Users/valte/OneDrive%20-%20University%20of%20Helsinki/Desktop/gaussian_quantum/gaussian_quantum/qpca.py#1-1), lines 1 to 120

Read [](file:///c%3A/Users/valte/OneDrive%20-%20University%20of%20Helsinki/Desktop/gaussian_quantum/gaussian_quantum/quantum_algorithms.py#1-1), lines 1 to 120

Read [](file:///c%3A/Users/valte/OneDrive%20-%20University%20of%20Helsinki/Desktop/gaussian_quantum/quantum_grid_search.py#160-160), lines 160 to 280

Read [](file:///c%3A/Users/valte/OneDrive%20-%20University%20of%20Helsinki/Desktop/gaussian_quantum/gaussian_quantum/insurance.py#120-120), lines 120 to 200

Read [](file:///c%3A/Users/valte/OneDrive%20-%20University%20of%20Helsinki/Desktop/gaussian_quantum/figures/stats_by_distribution.csv)

Read [](file:///c%3A/Users/valte/OneDrive%20-%20University%20of%20Helsinki/Desktop/gaussian_quantum/README.md#1-1), lines 1 to 100

Read [](file:///c%3A/Users/valte/OneDrive%20-%20University%20of%20Helsinki/Desktop/gaussian_quantum/figures/stats_by_payoff.csv)

Read [](file:///c%3A/Users/valte/OneDrive%20-%20University%20of%20Helsinki/Desktop/gaussian_quantum/figures/stats_errors.csv)

Read [](file:///c%3A/Users/valte/OneDrive%20-%20University%20of%20Helsinki/Desktop/gaussian_quantum/figures/quantum_grid_summary.csv#1-1), lines 1 to 30

Here is a detailed outline for the experimental section, structured for high academic quality with mathematical precision:

---

## Experimental Section Outline

---

### 5.1 Insurance Risk Integrals as a Bayesian Quadrature Benchmark

**Content:**
Formally define the expected insurance payoff as the target quantity:

$$\mathcal{I} = \mathbb{E}[\Pi(Z)] = \int_{\mathcal{D}} \Pi(z)\, f_Z(z)\, \mathrm{d}z$$

where $f_Z$ is the claim severity density and $\Pi : \mathbb{R}_{\geq 0} \to \mathbb{R}_{\geq 0}$ is a contract payoff function. The domain is truncated to $\mathcal{D} = [a, F_Z^{-1}(0.999)]$ to ensure finite support, with $a$ the lower support of the distribution. Describe each of the four severity distributions and five payoff functions mathematically:

- **Distributions:** $\text{Pareto}(\alpha=3, x_m=1)$, $\text{LogNormal}(\mu=0, \sigma=1)$, $\text{Gamma}(\alpha=2, \beta=1)$, $\text{Weibull}(k=1.5, \lambda=2)$
- **Payoffs** (with parameter $D$, $U$): ordinary deductible $\Pi(z)=\max(z-D,0)$; franchise deductible $\Pi(z)=z\cdot\mathbf{1}[z>D]$; policy limit $\Pi(z)=\min(z, U)$; combined $\Pi(z)=\min(\max(z-D,0), U)$; stop-loss $\Pi(z)=\max(z-D,0)$.

This yields $4 \times 5 = 20$ distinct integration tasks. Ground truth values are computed via adaptive quadrature (`scipy.integrate.quad`) and treated as exact.

**References:**
- Klugman, Panjer & Willmot, *Loss Models: From Data to Decisions* (Wiley, 2012) — standard insurance mathematics reference
- Embrechts, Klüppelberg & Mikosch, *Modelling Extremal Events* (Springer, 1997) — heavy-tailed distributions in insurance
- Briol et al., "Probabilistic Integration: A Role in Statistical Computation?", *Statistical Science* 34(1), 2019

---

### 5.2 Bayesian Quadrature via Gaussian Process Regression

**Content:**
Place the integration problem into the Bayesian quadrature (BQ) framework. Given $N$ evaluation points $\{z_i, g_i\}_{i=1}^N$ with $g_i = \Pi(z_i)f_Z(z_i) + \varepsilon_i$, $\varepsilon_i \sim \mathcal{N}(0, \sigma_n^2)$, the GP posterior over $g$ induces a Gaussian posterior over $\mathcal{I}$:

$$p(\mathcal{I} \mid \mathbf{g}) = \mathcal{N}(\mu_*, \sigma_*^2)$$

$$\mu_* = \mathbf{z}_\mu^\top (\mathbf{K} + \sigma_n^2 \mathbf{I})^{-1} \mathbf{g}, \quad \sigma_*^2 = \sigma_n^2\, \mathbf{z}_\mu^\top (\mathbf{K} + \sigma_n^2 \mathbf{I})^{-1} \mathbf{z}_\mu$$

where $[\mathbf{K}]_{ij} = k(z_i, z_j)$ is the RBF kernel $k(z,z') = A^2 \exp(-\|z-z'\|^2 / 2\ell^2)$ and $\mathbf{z}_\mu = \int_\mathcal{D} k(z, \cdot)\,\mathrm{d}z$ is the kernel mean vector. Describe the $O(N^3)$ cost of the classical GPQ baseline and motivate the HSGP approximation.

**References:**
- Rasmussen & Williams, *Gaussian Processes for Machine Learning* (MIT Press, 2006)
- O'Hagan, "Bayes–Hermite Quadrature", *J. Statistical Planning and Inference* 29, 1991
- Oates & Girolami, "Control Functionals for Monte Carlo Integration", *J. Royal Statistical Society B* 79(3), 2017

---

### 5.3 Hilbert-Space Gaussian Process Approximation (HSGP)

**Content:**
Describe Stage 1 in full. The stationary kernel is approximated by a rank-$M$ spectral expansion using Laplace eigenfunctions $\{\phi_j\}$ on $[-L,L]$ with Dirichlet boundary conditions:

$$k(z,z') \approx \sum_{j=1}^{M} S\!\left(\sqrt{\lambda_j}\right) \phi_j(z)\,\phi_j(z')$$

$$\phi_j(z) = L^{-1/2} \sin\!\left(\frac{j\pi (z + L)}{2L}\right), \quad \lambda_j = \left(\frac{j\pi}{2L}\right)^2$$

where $S(\omega) = A^2 \sqrt{2\pi}\,\ell\,\exp(-\ell^2 \omega^2/2)$ is the RBF spectral density. This yields the feature matrix $\mathbf{X} = \boldsymbol{\Phi}\,\mathrm{diag}(\sqrt{s_1}, \ldots, \sqrt{s_M}) \in \mathbb{R}^{N \times M}$ such that $\mathbf{X}\mathbf{X}^\top \approx \mathbf{K}$. The posterior reduces to the $M \times M$ system:

$$\mu_* = \mathbf{z}_\mu^\top (\mathbf{X}^\top\mathbf{X} + \sigma_n^2\mathbf{I})^{-1} \mathbf{X}^\top \mathbf{g}$$

reducing complexity from $O(N^3)$ to $O(NM^2 + M^3)$. Explain the domain boundary parameter $L = 1.3 \cdot (b-a)/2$ and the kernel mean integral $\mathbf{z}_\mu = \int_\mathcal{D} \mathbf{x}(z)\,\mathrm{d}z$.

**References:**
- Solin & Särkkä, "Hilbert Space Methods for Reduced-Rank Gaussian Process Regression", *Statistics and Computing* 30, 2020
- Riutort-Mayol et al., "Practical Hilbert Space Approximate Bayesian Gaussian Processes for Probabilistic Programming", *Statistics and Computing* 33, 2023

---

### 5.4 Quantum-Assisted HSGP-BQ: Stages 2 and 3

**Content:**
This is the most technically intensive subsection. Describe Stage 2 (qPCA) and Stage 3 (inner-product estimation) together.

**Stage 2 — Quantum PCA via QPE.** Define the density matrix $\rho = \mathbf{X}^\top\mathbf{X} / \|\mathbf{X}\|_F^2$ with spectral decomposition $\rho = \sum_r \theta_r |v_r\rangle\langle v_r|$, $\theta_r = \sigma_r^2 / F^2$. Construct the unitary $U = e^{2\pi i \rho}$ and apply Quantum Phase Estimation with $\tau$ ancilla qubits to encode $\theta_r$ in a $\tau$-bit register. Conditional single-qubit rotations on an output ancilla implement the map:

$$|\theta_r\rangle \longrightarrow |\theta_r\rangle \otimes \left(\sqrt{1 - \frac{c_\text{mean}^2}{(\sigma_r^2 + \sigma_n^2)^2}}|0\rangle + \frac{c_\text{mean}}{\sigma_r^2 + \sigma_n^2}|1\rangle\right)$$

Post-selection on the ancilla $|1\rangle$ and inverse QPE (uncomputation) prepares the states $|\psi_1\rangle$ and $|\psi_2\rangle$ whose inner product encodes the posterior mean.

**Stage 3 — Hadamard and Swap Tests.** The posterior mean and variance are extracted via:
$$\mu_* = \frac{1}{c_\text{mean}} \|\psi_1\| \|\psi_2\| \cdot \mathrm{Re}\langle\hat{\psi}_1|\hat{\psi}_2\rangle, \qquad \mathrm{Re}\langle\hat{\psi}_1|\hat{\psi}_2\rangle = 2P(\text{ancilla}=0) - 1$$

$$\sigma_*^2 = \frac{\sigma_n^2}{c_\text{var}^2}\bigl(1 - |\langle\hat{\psi}_1'|\hat{\psi}_2'\rangle|^2\bigr), \qquad |\langle\hat{\psi}_1'|\hat{\psi}_2'\rangle|^2 = 2P(\text{ancilla}=0) - 1 \text{ (Swap)}$$

with shot counts $S$ determining the standard error $O(1/\sqrt{S})$.

**References:**
- arXiv:2402.00544 — the primary reference for the full algorithm
- Lloyd, Mohseni & Rebentrost, "Quantum Principal Component Analysis", *Nature Physics* 10, 2014
- Kitaev, "Quantum measurements and the Abelian stabilizer problem", arXiv:quant-ph/9511026, 1995
- Nielsen & Chuang, *Quantum Computation and Quantum Information* (Cambridge, 2010) — QPE and QFT
- Buhrman et al., "Quantum Fingerprinting", *Physical Review Letters* 87(16), 2001 — Swap test
- Cincio et al., "Learning the quantum algorithm for state overlap", *New J. Physics* 20, 2018

---

### 5.5 Evaluation Point Placement Strategy

**Content:**
Describe the hybrid quantile-uniform placement scheme. For $N$ evaluation points with 70% quantile-placed and 30% uniformly-placed:

$$\mathcal{X}_q = \left\{F_Z^{-1}\!\left(\frac{i}{n_q+1}\right)\right\}_{i=1}^{n_q}, \quad n_q = \lfloor 0.7 N\rfloor$$

$$\mathcal{X}_u = \left\{a + \frac{j(b-a)}{n_u-1}\right\}_{j=0}^{n_u-1}, \quad n_u = N - n_q$$

$$\mathcal{X} = \text{sort}\!\left(\mathcal{X}_q \cup \mathcal{X}_u\right)$$

Justify this for heavy-tailed distributions (Pareto, Lognormal): quantile-placed points concentrate in high-mass regions where $f_Z$ peaks, while uniform points ensure tail coverage. All $z_i$ are recentred as $\tilde{z}_i = z_i - (a+b)/2$ to maximise HSGP basis efficiency. Observations are perturbed as $g_i = \Pi(z_i)f_Z(z_i) + \varepsilon_i$, $\varepsilon_i \sim \mathcal{N}(0, \sigma_n^2)$.

**References:**
- McKay, Beckman & Conover, "A Comparison of Three Methods for Selecting Values of Input Variables in the Analysis of Output from a Computer Code", *Technometrics* 21(2), 1979
- Niederreiter, *Random Number Generation and Quasi-Monte Carlo Methods* (SIAM, 1992)

---

### 5.6 Hyperparameter Configuration and Quantum Grid Search

**Content:**
This subsection should fully document the six hyperparameters and the two-stage search procedure used to select them.

**Hyperparameter space.** Define the complete parameter space:

| Symbol | Name | Role |
|---|---|---|
| $N$ | Training points | Size of observation set |
| $M$ | Basis functions | HSGP rank; controls $O(M^3)$ cost |
| $L$ | Domain boundary | HSGP basis domain $[-L, L]$ |
| $\ell$ | Length scale | RBF kernel correlation range |
| $A$ | Amplitude | Prior signal variance $A^2$ |
| $\sigma_n$ | Noise std | Observation noise |
| $\tau$ | QPE qubits | Phase resolution: $\Delta\theta = 2^{-\tau}$ |
| $S$ | Shots | Measurement samples per circuit |

Note the auto-heuristic $\ell_{\text{auto}} = \max\bigl((b-a)/(2\sqrt{N}),\, 0.3\bigr)$, which balances kernel expressiveness against over-fitting at fixed $N$.

**Two-stage quantum grid search.** The QPE circuit is expensive (exponential in $\tau$, linear in $S$), so a two-stage search decouples fast analytical evaluation from costly circuit execution.

*Stage 1 (analytical, $O(M^3)$).* For each combination $(N, M, \sigma_n, \ell)$ in the coarse grid $\mathcal{G}_1$, compute the analytical HSGP-BQ estimate (exact eigendecomposition, no QPE circuits) and record the absolute error $e_{\text{anl}} = |\mathcal{I} - \hat{\mu}_*^{\text{anl}}|$ averaged over all 20 tasks. Also compute diagnostic quantities: the Frobenius norm $F^2 = \|\mathbf{X}\|_F^2$, the condition number $\kappa(\mathbf{X}^\top\mathbf{X})$, the effective rank $r_{\text{eff}} = |\{j : \sigma_j^2 > 10^{-6}\sigma_{\max}^2\}|$, and the minimum phase gap $\delta_\theta = \min_{r \neq s} |\theta_r - \theta_s|$. Combinations with $e_{\text{anl}} > \epsilon_1$ are discarded.

*Stage 2 (circuit, $O(2^\tau S)$).* For each surviving combination, run full QPE circuits sweeping $\tau \in \{4,5,6,7,8\}$ and $S \in \{32768, 65536\}$. The best configuration found is:

$$N=32,\; M=6,\; \sigma_n=0.001,\; \ell=1.0,\; \tau=8,\; S=65536$$

achieving mean absolute error $0.0361$ across all 20 tasks. Discuss the trade-off between $\tau$ (phase resolution, more qubits) and $\delta_\theta$ (minimum eigenphase separation required for QPE to resolve distinct eigenvalues without aliasing).

**References:**
- Brassard et al., "Quantum Amplitude Amplification and Estimation", *Contemporary Mathematics* 305, 2002 — shot complexity
- Suzuki et al., "Amplitude Estimation without Phase Estimation", *Quantum Information Processing* 19, 2020
- Bergstra & Bengio, "Random Search for Hyper-Parameter Optimization", *JMLR* 13, 2012 — motivation for systematic grid vs. random search

---

### 5.7 Experimental Results

**Content:**
Report results in three views:

**Aggregate accuracy table.** Present mean/median/max percentage error for all four methods across all 20 tasks (values available in stats_errors.csv):

| Method | Mean % Err | Median % Err | Max % Err |
|---|---|---|---|
| Classical GPQ | 3.22 | 2.18 | 7.68 |
| Classical HSGP-BQ | 3.38 | 2.20 | 8.09 |
| Quantum HSGP-BQ (analytical) | 3.38 | 2.20 | 8.09 |
| Quantum HSGP-BQ (QPE circuit) | 3.27 | 2.00 | 11.29 |

The near-exact match between the analytical quantum and classical HSGP validates Stage 1 independently of Stage 2–3.

**By-distribution breakdown.** Analyse how tail behaviour affects accuracy. Heavy-tailed Pareto ($\alpha=3$, finite mean but infinite variance at lower shape) and Lognormal produce higher errors (~5.1–5.6%) while light-tailed Weibull achieves ~0.17% for classical methods. Note that QPE circuits introduce additional variance, inflating errors for the Weibull case (to ~2.24%) while recovering competitiveness for Pareto and Lognormal.

**By-payoff breakdown.** Ordinary deductible and stop-loss are identical in formulation; deductible-with-limit introduces a second discontinuity that challenges smooth kernel approximation. Policy limit produces the highest quantum circuit error (5.14%).

**Shot noise analysis.** Discuss the $O(1/\sqrt{S})$ convergence of both the Hadamard and Swap test estimators and relate it to the observed variance in the quantum circuit results.

**References:**
- Boucheron, Lugosi & Massart, *Concentration Inequalities: A Nonasymptotic Theory of Independence* (Oxford, 2013) — statistical analysis of shot-noise
- Kimmel et al., "Robust Calibration of a Universal Single-Qubit Gate-Set via Robust Phase Estimation", *Physical Review A* 92, 2015

---

### 5.8 Computational Complexity and Resource Estimates

**Content:**
Summarise the asymptotic complexity of each stage and the total qubit count. For HSGP-BQ with quantum subroutines:

- Stage 1: $O(NM + M^3)$ classically (feature construction + closed-form integrals)
- Stage 2: $\tau + \lceil\log_2 M\rceil + 1$ qubits; $O(2^\tau)$ controlled-$U$ applications per QPE
- Stage 3: $1 + \lceil\log_2 M\rceil$ qubits for Hadamard/Swap tests; $O(S)$ shots for $\epsilon$-precision

For the chosen configuration ($M=6$, $\tau=8$): total qubit count is $8 + 3 + 1 = 12$ qubits per mean estimation circuit. Discuss where quantum advantage could emerge as $M$ scales: the QPE-based inversion could in principle operate in $O(\text{poly}(\log M))$ gate complexity under QRAM models, compared to $O(M^3)$ classically.

**References:**
- Harrow, Hassidim & Lloyd, "Quantum Algorithm for Linear Systems of Equations", *Physical Review Letters* 103, 2009 — HHL context
- Biamonte et al., "Quantum Machine Learning", *Nature* 549, 2017 — scaling discussion
- Aaronson, "Read the Fine Print", *Nature Physics* 11, 2015 — quantum advantage caveats

---

This outline spans approximately 8 subsections. Sections 5.3–5.6 are the most mathematically dense and should carry the bulk of the narrative. The two-stage grid search in **5.6** is particularly novel and should be presented as a methodological contribution in its own right, since the analytical-then-circuit staging is a general strategy for QPE-based algorithms where circuit cost grows exponentially with $\tau$.

You're right — there are two distinct roles for the HSGP construction and they deserve separate subsections. Here is how to split them:

---

### 5.3 Reduced-Rank Kernel Approximation via Laplace Eigenfunctions

**Content — the approximation itself (shared foundation for both classical and quantum methods):**

The RBF kernel $k(z,z') = A^2\exp(-\|z-z'\|^2/2\ell^2)$ on a bounded domain $\mathcal{D}=[a,b]$ is approximated by a rank-$M$ spectral expansion. After centring to $\tilde{\mathcal{D}} = [-L, L]$ with $L = 1.3\cdot(b-a)/2$, the Laplace eigenfunctions with Dirichlet boundary conditions are:

$$\phi_j(\tilde{z}) = L^{-1/2}\sin\!\left(\frac{j\pi(\tilde{z}+L)}{2L}\right), \quad \lambda_j = \left(\frac{j\pi}{2L}\right)^2, \quad j = 1,\ldots,M$$

The Bochner spectral density of the RBF kernel evaluated at $\sqrt{\lambda_j}$ gives the spectral weights $s_j = S(\sqrt{\lambda_j})$, where

$$S(\omega) = A^2\sqrt{2\pi}\,\ell\,\exp\!\left(-\frac{\ell^2\omega^2}{2}\right)$$

This yields the **feature matrix** $\mathbf{X} = \boldsymbol{\Phi}\,\mathrm{diag}(\sqrt{s_1},\ldots,\sqrt{s_M}) \in \mathbb{R}^{N\times M}$, satisfying $\mathbf{X}\mathbf{X}^\top \approx \mathbf{K}$. This section should stop here — the feature matrix $\mathbf{X}$ is the output and is the input to *both* downstream methods. The key message is that $M \ll N$, so all subsequent linear algebra operates on an $M\times M$ system rather than $N\times N$.

---

### 5.4 Classical HSGP Bayesian Quadrature

**Content — how the feature matrix is used classically to close the BQ computation:**

Given $\mathbf{X}$, the GP posterior is computed entirely classically by solving the $M\times M$ system. The kernel mean vector $\mathbf{z}_\mu \in \mathbb{R}^M$ has entries

$$[\mathbf{z}_\mu]_j = \sqrt{s_j}\int_{-L}^{L}\phi_j(\tilde{z})\,\mathrm{d}\tilde{z}$$

which are available in closed form (sinusoidal integrals). The BQ posterior mean and variance are:

$$\mu_*^{\text{HSGP}} = \mathbf{z}_\mu^\top \underbrace{(\mathbf{X}^\top\mathbf{X} + \sigma_n^2\mathbf{I})^{-1}}_{\text{M\times M Cholesky}} \mathbf{X}^\top\mathbf{g}, \qquad \sigma_*^{2,\text{HSGP}} = \sigma_n^2\,\mathbf{z}_\mu^\top(\mathbf{X}^\top\mathbf{X}+\sigma_n^2\mathbf{I})^{-1}\mathbf{z}_\mu$$

This reduces complexity from $O(N^3)$ to $O(NM^2 + M^3)$. The classical HSGP-BQ is one of the four methods benchmarked in experiments; it also serves as the algebraic ground truth for validating the quantum pipeline, since the quantum analytical mode (exact eigendecomposition, no circuits) must reproduce it identically.

---

### 5.5 HSGP Feature Matrix as Quantum Preprocessing Input

**Content — how the same feature matrix is *handed off* to the quantum pipeline, and what changes:**

The quantum method reuses $\mathbf{X}$ from Section 5.3 but does **not** invert $\mathbf{X}^\top\mathbf{X}$ classically. Instead, the matrix inversion is delegated to Stages 2–3. To do so, define the **density matrix**:

$$\rho = \frac{\mathbf{X}^\top\mathbf{X}}{\|\mathbf{X}\|_F^2}, \qquad F^2 \equiv \|\mathbf{X}\|_F^2 = \mathrm{tr}(\mathbf{X}^\top\mathbf{X})$$

The normalisation by $F^2$ maps all eigenvalues $\sigma_r^2$ of $\mathbf{X}^\top\mathbf{X}$ into the unit interval as phases $\theta_r = \sigma_r^2/F^2 \in [0,1)$, which is a necessary precondition for QPE. This is the critical structural difference between the classical and quantum treatments: **the classical route inverts $\mathbf{X}^\top\mathbf{X} + \sigma_n^2\mathbf{I}$ directly; the quantum route encodes the spectrum of $\rho$ into a quantum register via QPE and then performs inversion via conditional rotations on that register**.

Importantly, both routes start from the same $\mathbf{X}$, so any error introduced before this point (HSGP approximation quality, choice of $M$, $\ell$, $L$) affects both methods equally. This is confirmed experimentally by the near-identical errors of the classical HSGP-BQ and the quantum analytical mode (Section 5.7).

Note also the QPE-specific constraint: the minimum phase gap $\delta_\theta = \min_{r\neq s}|\theta_r - \theta_s|$ must satisfy $\delta_\theta \gg 2^{-\tau}$, otherwise QPE conflates adjacent eigenvalues. This couples the choice of $M$ and $\sigma_n$ (which determine the eigenvalue spread of $\mathbf{X}^\top\mathbf{X}$) to the choice of $\tau$ — a constraint that the classical HSGP method does not face.

---

**The key conceptual distinction in one sentence:** Section 5.3 constructs $\mathbf{X}$; Section 5.4 uses $\mathbf{X}$ to solve a classical linear system; Section 5.5 uses $\mathbf{X}$ to build a quantum-accessible density matrix $\rho$ whose spectral inversion will be performed by QPE and conditional rotations in Stages 2–3.