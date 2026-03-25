"""Gaussian-quantum: classical and quantum GP regression algorithms.

Implements the full three-stage algorithm from arXiv:2402.00544:
    Stage 1 — Hilbert-space kernel approximation (hilbert_space_approx)
    Stage 2 — Quantum PCA with QPE and conditional rotations (qpca)
    Stage 3 — Hadamard / Swap tests for inner-product estimation (quantum_algorithms)
"""
