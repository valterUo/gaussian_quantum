"""Insurance Bayesian quadrature experiments.

Computes E[Π(Z)] = ∫ Π(z) f_Z(z) dz for every combination of
claim severity distribution × insurance payoff function using four methods:

    1. Exact integral  (scipy.integrate.quad)
    2. Classical GPQ   (full RBF kernel Bayesian quadrature)
    3. Classical HSGP-BQ (Hilbert-space GP Bayesian quadrature)
    4. Quantum HSGP-BQ  (quantum-assisted HSGP Bayesian quadrature)

Usage
-----
    python insurance_experiments.py                 # classical only (fast)
    python insurance_experiments.py --quantum       # include quantum method
    python insurance_experiments.py --plot           # save integrand plots
"""

import argparse

from gaussian_quantum.insurance import (
    make_integrand,
    PAYOFF_DEFAULTS,
)
from plot import (
    print_summary_table, 
    process_results_for_statistics, 
    save_plots
)
from experiments import run_all_experiments, EXPERIMENT_DIST_PARAMS


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Insurance Bayesian quadrature experiments",
    )
    parser.add_argument("--quantum", action="store_true", default=True,
                        help="Include quantum HSGP-BQ method (QPE circuits)")
    parser.add_argument("--quantum-analytical", action="store_true", default=True,
                        help="Include analytical quantum HSGP-BQ method (exact eigendecomposition)")
    parser.add_argument("--plot", action="store_true", default=True,
                        help="Save comparison plots to figures/")
    parser.add_argument("--N", type=int, default=32,
                        help="Number of evaluation points (default: 32)")
    parser.add_argument("--M", type=int, default=32,
                        help="Number of HSGP basis functions (default: 32)")
    parser.add_argument("--shots", type=int, default=65536,
                        help="Quantum measurement shots (default: 65536 = 16 qubits)")
    parser.add_argument("--seed", type=int, default=679,
                        help="Random seed (default: 679)")
    parser.add_argument("--noise-std", type=float, default=0.01,
                        help="Observation noise std (default: 0.01)")
    parser.add_argument("--point-strategy", type=str, default="hybrid",
                        choices=["hybrid", "quantile", "uniform"],
                        help="Evaluation point placement (default: hybrid)")
    parser.add_argument("--quantum-N", type=int, default=32,
                        help="Quantum evaluation points (default: 32)")
    parser.add_argument("--quantum-M", type=int, default=32,
                        help="Quantum HSGP basis functions (default: 32)")
    parser.add_argument("--n-eigenvalue-qubits", type=int, default=12,
                        help="QPE eigenvalue register qubits (default: 12)")
    parser.add_argument("--quantum-noise-std", type=float, default=0.01,
                        help="Quantum observation noise std (default: 0.01)")
    parser.add_argument("--quantum-length-scale", type=float, default=1.0,
                        help="Quantum kernel length scale (default: 1.0)")
    parser.add_argument("--load_from_file", type=str, default=None,
                        help="Path to JSON file with precomputed results (skip experiments and plotting)")
    args = parser.parse_args()

    print("Running insurance BQ experiments")
    print(f"  N={args.N}, M={args.M}, quantum={args.quantum}, "
          f"shots={args.shots}, seed={args.seed}, "
          f"noise_std={args.noise_std}, points={args.point_strategy}")
    if args.quantum or args.quantum_analytical:
        print(f"  quantum: N_q={args.quantum_N}, M_q={args.quantum_M}, "
              f"noise_std_q={args.quantum_noise_std}, "
              f"ls_q={args.quantum_length_scale}, "
              f"tau={args.n_eigenvalue_qubits}")
    print()

    load_from_file = args.load_from_file
    if load_from_file is not None:
        import json
        with open(load_from_file, "r") as f:
            results = json.load(f)
        print(f"Loaded results from {load_from_file}")
    else:
        results = run_all_experiments(
            run_quantum=args.quantum,
            run_quantum_analytical=args.quantum_analytical,
            N=args.N, M=args.M, shots=args.shots, seed=args.seed,
            noise_std=args.noise_std, point_strategy=args.point_strategy,
            quantum_N=args.quantum_N, quantum_M=args.quantum_M,
            quantum_noise_std=args.quantum_noise_std,
            quantum_length_scale=args.quantum_length_scale,
            n_eigenvalue_qubits=args.n_eigenvalue_qubits,
        )

    print_summary_table(results, run_quantum=args.quantum,
                        run_quantum_analytical=args.quantum_analytical)
    process_results_for_statistics(results, out_dir="figures")

    if args.plot:
        save_plots(
            results,
            make_integrand_fn=make_integrand,
            experiment_dist_params=EXPERIMENT_DIST_PARAMS,
            payoff_defaults=PAYOFF_DEFAULTS,
            N=args.N,
            noise_std=args.noise_std,
            seed=args.seed,
            point_strategy=args.point_strategy,
            run_quantum=args.quantum,
            run_quantum_analytical=args.quantum_analytical,
        )


if __name__ == "__main__":
    main()
