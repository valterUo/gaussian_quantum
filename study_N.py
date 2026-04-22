"""Study: effect of number of evaluation points N on quantum HSGP-BQ accuracy.

Fixed setup: Pareto(shape=3, scale=1), ordinary_deductible payoff, tau=8.
Varies N over N_VALUES and records all four methods for each value.

Usage
-----
    python study_N.py
    python study_N.py --tau 10 --out figures/N_study.pdf
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

from experiments import run_experiment, EXPERIMENT_DIST_PARAMS
from gaussian_quantum.insurance import PAYOFF_DEFAULTS

DIST_NAME     = "pareto"
DIST_PARAMS   = EXPERIMENT_DIST_PARAMS["pareto"]
PAYOFF_NAME   = "ordinary_deductible"
PAYOFF_PARAMS = PAYOFF_DEFAULTS.get(PAYOFF_NAME)

N_VALUES = [2, 4, 6, 8, 12, 16, 24, 32]

COLOR_EXACT = "k"
COLOR_GPQ   = "k"
COLOR_HSGP  = "#8f0606"
COLOR_Q     = "#008080"

PLT_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 16,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "figure.dpi": 600,
}


def run_N_study(tau=8, M=32, shots=65536, seed=679, noise_std=0.01,
                quantum_length_scale=1.0):
    records = []
    for N in N_VALUES:
        print(f"  N={N:3d} ...", end=" ", flush=True)
        res = run_experiment(
            dist_name=DIST_NAME,
            dist_params=DIST_PARAMS,
            payoff_name=PAYOFF_NAME,
            payoff_params=PAYOFF_PARAMS,
            N=N, M=M, noise_std=noise_std, seed=seed,
            point_strategy="hybrid",
            run_quantum=True,
            run_quantum_analytical=False,
            shots=shots,
            n_eigenvalue_qubits=tau,
            quantum_N=N,
            quantum_M=M,
            quantum_noise_std=noise_std,
            quantum_length_scale=quantum_length_scale,
        )
        mse_q    = (res["quantum_mean"] - res["exact"]) ** 2 + res["quantum_var"]
        mse_gpq  = (res["gpq_mean"]    - res["exact"]) ** 2 + res["gpq_var"]
        mse_hsgp = (res["hsgp_mean"]   - res["exact"]) ** 2 + res["hsgp_var"]
        print(f"Q_MSE={mse_q:.5f}  GPQ_MSE={mse_gpq:.5f}  HSGP_MSE={mse_hsgp:.5f}")
        records.append(res)
    return records


def plot_N_study(records, out_path="figures/N_study.pdf"):
    plt.rcParams.update(PLT_RC)

    n_vals = N_VALUES[:len(records)]
    exact  = records[0]["exact"]

    q_means    = np.array([r["quantum_mean"] for r in records])
    q_stds     = np.array([np.sqrt(max(r["quantum_var"], 0.0)) for r in records])
    gpq_means  = np.array([r["gpq_mean"]  for r in records])
    gpq_stds   = np.array([np.sqrt(max(r["gpq_var"],  0.0)) for r in records])
    hsgp_means = np.array([r["hsgp_mean"] for r in records])
    hsgp_stds  = np.array([np.sqrt(max(r["hsgp_var"], 0.0)) for r in records])

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.axhline(exact, color=COLOR_EXACT, lw=1.5, ls="-", label="Exact")

    # GPQ
    ax.fill_between(n_vals,
                    gpq_means - 2 * gpq_stds,
                    gpq_means + 2 * gpq_stds,
                    color=COLOR_GPQ, alpha=0.10)
    ax.plot(n_vals, gpq_means, color=COLOR_GPQ, lw=1.5,
            ls=":", marker="s", markersize=4, label="GPQ")

    # HSGP-BQ
    ax.fill_between(n_vals,
                    hsgp_means - 2 * hsgp_stds,
                    hsgp_means + 2 * hsgp_stds,
                    color=COLOR_HSGP, alpha=0.12)
    ax.plot(n_vals, hsgp_means, color=COLOR_HSGP, lw=1.5,
            ls="--", marker="^", markersize=4, label="HSGP-BQ")

    # Quantum
    ax.fill_between(n_vals,
                    q_means - 2 * q_stds,
                    q_means + 2 * q_stds,
                    color=COLOR_Q, alpha=0.20,
                    label=r"Quantum $\pm 2\sigma$")
    ax.plot(n_vals, q_means, color=COLOR_Q, lw=2,
            marker="o", markersize=5, label="Quantum mean")

    ax.set_xlabel(r"Number of evaluation points $N$", fontsize=16)
    ax.set_ylabel("Integral estimate", fontsize=16)
    ax.set_xticks(n_vals)
    # Make y-axis to log scale to better visualize differences at small N
    ax.set_yscale("log")
    ax.tick_params(direction="in", labelsize=14)
    ax.legend(loc="upper right", fontsize=16)
    ax.margins(x=0.03)
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"\nPlot saved to {out_path}")
    return fig


def main():
    parser = argparse.ArgumentParser(description="N evaluation-points study")
    parser.add_argument("--tau",                  type=int,   default=8)
    parser.add_argument("--M",                    type=int,   default=16)
    parser.add_argument("--shots",                type=int,   default=65536)
    parser.add_argument("--seed",                 type=int,   default=679)
    parser.add_argument("--noise-std",            type=float, default=0.01)
    parser.add_argument("--quantum-length-scale", type=float, default=1.0)
    parser.add_argument("--out",                  type=str,   default="figures/N_study.pdf")
    parser.add_argument("--load_from_file", type=str, default=None,
                        help="Path to JSON file with precomputed results (skip experiments and plotting)")
    args = parser.parse_args()

    print(f"N study: {DIST_NAME} x {PAYOFF_NAME}")
    print(f"  tau={args.tau}, M={args.M}, shots={args.shots}, seed={args.seed}\n")

    if args.load_from_file is not None:
        import json
        with open(args.load_from_file, "r") as f:
            records = json.load(f)
        print(f"Loaded results from {args.load_from_file}")
    else:

        records = run_N_study(
            tau=args.tau, M=args.M, shots=args.shots, seed=args.seed,
            noise_std=args.noise_std,
            quantum_length_scale=args.quantum_length_scale,
        )
        with open("figures/N_study.json", "w") as f:
            import json
            json.dump(records, f, indent=2)

    plot_N_study(records, out_path=args.out)


if __name__ == "__main__":
    main()