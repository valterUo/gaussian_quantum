"""Study: effect of QPE register size (tau) on quantum HSGP-BQ accuracy.

Fixed setup: Pareto(shape=3, scale=1), ordinary_deductible payoff, N=32.
Varies n_eigenvalue_qubits (tau) over TAU_VALUES and records the quantum
posterior mean and variance for each value.

Usage
-----
    python study_tau.py
    python study_tau.py --shots 131072 --out figures/tau_study.pdf
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

TAU_VALUES = [4, 5, 6, 7, 8, 9, 10, 11, 12]

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


def run_tau_study(N=16, M=16, shots=65536, seed=679, noise_std=0.01,
                  quantum_M=16, quantum_length_scale=1.0):
    records = []
    for tau in TAU_VALUES:
        print(f"  tau={tau:2d} ...", end=" ", flush=True)
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
            quantum_M=quantum_M,
            quantum_noise_std=noise_std,
            quantum_length_scale=quantum_length_scale,
        )
        mse_q = (res["quantum_mean"] - res["exact"]) ** 2 + res["quantum_var"]
        print(f"mean={res['quantum_mean']:.4f}  "
              f"std={np.sqrt(res['quantum_var']):.4f}  "
              f"MSE={mse_q:.6f}")
        records.append(res)
    return records


def plot_tau_study(records, out_path="figures/tau_study.pdf"):
    plt.rcParams.update(PLT_RC)

    tau_vals = TAU_VALUES[:len(records)]
    exact    = records[0]["exact"]

    q_means    = np.array([r["quantum_mean"] for r in records])
    q_stds     = np.array([np.sqrt(max(r["quantum_var"], 0.0)) for r in records])
    gpq_mean   = float(np.mean([r["gpq_mean"]  for r in records]))
    hsgp_mean  = float(np.mean([r["hsgp_mean"] for r in records]))

    fig, ax = plt.subplots(figsize=(9, 4))

    ax.axhline(exact,     color=COLOR_EXACT, lw=1.5, ls="-",  label="Exact")
    ax.axhline(gpq_mean,  color=COLOR_GPQ,  lw=1.2, ls=":",  label="GPQ",      alpha=0.7)
    ax.axhline(hsgp_mean, color=COLOR_HSGP, lw=1.2, ls="--", label="HSGP-BQ", alpha=0.7)

    ax.fill_between(tau_vals,
                    q_means - 2 * q_stds,
                    q_means + 2 * q_stds,
                    color=COLOR_Q, alpha=0.20,
                    label=r"Quantum $\pm 2\sigma$")
    ax.plot(tau_vals, q_means, color=COLOR_Q, lw=2,
            marker="o", markersize=5, label="Quantum mean")

    ax.set_xlabel(r"QPE register size $\tau$ (qubits)", fontsize=16)
    ax.set_ylabel("Integral estimate", fontsize=16)
    ax.set_xticks(tau_vals)
    ax.tick_params(direction="in", labelsize=14)
    ax.legend(loc="upper right", fontsize=13)
    ax.margins(x=0.03)
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"\nPlot saved to {out_path}")
    return fig


def main():
    parser = argparse.ArgumentParser(description="Tau-register size study")
    parser.add_argument("--N",                    type=int,   default=16)
    parser.add_argument("--M",                    type=int,   default=16)
    parser.add_argument("--shots",                type=int,   default=65536)
    parser.add_argument("--seed",                 type=int,   default=679)
    parser.add_argument("--noise-std",            type=float, default=0.01)
    parser.add_argument("--quantum-M",            type=int,   default=16)
    parser.add_argument("--quantum-length-scale", type=float, default=1.0)
    parser.add_argument("--out",                  type=str,   default="figures/tau_study.pdf")
    parser.add_argument("--load_from_file", type=str, default=None,
                        help="Path to JSON file with precomputed results (skip experiments and plotting)")
    args = parser.parse_args()

    print(f"Tau study: {DIST_NAME} x {PAYOFF_NAME}")
    print(f"  N={args.N}, M={args.M}, shots={args.shots}, seed={args.seed}\n")

    if args.load_from_file is not None:
        import json
        with open(args.load_from_file, "r") as f:
            records = json.load(f)
    else:

        records = run_tau_study(
            N=args.N, M=args.M, shots=args.shots, seed=args.seed,
            noise_std=args.noise_std, quantum_M=args.quantum_M,
            quantum_length_scale=args.quantum_length_scale,
        )
        # Save the records to a JSON file for later analysis (optional)
        with open("figures/tau_study.json", "w") as f:
            import json
            json.dump(records, f, indent=2)

    plot_tau_study(records, out_path=args.out)


if __name__ == "__main__":
    main()