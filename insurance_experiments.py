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
import time

import numpy as np

from gaussian_quantum.insurance import (
    make_integrand,
    exact_integral,
    DISTRIBUTIONS,
    PAYOFFS,
    PAYOFF_DEFAULTS,
)
from gaussian_quantum.classical import gpq_integral
from gaussian_quantum.hilbert_space_approx import hsgp_integral


# ---------------------------------------------------------------------------
# Single-experiment runner
# ---------------------------------------------------------------------------

def run_experiment(
    dist_name,
    dist_params=None,
    payoff_name="ordinary_deductible",
    payoff_params=None,
    N=16,
    M=10,
    L=None,
    noise_std=0.05,
    length_scale=1.0,
    amplitude=1.0,
    run_quantum=False,
    n_eigenvalue_qubits=6,
    shots=8192,
    seed=679,
):
    """Run a single BQ experiment for one (distribution, payoff) pair.

    Returns a dict with keys:
        exact, gpq_mean, gpq_var, hsgp_mean, hsgp_var,
        quantum_mean, quantum_var (if run_quantum),
        domain, dist_name, payoff_name, timings.
    """
    integrand, domain, dist = make_integrand(
        dist_name, dist_params, payoff_name, payoff_params,
    )

    # Default L: HSGP eigenfunctions live on [-L, L], so all data points
    # must satisfy |x| < L.  Pick L = max(|a|, |b|) + 20 % margin.
    if L is None:
        L = max(abs(domain[0]), abs(domain[1])) * 1.2
        L = max(L, 1.0)  # ensure L is not degenerate

    noise_var = noise_std ** 2

    # ── Evaluation points and noisy observations ──────────────────────────
    rng = np.random.default_rng(seed)
    X_eval = np.linspace(domain[0], domain[1], N).reshape(-1, 1)
    y_eval = np.squeeze(integrand(X_eval.ravel()))
    y_eval = y_eval + rng.normal(0.0, noise_std, size=y_eval.shape)

    result = {
        "dist_name": dist_name,
        "payoff_name": payoff_name,
        "domain": domain,
        "N": N,
        "M": M,
        "L": L,
    }
    timings = {}

    # ── 1. Exact integral ─────────────────────────────────────────────────
    t0 = time.perf_counter()
    exact_val, exact_err = exact_integral(integrand, domain)
    timings["exact"] = time.perf_counter() - t0
    result["exact"] = exact_val
    result["exact_err"] = exact_err

    # ── 2. Classical GPQ (full RBF kernel) ────────────────────────────────
    t0 = time.perf_counter()
    gpq_mean, gpq_var = gpq_integral(
        X_eval, y_eval, domain, noise_var,
        length_scale=length_scale, amplitude=amplitude,
    )
    timings["gpq"] = time.perf_counter() - t0
    result["gpq_mean"] = gpq_mean
    result["gpq_var"] = gpq_var

    # ── 3. Classical HSGP-BQ ──────────────────────────────────────────────
    t0 = time.perf_counter()
    hsgp_mean, hsgp_var = hsgp_integral(
        X_eval, y_eval, domain, M, L,
        noise_var=noise_var, length_scale=length_scale, amplitude=amplitude,
    )
    timings["hsgp"] = time.perf_counter() - t0
    result["hsgp_mean"] = hsgp_mean
    result["hsgp_var"] = hsgp_var

    # ── 4. Quantum HSGP-BQ (optional) ────────────────────────────────────
    if run_quantum:
        from gaussian_quantum.quantum_algorithms import quantum_hsgp_integral

        t0 = time.perf_counter()
        q_mean, q_var = quantum_hsgp_integral(
            X_eval, y_eval, domain, M, L, noise_var,
            length_scale=length_scale, amplitude=amplitude,
            n_eigenvalue_qubits=n_eigenvalue_qubits, shots=shots,
        )
        timings["quantum"] = time.perf_counter() - t0
        result["quantum_mean"] = q_mean
        result["quantum_var"] = q_var

    result["timings"] = timings
    return result


# ---------------------------------------------------------------------------
# All-experiments runner
# ---------------------------------------------------------------------------

# Distribution parameters to use in experiments
EXPERIMENT_DIST_PARAMS = {
    "pareto": {"shape": 3.0, "scale": 1.0},
    "lognormal": {"mu": 0.0, "sigma": 1.0},
    "gamma": {"shape": 2.0, "rate": 1.0},
    "weibull": {"shape": 1.5, "scale": 2.0},
    "poisson": {"lam": 5.0},
}


def run_all_experiments(
    dist_names=None,
    payoff_names=None,
    run_quantum=False,
    **kwargs,
):
    """Run experiments for every (distribution, payoff) combination.

    Args:
        dist_names: List of distribution names (default: all).
        payoff_names: List of payoff names (default: all).
        run_quantum: Whether to include the quantum method.
        **kwargs: Forwarded to :func:`run_experiment`.

    Returns:
        List of result dicts, one per combination.
    """
    if dist_names is None:
        dist_names = list(DISTRIBUTIONS.keys())
    if payoff_names is None:
        payoff_names = list(PAYOFFS.keys())

    results = []
    for dname in dist_names:
        for pname in payoff_names:
            print(f"  {dname:12s} × {pname:25s} ... ", end="", flush=True)
            res = run_experiment(
                dist_name=dname,
                dist_params=EXPERIMENT_DIST_PARAMS.get(dname),
                payoff_name=pname,
                payoff_params=PAYOFF_DEFAULTS.get(pname),
                run_quantum=run_quantum,
                **kwargs,
            )
            err_gpq = abs(res["exact"] - res["gpq_mean"])
            err_hsgp = abs(res["exact"] - res["hsgp_mean"])
            msg = f"exact={res['exact']:.6f}  GPQ_err={err_gpq:.6f}  HSGP_err={err_hsgp:.6f}"
            if run_quantum and "quantum_mean" in res:
                err_q = abs(res["exact"] - res["quantum_mean"])
                msg += f"  Q_err={err_q:.6f}"
            print(msg)
            results.append(res)
    return results


# ---------------------------------------------------------------------------
# Printing / Reporting
# ---------------------------------------------------------------------------

def print_summary_table(results, run_quantum=False):
    """Print a tabular summary of results."""
    header = (
        f"{'Distribution':>12s}  {'Payoff':>25s}  "
        f"{'Exact':>10s}  {'GPQ':>10s}  {'HSGP':>10s}"
    )
    if run_quantum:
        header += f"  {'Quantum':>10s}"
    header += f"  {'GPQ_err':>10s}  {'HSGP_err':>10s}"
    if run_quantum:
        header += f"  {'Q_err':>10s}"

    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))

    for r in results:
        line = (
            f"{r['dist_name']:>12s}  {r['payoff_name']:>25s}  "
            f"{r['exact']:10.6f}  {r['gpq_mean']:10.6f}  {r['hsgp_mean']:10.6f}"
        )
        if run_quantum and "quantum_mean" in r:
            line += f"  {r['quantum_mean']:10.6f}"
        elif run_quantum:
            line += f"  {'N/A':>10s}"
        line += (
            f"  {abs(r['exact'] - r['gpq_mean']):10.6f}"
            f"  {abs(r['exact'] - r['hsgp_mean']):10.6f}"
        )
        if run_quantum and "quantum_mean" in r:
            line += f"  {abs(r['exact'] - r['quantum_mean']):10.6f}"
        elif run_quantum:
            line += f"  {'N/A':>10s}"
        print(line)

    print("=" * len(header))


# ---------------------------------------------------------------------------
# Plotting helpers (optional; only imported if matplotlib available)
# ---------------------------------------------------------------------------

def plot_integrand(integrand, domain, X_eval, y_eval, title=""):
    """Plot the integrand, training points, and shaded integration area."""
    import matplotlib.pyplot as plt

    z = np.linspace(domain[0], domain[1], 500)
    g = integrand(z)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.fill_between(z, g, alpha=0.15, color="steelblue")
    ax.plot(z, g, "steelblue", lw=2, label=r"$\Pi(z)\,f_Z(z)$")
    ax.plot(X_eval.ravel(), y_eval, "rx", ms=7, label="Noisy observations")
    ax.set_xlabel("$z$")
    ax.set_ylabel("Integrand")
    ax.set_title(title or "Integrand and training data")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_comparison_gaussians(result, run_quantum=False):
    """Overlay Gaussian PDFs N(mean, var) for each BQ method."""
    import matplotlib.pyplot as plt
    from scipy.stats import norm

    exact = result["exact"]
    methods = [
        ("GPQ", result["gpq_mean"], result["gpq_var"], "black", "--"),
        ("HSGP-BQ", result["hsgp_mean"], result["hsgp_var"], "#8f0606", "-."),
    ]
    if run_quantum and "quantum_mean" in result:
        methods.append(
            ("Quantum", result["quantum_mean"], result["quantum_var"], "#008080", "-"),
        )

    # Determine plotting range
    all_means = [m[1] for m in methods]
    all_stds = [np.sqrt(max(m[2], 1e-12)) for m in methods]
    lo = min(m - 4 * s for m, s in zip(all_means, all_stds))
    hi = max(m + 4 * s for m, s in zip(all_means, all_stds))
    x = np.linspace(lo, hi, 500)

    fig, ax = plt.subplots(figsize=(8, 4))
    for label, mu, var, color, ls in methods:
        std = np.sqrt(max(var, 1e-12))
        ax.plot(x, norm.pdf(x, mu, std), color=color, ls=ls, lw=2, label=label)
        ax.axvline(mu, color=color, ls=":", alpha=0.5)
    ax.axvline(exact, color="green", lw=2, label="Exact")
    ax.set_xlabel("Integral value")
    ax.set_ylabel("Density")
    title = f"{result['dist_name']} × {result['payoff_name']}"
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Insurance Bayesian quadrature experiments",
    )
    parser.add_argument("--quantum", action="store_true",
                        help="Include quantum HSGP-BQ method")
    parser.add_argument("--plot", action="store_true",
                        help="Save comparison plots to figures/")
    parser.add_argument("--N", type=int, default=16,
                        help="Number of evaluation points (default: 16)")
    parser.add_argument("--M", type=int, default=10,
                        help="Number of HSGP basis functions (default: 10)")
    parser.add_argument("--shots", type=int, default=8192,
                        help="Quantum measurement shots (default: 8192)")
    parser.add_argument("--seed", type=int, default=679,
                        help="Random seed (default: 679)")
    args = parser.parse_args()

    print("Running insurance BQ experiments")
    print(f"  N={args.N}, M={args.M}, quantum={args.quantum}, "
          f"shots={args.shots}, seed={args.seed}\n")

    results = run_all_experiments(
        run_quantum=args.quantum,
        N=args.N, M=args.M, shots=args.shots, seed=args.seed,
    )

    print_summary_table(results, run_quantum=args.quantum)

    if args.plot:
        import os
        os.makedirs("figures", exist_ok=True)
        for r in results:
            integrand, domain, _ = make_integrand(
                r["dist_name"],
                EXPERIMENT_DIST_PARAMS.get(r["dist_name"]),
                r["payoff_name"],
                PAYOFF_DEFAULTS.get(r["payoff_name"]),
            )
            rng = np.random.default_rng(args.seed)
            X_eval = np.linspace(domain[0], domain[1], args.N).reshape(-1, 1)
            y_eval = integrand(X_eval.ravel()) + rng.normal(
                0.0, 0.05, size=args.N,
            )
            tag = f"{r['dist_name']}_{r['payoff_name']}"

            fig1 = plot_integrand(
                integrand, domain, X_eval, y_eval,
                title=f"{r['dist_name']} × {r['payoff_name']}",
            )
            fig1.savefig(f"figures/integrand_{tag}.pdf", bbox_inches="tight")

            fig2 = plot_comparison_gaussians(r, run_quantum=args.quantum)
            fig2.savefig(f"figures/comparison_{tag}.pdf", bbox_inches="tight")

            import matplotlib.pyplot as plt
            plt.close("all")

        print(f"\nPlots saved to figures/")


if __name__ == "__main__":
    main()
