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
    N=32,
    M=20,
    L=None,
    noise_std=0.01,
    length_scale=None,
    amplitude=1.0,
    run_quantum=False,
    run_quantum_analytical=False,
    n_eigenvalue_qubits=8,
    shots=65536,
    seed=679,
    point_strategy="hybrid",
    quantum_N=32,
    quantum_M=6,
    quantum_noise_std=0.001,
    quantum_length_scale=1.0,
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

    # ── Centre the domain for efficient HSGP basis usage ──────────────────
    a, b = domain
    midpoint = (a + b) / 2.0
    centered_domain = (a - midpoint, b - midpoint)
    half_width = (b - a) / 2.0

    # Default L: eigenfunctions live on [-L, L] centred on the domain
    # midpoint, with a 30 % margin beyond the half-width.
    if L is None:
        L = half_width * 1.3
        L = max(L, 1.0)  # ensure L is not degenerate

    noise_var = noise_std ** 2

    # ── Evaluation points and noisy observations ──────────────────────────
    rng = np.random.default_rng(seed)
    if point_strategy == "hybrid":
        # 70 % quantile points (dense where the PDF peaks) merged with
        # 30 % uniform points (to cover the tail of the domain).
        n_q = int(0.7 * N)
        n_u = N - n_q
        probs_q = np.linspace(1.0 / (n_q + 1), n_q / (n_q + 1), n_q)
        X_quantile = np.clip(dist.ppf(probs_q), a, b)
        X_uniform = np.linspace(a, b, n_u)
        X_raw = np.sort(np.unique(np.concatenate([X_quantile, X_uniform])))
        # Ensure exactly N points
        if len(X_raw) > N:
            idx = np.round(np.linspace(0, len(X_raw) - 1, N)).astype(int)
            X_raw = X_raw[idx]
        elif len(X_raw) < N:
            X_fill = np.linspace(a, b, N - len(X_raw) + 2)[1:-1]
            X_raw = np.sort(np.unique(np.concatenate([X_raw, X_fill])))[:N]
    elif point_strategy == "quantile":
        # Place points at distribution quantiles so they concentrate
        # where the PDF has the most mass (critical for heavy-tailed dists).
        probs = np.linspace(1.0 / (N + 1), N / (N + 1), N)
        X_raw = np.clip(dist.ppf(probs), a, b)
    else:
        X_raw = np.linspace(a, b, N)

    y_eval = np.squeeze(integrand(X_raw))
    y_eval = y_eval + rng.normal(0.0, noise_std, size=y_eval.shape)

    # Centre evaluation points to match the centred domain
    X_eval = (X_raw - midpoint).reshape(-1, 1)

    # ── Auto kernel length scale ──────────────────────────────────────────
    if length_scale is None:
        length_scale = half_width / np.sqrt(N)
        length_scale = max(length_scale, 0.3)

    result = {
        "dist_name": dist_name,
        "payoff_name": payoff_name,
        "domain": domain,
        "centered_domain": centered_domain,
        "N": N,
        "M": M,
        "L": L,
        "length_scale": length_scale,
        "point_strategy": point_strategy,
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
        X_eval, y_eval, centered_domain, noise_var,
        length_scale=length_scale, amplitude=amplitude,
    )
    timings["gpq"] = time.perf_counter() - t0
    result["gpq_mean"] = gpq_mean
    result["gpq_var"] = gpq_var

    # ── 3. Classical HSGP-BQ ──────────────────────────────────────────────
    t0 = time.perf_counter()
    hsgp_mean, hsgp_var = hsgp_integral(
        X_eval, y_eval, centered_domain, M, L,
        noise_var=noise_var, length_scale=length_scale, amplitude=amplitude,
    )
    timings["hsgp"] = time.perf_counter() - t0
    result["hsgp_mean"] = hsgp_mean
    result["hsgp_var"] = hsgp_var

    # ── 4. Quantum HSGP-BQ  ────────────────────────────────────
    #
    # The quantum circuit method uses QPE circuits whose accuracy degrades
    # when the Frobenius norm F² = trace(XᵀX) is too large or the noise
    # variance too small.  We therefore give the circuit method its own
    # QPE-compatible configuration.
    #
    # The *analytical* quantum method uses exact eigendecomposition, so it
    # is algebraically identical to classical HSGP.  To demonstrate this,
    # it shares the same data, domain, and hyperparameters as the HSGP
    # method above.
    if run_quantum_analytical:
        from gaussian_quantum.quantum_algorithms import quantum_hsgp_integral

        t0 = time.perf_counter()
        qa_mean, qa_var = quantum_hsgp_integral(
            X_eval, y_eval, centered_domain, M, L, noise_var,
            length_scale=length_scale, amplitude=amplitude,
            analytical=True,
        )
        timings["quantum_analytical"] = time.perf_counter() - t0
        result["quantum_analytical_mean"] = qa_mean
        result["quantum_analytical_var"] = qa_var

    if run_quantum:
        from gaussian_quantum.quantum_algorithms import quantum_hsgp_integral

        # Quantum-specific evaluation points.  Use the same hybrid strategy
        # as the classical methods: 70 % quantile-placed points (dense where
        # the PDF peaks, critical for heavy-tailed distributions such as
        # Pareto) merged with 30 % uniform points to cover the tail.
        q_n_q = int(0.7 * quantum_N)
        q_n_u = quantum_N - q_n_q
        q_probs = np.linspace(1.0 / (q_n_q + 1), q_n_q / (q_n_q + 1), q_n_q)
        q_X_quantile = np.clip(dist.ppf(q_probs), a, b)
        q_X_uniform = np.linspace(a, b, q_n_u)
        q_X_raw = np.sort(np.unique(np.concatenate([q_X_quantile, q_X_uniform])))
        if len(q_X_raw) > quantum_N:
            idx = np.round(np.linspace(0, len(q_X_raw) - 1, quantum_N)).astype(int)
            q_X_raw = q_X_raw[idx]
        elif len(q_X_raw) < quantum_N:
            q_X_fill = np.linspace(a, b, quantum_N - len(q_X_raw) + 2)[1:-1]
            q_X_raw = np.sort(
                np.unique(np.concatenate([q_X_raw, q_X_fill]))
            )[:quantum_N]

        q_y_eval = np.squeeze(integrand(q_X_raw))
        q_y_eval = q_y_eval + np.random.default_rng(seed).normal(
            0.0, quantum_noise_std, size=q_y_eval.shape,
        )
        q_noise_var = quantum_noise_std ** 2

        # Centre the quantum domain identically to the classical pipeline.
        # Using the raw uncentered domain causes eigenfunctions to cover
        # [-q_L, q_L] with data only in [0, b] — roughly half the support
        # is empty, halving F² = ‖X‖_F² and collapsing c_mean, which
        # amplifies shot noise by up to 1/c_mean ≈ 300× at the old defaults.
        q_midpoint = (a + b) / 2.0
        q_half_width = (b - a) / 2.0
        q_L = max(q_half_width * 1.3, 1.0)
        q_X_eval = (q_X_raw - q_midpoint).reshape(-1, 1)
        q_centered_domain = (a - q_midpoint, b - q_midpoint)

        t0 = time.perf_counter()
        q_mean, q_var = quantum_hsgp_integral(
            q_X_eval, q_y_eval, q_centered_domain, quantum_M, q_L, q_noise_var,
            length_scale=quantum_length_scale, amplitude=amplitude,
            n_eigenvalue_qubits=n_eigenvalue_qubits, shots=shots, seed=seed,
            noise_var_variance=noise_var,
            M_variance=M,
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
    run_quantum=True,
    run_quantum_analytical=False,
    **kwargs,
):
    """Run experiments for every (distribution, payoff) combination.

    Args:
        dist_names: List of distribution names (default: all).
        payoff_names: List of payoff names (default: all).
        run_quantum: Whether to include the quantum circuit method.
        run_quantum_analytical: Whether to include the analytical quantum method.
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
                run_quantum_analytical=run_quantum_analytical,
                **kwargs,
            )
            err_gpq = abs(res["exact"] - res["gpq_mean"])
            err_hsgp = abs(res["exact"] - res["hsgp_mean"])
            msg = f"exact={res['exact']:.6f}  GPQ_err={err_gpq:.6f}  HSGP_err={err_hsgp:.6f}"
            if run_quantum_analytical and "quantum_analytical_mean" in res:
                err_qa = abs(res["exact"] - res["quantum_analytical_mean"])
                msg += f"  Q_anl_err={err_qa:.6f}"
            if run_quantum and "quantum_mean" in res:
                err_q = abs(res["exact"] - res["quantum_mean"])
                msg += f"  Q_err={err_q:.6f}"
            print(msg)
            results.append(res)
    return results


# ---------------------------------------------------------------------------
# Printing / Reporting
# ---------------------------------------------------------------------------

def print_summary_table(results, run_quantum=False, run_quantum_analytical=False):
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
    if run_quantum_analytical:
        header += f"  {'QA_err':>10s}"

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
        if run_quantum_analytical and "quantum_analytical_mean" in r:
            line += f"  {abs(r['exact'] - r['quantum_analytical_mean']):10.6f}"
        elif run_quantum_analytical:
            line += f"  {'N/A':>10s}"
        print(line)

    print("=" * len(header))

def process_results_for_statistics(results, out_dir="figures"):
    import json
    import os
    import pandas as pd

    os.makedirs(out_dir, exist_ok=True)

    # ── Raw results → JSON ────────────────────────────────────────────────
    # Convert non-serialisable values (tuples, numpy scalars) to plain Python
    def _serialisable(r):
        out = {}
        for k, v in r.items():
            if isinstance(v, tuple):
                out[k] = list(v)
            elif isinstance(v, dict):
                out[k] = {kk: float(vv) for kk, vv in v.items()}
            elif hasattr(v, "item"):          # numpy scalar
                out[k] = v.item()
            else:
                out[k] = v
        return out

    with open(os.path.join(out_dir, "results_raw.json"), "w") as f:
        json.dump([_serialisable(r) for r in results], f, indent=2)

    # ── Percentage error statistics per method ──────────────────────────
    stats = {}
    error_rows = []
    for method in ["gpq", "hsgp", "quantum_analytical", "quantum"]:
        method_key = f"{method}_mean"
        if method_key in results[0]:
            pct_errors = [
                abs(r["exact"] - r[method_key]) / abs(r["exact"]) * 100
                if r["exact"] != 0 else float("nan")
                for r in results
            ]
            pct_errors = np.array(pct_errors, dtype=float)
            stats[method] = {
                "mean_pct_err": float(np.nanmean(pct_errors)),
                "median_pct_err": float(np.nanmedian(pct_errors)),
                "max_pct_err": float(np.nanmax(pct_errors)),
                "min_pct_err": float(np.nanmin(pct_errors)),
            }
            error_rows.append({"method": method, **stats[method]})

    pd.DataFrame(error_rows).to_csv(
        os.path.join(out_dir, "stats_errors.csv"), index=False,
    )

    # ── Flat DataFrame (drop non-scalar columns) ──────────────────────────
    scalar_keys = [
        k for k in results[0]
        if not isinstance(results[0][k], (dict, tuple, list))
    ]
    df = pd.DataFrame([{k: r[k] for k in scalar_keys} for r in results])

    mean_cols = ["exact"] + [c for c in df.columns if c.endswith("_mean")]

    # Relative percentage errors compared to exact for each method
    error_cols = [c for c in df.columns if c.endswith("_mean")]
    err_df = df[["dist_name", "payoff_name"]].copy()
    for col in error_cols:
        err_df[col.replace("_mean", "_pct_err")] = (
            (df[col] - df["exact"]).abs() / df["exact"].abs().replace(0, np.nan) * 100
        )
    pct_err_cols = [c for c in err_df.columns if c.endswith("_pct_err")]

    # Mean percentage error grouped by distribution
    stats["mean_pct_err_by_distribution"] = err_df.groupby("dist_name")[pct_err_cols].mean()
    stats["mean_pct_err_by_distribution"].to_csv(
        os.path.join(out_dir, "stats_by_distribution.csv"),
    )

    # Mean percentage error grouped by payoff type
    stats["mean_pct_err_by_payoff"] = err_df.groupby("payoff_name")[pct_err_cols].mean()
    stats["mean_pct_err_by_payoff"].to_csv(
        os.path.join(out_dir, "stats_by_payoff.csv"),
    )

    return stats


# ---------------------------------------------------------------------------
# Plotting helpers (optional; only imported if matplotlib available)
# ---------------------------------------------------------------------------

def plot_integrand(integrand, domain, X_eval, y_eval, title=""):
    """Plot the integrand, training points, and shaded integration area."""
    import matplotlib.pyplot as plt

    z = np.linspace(domain[0], domain[1], 500)
    g = integrand(z)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(z, g, alpha=0.15, color="steelblue")
    ax.plot(z, g, "steelblue", lw=2, label=r"$\Pi(z)\,f_Z(z)$")
    ax.plot(X_eval.ravel(), y_eval, "rx", ms=7, label="Noisy observations")
    #ax.set_yscale("log")
    ax.set_xlabel("$z$", fontsize=16)
    ax.set_ylabel("Integrand", fontsize=16)
    #ax.set_title(title or "Integrand and training data")
    ax.tick_params(direction='in', labelsize=14)
    ax.legend(fontsize=14)
    fig.tight_layout()
    return fig


def plot_comparison_gaussians(
    result,
    run_quantum=False,
    run_quantum_analytical=False,
    normalize=True,
    use_log=False,
    clip_min=1e-300,
):
    """Overlay Gaussian PDFs N(mean, var) for each BQ method.

    Args:
        normalize: if True, scale each PDF by its peak (so peaks=1) to
            compare shapes regardless of absolute height.
        use_log: if True, use a log y-axis (clip values below `clip_min`).
        clip_min: minimum y-value to use when plotting on a log scale.
    """
    import matplotlib.pyplot as plt
    from scipy.stats import norm

    exact = result["exact"]
    color_GPQ = "k"
    color_HSGP = "#8f0606"
    color_Q = "#008080"
    color_QA = "#e07b00"

    methods = [
        ("GPQ", result["gpq_mean"], result["gpq_var"], color_GPQ, ":"),
        ("HSGP-BQ", result["hsgp_mean"], result["hsgp_var"], color_HSGP, "--"),
    ]
    if run_quantum_analytical and "quantum_analytical_mean" in result and False:
        methods.append(
            ("Q-Analytical", result["quantum_analytical_mean"],
             result["quantum_analytical_var"], color_QA, "-."),
        )
    if run_quantum and "quantum_mean" in result:
        methods.append(
            ("Quantum", result["quantum_mean"], result["quantum_var"], color_Q, "-"),
        )

    # Determine plotting range
    all_means = [m[1] for m in methods]
    all_stds = [np.sqrt(max(m[2], 1e-12)) for m in methods]
    lo = min(m - 4 * s for m, s in zip(all_means, all_stds))
    hi = max(m + 4 * s for m, s in zip(all_means, all_stds))
    x = np.linspace(lo, hi, 500)

    fig = plt.figure(figsize=(10, 4))
    plt.axvline(x=exact, color='k', linestyle='-', linewidth=1.5, label="Exact")
    for label, mu, var, color, ls in methods:
        std = np.sqrt(max(var, 1e-12))
        pdf = norm.pdf(x, mu, std)
        peak = norm.pdf(mu, mu, std)
        if normalize:
            pdf = pdf / peak
            peak_plot = 1.0
            label_plot = f"{label} (norm)"
        else:
            peak_plot = peak
            label_plot = label

        # Avoid zeros / negative infinities on log scale
        if use_log:
            pdf = np.clip(pdf, clip_min, None)

        plt.fill_between(x, pdf, alpha=0.15, color=color)
        plt.plot(x, pdf, color=color, ls=ls, lw=2, label=label_plot)
        vbottom = clip_min if use_log else 0
        plt.vlines(mu, vbottom, peak_plot, color=color, linewidth=1.0, linestyle='-', alpha=0.6)
    plt.xlabel("Integral value", fontsize=16)
    plt.ylabel("Probability density", fontsize=16)
    if use_log:
        plt.yscale("log")
    else:
        plt.yscale("linear")
    #plt.title(f"{result['dist_name']} × {result['payoff_name']}")
    plt.tick_params(direction='in', labelsize=14)
    plt.legend(loc="upper right", fontsize=14)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Insurance Bayesian quadrature experiments",
    )
    parser.add_argument("--quantum", action="store_true",
                        help="Include quantum HSGP-BQ method (QPE circuits)")
    parser.add_argument("--quantum-analytical", action="store_true",
                        help="Include analytical quantum HSGP-BQ method (exact eigendecomposition)")
    parser.add_argument("--plot", action="store_true",
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
    parser.add_argument("--quantum-M", type=int, default=6,
                        help="Quantum HSGP basis functions (default: 6)")
    parser.add_argument("--n-eigenvalue-qubits", type=int, default=8,
                        help="QPE eigenvalue register qubits (default: 8)")
    parser.add_argument("--quantum-noise-std", type=float, default=0.001,
                        help="Quantum observation noise std (default: 0.001)")
    parser.add_argument("--quantum-length-scale", type=float, default=1.0,
                        help="Quantum kernel length scale (default: 1.0)")
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
        import os
        os.makedirs("figures", exist_ok=True)
        for r in results:
            integrand, domain, dist_obj = make_integrand(
                r["dist_name"],
                EXPERIMENT_DIST_PARAMS.get(r["dist_name"]),
                r["payoff_name"],
                PAYOFF_DEFAULTS.get(r["payoff_name"]),
            )
            rng = np.random.default_rng(args.seed)
            a, b = domain[0], domain[1]
            if args.point_strategy == "hybrid":
                n_q = int(0.7 * args.N)
                n_u = args.N - n_q
                probs_q = np.linspace(1.0/(n_q+1), n_q/(n_q+1), n_q)
                Xq = np.clip(dist_obj.ppf(probs_q), a, b)
                Xu = np.linspace(a, b, n_u)
                X_raw = np.sort(np.unique(np.concatenate([Xq, Xu])))
                if len(X_raw) > args.N:
                    idx = np.round(np.linspace(0, len(X_raw)-1, args.N)).astype(int)
                    X_raw = X_raw[idx]
                X_eval = X_raw.reshape(-1, 1)
            elif args.point_strategy == "quantile":
                probs = np.linspace(1.0/(args.N+1), args.N/(args.N+1), args.N)
                X_eval = np.clip(
                    dist_obj.ppf(probs), a, b,
                ).reshape(-1, 1)
            else:
                X_eval = np.linspace(a, b, args.N).reshape(-1, 1)
            y_eval = integrand(X_eval.ravel()) + rng.normal(
                0.0, args.noise_std, size=len(X_eval),
            )
            tag = f"{r['dist_name']}_{r['payoff_name']}"

            fig1 = plot_integrand(
                integrand, domain, X_eval, y_eval,
                title=f"{r['dist_name']} × {r['payoff_name']}",
            )
            fig1.savefig(f"figures/integrand_{tag}.pdf", bbox_inches="tight")

            fig2 = plot_comparison_gaussians(r, run_quantum=args.quantum,
                                             run_quantum_analytical=args.quantum_analytical)
            fig2.savefig(f"figures/comparison_{tag}.pdf", bbox_inches="tight")

            import matplotlib.pyplot as plt
            plt.close("all")

        print(f"\nPlots saved to figures/")


if __name__ == "__main__":
    main()
