# ---------------------------------------------------------------------------
# Plotting helpers (optional; only imported if matplotlib available)
# ---------------------------------------------------------------------------

import numpy as np


def plot_integrand(integrand, domain, X_eval, y_eval):
    """Plot the integrand, training points, and shaded integration area."""
    import matplotlib.pyplot as plt

    plt.rcParams.update({
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
    })

    z = np.linspace(domain[0], domain[1], 500)
    g = integrand(z)

    fig, ax = plt.subplots(figsize=(10, 5))
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
    normalize=False,
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

    # IEEE font settings - Times New Roman
    plt.rcParams.update({
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
    })

    exact = result["exact"]
    color_GPQ = "k"
    color_HSGP = "#8f0606"
    color_Q = "#008080"
    color_QA = "#e07b00"

    def _label(name, mean, var):
        mse = (mean - exact) ** 2 + var
        return f"{name} (MSE={mse:.4f})"

    methods = [
        (_label("GPQ", result["gpq_mean"], result["gpq_var"]),
         result["gpq_mean"], result["gpq_var"], color_GPQ, ":"),
        (_label("HSGP-BQ", result["hsgp_mean"], result["hsgp_var"]),
         result["hsgp_mean"], result["hsgp_var"], color_HSGP, "--"),
    ]
    if run_quantum_analytical and "quantum_analytical_mean" in result and False:
        methods.append(
            (_label("Q-Analytical", result["quantum_analytical_mean"],
                    result["quantum_analytical_var"]),
             result["quantum_analytical_mean"],
             result["quantum_analytical_var"], color_QA, "-."),
        )
    if run_quantum and "quantum_mean" in result:
        methods.append(
            (_label("Quantum", result["quantum_mean"], result["quantum_var"]),
             result["quantum_mean"], result["quantum_var"], color_Q, "-"),
        )

    # Determine plotting range
    all_means = [m[1] for m in methods]
    all_stds = [np.sqrt(max(m[2], 1e-12)) for m in methods]
    lo = min(m - 4 * s for m, s in zip(all_means, all_stds))
    hi = max(m + 4 * s for m, s in zip(all_means, all_stds))
    x = np.linspace(lo, hi, 500)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axvline(x=exact, color="k", linestyle="-", linewidth=1.5, label="Exact")

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


        if use_log:
            pdf = np.clip(pdf, clip_min, None)

        ax.fill_between(x, pdf, alpha=0.15, color=color)
        ax.plot(x, pdf, color=color, ls=ls, lw=2, label=label_plot)
        vbottom = clip_min if use_log else 0.0
        ax.vlines(mu, vbottom, peak_plot, color=color, linewidth=1.0, linestyle="-", alpha=0.6)

    ax.set_xlabel("Integral value", fontsize=16)
    ax.set_ylabel("Probability density", fontsize=16)

    if use_log:
        ax.set_yscale("log")
        ax.set_ylim(bottom=clip_min)
    else:
        ax.set_yscale("linear")
        ax.set_ylim(bottom=0.0)

    # Remove default autoscale padding so curves touch the x-axis baseline
    ax.margins(x=0, y=0)

    ax.tick_params(direction="in", labelsize=14)
    ax.legend(loc="upper right", fontsize=14)
    fig.tight_layout()
    return fig

# ---------------------------------------------------------------------------
# Printing / Reporting
# ---------------------------------------------------------------------------

def _mse(r, method):
    """Compute MSE = bias² + variance for a result dict and method prefix."""
    return (r[f"{method}_mean"] - r["exact"]) ** 2 + r[f"{method}_var"]


def print_summary_table(results, run_quantum=False, run_quantum_analytical=False):
    """Print a tabular summary of results using MSE = bias² + variance."""
    header = (
        f"{'Distribution':>12s}  {'Payoff':>25s}  "
        f"{'Exact':>10s}  {'GPQ_MSE':>10s}  {'HSGP_MSE':>10s}"
    )
    if run_quantum_analytical:
        header += f"  {'QA_MSE':>10s}"
    if run_quantum:
        header += f"  {'Q_MSE':>10s}"

    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))

    for r in results:
        line = (
            f"{r['dist_name']:>12s}  {r['payoff_name']:>25s}  "
            f"{r['exact']:10.6f}  {_mse(r, 'gpq'):10.6f}  {_mse(r, 'hsgp'):10.6f}"
        )
        if run_quantum_analytical and "quantum_analytical_mean" in r:
            line += f"  {_mse(r, 'quantum_analytical'):10.6f}"
        elif run_quantum_analytical:
            line += f"  {'N/A':>10s}"
        if run_quantum and "quantum_mean" in r:
            line += f"  {_mse(r, 'quantum'):10.6f}"
        elif run_quantum:
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

    # ── MSE and percentage-error statistics per method ───────────────────
    stats = {}
    error_rows = []
    for method in ["gpq", "hsgp", "quantum_analytical", "quantum"]:
        method_key = f"{method}_mean"
        var_key = f"{method}_var"
        if method_key in results[0]:
            pct_errors = [
                abs(r["exact"] - r[method_key]) / abs(r["exact"]) * 100
                if r["exact"] != 0 else float("nan")
                for r in results
            ]
            mse_vals = [
                (r[method_key] - r["exact"]) ** 2 + r.get(var_key, 0.0)
                for r in results
            ]
            pct_errors = np.array(pct_errors, dtype=float)
            mse_vals = np.array(mse_vals, dtype=float)
            stats[method] = {
                "mean_pct_err": float(np.nanmean(pct_errors)),
                "median_pct_err": float(np.nanmedian(pct_errors)),
                "max_pct_err": float(np.nanmax(pct_errors)),
                "min_pct_err": float(np.nanmin(pct_errors)),
                "mean_mse": float(np.nanmean(mse_vals)),
                "median_mse": float(np.nanmedian(mse_vals)),
                "max_mse": float(np.nanmax(mse_vals)),
                "min_mse": float(np.nanmin(mse_vals)),
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

    # Relative percentage errors and MSE compared to exact for each method
    error_cols = [c for c in df.columns if c.endswith("_mean")]
    err_df = df[["dist_name", "payoff_name"]].copy()
    for col in error_cols:
        method = col[: -len("_mean")]
        err_df[f"{method}_pct_err"] = (
            (df[col] - df["exact"]).abs() / df["exact"].abs().replace(0, np.nan) * 100
        )
        var_col = f"{method}_var"
        if var_col in df.columns:
            err_df[f"{method}_mse"] = (
                (df[col] - df["exact"]) ** 2 + df[var_col]
            )
    pct_err_cols = [c for c in err_df.columns if c.endswith("_pct_err")]
    mse_cols = [c for c in err_df.columns if c.endswith("_mse")]

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

    # Mean MSE grouped by distribution and payoff type
    if mse_cols:
        stats["mean_mse_by_distribution"] = err_df.groupby("dist_name")[mse_cols].mean()
        stats["mean_mse_by_distribution"].to_csv(
            os.path.join(out_dir, "stats_mse_by_distribution.csv"),
        )
        stats["mean_mse_by_payoff"] = err_df.groupby("payoff_name")[mse_cols].mean()
        stats["mean_mse_by_payoff"].to_csv(
            os.path.join(out_dir, "stats_mse_by_payoff.csv"),
        )

    return stats


def save_plots(
    results,
    make_integrand_fn,
    experiment_dist_params,
    payoff_defaults,
    N=32,
    noise_std=0.01,
    seed=679,
    point_strategy="hybrid",
    run_quantum=False,
    run_quantum_analytical=False,
    out_dir="figures",
):
    """Generate and save integrand and comparison plots for all results.

    Args:
        results: List of result dicts from run_all_experiments.
        make_integrand_fn: The make_integrand callable.
        experiment_dist_params: Dict mapping dist_name to parameter dicts.
        payoff_defaults: Dict mapping payoff_name to parameter dicts.
        N: Number of evaluation points used in the experiments.
        noise_std: Observation noise std used in the experiments.
        seed: Random seed used in the experiments.
        point_strategy: Point placement strategy ('hybrid', 'quantile', 'uniform').
        run_quantum: Whether quantum results are present.
        run_quantum_analytical: Whether analytical quantum results are present.
        out_dir: Directory to write figures into.
    """
    import os
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    for r in results:
        integrand, domain, dist_obj = make_integrand_fn(
            r["dist_name"],
            experiment_dist_params.get(r["dist_name"]),
            r["payoff_name"],
            payoff_defaults.get(r["payoff_name"]),
        )
        a, b = domain[0], domain[1]

        # Mirror the sampling logic from experiments._sample_points so the
        # plot training points match what was actually used in the experiment.
        from experiments import _sample_points
        X_raw = _sample_points(dist_obj, a, b, N, point_strategy)
        X_eval = X_raw.reshape(-1, 1)

        y_eval = integrand(X_raw) + rng.normal(0.0, noise_std, size=len(X_raw))
        tag = f"{r['dist_name']}_{r['payoff_name']}"

        fig1 = plot_integrand(
            integrand, domain, X_eval, y_eval
        )
        fig1.savefig(os.path.join(out_dir, f"integrand_{tag}.pdf"), bbox_inches="tight")

        fig2 = plot_comparison_gaussians(
            r,
            run_quantum=run_quantum,
            run_quantum_analytical=run_quantum_analytical,
        )
        fig2.savefig(os.path.join(out_dir, f"comparison_{tag}.pdf"), bbox_inches="tight")

        plt.close("all")

    print(f"\nPlots saved to {out_dir}/")