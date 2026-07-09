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
        "font.size": 20,
        "axes.labelsize": 20,
        "axes.titlesize": 20,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
        "figure.dpi": 600,
    })

    z = np.linspace(domain[0], domain[1], 500)
    g = integrand(z)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(z, g, 0.0, alpha=0.15, color="steelblue")
    ax.plot(z, g, "steelblue", lw=2, label=r"$\Pi(z)\,f_Z(z)$")
    ax.plot(X_eval.ravel(), y_eval, "rx", ms=7, label="Noisy observations")

    ax.set_xlabel("$z$", fontsize=20)
    ax.set_ylabel("Integrand", fontsize=20)
    ax.tick_params(direction="in", labelsize=16)
    ax.legend(fontsize=20)

    # Add small horizontal padding so the curve/points do not touch plot borders.
    x_all = np.concatenate([np.asarray(z).ravel(), np.asarray(X_eval).ravel()])
    x_min = np.nanmin(x_all)
    x_max = np.nanmax(x_all)
    x_span = x_max - x_min
    x_pad = 0.04 * (x_span if x_span > 0 else (abs(x_max) if x_max != 0 else 1.0))
    ax.set_xlim(x_min - x_pad, x_max + x_pad)

    # Keep the baseline at y=0 when data is nonnegative so the shaded area
    # touches the x-axis (no floating effect).
    y_all = np.concatenate([np.asarray(g).ravel(), np.asarray(y_eval).ravel()])
    y_min = np.nanmin(y_all)
    y_max = np.nanmax(y_all)
    y_span = y_max - y_min
    y_pad = 0.08 * (y_span if y_span > 0 else (abs(y_max) if y_max != 0 else 1.0))

    if y_min >= 0:
        ax.set_ylim(0.0, y_max + y_pad)
    else:
        ax.set_ylim(y_min - y_pad, y_max + y_pad)

    # Remove extra autoscale padding on y so baseline aligns exactly.
    ax.margins(y=0)

    fig.tight_layout()
    return fig


_PLT_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 20,
    "axes.labelsize": 20,
    "axes.titlesize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
    "figure.dpi": 600,
}

COLOR_GPQ = "k"
COLOR_HSGP = "#8f0606"
COLOR_HSGP_Q = "#e07b00"
COLOR_Q = "#008080"


def _draw_gaussians(ax, exact, methods, normalize=False, use_log=False,
                    clip_min=1e-300):
    """Draw N(mean, var) PDFs for a list of (label, mean, var, color, ls, alpha).

    Returns nothing; mutates *ax*.  The x-range spans all means ± 4σ.
    """
    from scipy.stats import norm

    all_means = [m[1] for m in methods]
    all_stds = [np.sqrt(max(m[2], 1e-12)) for m in methods]
    lo = min(exact, min(m - 4 * s for m, s in zip(all_means, all_stds)))
    hi = max(exact, max(m + 4 * s for m, s in zip(all_means, all_stds)))
    x = np.linspace(lo, hi, 600)

    ax.axvline(x=exact, color="k", linestyle="-", linewidth=1.5, label="Exact")
    for label, mu, var, color, ls, alpha in methods:
        std = np.sqrt(max(var, 1e-12))
        pdf = norm.pdf(x, mu, std)
        peak = norm.pdf(mu, mu, std)
        if normalize:
            pdf = pdf / peak
            peak = 1.0
        if use_log:
            pdf = np.clip(pdf, clip_min, None)
        ax.fill_between(x, pdf, alpha=0.12 * alpha, color=color)
        ax.plot(x, pdf, color=color, ls=ls, lw=2, alpha=alpha, label=label)
        ax.vlines(mu, clip_min if use_log else 0.0, peak,
                  color=color, linewidth=1.0, alpha=0.6 * alpha)

    ax.set_xlabel("Integral value", fontsize=20)
    ax.set_ylabel("Probability density", fontsize=20)
    if use_log:
        ax.set_yscale("log")
        ax.set_ylim(bottom=clip_min)
    else:
        ax.set_ylim(bottom=0.0)
    ax.margins(x=0, y=0)
    ax.tick_params(direction="in", labelsize=16)
    ax.legend(loc="upper left", fontsize=16)


def plot_comparison_gaussians(
    result,
    run_quantum=False,
    run_quantum_analytical=False,
    normalize=False,
    use_log=False,
    clip_min=1e-300,
):
    """Overlay Gaussian PDFs N(mean, var) for each BQ method.

    Draws the full-resolution classical baselines (GPQ, HSGP-BQ at σ), the
    matched classical target (HSGP-BQ at the quantum σ_q, M_q — the curve the
    quantum method converges to), and the quantum estimate.  Because the
    classical baselines use the (smaller) experiment σ while the quantum uses
    σ_q, the classical curves are sharp and the quantum is broad; the matched
    curve shows the quantum lands on its correct classical target despite that
    width difference.  For the paper-style figure where every method shares
    one σ and the distributions overlap, see :func:`plot_comparison_matched`.

    Args:
        normalize: if True, scale each PDF by its peak (so peaks=1).
        use_log: if True, use a log y-axis (clip values below `clip_min`).
        clip_min: minimum y-value to use when plotting on a log scale.
    """
    import matplotlib.pyplot as plt

    plt.rcParams.update(_PLT_RC)
    exact = result["exact"]

    methods = [
        ("GPQ", result["gpq_mean"], result["gpq_var"], COLOR_GPQ, ":", 1.0),
        ("HSGP-BQ", result["hsgp_mean"], result["hsgp_var"], COLOR_HSGP, "--", 1.0),
    ]
    if (run_quantum or run_quantum_analytical) and "hsgp_q_mean" in result:
        methods.append(
            ("HSGP-BQ (matched)", result["hsgp_q_mean"], result["hsgp_q_var"],
             COLOR_HSGP_Q, "-.", 1.0),
        )
    if run_quantum and "quantum_mean" in result:
        methods.append(
            ("Quantum", result["quantum_mean"], result["quantum_var"],
             COLOR_Q, "-", 1.0),
        )

    fig, ax = plt.subplots(figsize=(10, 5))
    _draw_gaussians(ax, exact, methods, normalize, use_log, clip_min)
    fig.tight_layout()
    return fig


def plot_comparison_matched(result, use_circuit=True):
    """Paper-style comparison: every method at the *same* hyperparameters.

    Reproduces the reference paper's Fig. 5 layout.  GPQ, HSGP-BQ, and the
    quantum quadrature are all evaluated at the quantum configuration
    (M_q, σ_q, ℓ_q) — a single shared σ — so their posterior variances are
    comparable and the Gaussian estimates overlap around the integral value,
    rather than the sharp-classical / broad-quantum split that appears when
    the classical baselines use a much smaller σ.  The quantum low-rank
    sweep (R = 1, 2, …) is drawn with increasing opacity, showing the higher
    ranks converging onto the classical HSGP-BQ curve.

    Args:
        result: One result dict from :func:`experiments.run_experiment`
            (must have been run with run_quantum_analytical=True so the
            matched baselines and rank sweep are present).
        use_circuit: also overlay the shot-based circuit quantum estimate.

    Returns:
        Matplotlib figure, or None if the matched fields are absent.
    """
    import matplotlib.pyplot as plt

    if "hsgp_q_mean" not in result:
        return None
    plt.rcParams.update(_PLT_RC)
    exact = result["exact"]

    methods = [
        ("GPQ", result["gpq_q_mean"], result["gpq_q_var"], COLOR_GPQ, ":", 1.0),
        ("HSGP-BQ", result["hsgp_q_mean"], result["hsgp_q_var"],
         COLOR_HSGP, "--", 1.0),
    ]
    ranks = result.get("rank_values", [])
    rank_means = result.get("quantum_rank_means", [])
    rank_vars = result.get("quantum_rank_vars", [])
    r_max = max(ranks) if ranks else 1
    for R, m, v in zip(ranks, rank_means, rank_vars):
        alpha = (R / r_max) ** 1.6
        methods.append(
            (f"Quantum R={R}", m, v, COLOR_Q, "-", alpha),
        )
    if use_circuit and "quantum_mean" in result:
        methods.append(
            ("Quantum (circuit)", result["quantum_mean"], result["quantum_var"],
             COLOR_HSGP_Q, "-", 1.0),
        )

    fig, ax = plt.subplots(figsize=(10, 5))
    _draw_gaussians(ax, exact, methods)
    fig.tight_layout()
    return fig

# ---------------------------------------------------------------------------
# Printing / Reporting
# ---------------------------------------------------------------------------

def _mse(r, method):
    """Compute MSE = bias² + variance for a result dict and method prefix."""
    return (r[f"{method}_mean"] - r["exact"]) ** 2 + r[f"{method}_var"]


def _abs_err(r, method):
    """Point-estimate error |mean − exact| for a result dict and method."""
    return abs(r[f"{method}_mean"] - r["exact"])


def print_summary_table(results, run_quantum=False, run_quantum_analytical=False):
    """Print a tabular summary of results.

    Columns
    -------
    Tail%      Domain-truncation remainder ∫_b^∞ Π f dz as a percentage of the
               untruncated expectation (truncation error vs. method error).
    *_MSE      Bayesian risk bias² + posterior variance.  Note the posterior
               variance scales with σ², so methods run at a larger σ have a
               larger MSE even with an identical point estimate.
    HSGPq_MSE  Classical HSGP-BQ at the *quantum* hyperparameters (M_q, σ_q):
               the matched target the quantum method converges to.  QA_MSE
               sits on this to machine precision — the gap to HSGP_MSE is the
               M/σ config difference, not a quantum effect.
    Q|err|     Point-estimate error |Q_mean − exact| of the quantum circuit,
               which reflects estimate accuracy without the σ²-inflated
               variance term.
    """
    has_tail = bool(results) and "exact_tail" in results[0]
    has_matched = bool(results) and "hsgp_q_mean" in results[0]
    header = (
        f"{'Distribution':>12s}  {'Payoff':>25s}  "
        f"{'Exact':>10s}"
    )
    if has_tail:
        header += f"  {'Tail%':>6s}"
    header += f"  {'GPQ_MSE':>10s}  {'HSGP_MSE':>10s}"
    if has_matched and (run_quantum or run_quantum_analytical):
        header += f"  {'HSGPq_MSE':>10s}"
    if run_quantum_analytical:
        header += f"  {'QA_MSE':>10s}"
    if run_quantum:
        header += f"  {'Q_MSE':>10s}  {'Q|err|':>8s}"

    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))

    for r in results:
        line = (
            f"{r['dist_name']:>12s}  {r['payoff_name']:>25s}  "
            f"{r['exact']:10.6f}"
        )
        if has_tail:
            tail = r.get("exact_tail", 0.0)
            full = r["exact"] + tail
            line += f"  {tail / full * 100 if full else 0.0:6.3f}"
        line += f"  {_mse(r, 'gpq'):10.6f}  {_mse(r, 'hsgp'):10.6f}"
        if has_matched and (run_quantum or run_quantum_analytical):
            if "hsgp_q_mean" in r:
                line += f"  {_mse(r, 'hsgp_q'):10.6f}"
            else:
                line += f"  {'N/A':>10s}"
        if run_quantum_analytical and "quantum_analytical_mean" in r:
            line += f"  {_mse(r, 'quantum_analytical'):10.6f}"
        elif run_quantum_analytical:
            line += f"  {'N/A':>10s}"
        if run_quantum and "quantum_mean" in r:
            line += f"  {_mse(r, 'quantum'):10.6f}  {_abs_err(r, 'quantum'):8.4f}"
        elif run_quantum:
            line += f"  {'N/A':>10s}  {'N/A':>8s}"
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

    for r in results:
        # Fresh RNG per result so the plotted noise matches the noise each
        # experiment actually trained on (run_experiment reseeds with the same
        # seed for every (dist, payoff) pair rather than continuing one stream).
        rng = np.random.default_rng(seed)
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

        # Paper-style matched comparison (all methods at the quantum config),
        # only when the matched baselines / rank sweep are present.
        if run_quantum_analytical and "hsgp_q_mean" in r:
            fig3 = plot_comparison_matched(r, use_circuit=run_quantum)
            if fig3 is not None:
                fig3.savefig(
                    os.path.join(out_dir, f"matched_{tag}.pdf"),
                    bbox_inches="tight",
                )

        plt.close("all")

    print(f"\nPlots saved to {out_dir}/")