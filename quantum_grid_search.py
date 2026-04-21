"""Two-stage grid search for quantum HSGP-BQ parameters.

Stage 1 (fast):  Sweep (N, M, noise_std, length_scale) and evaluate only
  the *analytical* quantum result (exact eigendecomposition — no circuits).
  This quickly identifies which parameter combos give a good HSGP
  approximation.  Combos with analytical error > threshold are discarded.

Stage 2 (slow):  For each surviving combo, sweep τ (QPE qubits) and shots,
  running the full QPE-circuit quantum method.  Diagnostics on eigenphase
  structure are logged so we can see *why* certain combos work.

Usage
-----
    python quantum_grid_search.py                   # default (fast)
    python quantum_grid_search.py --full             # wider grid
    python quantum_grid_search.py --analytical-only  # Stage 1 only
"""

import argparse
import itertools
import os
import time

import numpy as np
import pandas as pd

from gaussian_quantum.insurance import (
    make_integrand,
    exact_integral,
    DISTRIBUTIONS,
    PAYOFFS,
    PAYOFF_DEFAULTS,
)
from gaussian_quantum.hilbert_space_approx import (
    hilbert_space_features,
    kernel_mean_features,
)
from gaussian_quantum.qpca import (
    build_density_matrix_unitary,
    conditional_rotation_mean,
    prepare_mean_states,
    prepare_mean_states_analytical,
)

# Distribution parameters (same as insurance_experiments.py)
EXPERIMENT_DIST_PARAMS = {
    "pareto": {"shape": 3.0, "scale": 1.0},
    "lognormal": {"mu": 0.0, "sigma": 1.0},
    "gamma": {"shape": 2.0, "rate": 1.0},
    "weibull": {"shape": 1.5, "scale": 2.0},
}

# All (dist, payoff) combinations
ALL_CASES = [
    (d, p)
    for d in DISTRIBUTIONS
    for p in PAYOFFS
]


# ---------------------------------------------------------------------------
# Shared helper: build centred evaluation data
# ---------------------------------------------------------------------------

def _build_quantum_data(dist_name, payoff_name, quantum_N, quantum_noise_std,
                         seed=679):
    """Build centred evaluation points and noisy observations."""
    dist_params = EXPERIMENT_DIST_PARAMS.get(dist_name)
    payoff_params = PAYOFF_DEFAULTS.get(payoff_name)
    integrand, domain, dist = make_integrand(
        dist_name, dist_params, payoff_name, payoff_params,
    )
    exact_val, _ = exact_integral(integrand, domain)

    a, b = domain
    midpoint = (a + b) / 2.0
    half_width = (b - a) / 2.0

    rng = np.random.default_rng(seed)
    n_q = int(0.7 * quantum_N)
    n_u = quantum_N - n_q
    probs_q = np.linspace(1.0 / (n_q + 1), n_q / (n_q + 1), n_q)
    X_quantile = np.clip(dist.ppf(probs_q), a, b)
    X_uniform = np.linspace(a, b, n_u)
    X_raw = np.sort(np.unique(np.concatenate([X_quantile, X_uniform])))
    if len(X_raw) > quantum_N:
        idx = np.round(np.linspace(0, len(X_raw) - 1, quantum_N)).astype(int)
        X_raw = X_raw[idx]
    elif len(X_raw) < quantum_N:
        X_fill = np.linspace(a, b, quantum_N - len(X_raw) + 2)[1:-1]
        X_raw = np.sort(np.unique(np.concatenate([X_raw, X_fill])))[:quantum_N]

    y_eval = np.squeeze(integrand(X_raw))
    y_eval = y_eval + rng.normal(0.0, quantum_noise_std, size=y_eval.shape)

    X_eval = (X_raw - midpoint).reshape(-1, 1)
    centered_domain = (a - midpoint, b - midpoint)
    L = max(half_width * 1.3, 1.0)

    return X_eval, y_eval, centered_domain, L, exact_val


# ---------------------------------------------------------------------------
# Stage 1: Analytical-only evaluation (fast)
# ---------------------------------------------------------------------------

def evaluate_analytical(
    dist_name, payoff_name,
    quantum_N, quantum_M, quantum_noise_std, quantum_length_scale,
    seed=679, amplitude=1.0,
):
    """Evaluate a single (dist, payoff) case analytically — no circuits."""
    X_eval, y_eval, centered_domain, L, exact_val = _build_quantum_data(
        dist_name, payoff_name, quantum_N, quantum_noise_std, seed,
    )
    noise_var = quantum_noise_std ** 2

    X_feat, _, _, _ = hilbert_space_features(
        X_eval, quantum_M, L, quantum_length_scale, amplitude,
    )
    z_mu, _, _, _ = kernel_mean_features(
        centered_domain, quantum_M, L, quantum_length_scale, amplitude,
    )

    # Eigenstructure diagnostics
    XtX = X_feat.T @ X_feat
    F_sq = float(np.trace(XtX))
    eigvals = np.linalg.eigvalsh(XtX)
    eigvals = np.maximum(eigvals, 0.0)
    phases = eigvals / F_sq if F_sq > 1e-12 else eigvals * 0
    sorted_nz = np.sort(phases[phases > 1e-10])
    if len(sorted_nz) > 1:
        min_gap = float(np.min(np.diff(sorted_nz)))
    else:
        min_gap = float("nan")

    # Analytical mean
    psi1, psi2, n1, n2, c_mean, sprob = prepare_mean_states_analytical(
        X_feat, y_eval, z_mu, noise_var,
    )
    if n1 < 1e-12 or n2 < 1e-12 or sprob < 1e-15:
        analytical_mean = 0.0
    else:
        inner = float(np.real(np.vdot(psi1, psi2)))
        analytical_mean = float((1.0 / c_mean) * n1 * n2 * inner)

    return {
        "dist": dist_name,
        "payoff": payoff_name,
        "exact": exact_val,
        "analytical_mean": analytical_mean,
        "analytical_err": abs(exact_val - analytical_mean),
        "N": quantum_N,
        "M": quantum_M,
        "noise_std": quantum_noise_std,
        "length_scale": quantum_length_scale,
        "F_sq": F_sq,
        "cond_XtX": eigvals[-1] / max(eigvals[0], 1e-15),
        "n_eff_eigvals": int(np.sum(eigvals > 1e-6 * eigvals[-1])),
        "min_phase_gap": min_gap,
        "phases": phases.tolist(),
    }


# ---------------------------------------------------------------------------
# Stage 2: Full QPE circuit evaluation
# ---------------------------------------------------------------------------

def evaluate_circuit(
    dist_name, payoff_name,
    quantum_N, quantum_M, n_eigenvalue_qubits,
    quantum_noise_std, quantum_length_scale, shots,
    seed=679, amplitude=1.0,
):
    """Run the full QPE-circuit quantum method."""
    X_eval, y_eval, centered_domain, L, exact_val = _build_quantum_data(
        dist_name, payoff_name, quantum_N, quantum_noise_std, seed,
    )
    noise_var = quantum_noise_std ** 2

    from gaussian_quantum.quantum_algorithms import quantum_hsgp_integral

    t0 = time.perf_counter()
    q_mean, q_var = quantum_hsgp_integral(
        X_eval, y_eval, centered_domain, quantum_M, L, noise_var,
        length_scale=quantum_length_scale, amplitude=amplitude,
        n_eigenvalue_qubits=n_eigenvalue_qubits, shots=shots, seed=seed,
    )
    elapsed = time.perf_counter() - t0

    abs_err = abs(exact_val - q_mean)
    return {
        "dist": dist_name,
        "payoff": payoff_name,
        "exact": exact_val,
        "q_mean": q_mean,
        "abs_err": abs_err,
        "pct_err": abs_err / abs(exact_val) * 100 if exact_val != 0 else float("nan"),
        "N": quantum_N,
        "M": quantum_M,
        "tau": n_eigenvalue_qubits,
        "noise_std": quantum_noise_std,
        "length_scale": quantum_length_scale,
        "shots": shots,
        "time_s": elapsed,
    }


# ---------------------------------------------------------------------------
# Main two-stage search
# ---------------------------------------------------------------------------

def run_grid_search(full=False, analytical_only=False):
    """Two-stage parameter search."""

    # ── Stage 1 grid (N, M, noise_std, length_scale) ─────────────────────
    if full:
        stage1_grid = {
            "quantum_N": [8, 12, 16, 24, 32],
            "quantum_M": [2, 3, 4, 6, 8, 12, 16],
            "quantum_noise_std": [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5],
            "quantum_length_scale": [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0],
        }
    else:
        stage1_grid = {
            "quantum_N": [8, 16, 32],
            "quantum_M": [2, 3, 4, 6, 8],
            "quantum_noise_std": [0.001, 0.01, 0.05, 0.1, 0.5],
            "quantum_length_scale": [0.5, 1.0, 2.0, 3.0, 5.0],
        }

    s1_names = list(stage1_grid.keys())
    s1_combos = list(itertools.product(*stage1_grid.values()))
    # Filter: M <= N
    s1_combos = [c for c in s1_combos if c[1] <= c[0]]

    print(f"=== STAGE 1: Analytical evaluation ===")
    print(f"{len(s1_combos)} combos × {len(ALL_CASES)} cases "
          f"= {len(s1_combos) * len(ALL_CASES)} evaluations")
    print()

    stage1_rows = []
    combo_anl_errors = []

    for ci, combo in enumerate(s1_combos):
        params = dict(zip(s1_names, combo))
        errs = []
        for dist_name, payoff_name in ALL_CASES:
            res = evaluate_analytical(
                dist_name, payoff_name,
                quantum_N=params["quantum_N"],
                quantum_M=params["quantum_M"],
                quantum_noise_std=params["quantum_noise_std"],
                quantum_length_scale=params["quantum_length_scale"],
            )
            stage1_rows.append(res)
            errs.append(res["analytical_err"])

        mean_err = float(np.nanmean(errs))
        max_err = float(np.nanmax(errs))
        combo_anl_errors.append((params, mean_err, max_err, errs))

        if (ci + 1) % 20 == 0 or ci == 0:
            print(f"  [{ci+1:4d}/{len(s1_combos)}]  "
                  f"N={params['quantum_N']:2d} M={params['quantum_M']:2d} "
                  f"σ_n={params['quantum_noise_std']:.3f} "
                  f"ls={params['quantum_length_scale']:.1f}  "
                  f"anl_mean_err={mean_err:.6f}  anl_max_err={max_err:.6f}")

    # Sort and print Stage 1 results
    combo_anl_errors.sort(key=lambda x: x[1])

    print(f"\n{'='*100}")
    print("STAGE 1: TOP 30 ANALYTICAL COMBOS (by mean error)")
    print(f"{'='*100}")
    print(f"{'Rank':>4s}  {'N':>3s} {'M':>3s} {'σ_n':>8s} {'ls':>6s}  "
          f"{'mean_err':>10s} {'max_err':>10s}")
    print("-" * 60)
    for rank, (params, mean_err, max_err, _) in enumerate(combo_anl_errors[:30]):
        print(f"{rank+1:4d}  "
              f"{params['quantum_N']:3d} {params['quantum_M']:3d} "
              f"{params['quantum_noise_std']:8.3f} "
              f"{params['quantum_length_scale']:6.1f}  "
              f"{mean_err:10.6f} {max_err:10.6f}")

    # Save Stage 1
    os.makedirs("figures", exist_ok=True)
    s1_df = pd.DataFrame(stage1_rows)
    for col in ["phases"]:
        if col in s1_df.columns:
            s1_df = s1_df.drop(columns=[col])
    s1_df.to_csv("figures/quantum_grid_stage1.csv", index=False)

    if analytical_only:
        print(f"\nStage 1 results saved to figures/quantum_grid_stage1.csv")
        return combo_anl_errors, stage1_rows, [], []

    # ── Stage 2: Circuit evaluation on promising combos ──────────────────
    # Keep combos with analytical max_err < threshold
    THRESHOLD = 0.10  # analytical error must be < 10% of typical exact values
    promising = [x for x in combo_anl_errors if x[2] < THRESHOLD]
    if not promising:
        # Fall back to top-20
        promising = combo_anl_errors[:20]
    else:
        promising = promising[:40]  # cap to avoid excessive runtime

    # QPE grid
    tau_values = [4, 5, 6, 7, 8]
    shots_values = [32768, 65536]

    n_circuit_evals = len(promising) * len(tau_values) * len(shots_values) * len(ALL_CASES)
    print(f"\n=== STAGE 2: Circuit evaluation ===")
    print(f"{len(promising)} promising combos × {len(tau_values)} τ values "
          f"× {len(shots_values)} shots × {len(ALL_CASES)} cases "
          f"= {n_circuit_evals} evaluations")
    print()

    stage2_rows = []
    circuit_combo_errors = []
    eval_count = 0

    for pi, (params, anl_mean, anl_max, _) in enumerate(promising):
        for tau in tau_values:
            # Skip if n_state_qubits + tau + 1 > ~16 (too slow)
            n_state_q = max(1, int(np.ceil(np.log2(params["quantum_M"]))))
            total_qubits = tau + n_state_q + 1
            if total_qubits > 14:
                continue

            for shots in shots_values:
                errs = []
                for dist_name, payoff_name in ALL_CASES:
                    try:
                        res = evaluate_circuit(
                            dist_name, payoff_name,
                            quantum_N=params["quantum_N"],
                            quantum_M=params["quantum_M"],
                            n_eigenvalue_qubits=tau,
                            quantum_noise_std=params["quantum_noise_std"],
                            quantum_length_scale=params["quantum_length_scale"],
                            shots=shots,
                        )
                        stage2_rows.append(res)
                        errs.append(res["abs_err"])
                    except Exception as e:
                        print(f"  FAIL: {e}")
                        errs.append(float("nan"))

                eval_count += len(ALL_CASES)
                mean_err = float(np.nanmean(errs))
                max_err = float(np.nanmax(errs))
                full_params = dict(params)
                full_params["n_eigenvalue_qubits"] = tau
                full_params["shots"] = shots
                circuit_combo_errors.append((full_params, mean_err, max_err))

                if eval_count % (len(ALL_CASES) * 5) == 0 or eval_count == len(ALL_CASES):
                    print(f"  [{eval_count:5d}/{n_circuit_evals}]  "
                          f"N={params['quantum_N']:2d} M={params['quantum_M']:2d} "
                          f"τ={tau} σ_n={params['quantum_noise_std']:.3f} "
                          f"ls={params['quantum_length_scale']:.1f} "
                          f"shots={shots}  "
                          f"mean_err={mean_err:.6f}  max_err={max_err:.6f}")

    # Sort and display Stage 2 results
    circuit_combo_errors.sort(key=lambda x: x[1])

    print(f"\n{'='*120}")
    print("STAGE 2: TOP 20 CIRCUIT COMBOS (by mean absolute error)")
    print(f"{'='*120}")
    header = (f"{'Rank':>4s}  {'N':>3s} {'M':>3s} {'τ':>3s} "
              f"{'σ_noise':>8s} {'l_scale':>8s} {'shots':>7s}  "
              f"{'mean_err':>10s} {'max_err':>10s}")
    print(header)
    print("-" * 80)

    for rank, (params, mean_err, max_err) in enumerate(circuit_combo_errors[:20]):
        print(f"{rank+1:4d}  "
              f"{params['quantum_N']:3d} {params['quantum_M']:3d} "
              f"{params['n_eigenvalue_qubits']:3d} "
              f"{params['quantum_noise_std']:8.3f} "
              f"{params['quantum_length_scale']:8.2f} "
              f"{params['shots']:7d}  "
              f"{mean_err:10.6f} {max_err:10.6f}")
    print(f"{'='*120}")

    # Top 10 by worst-case
    by_max = sorted(circuit_combo_errors, key=lambda x: x[2])
    print("\nTOP 10 BY WORST-CASE (max) ERROR:")
    print("-" * 80)
    for rank, (params, mean_err, max_err) in enumerate(by_max[:10]):
        print(f"{rank+1:4d}  "
              f"N={params['quantum_N']:3d} M={params['quantum_M']:3d} "
              f"τ={params['n_eigenvalue_qubits']:3d} "
              f"σ_n={params['quantum_noise_std']:.3f} "
              f"ls={params['quantum_length_scale']:.2f} "
              f"shots={params['shots']:7d}  "
              f"mean={mean_err:.6f} max={max_err:.6f}")

    # Detailed results for top-3
    print("\n\nDETAILED PER-CASE RESULTS FOR TOP-3 COMBOS:")
    for rank, (params, mean_err, max_err) in enumerate(circuit_combo_errors[:3]):
        print(f"\n--- Rank {rank+1}: N={params['quantum_N']} M={params['quantum_M']} "
              f"τ={params['n_eigenvalue_qubits']} "
              f"σ_n={params['quantum_noise_std']} "
              f"ls={params['quantum_length_scale']} "
              f"shots={params['shots']} ---")
        rows = [r for r in stage2_rows
                if r["N"] == params["quantum_N"]
                and r["M"] == params["quantum_M"]
                and r["tau"] == params["n_eigenvalue_qubits"]
                and r["noise_std"] == params["quantum_noise_std"]
                and r["length_scale"] == params["quantum_length_scale"]
                and r["shots"] == params["shots"]]
        for r in rows:
            print(f"  {r['dist']:>12s} × {r['payoff']:>25s}  "
                  f"exact={r['exact']:.6f}  q={r['q_mean']:.6f}  "
                  f"err={r['abs_err']:.6f}  ({r['pct_err']:.2f}%)")

    # Save
    s2_df = pd.DataFrame(stage2_rows)
    s2_df.to_csv("figures/quantum_grid_stage2.csv", index=False)

    summary_rows = []
    for params, mean_err, max_err in circuit_combo_errors:
        row = dict(params)
        row["mean_abs_err"] = mean_err
        row["max_abs_err"] = max_err
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(
        "figures/quantum_grid_summary.csv", index=False,
    )
    print(f"\nResults saved to figures/quantum_grid_stage1.csv, "
          f"quantum_grid_stage2.csv, quantum_grid_summary.csv")

    return combo_anl_errors, stage1_rows, circuit_combo_errors, stage2_rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantum parameter grid search")
    parser.add_argument("--full", action="store_true",
                        help="Run the full (larger) grid")
    parser.add_argument("--analytical-only", action="store_true",
                        help="Only run Stage 1 (analytical, fast)")
    args = parser.parse_args()
    run_grid_search(full=args.full, analytical_only=args.analytical_only)
