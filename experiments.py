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

# Distribution parameters to use in experiments
EXPERIMENT_DIST_PARAMS = {
    "pareto": {"shape": 3.0, "scale": 1.0},
    "lognormal": {"mu": 0.0, "sigma": 1.0},
    "gamma": {"shape": 2.0, "rate": 1.0},
    "weibull": {"shape": 1.5, "scale": 2.0},
    "poisson": {"lam": 5.0},
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sample_points(dist, a, b, N, strategy):
    """Sample N evaluation points covering [a, b] using the given strategy.

    For the hybrid strategy the uniform component's lower bound is clipped to
    the 1e-6 quantile of *dist*, avoiding near-zero-density regions such as
    x≈0 for lognormal or gamma where the integrand is effectively zero.
    """
    if strategy == "hybrid":
        n_q = int(0.7 * N)
        n_u = N - n_q
        probs_q = np.linspace(1.0 / (n_q + 1), n_q / (n_q + 1), n_q)
        X_quantile = np.clip(dist.ppf(probs_q), a, b)
        # Clip the uniform grid's lower bound to the effective support so we
        # don't waste points in the zero-density region (e.g. x≈0 for lognormal).
        a_u = max(float(a), float(dist.ppf(1e-6)))
        X_uniform = np.linspace(a_u, b, n_u)
        X_raw = np.sort(np.unique(np.concatenate([X_quantile, X_uniform])))
        if len(X_raw) > N:
            idx = np.round(np.linspace(0, len(X_raw) - 1, N)).astype(int)
            X_raw = X_raw[idx]
        elif len(X_raw) < N:
            X_fill = np.linspace(a, b, N - len(X_raw) + 2)[1:-1]
            X_raw = np.sort(np.unique(np.concatenate([X_raw, X_fill])))[:N]
    elif strategy == "quantile":
        probs = np.linspace(1.0 / (N + 1), N / (N + 1), N)
        X_raw = np.clip(dist.ppf(probs), a, b)
    else:
        X_raw = np.linspace(a, b, N)
    return X_raw


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
    amplitude=None,
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
    X_raw = _sample_points(dist, a, b, N, point_strategy)

    y_eval = np.squeeze(integrand(X_raw))
    y_eval = y_eval + rng.normal(0.0, noise_std, size=y_eval.shape)

    # Centre evaluation points to match the centred domain
    X_eval = (X_raw - midpoint).reshape(-1, 1)

    # ── Auto kernel hyperparameters ────────────────────────────────────────
    if length_scale is None:
        length_scale = half_width / np.sqrt(N)
        length_scale = max(length_scale, 0.3)

    # Set amplitude to the empirical std of the (clean) integrand so the GP
    # prior is on the right scale.  A prior with amplitude=1 is wildly wrong
    # for integrands whose values are O(0.01)–O(0.1).
    if amplitude is None:
        amplitude = float(np.std(np.squeeze(integrand(X_raw))))
        amplitude = max(amplitude, 1e-3)  # guard against near-zero integrands

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

        # Reuse the classical evaluation points when quantum_N matches N so
        # all methods train on identical inputs.  When quantum_N differs,
        # generate a consistent set via the same _sample_points helper.
        q_X_raw = X_raw if quantum_N == N else _sample_points(
            dist, a, b, quantum_N, point_strategy
        )

        q_y_eval = np.squeeze(integrand(q_X_raw))
        q_y_eval = q_y_eval + np.random.default_rng(seed).normal(
            0.0, quantum_noise_std, size=q_y_eval.shape,
        )
        q_noise_var = quantum_noise_std ** 2

        q_X_eval = (q_X_raw - midpoint).reshape(-1, 1)

        t0 = time.perf_counter()
        q_mean, q_var = quantum_hsgp_integral(
            q_X_eval, q_y_eval, centered_domain, quantum_M, L, q_noise_var,
            length_scale=length_scale, amplitude=amplitude,
            n_eigenvalue_qubits=n_eigenvalue_qubits, shots=shots, seed=seed,
        )
        timings["quantum"] = time.perf_counter() - t0
        result["quantum_mean"] = q_mean
        result["quantum_var"] = q_var

    result["timings"] = timings
    return result


# ---------------------------------------------------------------------------
# All-experiments runner
# ---------------------------------------------------------------------------


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
            mse_gpq = (res["gpq_mean"] - res["exact"]) ** 2 + res["gpq_var"]
            mse_hsgp = (res["hsgp_mean"] - res["exact"]) ** 2 + res["hsgp_var"]
            msg = (
                f"exact={res['exact']:.6f}  "
                f"GPQ_MSE={mse_gpq:.6f}  "
                f"HSGP_MSE={mse_hsgp:.6f}"
            )
            if run_quantum_analytical and "quantum_analytical_mean" in res:
                mse_qa = (
                    (res["quantum_analytical_mean"] - res["exact"]) ** 2
                    + res["quantum_analytical_var"]
                )
                msg += f"  Q_anl_MSE={mse_qa:.6f}"
            if run_quantum and "quantum_mean" in res:
                mse_q = (res["quantum_mean"] - res["exact"]) ** 2 + res["quantum_var"]
                msg += f"  Q_MSE={mse_q:.6f}"
            print(msg)
            results.append(res)
    return results


