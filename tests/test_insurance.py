"""Tests for the insurance module and classical GPQ baseline."""

import numpy as np
import pytest
from scipy import stats

from gaussian_quantum.insurance import (
    ordinary_deductible,
    franchise_deductible,
    policy_limit,
    deductible_with_limit,
    stop_loss,
    make_integrand,
    exact_integral,
    get_distribution,
    DISTRIBUTIONS,
    PAYOFFS,
)
from gaussian_quantum.classical import (
    gpq_integral,
    rbf_kernel_mean_embedding,
    rbf_kernel_double_integral,
)
from gaussian_quantum.hilbert_space_approx import hsgp_integral


# ---------------------------------------------------------------------------
# Payoff function boundary-value tests
# ---------------------------------------------------------------------------

class TestPayoffFunctions:
    """Verify payoff functions at key boundary values."""

    def test_ordinary_deductible_below(self):
        assert ordinary_deductible(0.5, D=1.0) == 0.0

    def test_ordinary_deductible_at(self):
        assert ordinary_deductible(1.0, D=1.0) == 0.0

    def test_ordinary_deductible_above(self):
        assert ordinary_deductible(5.0, D=1.0) == pytest.approx(4.0)

    def test_ordinary_deductible_vectorized(self):
        z = np.array([0.5, 1.0, 3.0, 10.0])
        expected = np.array([0.0, 0.0, 2.0, 9.0])
        np.testing.assert_array_almost_equal(ordinary_deductible(z, D=1.0), expected)

    def test_franchise_deductible_below(self):
        assert franchise_deductible(0.5, D=1.0) == 0.0

    def test_franchise_deductible_at(self):
        assert franchise_deductible(1.0, D=1.0) == 0.0

    def test_franchise_deductible_above(self):
        assert franchise_deductible(5.0, D=1.0) == pytest.approx(5.0)

    def test_policy_limit_below(self):
        assert policy_limit(2.0, U=5.0) == pytest.approx(2.0)

    def test_policy_limit_above(self):
        assert policy_limit(10.0, U=5.0) == pytest.approx(5.0)

    def test_deductible_with_limit_below_D(self):
        assert deductible_with_limit(0.5, D=1.0, U=5.0) == 0.0

    def test_deductible_with_limit_between(self):
        assert deductible_with_limit(3.0, D=1.0, U=5.0) == pytest.approx(2.0)

    def test_deductible_with_limit_above_U(self):
        assert deductible_with_limit(100.0, D=1.0, U=5.0) == pytest.approx(5.0)

    def test_stop_loss_equals_ordinary_deductible(self):
        z = np.array([0.5, 1.0, 3.0, 10.0])
        np.testing.assert_array_equal(
            stop_loss(z, D=2.0), ordinary_deductible(z, D=2.0)
        )


# ---------------------------------------------------------------------------
# Distribution tests
# ---------------------------------------------------------------------------

class TestDistributions:
    """Verify distribution factory returns valid objects."""

    @pytest.mark.parametrize("name", list(DISTRIBUTIONS.keys()))
    def test_get_distribution_returns_domain(self, name):
        dist, domain = get_distribution(name)
        a, b = domain
        assert a < b
        assert np.isfinite(a)
        assert np.isfinite(b)

    @pytest.mark.parametrize("name", ["pareto", "lognormal", "gamma", "weibull"])
    def test_continuous_pdf_integrates_near_one(self, name):
        """PDF should integrate to ~1 over the returned domain."""
        dist, domain = get_distribution(name)
        from scipy.integrate import quad
        val, _ = quad(dist.pdf, domain[0], domain[1])
        assert val > 0.95  # truncated domain captures ≥95 % of mass


# ---------------------------------------------------------------------------
# Integrand factory tests
# ---------------------------------------------------------------------------

class TestIntegrandFactory:
    """Verify the integrand g(z) = Π(z)·f_Z(z) is constructed correctly."""

    def test_returns_callable_and_domain(self):
        g, domain, dist = make_integrand("pareto")
        assert callable(g)
        a, b = domain
        assert a < b

    def test_integrand_nonnegative_for_deductible(self):
        """Π(z) ≥ 0 and f_Z(z) ≥ 0, so g(z) ≥ 0."""
        g, domain, _ = make_integrand(
            "lognormal", payoff_name="ordinary_deductible",
        )
        z = np.linspace(domain[0] + 1e-6, domain[1], 200)
        assert np.all(g(z) >= -1e-12)

    @pytest.mark.parametrize("payoff_name", list(PAYOFFS.keys()))
    def test_exact_integral_is_finite(self, payoff_name):
        g, domain, _ = make_integrand("gamma", payoff_name=payoff_name)
        val, err = exact_integral(g, domain)
        assert np.isfinite(val)
        assert val >= 0.0


# ---------------------------------------------------------------------------
# Classical GPQ tests
# ---------------------------------------------------------------------------

class TestClassicalGPQ:
    """Verify classical full-kernel GPQ integral estimation."""

    def test_kernel_mean_embedding_shape(self):
        X = np.linspace(0, 5, 20).reshape(-1, 1)
        k_mu = rbf_kernel_mean_embedding(X, domain=(0.0, 5.0))
        assert k_mu.shape == (20,)

    def test_kernel_double_integral_positive(self):
        val = rbf_kernel_double_integral(domain=(0.0, 5.0))
        assert val > 0.0

    def test_gpq_returns_scalars(self):
        rng = np.random.default_rng(42)
        X = np.linspace(0, np.pi, 20).reshape(-1, 1)
        y = np.sin(X.ravel()) + rng.normal(0, 0.01, 20)
        mean, var = gpq_integral(X, y, domain=(0.0, np.pi), noise_var=0.01 ** 2)
        assert np.isfinite(mean)
        assert np.isfinite(var)

    def test_gpq_sin_integral_close_to_exact(self):
        """∫_0^π sin(x) dx = 2.  GPQ should get reasonably close."""
        rng = np.random.default_rng(42)
        X = np.linspace(0, np.pi, 30).reshape(-1, 1)
        y = np.sin(X.ravel()) + rng.normal(0, 0.01, 30)
        mean, var = gpq_integral(X, y, domain=(0.0, np.pi), noise_var=0.01 ** 2)
        assert abs(mean - 2.0) < 0.3  # generous tolerance

    def test_gpq_variance_nonnegative(self):
        rng = np.random.default_rng(42)
        X = np.linspace(0, 5, 20).reshape(-1, 1)
        y = np.exp(-X.ravel()) + rng.normal(0, 0.01, 20)
        _, var = gpq_integral(X, y, domain=(0.0, 5.0), noise_var=0.01 ** 2)
        assert var >= -1e-10


# ---------------------------------------------------------------------------
# GPQ vs HSGP-BQ consistency
# ---------------------------------------------------------------------------

class TestGPQvsHSGP:
    """Classical GPQ and HSGP-BQ should agree on smooth functions."""

    def test_sin_integral_both_close(self):
        rng = np.random.default_rng(42)
        X = np.linspace(0.01, np.pi, 30).reshape(-1, 1)
        y = np.sin(X.ravel()) + rng.normal(0, 0.01, 30)
        nv = 0.01 ** 2
        domain = (0.01, np.pi)

        gpq_mean, _ = gpq_integral(X, y, domain, noise_var=nv)
        hsgp_mean, _ = hsgp_integral(X, y, domain, M=15, L=3.0, noise_var=nv)

        # Both should be within 0.5 of exact value 2.0
        assert abs(gpq_mean - 2.0) < 0.5
        assert abs(hsgp_mean - 2.0) < 0.5


# ---------------------------------------------------------------------------
# Insurance experiment end-to-end test
# ---------------------------------------------------------------------------

class TestInsuranceExperiment:
    """Quick smoke test of the full experiment pipeline (classical only)."""

    def test_pareto_ordinary_deductible(self):
        from insurance_experiments import run_experiment

        result = run_experiment(
            dist_name="pareto",
            dist_params={"shape": 3.0, "scale": 1.0},
            payoff_name="ordinary_deductible",
            payoff_params={"D": 1.0},
            N=16, M=8,
            run_quantum=False,
            seed=42,
        )
        assert np.isfinite(result["exact"])
        assert np.isfinite(result["gpq_mean"])
        assert np.isfinite(result["hsgp_mean"])
        # All methods should produce a non-negative expected payoff
        assert result["exact"] >= 0.0

    def test_lognormal_policy_limit(self):
        from insurance_experiments import run_experiment

        result = run_experiment(
            dist_name="lognormal",
            dist_params={"mu": 0.0, "sigma": 1.0},
            payoff_name="policy_limit",
            payoff_params={"U": 5.0},
            N=20, M=10,
            run_quantum=False,
            seed=42,
        )
        assert np.isfinite(result["exact"])
        assert result["exact"] > 0.0
