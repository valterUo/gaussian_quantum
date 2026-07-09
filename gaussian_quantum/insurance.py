"""Insurance claim severity distributions and payoff functions.

Provides the building blocks for computing expected insurance payoffs

    I = E[Π(Z)] = ∫ Π(z) f_Z(z) dz

where f_Z is the claim severity density and Π is the payoff (contract) function.

Distributions
-------------
Pareto, Lognormal, Gamma, Weibull, and Poisson (continuous relaxation).

Payoff functions
----------------
Ordinary deductible, franchise deductible, policy limit,
deductible with policy limit, and stop-loss.
"""

import numpy as np
from scipy import stats, integrate


# ---------------------------------------------------------------------------
# Claim severity distributions
# ---------------------------------------------------------------------------

DISTRIBUTIONS = {
    "pareto": {
        "scipy": lambda p: stats.pareto(b=p["shape"], scale=p.get("scale", 1.0)),
        "default_params": {"shape": 3.0, "scale": 1.0},
    },
    "lognormal": {
        "scipy": lambda p: stats.lognorm(s=p["sigma"], scale=np.exp(p.get("mu", 0.0))),
        "default_params": {"mu": 0.0, "sigma": 1.0},
    },
    "gamma": {
        "scipy": lambda p: stats.gamma(a=p["shape"], scale=1.0 / p.get("rate", 1.0)),
        "default_params": {"shape": 2.0, "rate": 1.0},
    },
    "weibull": {
        "scipy": lambda p: stats.weibull_min(c=p["shape"], scale=p.get("scale", 1.0)),
        "default_params": {"shape": 1.5, "scale": 2.0},
    },
    "poisson": {
        "scipy": lambda p: stats.poisson(mu=p["lam"]),
        "default_params": {"lam": 5.0},
    }
}


# Upper truncation quantile of the integration domain.  The BQ target is the
# *truncated* integral; the discarded tail ∫_b^∞ Π f dz is computed by
# tail_mass() and reported alongside the results (0.6–3.3 % of the full
# expectation for the default distributions).  Raising the quantile shrinks
# the tail but widens the domain, which degrades the HSGP basis resolution
# and the length-scale heuristics far more than the tail gains — 0.999 is
# the better trade-off; see figures/stats tables for the reported tails.
TRUNCATION_QUANTILE = 0.999


def get_distribution(name, params=None, q=TRUNCATION_QUANTILE):
    """Return a frozen scipy distribution and a finite integration domain.

    For continuous distributions the domain is [support_lower, q-quantile].
    For Poisson (discrete), the domain is (-1/2, K + 1/2) with K = ppf(q),
    i.e. unit bins centred on the integer support points, matching the
    midpoint relaxation in :func:`distribution_pdf` / :func:`make_integrand`.

    Args:
        name: One of 'pareto', 'lognormal', 'gamma', 'weibull', 'poisson'.
        params: Dict of distribution parameters.  Uses defaults if *None*.
        q: Truncation quantile for the upper integration limit.

    Returns:
        dist: Frozen scipy distribution (continuous or discrete).
        domain: (a, b) finite integration limits.
    """
    if name not in DISTRIBUTIONS:
        raise ValueError(
            f"Unknown distribution '{name}'. "
            f"Choose from {list(DISTRIBUTIONS.keys())}."
        )
    spec = DISTRIBUTIONS[name]
    params = params or spec["default_params"]
    dist = spec["scipy"](params)

    if name == "poisson":
        K = float(dist.ppf(q))
        if K < 1.0:
            K = 20.0
        # Bins [k - 1/2, k + 1/2) centred on the integers 0 … K, so that the
        # midpoint relaxation integrates to Σ_k Π(k) pmf(k) exactly.
        a, b = -0.5, K + 0.5
    else:
        a = float(dist.support()[0])
        b = float(dist.ppf(q))
        # Ensure a finite upper bound
        if not np.isfinite(b) or b > 1e6:
            b = float(dist.ppf(0.99))
        if not np.isfinite(b):
            b = 100.0
    return dist, (a, b)


def _nearest_integer(z):
    """Round to the nearest integer with half-up ties (bin midpoints)."""
    return np.floor(np.asarray(z, dtype=float) + 0.5)


def distribution_pdf(dist, name):
    """Return a callable PDF for *dist*.

    For discrete Poisson, returns the piecewise-constant relaxation
    f(z) = pmf(k) for z in the unit bin [k - 1/2, k + 1/2) centred on the
    integer k, so that ∫ f(z) dz = Σ_k pmf(k) over the centred domain.

    Args:
        dist: Frozen scipy distribution.
        name: Distribution name string.

    Returns:
        Callable f(z) -> ndarray.
    """
    if name == "poisson":
        def _poisson_pdf(z):
            k = np.maximum(_nearest_integer(z), 0.0)
            return dist.pmf(k.astype(int)).astype(float)
        return _poisson_pdf
    else:
        return dist.pdf


# ---------------------------------------------------------------------------
# Payoff (contract) functions
# ---------------------------------------------------------------------------

def ordinary_deductible(z, D):
    """Π(z) = max(z − D, 0)."""
    return np.maximum(np.asarray(z, dtype=float) - D, 0.0)


def franchise_deductible(z, D):
    """Π(z) = 0 if z ≤ D, else z."""
    z = np.asarray(z, dtype=float)
    return np.where(z > D, z, 0.0)


def policy_limit(z, U):
    """Π(z) = min(z, U)."""
    return np.minimum(np.asarray(z, dtype=float), U)


def deductible_with_limit(z, D, U):
    """Π(z) = min(max(z − D, 0), U)."""
    return np.minimum(np.maximum(np.asarray(z, dtype=float) - D, 0.0), U)


def stop_loss(z, D):
    """Π(z) = max(z − D, 0).  (Alias for ordinary deductible on aggregate.)"""
    return np.maximum(np.asarray(z, dtype=float) - D, 0.0)


PAYOFFS = {
    "ordinary_deductible": ordinary_deductible,
    "franchise_deductible": franchise_deductible,
    "policy_limit": policy_limit,
    "deductible_with_limit": deductible_with_limit,
    "stop_loss": stop_loss,
}

# Default parameters for each payoff type (D = deductible, U = limit)
PAYOFF_DEFAULTS = {
    "ordinary_deductible": {"D": 1.0},
    "franchise_deductible": {"D": 1.0},
    "policy_limit": {"U": 5.0},
    "deductible_with_limit": {"D": 1.0, "U": 5.0},
    "stop_loss": {"D": 1.0},
}


# ---------------------------------------------------------------------------
# Integrand factory
# ---------------------------------------------------------------------------

def make_integrand(dist_name, dist_params=None,
                   payoff_name="ordinary_deductible", payoff_params=None,
                   q=TRUNCATION_QUANTILE):
    """Build the integrand g(z) = Π(z) · f_Z(z) and its integration domain.

    For continuous distributions g(z) = Π(z) f(z).  For the discrete Poisson
    distribution both factors are evaluated at the midpoint k of the unit bin
    [k - 1/2, k + 1/2) containing z:

        g(z) = Π(k) pmf(k),   k = round(z),

    so that ∫_{-1/2}^{K+1/2} g(z) dz = Σ_{k=0}^{K} Π(k) pmf(k) *exactly* —
    the continuous relaxation and the discrete expectation coincide up to
    tail truncation, for arbitrary payoff functions.

    Args:
        dist_name: Distribution name (e.g. 'pareto').
        dist_params: Dict of distribution parameters (or None for defaults).
        payoff_name: Payoff function name (e.g. 'ordinary_deductible').
        payoff_params: Dict of payoff parameters (or None for defaults).
        q: Truncation quantile for the upper integration limit.

    Returns:
        integrand: Callable g(z) → ndarray.
        domain: (a, b) finite integration limits.
        dist: Frozen scipy distribution.
    """
    dist, domain = get_distribution(dist_name, dist_params, q=q)
    payoff_fn = PAYOFFS[payoff_name]
    payoff_params = payoff_params or PAYOFF_DEFAULTS[payoff_name]

    if dist_name == "poisson":
        def integrand(z):
            k = np.maximum(_nearest_integer(z), 0.0)
            return payoff_fn(k, **payoff_params) * dist.pmf(k.astype(int))
    else:
        pdf = distribution_pdf(dist, dist_name)

        def integrand(z):
            return payoff_fn(z, **payoff_params) * pdf(z)

    return integrand, domain, dist


def quad_breakpoints(dist_name, domain, payoff_params=None):
    """Breakpoints where the integrand is non-smooth, for adaptive quadrature.

    Returns the payoff kink locations (deductible D, limit U) inside the
    domain and, for Poisson, the half-integer bin edges of the
    piecewise-constant relaxation.

    Args:
        dist_name: Distribution name string.
        domain: (a, b) integration limits.
        payoff_params: Payoff parameter dict (values are kink locations).

    Returns:
        Sorted list of interior breakpoints, or None if there are none.
    """
    a, b = float(domain[0]), float(domain[1])
    pts = []
    if dist_name == "poisson":
        pts.extend(np.arange(np.floor(a) + 0.5, b, 1.0))
    if payoff_params:
        pts.extend(float(v) for v in payoff_params.values())
    pts = sorted({p for p in pts if a < p < b})
    return pts or None


def exact_integral(integrand, domain, points=None):
    """Compute ∫_a^b g(z) dz via adaptive quadrature.

    Args:
        integrand: Callable g(z).
        domain: (a, b) integration limits.
        points: Optional interior breakpoints (kinks/discontinuities) to
            pass to the adaptive quadrature, see :func:`quad_breakpoints`.

    Returns:
        result: Scalar integral value.
        error: Estimated absolute error.
    """
    result, error = integrate.quad(integrand, domain[0], domain[1],
                                   limit=200, points=points)
    return float(result), float(error)


def tail_mass(integrand, domain, dist_name=None, max_terms=400):
    """Integral of *integrand* beyond the truncated domain, ∫_b^∞ g(z) dz.

    Quantifies how much of the untruncated expectation E[Π(Z)] the finite
    domain discards, so the truncation error can be reported alongside the
    method errors.  For Poisson the integrand is piecewise constant on unit
    bins, so the remainder is the discrete sum Σ_{k>K} Π(k) pmf(k).

    Args:
        integrand: Callable g(z) from :func:`make_integrand`.
        domain: (a, b) truncated integration limits.
        dist_name: Distribution name; 'poisson' switches to the discrete sum.
        max_terms: Number of tail terms to sum in the Poisson case.

    Returns:
        Scalar tail remainder (non-negative for non-negative payoffs).
    """
    b = float(domain[1])
    if dist_name == "poisson":
        # Domain ends at b = K + 1/2, so the first fully-excluded bin is K+1.
        k_start = np.floor(b) + 1.0
        ks = k_start + np.arange(max_terms)
        return float(np.sum(integrand(ks)))
    remainder, _ = integrate.quad(integrand, b, np.inf, limit=400)
    return float(remainder)
