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


def get_distribution(name, params=None):
    """Return a frozen scipy distribution and a finite integration domain.

    For continuous distributions the domain is [support_lower, q_{0.999}].
    For Poisson (discrete), a piecewise-constant continuous PDF is returned
    alongside an integer-valued domain suitable for midpoint quadrature.

    Args:
        name: One of 'pareto', 'lognormal', 'gamma', 'weibull', 'poisson'.
        params: Dict of distribution parameters.  Uses defaults if *None*.

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
        a = 0.0
        b = float(dist.ppf(0.999))
        if b < 1.0:
            b = 20.0
    else:
        a = float(dist.support()[0])
        b = float(dist.ppf(0.999))
        # Ensure a finite upper bound
        if not np.isfinite(b) or b > 1e6:
            b = float(dist.ppf(0.99))
        if not np.isfinite(b):
            b = 100.0
    return dist, (a, b)


def distribution_pdf(dist, name):
    """Return a callable PDF for *dist*.

    For discrete Poisson, returns a piecewise-constant function that
    interpolates the PMF so that ∫ f(z) dz ≈ 1 over the domain.

    Args:
        dist: Frozen scipy distribution.
        name: Distribution name string.

    Returns:
        Callable f(z) -> ndarray.
    """
    if name == "poisson":
        def _poisson_pdf(z):
            z = np.asarray(z, dtype=float)
            return dist.pmf(np.floor(z).astype(int)).astype(float)
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
                   payoff_name="ordinary_deductible", payoff_params=None):
    """Build the integrand g(z) = Π(z) · f_Z(z) and its integration domain.

    Args:
        dist_name: Distribution name (e.g. 'pareto').
        dist_params: Dict of distribution parameters (or None for defaults).
        payoff_name: Payoff function name (e.g. 'ordinary_deductible').
        payoff_params: Dict of payoff parameters (or None for defaults).

    Returns:
        integrand: Callable g(z) → ndarray.
        domain: (a, b) finite integration limits.
        dist: Frozen scipy distribution.
    """
    dist, domain = get_distribution(dist_name, dist_params)
    pdf = distribution_pdf(dist, dist_name)

    payoff_fn = PAYOFFS[payoff_name]
    payoff_params = payoff_params or PAYOFF_DEFAULTS[payoff_name]

    def integrand(z):
        return payoff_fn(z, **payoff_params) * pdf(z)

    return integrand, domain, dist


def exact_integral(integrand, domain):
    """Compute ∫_a^b g(z) dz via adaptive quadrature.

    Args:
        integrand: Callable g(z).
        domain: (a, b) integration limits.

    Returns:
        result: Scalar integral value.
        error: Estimated absolute error.
    """
    result, error = integrate.quad(integrand, domain[0], domain[1],
                                   limit=200)
    return float(result), float(error)
