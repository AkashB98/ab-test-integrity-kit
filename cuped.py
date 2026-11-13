"""CUPED variance reduction.

**CUPED** — Controlled-experiment Using Pre-Experiment Data — uses a
pre-period **covariate** (a measurement taken *before* the experiment
started, so the treatment cannot have influenced it) to soak up
outcome variance that was never caused by the treatment:

    theta        = Cov(Y, X) / Var(X)
    Y_adjusted   = Y - theta * (X - mean(X))

The adjustment leaves the expected treatment effect unchanged (X has
the same distribution in both arms) but shrinks the outcome variance
by a factor of (1 - rho^2), where rho is the pre-period correlation
between covariate and outcome. At rho = 0.7 that is a 49% variance
reduction — roughly the same precision as doubling the sample, for
free. At rho = 0, theta is 0 and CUPED correctly does nothing.
"""

from __future__ import annotations

from typing import Dict, List, Sequence


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs)


def variance(xs: Sequence[float]) -> float:
    """Population variance (denominator n) — the CUPED convention."""
    if len(xs) < 2:
        raise ValueError("need at least 2 observations")
    m = _mean(xs)
    return sum((x - m) ** 2 for x in xs) / len(xs)


def covariance(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        raise ValueError("paired samples of length >= 2 required")
    mx, my = _mean(xs), _mean(ys)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / len(xs)


def correlation(xs: Sequence[float], ys: Sequence[float]) -> float:
    vx, vy = variance(xs), variance(ys)
    if vx == 0.0 or vy == 0.0:
        raise ValueError("correlation undefined for a constant sample")
    return covariance(xs, ys) / (vx * vy) ** 0.5


def cuped_theta(covariate: Sequence[float], outcome: Sequence[float]) -> float:
    """theta = Cov(Y, X) / Var(X), from pre-period (or pooled) data."""
    vx = variance(covariate)
    if vx == 0.0:
        raise ValueError("covariate is constant; CUPED theta undefined")
    return covariance(covariate, outcome) / vx


def apply_cuped(covariate: Sequence[float], outcome: Sequence[float],
                theta: float | None = None) -> Dict:
    """Adjust outcomes and report the variance reduction achieved."""
    if len(covariate) != len(outcome):
        raise ValueError("covariate and outcome length mismatch")
    if theta is None:
        theta = cuped_theta(covariate, outcome)
    mx = _mean(covariate)
    adjusted: List[float] = [y - theta * (x - mx)
                             for x, y in zip(covariate, outcome)]
    var_raw = variance(outcome)
    var_adj = variance(adjusted)
    rho = correlation(covariate, outcome)
    reduction = 1.0 - var_adj / var_raw if var_raw > 0 else 0.0
    return {
        "theta": round(theta, 6),
        "rho": round(rho, 6),
        "rho_squared": round(rho * rho, 6),
        "variance_before": round(var_raw, 6),
        "variance_after": round(var_adj, 6),
        "variance_reduction": round(reduction, 6),
        "adjusted": adjusted,
    }
