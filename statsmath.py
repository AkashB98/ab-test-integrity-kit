"""Small statistical primitives, stdlib only.

The centrepiece is the regularised incomplete gamma function, because
the chi-square survival function behind the SRM test is

    P(chi2_df >= x) = Q(df/2, x/2)

where Q is the regularised *upper* incomplete gamma. There is no
two-arm shortcut anywhere: the same gamma code serves a 2-arm test
(df=1), a 3-arm test (df=2), and any other arm count.

Implementation follows Numerical Recipes: a series expansion for
P(a, x) when x < a + 1, and a Lentz continued fraction for Q(a, x)
otherwise. Both branches share one code path for every df.
"""

from __future__ import annotations

import math
from statistics import NormalDist

_NORM = NormalDist()
_EPS = 3e-14
_FPMIN = 1e-300
_MAXIT = 1000


def _gamma_p_series(a: float, x: float) -> float:
    """Regularised lower incomplete gamma P(a, x) via series."""
    if x == 0.0:
        return 0.0
    ap = a
    term = 1.0 / a
    total = term
    for _ in range(_MAXIT):
        ap += 1.0
        term *= x / ap
        total += term
        if abs(term) < abs(total) * _EPS:
            break
    return total * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _gamma_q_cf(a: float, x: float) -> float:
    """Regularised upper incomplete gamma Q(a, x) via continued fraction."""
    b = x + 1.0 - a
    c = 1.0 / _FPMIN
    d = 1.0 / b
    h = d
    for i in range(1, _MAXIT + 1):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = b + an / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPS:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def gammainc_reg(a: float, x: float) -> float:
    """Regularised lower incomplete gamma P(a, x), in [0, 1]."""
    if a <= 0:
        raise ValueError("a must be > 0")
    if x < 0:
        raise ValueError("x must be >= 0")
    if x == 0.0:
        return 0.0
    if x < a + 1.0:
        return _gamma_p_series(a, x)
    return 1.0 - _gamma_q_cf(a, x)


def gammaincc_reg(a: float, x: float) -> float:
    """Regularised upper incomplete gamma Q(a, x) = 1 - P(a, x)."""
    if a <= 0:
        raise ValueError("a must be > 0")
    if x < 0:
        raise ValueError("x must be >= 0")
    if x == 0.0:
        return 1.0
    if x < a + 1.0:
        return 1.0 - _gamma_p_series(a, x)
    return _gamma_q_cf(a, x)


def chi2_sf(x: float, df: int) -> float:
    """Survival function P(chi2_df >= x) for a chi-square statistic."""
    if df < 1:
        raise ValueError("df must be >= 1")
    if x < 0:
        raise ValueError("x must be >= 0")
    if x == 0.0:
        return 1.0
    return gammaincc_reg(df / 2.0, x / 2.0)


def normal_cdf(x: float) -> float:
    return _NORM.cdf(x)


def normal_sf(x: float) -> float:
    return 1.0 - _NORM.cdf(x)


def z_value(prob: float) -> float:
    """Standard-normal quantile (inverse CDF) via statistics.NormalDist."""
    if not 0.0 < prob < 1.0:
        raise ValueError("prob must be in (0, 1)")
    return _NORM.inv_cdf(prob)


def two_sided_p_from_z(z: float) -> float:
    return 2.0 * normal_sf(abs(z))
