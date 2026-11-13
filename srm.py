"""Sample-ratio-mismatch (SRM) detection.

An **SRM** — sample-ratio mismatch, when the split of users across
arms is far from the split the experiment was configured with — is
almost never bad luck. It means the randomiser, a redirect, a bot
filter, or a logging join treated the arms differently, so the arms
are no longer comparable and no effect estimate from the experiment
can be trusted until the cause is found.

Test: Pearson's chi-square goodness-of-fit of the observed arm counts
against the expected counts implied by the configured proportions,
with df = (number of arms - 1) and the p-value from the chi-square
survival function in ``statsmath`` (regularised incomplete gamma —
one code path for two arms, three arms, or more).

Threshold: p < 0.001. The industry convention (used by major
experimentation platforms) is deliberately far stricter than 0.05:
an SRM check runs on *every* experiment, so at 0.05 one in twenty
healthy experiments would be flagged and the alarm would be ignored;
at 0.001 a flag is rare enough to stop the line for — and a genuine
assignment bug typically produces a mismatch so large that the
stricter line still catches it instantly. See README.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

from statsmath import chi2_sf

SRM_ALPHA = 0.001


def srm_check(observed: Sequence[int],
              expected_proportions: Sequence[float],
              alpha: float = SRM_ALPHA) -> Dict:
    """Chi-square SRM test for any arm count >= 2.

    ``observed`` — per-arm user counts.
    ``expected_proportions`` — the configured split (need not sum to
    exactly 1; it is normalised, but must be positive per arm).
    """
    if len(observed) != len(expected_proportions):
        raise ValueError("observed and expected_proportions length mismatch")
    if len(observed) < 2:
        raise ValueError("need at least 2 arms")
    if any((not isinstance(o, int)) or isinstance(o, bool) or o < 0
           for o in observed):
        raise ValueError("observed counts must be non-negative ints")
    total = sum(observed)
    if total <= 0:
        raise ValueError("observed counts must sum to > 0")
    if any(p <= 0 for p in expected_proportions):
        raise ValueError("expected proportions must be > 0")
    prop_sum = float(sum(expected_proportions))
    props = [p / prop_sum for p in expected_proportions]

    chi2 = 0.0
    expected_counts: List[float] = []
    for obs, prop in zip(observed, props):
        exp = total * prop
        expected_counts.append(round(exp, 4))
        chi2 += (obs - exp) ** 2 / exp
    df = len(observed) - 1
    p_value = chi2_sf(chi2, df)
    return {
        "observed": list(observed),
        "expected_counts": expected_counts,
        "expected_proportions": [round(p, 6) for p in props],
        "chi2": round(chi2, 6),
        "df": df,
        "p_value": p_value,
        "alpha": alpha,
        "srm_detected": bool(p_value < alpha),
    }


def srm_from_counts(counts: Dict[str, int],
                    expected_proportions: Dict[str, float],
                    alpha: float = SRM_ALPHA) -> Dict:
    """SRM test keyed by arm name; arm order follows ``counts``."""
    arms = list(counts.keys())
    if set(arms) != set(expected_proportions.keys()):
        raise ValueError("counts and expected_proportions arms mismatch")
    result = srm_check([counts[a] for a in arms],
                       [expected_proportions[a] for a in arms], alpha=alpha)
    result["arms"] = arms
    return result
