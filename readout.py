"""The readout — and the refusal gate that makes this a kit.

A readout answers "can we ship the treatment?" Most experiment
notebooks print an effect and a p-value and leave the integrity
questions to the reader's discipline. This readout refuses:

  REFUSAL 1 — SRM fires (p < 0.001): the arms are not comparable;
              the effect estimate is not evidence about anything.
  REFUSAL 2 — the experiment was peeked at (looks > 1) without an
              alpha-spending correction: the quoted p-value understates
              the false-positive risk by roughly 3x at 5 looks.
  REFUSAL 3 — a **guardrail metric** (a metric that must not get
              worse, here the checkout error rate) breaches its
              margin in a treatment arm: a real primary lift bought
              with broken checkouts is not a win.

Only when no refusal fires does the primary test decide:
SHIP (positive, significant at alpha) or INCONCLUSIVE.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence

from cuped import apply_cuped, cuped_theta
from peeking import peeking_verdict
from srm import srm_from_counts
from statsmath import two_sided_p_from_z

DEFAULT_ALPHA = 0.05
DEFAULT_GUARDRAIL_MARGIN = 0.01


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs)


def _sample_var(xs: Sequence[float], m: float) -> float:
    if len(xs) < 2:
        return 0.0
    return sum((x - m) ** 2 for x in xs) / (len(xs) - 1)


def two_sample_z(a: Sequence[float], b: Sequence[float]) -> Dict:
    """Welch-style two-sample z on means (b minus a)."""
    ma, mb = _mean(a), _mean(b)
    va, vb = _sample_var(a, ma), _sample_var(b, mb)
    se = math.sqrt(va / len(a) + vb / len(b))
    diff = mb - ma
    if se > 0:
        z = diff / se
    elif diff == 0.0:
        z = 0.0
    else:
        # Zero within-arm variance with a non-zero difference (a
        # perfectly linear hand-built dataset after CUPED) is the
        # strongest possible evidence, not p=1 — caught by test.
        z = math.copysign(math.inf, diff)
    return {"mean_a": round(ma, 6), "mean_b": round(mb, 6),
            "difference": round(diff, 6), "se": round(se, 6),
            "z": round(z, 6), "p_value": two_sided_p_from_z(z)}


def evaluate_readout(rows: Sequence[Dict],
                     expected_proportions: Dict[str, float] | None = None,
                     looks: int = 1, peeking_corrected: bool = True,
                     guardrail_margin: float = DEFAULT_GUARDRAIL_MARGIN,
                     alpha: float = DEFAULT_ALPHA,
                     use_cuped: bool = True) -> Dict:
    """Full integrity readout over per-user rows.

    Arms are ordered by first appearance; arm[0] is the control and
    arm[1] the primary treatment. ``expected_proportions`` defaults
    to an equal split across the arms present.
    """
    arms: List[str] = []
    for r in rows:
        if r["arm"] not in arms:
            arms.append(r["arm"])
    if len(arms) < 2:
        raise ValueError("readout needs at least 2 arms")
    # Control is the baseline wherever it appears: ordering arms by
    # first appearance once flipped the effect's sign when the first
    # user happened to land in treatment (caught by demo stage 5).
    if "control" in arms:
        arms.sort(key=lambda a: a != "control")
    if expected_proportions is None:
        expected_proportions = {a: 1.0 / len(arms) for a in arms}

    counts = {a: 0 for a in arms}
    by_arm: Dict[str, List[Dict]] = {a: [] for a in arms}
    for r in rows:
        counts[r["arm"]] += 1
        by_arm[r["arm"]].append(r)

    srm = srm_from_counts(counts, expected_proportions)
    peek = peeking_verdict(looks, peeking_corrected)

    # Guardrail: worst treatment-arm error-rate excess over control.
    rates = {a: _mean([r["guardrail_error"] for r in by_arm[a]]) for a in arms}
    control_rate = rates[arms[0]]
    worst_excess = max(rates[a] - control_rate for a in arms[1:])
    guardrail = {
        "rates": {a: round(rates[a], 6) for a in arms},
        "margin": guardrail_margin,
        "worst_excess": round(worst_excess, 6),
        "breached": bool(worst_excess > guardrail_margin),
    }

    # Primary effect: CUPED-adjusted outcome when requested (theta is
    # estimated on the pooled sample; the covariate is pre-period, so
    # pooling cannot leak the treatment into theta's inputs' timing —
    # it is still reported so the choice is auditable).
    cov_all = [r["covariate_pre"] for r in rows]
    out_all = [r["outcome"] for r in rows]
    cuped_info = None
    if use_cuped:
        theta = cuped_theta(cov_all, out_all)
        adj = apply_cuped(cov_all, out_all, theta=theta)
        cuped_info = {k: adj[k] for k in
                      ("theta", "rho", "rho_squared", "variance_before",
                       "variance_after", "variance_reduction")}
        adjusted = adj["adjusted"]
        values = {a: [] for a in arms}
        for r, y_adj in zip(rows, adjusted):
            values[r["arm"]].append(y_adj)
    else:
        values = {a: [r["outcome"] for r in by_arm[a]] for a in arms}
    effect = two_sample_z(values[arms[0]], values[arms[1]])
    effect["metric"] = "cuped_adjusted_outcome" if use_cuped else "outcome"

    reasons: List[str] = []
    if srm["srm_detected"]:
        reasons.append(
            f"SRM detected (chi2={srm['chi2']}, df={srm['df']}, "
            f"p={srm['p_value']:.3g} < 0.001): assignment is broken, "
            "the arms are not comparable")
    if peek["uncorrected_peeking"]:
        reasons.append(
            f"peeked at {looks} times without an alpha-spending "
            "correction: the quoted p-value understates the "
            "false-positive risk")
    if guardrail["breached"]:
        reasons.append(
            f"guardrail breach: error rate exceeds control by "
            f"{guardrail['worst_excess']:.4f} > margin "
            f"{guardrail_margin}")

    if reasons:
        verdict = "DO_NOT_SHIP"
    elif effect["p_value"] < alpha and effect["difference"] > 0:
        verdict = "SHIP"
    else:
        verdict = "INCONCLUSIVE"

    return {
        "verdict": verdict,
        "refusal_reasons": reasons,
        "arms": arms,
        "counts": counts,
        "srm": srm,
        "peeking": peek,
        "guardrail": guardrail,
        "effect": effect,
        "cuped": cuped_info,
        "alpha": alpha,
    }
