"""Peeking: repeated looks at an A/A test inflate false positives,
and an O'Brien-Fleming correction brings the rate back to alpha.

**Peeking** — checking a running experiment's p-value repeatedly and
stopping when it first looks significant — quietly multiplies the
false-positive rate: each look is another chance for noise to cross
the line. With 5 looks at the usual 0.05 line, a null (A/A) test
"finds an effect" roughly 14% of the time instead of 5%.

The fix used here is **alpha spending** — pre-committing how much of
the total false-positive budget (alpha) may be spent at each look.
The O'Brien-Fleming shape spends almost nothing early (when the data
are thin and noise dominates) and almost everything at the final
look, so early stopping requires overwhelming evidence while the
final-look bar stays close to the familiar 1.96. The critical z
values below are the standard tabulated O'Brien-Fleming boundaries
for two-sided alpha = 0.05 with equally spaced looks (Jennison &
Turnbull); each is converted to a per-look nominal p-value with
``statistics.NormalDist`` (via ``statsmath``). For look counts other
than 1-5, a Lan-DeMets-style approximation ``z / sqrt(t)`` is used
and labelled as such.

The Monte Carlo is a seeded A/A simulation. Under the null the
two-arm difference, standardised per look, is a cumulative sum of
independent standard-normal increments divided by sqrt(k) — exactly
the joint distribution of sequential z-statistics under the normal
approximation — so a simulation needs only ``random.Random.gauss``
increments and runs in milliseconds while exhibiting the true
peeking behaviour (correlated looks, not independent ones).
"""

from __future__ import annotations

import random
from typing import Dict, List

from statsmath import two_sided_p_from_z, z_value

DEFAULT_ALPHA = 0.05
DEFAULT_LOOKS = 5
DEFAULT_SIMS = 2000
DEFAULT_SEED = 7

# Standard O'Brien-Fleming critical z values, two-sided alpha=0.05,
# equally spaced looks (Jennison & Turnbull tables).
_OBF_Z = {
    1: (1.959964,),
    2: (2.962593, 1.968572),
    3: (3.710301, 2.511442, 1.992739),
    4: (4.332720, 2.963095, 2.359012, 2.013710),
    5: (4.561755, 3.225559, 2.633935, 2.280593, 2.039549),
}


def obf_boundaries(looks: int, alpha: float = DEFAULT_ALPHA) -> Dict:
    """O'Brien-Fleming boundaries for ``looks`` equally spaced looks.

    Returns critical z values and the per-look nominal two-sided
    p-value thresholds (from ``statistics.NormalDist``). For
    alpha != 0.05 the z values are scaled by z_{1-alpha/2} / z_{0.975},
    an approximation that is exact at alpha = 0.05.
    """
    if looks < 1:
        raise ValueError("looks must be >= 1")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    scale = z_value(1.0 - alpha / 2.0) / z_value(0.975)
    if looks in _OBF_Z:
        zs = [round(z * scale, 6) for z in _OBF_Z[looks]]
        source = "tabulated"
    else:
        base = z_value(1.0 - alpha / 2.0)
        zs = [round(base / ((k + 1) / looks) ** 0.5, 6) for k in range(looks)]
        source = "lan-demets-approx"
    return {
        "looks": looks,
        "alpha": alpha,
        "critical_z": zs,
        "nominal_p": [round(two_sided_p_from_z(z), 6) for z in zs],
        "source": source,
    }


def simulate_peeking(n_sims: int = DEFAULT_SIMS, looks: int = DEFAULT_LOOKS,
                     alpha: float = DEFAULT_ALPHA,
                     seed: int = DEFAULT_SEED) -> Dict:
    """Seeded A/A Monte Carlo: false-positive rate, uncorrected vs
    O'Brien-Fleming-corrected, over the same simulated experiments."""
    if n_sims < 1:
        raise ValueError("n_sims must be >= 1")
    bounds = obf_boundaries(looks, alpha)
    crit = bounds["critical_z"]
    rng = random.Random(seed)
    fp_raw = 0
    fp_corrected = 0
    for _ in range(n_sims):
        cumsum = 0.0
        raw_hit = False
        corr_hit = False
        for k in range(looks):
            cumsum += rng.gauss(0.0, 1.0)
            z = cumsum / ((k + 1) ** 0.5)
            if not raw_hit and two_sided_p_from_z(z) < alpha:
                raw_hit = True
            if not corr_hit and abs(z) > crit[k]:
                corr_hit = True
        fp_raw += raw_hit
        fp_corrected += corr_hit
    return {
        "n_sims": n_sims,
        "looks": looks,
        "alpha": alpha,
        "seed": seed,
        "uncorrected_false_positives": fp_raw,
        "uncorrected_fp_rate": round(fp_raw / n_sims, 6),
        "corrected_false_positives": fp_corrected,
        "corrected_fp_rate": round(fp_corrected / n_sims, 6),
        "boundaries": bounds,
    }


def peeking_verdict(looks: int, corrected: bool) -> Dict:
    """Gate helper: was repeated looking accounted for?"""
    return {
        "looks": looks,
        "corrected": bool(corrected),
        "uncorrected_peeking": bool(looks > 1 and not corrected),
    }
