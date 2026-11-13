"""Deterministic synthetic experiment generator — Northwind Pantry (fictional).

SYNTHETIC DATA ONLY. No real company, customer, or revenue data is
ever used. Every row is fabricated from a seeded PRNG so the same
seed always yields the same dataset, byte-for-byte.

One row per user:

  user_id, arm, covariate_pre, outcome, guardrail_error

- ``covariate_pre`` — the user's pre-period spend (the CUPED
  covariate), drawn before assignment so treatment cannot touch it.
- ``outcome`` — post-period spend. Built as
  ``50 + 0.7 * (X - 50) + noise`` (noise sd = 10*sqrt(1-0.49)), so the
  covariate/outcome correlation is rho = 0.7 by construction and the
  CUPED variance reduction should land near rho^2 = 0.49. Treatment
  adds ``effect`` (default +3.0).
- ``guardrail_error`` — 1 if the user's session hit the guardrail
  error (a checkout failure), else 0. Base rate 0.02 in every arm;
  a scenario may add ``guardrail_lift`` to the treatment arm.

Scenarios plant the integrity failures the kit exists to catch:

- ``clean``     — observed split matches the configured split.
- ``srm``       — observed split is skewed (default 70/30 against a
                  configured 50/50): an assignment bug.
- ``guardrail`` — split is clean, but the treatment arm's guardrail
                  error rate is lifted by +0.03.
"""

from __future__ import annotations

import csv
import io
import json
import math
import random
from typing import Dict, List, Sequence

DEFAULT_SEED = 42
DEFAULT_N_USERS = 4000
DEFAULT_EFFECT = 3.0
DEFAULT_RHO = 0.7
BASE_GUARDRAIL_RATE = 0.02

ROW_FIELDS = ["user_id", "arm", "covariate_pre", "outcome", "guardrail_error"]

SCENARIOS = ("clean", "srm", "guardrail")


def generate_experiment(seed: int = DEFAULT_SEED,
                        n_users: int = DEFAULT_N_USERS,
                        arms: Sequence[str] = ("control", "treatment"),
                        expected_proportions: Sequence[float] = (0.5, 0.5),
                        observed_weights: Sequence[float] | None = None,
                        effect: float = DEFAULT_EFFECT,
                        rho: float = DEFAULT_RHO,
                        guardrail_lift: float = 0.0) -> List[Dict]:
    """Return synthetic per-user rows, deterministic for ``seed``."""
    if n_users < 10:
        raise ValueError("n_users must be >= 10")
    if len(arms) < 2 or len(arms) != len(expected_proportions):
        raise ValueError("arms and expected_proportions must align (>= 2)")
    weights = list(observed_weights) if observed_weights is not None \
        else list(expected_proportions)
    if len(weights) != len(arms) or any(w <= 0 for w in weights):
        raise ValueError("observed_weights must be positive and aligned")
    if not -1.0 < rho < 1.0:
        raise ValueError("rho must be in (-1, 1)")
    wsum = float(sum(weights))
    cum: List[float] = []
    acc = 0.0
    for w in weights:
        acc += w / wsum
        cum.append(acc)

    rng = random.Random(seed)
    noise_sd = 10.0 * math.sqrt(1.0 - rho * rho)
    slope = rho * 10.0 / 10.0  # rho * sigma_Y / sigma_X, both sigmas 10
    rows: List[Dict] = []
    for i in range(n_users):
        u = rng.random()
        arm_idx = next(j for j, c in enumerate(cum) if u <= c)
        x = rng.gauss(50.0, 10.0)
        y = 50.0 + slope * (x - 50.0) + rng.gauss(0.0, noise_sd)
        if arm_idx == 1:
            y += effect
        g_rate = BASE_GUARDRAIL_RATE + (guardrail_lift if arm_idx == 1 else 0.0)
        rows.append({
            "user_id": f"user_{i + 1:05d}",
            "arm": arms[arm_idx],
            "covariate_pre": round(x, 4),
            "outcome": round(y, 4),
            "guardrail_error": 1 if rng.random() < g_rate else 0,
        })
    return rows


def generate_scenario(scenario: str = "clean", seed: int = DEFAULT_SEED,
                      n_users: int = DEFAULT_N_USERS) -> List[Dict]:
    """Generate one of the named planted scenarios (see module docstring)."""
    if scenario == "clean":
        return generate_experiment(seed=seed, n_users=n_users)
    if scenario == "srm":
        return generate_experiment(seed=seed, n_users=n_users,
                                   observed_weights=(0.7, 0.3))
    if scenario == "guardrail":
        return generate_experiment(seed=seed, n_users=n_users,
                                   guardrail_lift=0.03)
    raise ValueError(f"unknown scenario: {scenario!r} (choose from {SCENARIOS})")


def arm_counts(rows: Sequence[Dict]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for r in rows:
        counts[r["arm"]] = counts.get(r["arm"], 0) + 1
    return counts


def rows_to_jsonl(rows: Sequence[Dict]) -> str:
    return "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)


def rows_to_csv(rows: Sequence[Dict]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=ROW_FIELDS)
    writer.writeheader()
    for r in rows:
        writer.writerow(r)
    return buf.getvalue()


def _coerce_row(raw: Dict, lineno: int) -> Dict:
    try:
        return {
            "user_id": str(raw["user_id"]),
            "arm": str(raw["arm"]),
            "covariate_pre": float(raw["covariate_pre"]),
            "outcome": float(raw["outcome"]),
            "guardrail_error": int(raw["guardrail_error"]),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"line {lineno}: bad row {raw!r}: {exc}") from exc


def load_rows(path: str) -> List[Dict]:
    """Load rows from .jsonl, .csv, or a JSON array file."""
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    if path.endswith(".csv"):
        reader = csv.DictReader(io.StringIO(text))
        return [_coerce_row(r, i + 2) for i, r in enumerate(reader)]
    stripped = text.strip()
    if stripped.startswith("["):
        data = json.loads(stripped)
        if not isinstance(data, list):
            raise ValueError("JSON input must be an array of rows")
        return [_coerce_row(r, i + 1) for i, r in enumerate(data)]
    rows = []
    for i, line in enumerate(text.splitlines()):
        if line.strip():
            rows.append(_coerce_row(json.loads(line), i + 1))
    return rows


def write_rows(rows: Sequence[Dict], path: str, fmt: str = "jsonl") -> None:
    text = rows_to_csv(rows) if fmt == "csv" else rows_to_jsonl(rows)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
