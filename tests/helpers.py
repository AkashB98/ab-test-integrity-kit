"""Shared row builders for the test suite (not a test module)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def row(user_id="user_00001", arm="control", covariate_pre=50.0,
        outcome=50.0, guardrail_error=0):
    return {"user_id": user_id, "arm": arm,
            "covariate_pre": covariate_pre, "outcome": outcome,
            "guardrail_error": guardrail_error}


def two_arm(n_per_arm=100, effect=5.0, guardrail_lift=0):
    """Deterministic hand-built two-arm dataset (no randomness)."""
    rows = []
    i = 0
    for arm in ("control", "treatment"):
        for j in range(n_per_arm):
            i += 1
            x = 40.0 + (j % 20)
            y = 50.0 + 0.5 * (x - 50.0) + (effect if arm == "treatment" else 0.0)
            g = 1 if (arm == "treatment" and j < guardrail_lift) else 0
            rows.append(row(f"user_{i:05d}", arm, x, y, g))
    return rows
