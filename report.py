"""Full report assembly: SRM + peeking calibration + CUPED + the
gated readouts of all three planted scenarios, one document."""

from __future__ import annotations

from typing import Dict, List, Sequence

from cuped import apply_cuped
from generator import generate_scenario
from peeking import simulate_peeking
from readout import evaluate_readout
from srm import srm_check


def build_report(rows: Sequence[Dict] | None = None) -> Dict:
    clean = list(rows) if rows is not None else generate_scenario("clean")
    srm_rows = generate_scenario("srm")
    guard_rows = generate_scenario("guardrail")

    cov = [r["covariate_pre"] for r in clean]
    out = [r["outcome"] for r in clean]
    cuped = apply_cuped(cov, out)
    cuped_summary = {k: cuped[k] for k in
                     ("theta", "rho", "rho_squared", "variance_before",
                      "variance_after", "variance_reduction")}

    from generator import arm_counts
    clean_counts = arm_counts(clean)
    arms = list(clean_counts.keys())
    multi = srm_check([1200, 1150, 1650], [1 / 3, 1 / 3, 1 / 3])

    return {
        "dataset": {
            "rows": len(clean),
            "counts": clean_counts,
            "synthetic": True,
            "company": "Northwind Pantry (fictional)",
        },
        "srm_two_arm_clean": srm_check([clean_counts[a] for a in arms],
                                       [0.5, 0.5]),
        "srm_multi_arm_planted": multi,
        "peeking": simulate_peeking(),
        "cuped": cuped_summary,
        "readouts": {
            "clean": evaluate_readout(clean),
            "clean_peeked_uncorrected": evaluate_readout(
                clean, looks=5, peeking_corrected=False),
            "srm": evaluate_readout(srm_rows),
            "guardrail": evaluate_readout(guard_rows),
        },
    }
