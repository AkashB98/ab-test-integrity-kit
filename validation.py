"""Data-quality gate for experiment rows.

Reports issues, never raises on bad data, so the CLI ``check`` gate
can map data problems to exit code 2 — distinct from an integrity
refusal (exit 1) and from a pass (exit 0).
"""

from __future__ import annotations

from typing import Dict, List, Sequence

REQUIRED_FIELDS = ("user_id", "arm", "covariate_pre", "outcome",
                   "guardrail_error")


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def validate_row(row, index: int = 0) -> List[str]:
    issues: List[str] = []
    if not isinstance(row, dict):
        return [f"row {index}: not an object"]
    for field in REQUIRED_FIELDS:
        if field not in row:
            issues.append(f"row {index}: missing field {field!r}")
    if issues:
        return issues
    if not isinstance(row["user_id"], str) or not row["user_id"]:
        issues.append(f"row {index}: user_id must be a non-empty string")
    if not isinstance(row["arm"], str) or not row["arm"]:
        issues.append(f"row {index}: arm must be a non-empty string")
    for field in ("covariate_pre", "outcome"):
        if not _is_number(row[field]):
            issues.append(f"row {index}: {field} must be a number")
    g = row["guardrail_error"]
    if isinstance(g, bool) or g not in (0, 1):
        issues.append(f"row {index}: guardrail_error must be 0 or 1")
    return issues


def validate_dataset(rows: Sequence[Dict]) -> List[str]:
    issues: List[str] = []
    if not rows:
        return ["dataset: no rows"]
    seen = set()
    arms = set()
    for i, row in enumerate(rows):
        issues.extend(validate_row(row, i))
        if isinstance(row, dict):
            uid = row.get("user_id")
            if uid in seen:
                issues.append(f"row {i}: duplicate user_id {uid!r}")
            seen.add(uid)
            if isinstance(row.get("arm"), str):
                arms.add(row["arm"])
        if len(issues) > 50:
            issues.append("dataset: too many issues, truncated")
            break
    if len(arms) < 2 and not issues:
        issues.append("dataset: fewer than 2 arms present")
    return issues
