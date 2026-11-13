"""CLI: generate | srm | peeking | cuped | readout | report | check | demo | serve.

SYNTHETIC DATA ONLY - Northwind Pantry is a fictional company and
every dataset handled here is fabricated sample data.

Exit codes:
  check : 0 = pass (readout verdict SHIP or INCONCLUSIVE, no refusal),
          1 = refusal (SRM, uncorrected peeking, or guardrail breach),
          2 = usage / IO / data-quality error
  others: 0 = success, 2 = usage / IO error
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cuped import apply_cuped  # noqa: E402
from generator import (DEFAULT_N_USERS, DEFAULT_SEED, SCENARIOS,  # noqa: E402
                       arm_counts, generate_scenario, load_rows, write_rows)
from peeking import simulate_peeking  # noqa: E402
from readout import evaluate_readout  # noqa: E402
from report import build_report  # noqa: E402
from srm import srm_check  # noqa: E402
from validation import validate_dataset  # noqa: E402

EXIT_OK, EXIT_FAIL, EXIT_ERROR = 0, 1, 2


def canonical(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True)


def _load_or_error(path: str) -> Optional[List[Dict]]:
    try:
        return load_rows(path)
    except FileNotFoundError:
        print(f"error: file not found: {path}", file=sys.stderr)
    except (ValueError, OSError) as exc:
        print(f"error: cannot read {path}: {exc}", file=sys.stderr)
    return None


def _validated_or_error(path: str):
    rows = _load_or_error(path)
    if rows is None:
        return None, EXIT_ERROR
    issues = validate_dataset(rows)
    if issues:
        for issue in issues[:20]:
            print(f"data-quality: {issue}", file=sys.stderr)
        return None, EXIT_ERROR
    return rows, EXIT_OK


def _write_out(args, text: str, summary: str) -> int:
    if getattr(args, "out", None):
        try:
            with open(args.out, "w", encoding="utf-8") as fh:
                fh.write(text)
        except OSError as exc:
            print(f"error: cannot write {args.out}: {exc}", file=sys.stderr)
            return EXIT_ERROR
        print(summary)
    else:
        print(text, end="")
    return EXIT_OK


def _expected(args, arms: List[str]) -> Optional[Dict[str, float]]:
    if not getattr(args, "expected", None):
        return None
    parts = [float(p) for p in args.expected.split(",")]
    if len(parts) != len(arms):
        print("error: --expected must give one proportion per arm",
              file=sys.stderr)
        return {}
    return dict(zip(arms, parts))


def cmd_generate(args) -> int:
    rows = generate_scenario(args.scenario, seed=args.seed, n_users=args.n_users)
    fmt = "csv" if (args.out and args.out.endswith(".csv")) else "jsonl"
    if args.out:
        try:
            write_rows(rows, args.out, fmt=fmt)
        except OSError as exc:
            print(f"error: cannot write {args.out}: {exc}", file=sys.stderr)
            return EXIT_ERROR
        print(f"generated {len(rows)} synthetic rows "
              f"(scenario={args.scenario}, seed={args.seed}) -> {args.out}")
        return EXIT_OK
    from generator import rows_to_csv, rows_to_jsonl
    print(rows_to_csv(rows) if fmt == "csv" else rows_to_jsonl(rows), end="")
    return EXIT_OK


def cmd_srm(args) -> int:
    observed = [int(x) for x in args.observed.split(",")]
    expected = [float(x) for x in args.expected.split(",")]
    try:
        result = srm_check(observed, expected)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    return _write_out(args, canonical(result) + "\n", f"srm -> {args.out}")


def cmd_peeking(args) -> int:
    result = simulate_peeking(n_sims=args.sims, looks=args.looks,
                              alpha=args.alpha, seed=args.seed)
    return _write_out(args, canonical(result) + "\n", f"peeking -> {args.out}")


def cmd_cuped(args) -> int:
    rows, code = _validated_or_error(args.input)
    if rows is None:
        return code
    result = apply_cuped([r["covariate_pre"] for r in rows],
                         [r["outcome"] for r in rows])
    result.pop("adjusted", None)
    return _write_out(args, canonical(result) + "\n", f"cuped -> {args.out}")


def cmd_readout(args) -> int:
    rows, code = _validated_or_error(args.input)
    if rows is None:
        return code
    counts = arm_counts(rows)
    expected = _expected(args, list(counts.keys()))
    if expected == {}:
        return EXIT_ERROR
    try:
        result = evaluate_readout(
            rows, expected_proportions=expected, looks=args.looks,
            peeking_corrected=not args.uncorrected,
            guardrail_margin=args.guardrail_margin)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    return _write_out(args, canonical(result) + "\n", f"readout -> {args.out}")


def cmd_report(args) -> int:
    rows = None
    if args.input:
        rows, code = _validated_or_error(args.input)
        if rows is None:
            return code
    report = build_report(rows)
    return _write_out(args, canonical(report) + "\n", f"report -> {args.out}")


def cmd_check(args) -> int:
    rows = _load_or_error(args.input)
    if rows is None:
        return EXIT_ERROR
    issues = validate_dataset(rows)
    if issues:
        for issue in issues[:20]:
            print(f"data-quality: {issue}")
        print(f"check FAIL (data quality): {len(issues)} issue(s)")
        return EXIT_ERROR
    counts = arm_counts(rows)
    expected = _expected(args, list(counts.keys()))
    if expected == {}:
        return EXIT_ERROR
    result = evaluate_readout(
        rows, expected_proportions=expected, looks=args.looks,
        peeking_corrected=not args.uncorrected,
        guardrail_margin=args.guardrail_margin)
    print(f"verdict={result['verdict']} "
          f"srm_p={result['srm']['p_value']:.6g} "
          f"srm_detected={result['srm']['srm_detected']} "
          f"guardrail_breached={result['guardrail']['breached']} "
          f"effect_diff={result['effect']['difference']} "
          f"effect_p={result['effect']['p_value']:.6g}")
    if result["verdict"] == "DO_NOT_SHIP":
        for reason in result["refusal_reasons"]:
            print(f"refusal: {reason}")
        print("check FAIL (readout refused)")
        return EXIT_FAIL
    print("check PASS")
    return EXIT_OK


def cmd_demo(args) -> int:
    import demo as demo_mod
    try:
        demo_mod.main()
    except AssertionError as exc:
        print(f"demo FAILED: {exc}", file=sys.stderr)
        return EXIT_FAIL
    return EXIT_OK


# ------------------------------------------------------------------- serve

class _State:
    def __init__(self):
        self.rows: List[Dict] = generate_scenario("clean")


_STATE = _State()


def get_state() -> _State:
    return _STATE


def reset_state(rows: Optional[List[Dict]] = None) -> None:
    _STATE.rows = list(rows) if rows is not None else generate_scenario("clean")


class _Handler(BaseHTTPRequestHandler):
    def _send_json(self, code: int, payload: Any) -> None:
        body = (canonical(payload) + "\n").encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if self.path == "/health":
            return self._send_json(200, {"status": "ok"})
        if self.path == "/report":
            try:
                return self._send_json(200, build_report(_STATE.rows))
            except ValueError as exc:
                return self._send_json(400, {"error": str(exc)})
        return self._send_json(404, {"error": "unknown endpoint"})

    def do_POST(self):  # noqa: N802
        if self.path != "/readout":
            return self._send_json(404, {"error": "unknown endpoint"})
        length = int(self.headers.get("Content-Length", 0) or 0)
        try:
            raw = self.rfile.read(length) if length else b""
            body = json.loads(raw) if raw.strip() else {}
            if not isinstance(body, dict):
                raise ValueError("expected a JSON object")
            rows = body.get("rows", _STATE.rows)
            result = evaluate_readout(
                rows, looks=int(body.get("looks", 1)),
                peeking_corrected=bool(body.get("peeking_corrected", True)))
            return self._send_json(200, result)
        except (ValueError, TypeError, KeyError) as exc:
            return self._send_json(400, {"error": f"bad request: {exc}"})

    def log_message(self, *a: Any) -> None:
        pass


def make_server(port: int) -> HTTPServer:
    return HTTPServer(("127.0.0.1", port), _Handler)


def cmd_serve(args) -> int:
    if args.input:
        rows = _load_or_error(args.input)
        if rows is None:
            return EXIT_ERROR
        reset_state(rows)
    server = make_server(args.port)
    print(f"serving on http://127.0.0.1:{args.port}  "
          f"(POST /readout, GET /report, GET /health)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ab-test-integrity-kit",
        description="A/B test integrity checker (Northwind Pantry sample data).")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="generate a synthetic experiment dataset")
    g.add_argument("--scenario", choices=SCENARIOS, default="clean")
    g.add_argument("--seed", type=int, default=DEFAULT_SEED)
    g.add_argument("--n-users", type=int, default=DEFAULT_N_USERS)
    g.add_argument("--out", default=None)
    g.set_defaults(func=cmd_generate)

    s = sub.add_parser("srm", help="SRM chi-square check on arm counts")
    s.add_argument("--observed", required=True,
                   help="comma-separated observed counts, e.g. 2010,1990")
    s.add_argument("--expected", default="0.5,0.5",
                   help="comma-separated expected proportions")
    s.add_argument("--out", default=None)
    s.set_defaults(func=cmd_srm)

    pk = sub.add_parser("peeking", help="seeded A/A peeking Monte Carlo")
    pk.add_argument("--sims", type=int, default=2000)
    pk.add_argument("--looks", type=int, default=5)
    pk.add_argument("--alpha", type=float, default=0.05)
    pk.add_argument("--seed", type=int, default=7)
    pk.add_argument("--out", default=None)
    pk.set_defaults(func=cmd_peeking)

    for name, fn, helptext in (
            ("cuped", cmd_cuped, "CUPED theta + variance reduction"),
            ("readout", cmd_readout, "gated readout of an experiment file"),
            ("report", cmd_report, "full integrity report, one document")):
        sp = sub.add_parser(name, help=helptext)
        sp.add_argument("--input", required=(name != "report"))
        sp.add_argument("--out", default=None)
        if name == "readout":
            sp.add_argument("--expected", default=None)
            sp.add_argument("--looks", type=int, default=1)
            sp.add_argument("--uncorrected", action="store_true",
                            help="the looks were NOT alpha-spending corrected")
            sp.add_argument("--guardrail-margin", type=float, default=0.01)
        sp.set_defaults(func=fn)

    c = sub.add_parser("check", help="CI gate: exit 0 pass / 1 refusal / 2 error")
    c.add_argument("--input", required=True)
    c.add_argument("--expected", default=None)
    c.add_argument("--looks", type=int, default=1)
    c.add_argument("--uncorrected", action="store_true")
    c.add_argument("--guardrail-margin", type=float, default=0.01)
    c.set_defaults(func=cmd_check)

    sub.add_parser("demo", help="run the end-to-end demo").set_defaults(func=cmd_demo)

    sv = sub.add_parser("serve", help="localhost JSON API")
    sv.add_argument("--port", type=int, default=8765)
    sv.add_argument("--input", default=None)
    sv.set_defaults(func=cmd_serve)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
