"""End-to-end demo: generate -> SRM -> peeking -> CUPED -> gated
readouts (clean ships, planted failures refused) -> check -> serve.

SYNTHETIC DATA ONLY (fictional Northwind Pantry dataset). Runs fully
offline; the serve round-trip is loopback-only on an ephemeral port.
Every stage asserts.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cli import EXIT_OK, main as cli_main, make_server, reset_state  # noqa: E402
from cuped import apply_cuped  # noqa: E402
from generator import generate_scenario, rows_to_jsonl  # noqa: E402
from peeking import simulate_peeking  # noqa: E402
from readout import evaluate_readout  # noqa: E402
from srm import srm_check  # noqa: E402
from validation import validate_dataset  # noqa: E402


def _request(port: int, path: str, payload=None, method: str = "GET"):
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"} if data else {}
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
                                 data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=10) as resp:
        return resp.status, json.loads(resp.read())


def main() -> None:
    print("== 1. generate synthetic Northwind Pantry experiment (seed=42) ==")
    rows = generate_scenario("clean")
    again = generate_scenario("clean")
    assert rows == again, "generator must be deterministic"
    assert len(rows) == 4000, len(rows)
    assert validate_dataset(rows) == [], validate_dataset(rows)[:3]
    print(f"   {len(rows)} users, deterministic, clean")

    print("\n== 2. SRM: clean split passes, planted 70/30 fires ==")
    clean_srm = srm_check([2000, 2000], [0.5, 0.5])
    planted = evaluate_readout(generate_scenario("srm"))
    assert not clean_srm["srm_detected"]
    assert planted["srm"]["srm_detected"], planted["srm"]
    print(f"   clean p={clean_srm['p_value']:.4f}; planted SRM "
          f"chi2={planted['srm']['chi2']} p={planted['srm']['p_value']:.3g} "
          "-> readout refuses")

    print("\n== 3. peeking: 5 uncorrected looks inflate false positives ==")
    sim = simulate_peeking()
    assert sim["uncorrected_fp_rate"] > 0.10, sim
    assert abs(sim["corrected_fp_rate"] - 0.05) <= 0.03, sim
    print(f"   A/A false positives: uncorrected "
          f"{sim['uncorrected_fp_rate']:.3f} vs O'Brien-Fleming "
          f"{sim['corrected_fp_rate']:.3f} (alpha=0.05)")

    print("\n== 4. CUPED: variance reduction near rho^2 = 0.49 ==")
    cuped = apply_cuped([r["covariate_pre"] for r in rows],
                        [r["outcome"] for r in rows])
    assert abs(cuped["variance_reduction"] - cuped["rho_squared"]) < 0.05, cuped
    print(f"   theta={cuped['theta']} rho={cuped['rho']} "
          f"reduction={cuped['variance_reduction']:.3f} "
          f"(rho^2={cuped['rho_squared']:.3f})")

    print("\n== 5. readout gate: clean ships, failures refused ==")
    clean = evaluate_readout(rows)
    assert clean["verdict"] == "SHIP", clean["verdict"]
    peeked = evaluate_readout(rows, looks=5, peeking_corrected=False)
    assert peeked["verdict"] == "DO_NOT_SHIP", peeked["verdict"]
    guard = evaluate_readout(generate_scenario("guardrail"))
    assert guard["verdict"] == "DO_NOT_SHIP", guard["verdict"]
    assert planted["verdict"] == "DO_NOT_SHIP"
    print(f"   clean=SHIP; srm / peeked / guardrail = DO_NOT_SHIP "
          f"(guardrail excess={guard['guardrail']['worst_excess']})")

    print("\n== 6. CLI check gate ==")
    import tempfile
    tmp = tempfile.mkdtemp(prefix="abtest-demo-")
    clean_path = os.path.join(tmp, "clean.jsonl")
    srm_path = os.path.join(tmp, "srm.jsonl")
    with open(clean_path, "w", encoding="utf-8") as fh:
        fh.write(rows_to_jsonl(rows))
    with open(srm_path, "w", encoding="utf-8") as fh:
        fh.write(rows_to_jsonl(generate_scenario("srm")))
    code_ok = cli_main(["check", "--input", clean_path])
    code_refuse = cli_main(["check", "--input", srm_path])
    code_err = cli_main(["check", "--input", "/tmp/no-such-abtest-xyz.jsonl"])
    print(f"   clean={code_ok} srm={code_refuse} missing-file={code_err}")
    assert (code_ok, code_refuse, code_err) == (EXIT_OK, 1, 2)

    print("\n== 7. live localhost serve round-trip ==")
    reset_state(rows)
    server = make_server(0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, health = _request(port, "/health")
        assert status == 200 and health == {"status": "ok"}
        status, served = _request(port, "/readout", payload={}, method="POST")
        assert status == 200 and served == clean, (served, clean)
        status, served_report = _request(port, "/report")
        assert status == 200 and served_report["readouts"]["clean"] == clean
        print("   POST /readout and GET /report match local computation")
    finally:
        server.shutdown()
        server.server_close()
        reset_state()

    print("\nDemo OK: generate, SRM, peeking, CUPED, gated readout, "
          "check and serve verified.")


if __name__ == "__main__":
    main()
