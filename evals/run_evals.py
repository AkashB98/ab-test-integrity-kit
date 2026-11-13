"""Golden evals for ab-test-integrity-kit.

SYNTHETIC DATA ONLY (fictional Northwind Pantry dataset). Every eval
is deterministic: run_evals.py writes eval_report.json and the file
must be byte-identical across runs. No timestamps, ports, or temp
paths enter the report - only names, booleans, pinned counts and
hashes.

E1-E7 pin the SRM/gamma machinery against published chi-square
critical values and hand-computed cases. E8-E9 are the peeking
calibration: a seeded 2,000-experiment A/A Monte Carlo whose
false-positive counts are pinned exactly (278 uncorrected vs 91
O'Brien-Fleming-corrected at alpha=0.05). E10-E11 pin CUPED against
the planted rho=0.7 and hand-computed cases. E12-E16 pin the refusal
gate, the CLI exit codes, the serve round-trip, the fixture, and
report determinism.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import sys
import threading
import urllib.request
from contextlib import redirect_stderr, redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cli import EXIT_ERROR, EXIT_FAIL, EXIT_OK, main as cli_main  # noqa: E402
from cli import make_server, reset_state
from cuped import apply_cuped, cuped_theta  # noqa: E402
from generator import (arm_counts, generate_scenario, load_rows,  # noqa: E402
                       rows_to_jsonl)
from peeking import obf_boundaries, simulate_peeking  # noqa: E402
from readout import evaluate_readout  # noqa: E402
from report import build_report  # noqa: E402
from srm import srm_check  # noqa: E402
from statsmath import chi2_sf, gammaincc_reg  # noqa: E402
from validation import validate_dataset  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIXTURE = os.path.join(ROOT, "fixtures", "sample_experiment.jsonl")
RESULTS = []


def check(name, passed, detail=""):
    RESULTS.append({"name": name, "passed": bool(passed), "detail": detail})
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))
    return passed


def _run_cli(argv):
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        return cli_main(argv)


def main():
    print("== golden evals: ab-test-integrity-kit ==")
    clean = generate_scenario("clean")

    # E1: generator determinism (bytes + counts pinned)
    text1 = rows_to_jsonl(generate_scenario("clean"))
    md5 = hashlib.md5(text1.encode()).hexdigest()
    check("E1 generator determinism (seed=42, 4000 users)",
          text1 == rows_to_jsonl(generate_scenario("clean"))
          and len(clean) == 4000
          and arm_counts(clean) == {"treatment": 2015, "control": 1985}
          and md5 == "b5c211706907cac595835dcf392cb87f",
          f"jsonl_md5={md5}")

    # E2: gamma / chi-square against published critical values
    crit_ok = all(abs(chi2_sf(x, df) - 0.05) < 1e-9 for x, df in
                  ((3.841458820694124, 1), (5.991464547107979, 2),
                   (7.814727903251179, 3), (18.307038053275146, 10)))
    check("E2 chi-square SF at published 95th-percentile critical values",
          crit_ok and abs(chi2_sf(9.0, 2) - math.exp(-4.5)) < 1e-12
          and abs(gammaincc_reg(0.5, 2.0) - math.erfc(math.sqrt(2.0))) < 1e-12)

    # E3: SRM two-arm clean (seeded split, no fire; perfect split p=1)
    r = srm_check([1985, 2015], [0.5, 0.5])
    check("E3 SRM two-arm clean (p=0.635256, no fire)",
          abs(r["p_value"] - 0.6352562959972485) < 1e-12
          and not r["srm_detected"]
          and srm_check([2000, 2000], [0.5, 0.5])["p_value"] == 1.0,
          f"p={r['p_value']}")

    # E4: SRM planted two-arm fires (seeded 70/30 scenario counts)
    counts = arm_counts(generate_scenario("srm"))
    r = srm_check([counts["control"], counts["treatment"]], [0.5, 0.5])
    check("E4 SRM planted 70/30 fires (chi2=611.524, p<1e-100)",
          counts == {"control": 2782, "treatment": 1218}
          and r["chi2"] == 611.524 and r["p_value"] < 1e-100
          and r["srm_detected"],
          f"chi2={r['chi2']} p={r['p_value']:.3g}")

    # E5: SRM multi-arm (same gamma path, df=2): clean vs planted
    rc = srm_check([1330, 1340, 1330], [1 / 3, 1 / 3, 1 / 3])
    rp = srm_check([1200, 1150, 1650], [1 / 3, 1 / 3, 1 / 3])
    check("E5 SRM multi-arm (clean chi2=0.05 no fire; planted 113.75 fires)",
          rc["chi2"] == 0.05 and rc["df"] == 2 and not rc["srm_detected"]
          and rp["chi2"] == 113.75 and rp["srm_detected"],
          f"planted_p={rp['p_value']:.3g}")

    # E6: threshold discipline - p=0.0047 is significant at 0.05 but
    # must NOT fire the 0.001 SRM line; p<0.001 must.
    mid = srm_check([10200, 9800], [0.5, 0.5])
    hard = srm_check([1100, 900], [0.5, 0.5])
    check("E6 SRM threshold 0.001 (p~0.0047 no fire, p~7.7e-6 fires)",
          mid["chi2"] == 8.0 and 0.001 < mid["p_value"] < 0.05
          and not mid["srm_detected"] and hard["srm_detected"],
          f"mid_p={mid['p_value']:.6f}")

    # E7: O'Brien-Fleming boundaries pinned (z + NormalDist nominal p)
    b = obf_boundaries(5)
    check("E7 OBF boundaries pinned (K=5, final z=2.039549)",
          b["critical_z"] == [4.561755, 3.225559, 2.633935, 2.280593, 2.039549]
          and b["nominal_p"] == [5e-06, 0.001257, 0.00844, 0.022573, 0.041395]
          and obf_boundaries(1)["critical_z"] == [1.959964])

    # E8 + E9: peeking Monte Carlo (one simulation, two evals)
    sim = simulate_peeking()
    check("E8 peeking uncorrected FP pinned (278/2000 = 0.139)",
          sim["uncorrected_false_positives"] == 278
          and sim["uncorrected_fp_rate"] == 0.139
          and sim["uncorrected_fp_rate"] > 0.10,
          f"rate={sim['uncorrected_fp_rate']}")
    check("E9 peeking corrected FP back to alpha (91/2000 = 0.0455, tol 0.02)",
          sim["corrected_false_positives"] == 91
          and sim["corrected_fp_rate"] == 0.0455
          and abs(sim["corrected_fp_rate"] - 0.05) <= 0.02,
          f"rate={sim['corrected_fp_rate']} tol=|rate-0.05|<=0.02")

    # E10: CUPED on the seeded sample vs planted rho=0.7 (rho^2=0.49)
    cuped = apply_cuped([row["covariate_pre"] for row in clean],
                        [row["outcome"] for row in clean])
    check("E10 CUPED seeded (theta=0.70491, reduction 0.474941 vs rho^2)",
          cuped["theta"] == 0.70491
          and cuped["variance_reduction"] == 0.474941
          and cuped["variance_reduction"] == cuped["rho_squared"]
          and abs(cuped["variance_reduction"] - 0.49) <= 0.03,
          f"reduction={cuped['variance_reduction']} planted_rho2=0.49")

    # E11: CUPED hand cases (perfect line -> theta 2, reduction 1;
    # uncorrelated -> theta 0, reduction 0)
    perfect = apply_cuped([1.0, 2.0, 3.0, 4.0, 5.0],
                          [2.0, 4.0, 6.0, 8.0, 10.0])
    zero = apply_cuped([1.0, 2.0, 3.0, 4.0], [1.0, -1.0, -1.0, 1.0])
    check("E11 CUPED hand-computed (theta 2.0 / 0.0)",
          cuped_theta([1.0, 2.0, 3.0, 4.0, 5.0],
                      [2.0, 4.0, 6.0, 8.0, 10.0]) == 2.0
          and perfect["variance_reduction"] == 1.0
          and abs(zero["theta"]) < 1e-12
          and abs(zero["variance_reduction"]) < 1e-12)

    # E12: the refusal gate - every planted failure refused, clean ships
    verdicts = {
        "clean": evaluate_readout(clean)["verdict"],
        "srm": evaluate_readout(generate_scenario("srm"))["verdict"],
        "guardrail": evaluate_readout(generate_scenario("guardrail"))["verdict"],
        "peeked": evaluate_readout(clean, looks=5,
                                   peeking_corrected=False)["verdict"],
        "peeked_corrected": evaluate_readout(
            clean, looks=5, peeking_corrected=True)["verdict"],
    }
    check("E12 readout gate verdicts (SHIP / 3x DO_NOT_SHIP / SHIP)",
          verdicts == {"clean": "SHIP", "srm": "DO_NOT_SHIP",
                       "guardrail": "DO_NOT_SHIP", "peeked": "DO_NOT_SHIP",
                       "peeked_corrected": "SHIP"},
          f"verdicts={verdicts}")

    # E13: CLI check exit codes 0 / 1 / 2
    import tempfile
    tmp = tempfile.mkdtemp(prefix="abtest-evals-")
    clean_path = os.path.join(tmp, "clean.jsonl")
    srm_path = os.path.join(tmp, "srm.jsonl")
    with open(clean_path, "w", encoding="utf-8") as fh:
        fh.write(rows_to_jsonl(clean))
    with open(srm_path, "w", encoding="utf-8") as fh:
        fh.write(rows_to_jsonl(generate_scenario("srm")))
    codes = (_run_cli(["check", "--input", clean_path]),
             _run_cli(["check", "--input", srm_path]),
             _run_cli(["check", "--input", clean_path,
                       "--looks", "5", "--uncorrected"]),
             _run_cli(["check", "--input", "/tmp/no-such-abtest-xyz.jsonl"]))
    check("E13 CLI check exit codes 0/1/1/2",
          codes == (EXIT_OK, EXIT_FAIL, EXIT_FAIL, EXIT_ERROR),
          f"codes={codes}")

    # E14: serve round-trip matches local computation
    reset_state(clean)
    server = make_server(0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/readout", data=b"{}",
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=10) as resp:
            served = json.loads(resp.read())
        serve_ok = served == evaluate_readout(clean)
    finally:
        server.shutdown()
        server.server_close()
        reset_state()
    check("E14 serve round-trip matches local readout", serve_ok)

    # E15: bundled fixture pinned (4000 rows, totals, clean, same readout)
    fixture = load_rows(FIXTURE)
    total = round(sum(r["outcome"] for r in fixture), 2)
    check("E15 fixture pinned (4000 rows, total outcome 206141.52, clean)",
          len(fixture) == 4000 and total == 206141.52
          and validate_dataset(fixture) == []
          and evaluate_readout(fixture)["verdict"] == "SHIP",
          f"total_outcome={total}")

    # E16: full report deterministic and internally consistent
    rep1, rep2 = build_report(), build_report()
    check("E16 report deterministic (verdicts + peeking + cuped pinned)",
          rep1 == rep2
          and rep1["readouts"]["clean"]["verdict"] == "SHIP"
          and rep1["peeking"]["corrected_fp_rate"] == 0.0455
          and rep1["cuped"]["variance_reduction"] == 0.474941)

    passed = sum(1 for r in RESULTS if r["passed"])
    total_n = len(RESULTS)
    summary_doc = {"passed": passed, "total": total_n,
                   "all_passed": passed == total_n, "evals": RESULTS}
    with open(os.path.join(HERE, "eval_report.json"), "w", encoding="utf-8") as fh:
        fh.write(json.dumps(summary_doc, sort_keys=True) + "\n")
    print(f"\n{passed}/{total_n} evals passed -> evals/eval_report.json")
    return 0 if passed == total_n else 1


if __name__ == "__main__":
    sys.exit(main())
