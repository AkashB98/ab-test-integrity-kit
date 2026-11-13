import io
import json
import os
import sys
import tempfile
import threading
import unittest
import urllib.request
from contextlib import redirect_stderr, redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cli import EXIT_ERROR, EXIT_FAIL, EXIT_OK, main as cli_main  # noqa: E402
from cli import make_server, reset_state  # noqa: E402
from generator import generate_scenario, rows_to_jsonl  # noqa: E402


def run_cli(argv):
    with redirect_stdout(io.StringIO()) as out, redirect_stderr(io.StringIO()):
        code = cli_main(argv)
    return code, out.getvalue()


class TestCLI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="abtest-cli-")
        cls.clean = os.path.join(cls.tmp, "clean.jsonl")
        cls.srm = os.path.join(cls.tmp, "srm.jsonl")
        cls.guard = os.path.join(cls.tmp, "guard.jsonl")
        for path, scen in ((cls.clean, "clean"), (cls.srm, "srm"),
                           (cls.guard, "guardrail")):
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(rows_to_jsonl(generate_scenario(scen)))

    def test_generate_to_file(self):
        out = os.path.join(self.tmp, "gen.jsonl")
        code, _ = run_cli(["generate", "--out", out])
        self.assertEqual(code, EXIT_OK)
        with open(out, encoding="utf-8") as fh:
            self.assertEqual(len(fh.read().strip().splitlines()), 4000)

    def test_generate_srm_scenario(self):
        out = os.path.join(self.tmp, "gen2.jsonl")
        code, _ = run_cli(["generate", "--scenario", "srm", "--out", out])
        self.assertEqual(code, EXIT_OK)

    def test_srm_command(self):
        code, text = run_cli(["srm", "--observed", "2782,1218",
                              "--expected", "0.5,0.5"])
        self.assertEqual(code, EXIT_OK)
        payload = json.loads(text)
        self.assertTrue(payload["srm_detected"])
        self.assertEqual(payload["chi2"], 611.524)

    def test_srm_command_bad_input(self):
        code, _ = run_cli(["srm", "--observed", "100", "--expected", "1.0"])
        self.assertEqual(code, EXIT_ERROR)

    def test_peeking_command(self):
        code, text = run_cli(["peeking", "--sims", "500", "--seed", "7"])
        self.assertEqual(code, EXIT_OK)
        payload = json.loads(text)
        self.assertEqual(payload["n_sims"], 500)

    def test_cuped_command(self):
        code, text = run_cli(["cuped", "--input", self.clean])
        self.assertEqual(code, EXIT_OK)
        payload = json.loads(text)
        self.assertEqual(payload["theta"], 0.70491)
        self.assertNotIn("adjusted", payload)

    def test_readout_command(self):
        code, text = run_cli(["readout", "--input", self.clean])
        self.assertEqual(code, EXIT_OK)
        self.assertEqual(json.loads(text)["verdict"], "SHIP")

    def test_readout_uncorrected_flag(self):
        code, text = run_cli(["readout", "--input", self.clean,
                              "--looks", "5", "--uncorrected"])
        self.assertEqual(json.loads(text)["verdict"], "DO_NOT_SHIP")

    def test_report_command(self):
        code, text = run_cli(["report"])
        self.assertEqual(code, EXIT_OK)
        self.assertEqual(json.loads(text)["dataset"]["rows"], 4000)

    def test_check_exit_codes(self):
        self.assertEqual(run_cli(["check", "--input", self.clean])[0], EXIT_OK)
        self.assertEqual(run_cli(["check", "--input", self.srm])[0], EXIT_FAIL)
        self.assertEqual(run_cli(["check", "--input", self.guard])[0], EXIT_FAIL)
        self.assertEqual(run_cli(["check", "--input",
                                   "/tmp/no-such-abtest-xyz.jsonl"])[0],
                         EXIT_ERROR)

    def test_check_peeking_refusal(self):
        code, _ = run_cli(["check", "--input", self.clean,
                           "--looks", "5", "--uncorrected"])
        self.assertEqual(code, EXIT_FAIL)

    def test_check_data_quality_is_error(self):
        bad = os.path.join(self.tmp, "bad.jsonl")
        with open(bad, "w", encoding="utf-8") as fh:
            fh.write('{"user_id": "u1", "arm": "control"}\n')
        self.assertEqual(run_cli(["check", "--input", bad])[0], EXIT_ERROR)

    def test_missing_file_readout(self):
        code, _ = run_cli(["readout", "--input", "/tmp/nope-xyz.jsonl"])
        self.assertEqual(code, EXIT_ERROR)


class TestServe(unittest.TestCase):
    def test_round_trip(self):
        rows = generate_scenario("clean")
        reset_state(rows)
        server = make_server(0)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health",
                                        timeout=10) as resp:
                self.assertEqual(json.loads(resp.read()), {"status": "ok"})
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/readout", data=b"{}",
                headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=10) as resp:
                served = json.loads(resp.read())
            from readout import evaluate_readout
            self.assertEqual(served, evaluate_readout(rows))
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/report",
                                        timeout=10) as resp:
                report = json.loads(resp.read())
            self.assertEqual(report["readouts"]["clean"]["verdict"], "SHIP")
        finally:
            server.shutdown()
            server.server_close()
            reset_state()


if __name__ == "__main__":
    unittest.main()
