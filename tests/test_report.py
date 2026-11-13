import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from report import build_report  # noqa: E402


class TestReport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = build_report()

    def test_deterministic(self):
        self.assertEqual(build_report(), self.report)

    def test_sections_present(self):
        for key in ("dataset", "srm_two_arm_clean", "srm_multi_arm_planted",
                    "peeking", "cuped", "readouts"):
            self.assertIn(key, self.report)

    def test_verdicts(self):
        r = self.report["readouts"]
        self.assertEqual(r["clean"]["verdict"], "SHIP")
        self.assertEqual(r["clean_peeked_uncorrected"]["verdict"], "DO_NOT_SHIP")
        self.assertEqual(r["srm"]["verdict"], "DO_NOT_SHIP")
        self.assertEqual(r["guardrail"]["verdict"], "DO_NOT_SHIP")

    def test_multi_arm_srm_planted(self):
        self.assertTrue(self.report["srm_multi_arm_planted"]["srm_detected"])
        self.assertEqual(self.report["srm_multi_arm_planted"]["df"], 2)

    def test_peeking_section(self):
        p = self.report["peeking"]
        self.assertEqual(p["uncorrected_fp_rate"], 0.139)
        self.assertEqual(p["corrected_fp_rate"], 0.0455)


if __name__ == "__main__":
    unittest.main()
