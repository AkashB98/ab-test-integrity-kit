import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generator import generate_scenario  # noqa: E402
from helpers import two_arm  # noqa: E402
from readout import evaluate_readout, two_sample_z  # noqa: E402


class TestTwoSampleZ(unittest.TestCase):
    def test_identical_samples(self):
        out = two_sample_z([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        self.assertEqual(out["difference"], 0.0)
        self.assertEqual(out["p_value"], 1.0)

    def test_known_shift(self):
        out = two_sample_z([0.0] * 50 + [2.0] * 50, [3.0] * 50 + [5.0] * 50)
        self.assertEqual(out["difference"], 3.0)
        self.assertLess(out["p_value"], 0.001)


class TestGate(unittest.TestCase):
    def test_clean_generated_ships(self):
        out = evaluate_readout(generate_scenario("clean"))
        self.assertEqual(out["verdict"], "SHIP")
        self.assertEqual(out["refusal_reasons"], [])
        self.assertEqual(out["arms"], ["control", "treatment"])

    def test_control_first_even_if_treatment_row_first(self):
        # Regression: the seeded dataset's first row is treatment; the
        # baseline must still be control or the effect sign flips.
        rows = generate_scenario("clean")
        self.assertEqual(rows[0]["arm"], "treatment")
        out = evaluate_readout(rows)
        self.assertGreater(out["effect"]["difference"], 0)

    def test_srm_refused(self):
        out = evaluate_readout(generate_scenario("srm"))
        self.assertEqual(out["verdict"], "DO_NOT_SHIP")
        self.assertTrue(any("SRM" in r for r in out["refusal_reasons"]))

    def test_uncorrected_peeking_refused(self):
        out = evaluate_readout(generate_scenario("clean"),
                               looks=5, peeking_corrected=False)
        self.assertEqual(out["verdict"], "DO_NOT_SHIP")
        self.assertTrue(any("peeked" in r for r in out["refusal_reasons"]))

    def test_corrected_peeking_not_refused(self):
        out = evaluate_readout(generate_scenario("clean"),
                               looks=5, peeking_corrected=True)
        self.assertEqual(out["verdict"], "SHIP")

    def test_single_look_uncorrected_not_refused(self):
        out = evaluate_readout(generate_scenario("clean"),
                               looks=1, peeking_corrected=False)
        self.assertEqual(out["verdict"], "SHIP")

    def test_guardrail_refused(self):
        out = evaluate_readout(generate_scenario("guardrail"))
        self.assertEqual(out["verdict"], "DO_NOT_SHIP")
        self.assertTrue(any("guardrail" in r for r in out["refusal_reasons"]))
        self.assertEqual(out["guardrail"]["worst_excess"], 0.034469)

    def test_no_effect_is_inconclusive_not_ship(self):
        out = evaluate_readout(two_arm(effect=0.0))
        self.assertEqual(out["verdict"], "INCONCLUSIVE")

    def test_hand_built_positive_effect_ships(self):
        out = evaluate_readout(two_arm(effect=5.0))
        self.assertEqual(out["verdict"], "SHIP")

    def test_negative_effect_not_ship(self):
        out = evaluate_readout(two_arm(effect=-5.0))
        self.assertEqual(out["verdict"], "INCONCLUSIVE")

    def test_guardrail_hand_built_breach(self):
        rows = two_arm(effect=5.0, guardrail_lift=30)  # 30% errors vs 0%
        out = evaluate_readout(rows)
        self.assertEqual(out["verdict"], "DO_NOT_SHIP")
        self.assertTrue(out["guardrail"]["breached"])

    def test_cuped_tightens_standard_error(self):
        rows = generate_scenario("clean")
        adj = evaluate_readout(rows, use_cuped=True)
        raw = evaluate_readout(rows, use_cuped=False)
        self.assertLess(adj["effect"]["se"], raw["effect"]["se"])
        self.assertIsNotNone(adj["cuped"])
        self.assertIsNone(raw["cuped"])

    def test_multiple_refusals_all_listed(self):
        out = evaluate_readout(generate_scenario("srm"),
                               looks=5, peeking_corrected=False)
        self.assertEqual(out["verdict"], "DO_NOT_SHIP")
        self.assertEqual(len(out["refusal_reasons"]), 2)

    def test_single_arm_raises(self):
        from helpers import row
        with self.assertRaises(ValueError):
            evaluate_readout([row("u1", "control"), row("u2", "control")])


if __name__ == "__main__":
    unittest.main()
