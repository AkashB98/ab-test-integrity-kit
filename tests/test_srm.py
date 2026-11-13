import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from srm import SRM_ALPHA, srm_check, srm_from_counts  # noqa: E402


class TestSRM(unittest.TestCase):
    def test_perfect_split(self):
        r = srm_check([2000, 2000], [0.5, 0.5])
        self.assertEqual(r["chi2"], 0.0)
        self.assertEqual(r["p_value"], 1.0)
        self.assertFalse(r["srm_detected"])

    def test_seeded_clean_split_no_fire(self):
        r = srm_check([1985, 2015], [0.5, 0.5])
        self.assertAlmostEqual(r["p_value"], 0.635256, places=5)
        self.assertFalse(r["srm_detected"])

    def test_planted_two_arm_fires(self):
        r = srm_check([2782, 1218], [0.5, 0.5])
        self.assertEqual(r["chi2"], 611.524)
        self.assertLess(r["p_value"], 1e-100)
        self.assertTrue(r["srm_detected"])

    def test_chi2_hand_computed(self):
        # (110-100)^2/100 + (90-100)^2/100 = 2.0
        r = srm_check([110, 90], [0.5, 0.5])
        self.assertEqual(r["chi2"], 2.0)
        self.assertEqual(r["df"], 1)

    def test_unequal_expected_proportions(self):
        r = srm_check([800, 200], [0.8, 0.2])
        self.assertEqual(r["chi2"], 0.0)
        self.assertFalse(r["srm_detected"])
        r2 = srm_check([500, 500], [0.8, 0.2])
        self.assertTrue(r2["srm_detected"])

    def test_multi_arm_clean(self):
        r = srm_check([1330, 1340, 1330], [1 / 3, 1 / 3, 1 / 3])
        self.assertEqual(r["df"], 2)
        self.assertEqual(r["chi2"], 0.05)
        self.assertFalse(r["srm_detected"])

    def test_multi_arm_planted_fires(self):
        r = srm_check([1200, 1150, 1650], [1 / 3, 1 / 3, 1 / 3])
        self.assertEqual(r["chi2"], 113.75)
        self.assertTrue(r["srm_detected"])

    def test_four_arm(self):
        r = srm_check([250, 250, 250, 250], [0.25] * 4)
        self.assertEqual(r["df"], 3)
        self.assertFalse(r["srm_detected"])

    def test_threshold_is_0001(self):
        self.assertEqual(SRM_ALPHA, 0.001)
        # A split significant at 0.05 but not 0.001 must NOT fire:
        # chi2 = 4.0 -> p ~= 0.0455 for df=1.
        r = srm_check([1020, 980], [0.5, 0.5])
        self.assertEqual(r["chi2"], 0.8)  # p ~= 0.37, clearly no fire
        self.assertFalse(r["srm_detected"])
        r2 = srm_check([1100, 900], [0.5, 0.5])  # chi2=20, p~7.7e-6
        self.assertTrue(r2["srm_detected"])

    def test_proportions_need_not_sum_to_one(self):
        r = srm_check([100, 300], [1.0, 3.0])
        self.assertEqual(r["chi2"], 0.0)

    def test_from_counts_keyed(self):
        r = srm_from_counts({"control": 2000, "treatment": 2000},
                            {"control": 0.5, "treatment": 0.5})
        self.assertEqual(r["arms"], ["control", "treatment"])
        self.assertFalse(r["srm_detected"])

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            srm_check([100], [1.0])
        with self.assertRaises(ValueError):
            srm_check([100, 100], [0.5])
        with self.assertRaises(ValueError):
            srm_check([0, 0], [0.5, 0.5])
        with self.assertRaises(ValueError):
            srm_check([100, -5], [0.5, 0.5])
        with self.assertRaises(ValueError):
            srm_check([100, 100], [0.5, 0.0])
        with self.assertRaises(ValueError):
            srm_from_counts({"a": 10, "b": 10}, {"a": 0.5, "c": 0.5})


if __name__ == "__main__":
    unittest.main()
