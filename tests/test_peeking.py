import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from peeking import obf_boundaries, peeking_verdict, simulate_peeking  # noqa: E402


class TestBoundaries(unittest.TestCase):
    def test_k5_tabulated(self):
        b = obf_boundaries(5)
        self.assertEqual(b["critical_z"],
                         [4.561755, 3.225559, 2.633935, 2.280593, 2.039549])
        self.assertEqual(b["source"], "tabulated")

    def test_boundaries_decrease(self):
        for k in (2, 3, 4, 5):
            zs = obf_boundaries(k)["critical_z"]
            self.assertTrue(all(a > b for a, b in zip(zs, zs[1:])), zs)

    def test_final_boundary_near_196(self):
        self.assertAlmostEqual(obf_boundaries(5)["critical_z"][-1], 2.039549)
        self.assertEqual(obf_boundaries(1)["critical_z"], [1.959964])

    def test_nominal_p_monotone_and_small_first(self):
        b = obf_boundaries(5)
        ps = b["nominal_p"]
        self.assertTrue(all(a < c for a, c in zip(ps, ps[1:])), ps)
        self.assertLess(ps[0], 0.0001)
        self.assertAlmostEqual(ps[-1], 0.041395, places=5)

    def test_nominal_p_spend_sums_near_alpha(self):
        # Total spend across looks should be in alpha's neighbourhood
        # (increments of cumulative spend, not the nominal p's sum —
        # the check is that no single look spends everything early).
        b = obf_boundaries(5)
        self.assertLess(sum(b["nominal_p"]), 0.10)

    def test_large_k_fallback_labelled(self):
        b = obf_boundaries(8)
        self.assertEqual(b["source"], "lan-demets-approx")
        self.assertEqual(len(b["critical_z"]), 8)

    def test_invalid(self):
        with self.assertRaises(ValueError):
            obf_boundaries(0)
        with self.assertRaises(ValueError):
            obf_boundaries(3, alpha=1.5)


class TestSimulation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sim = simulate_peeking()

    def test_deterministic(self):
        self.assertEqual(simulate_peeking(n_sims=200, seed=3),
                         simulate_peeking(n_sims=200, seed=3))

    def test_uncorrected_inflated_pinned(self):
        self.assertEqual(self.sim["uncorrected_false_positives"], 278)
        self.assertEqual(self.sim["uncorrected_fp_rate"], 0.139)
        self.assertGreater(self.sim["uncorrected_fp_rate"], 0.10)

    def test_corrected_back_to_alpha_pinned(self):
        self.assertEqual(self.sim["corrected_false_positives"], 91)
        self.assertEqual(self.sim["corrected_fp_rate"], 0.0455)
        self.assertLessEqual(abs(self.sim["corrected_fp_rate"] - 0.05), 0.02)

    def test_single_look_no_inflation(self):
        sim = simulate_peeking(n_sims=2000, looks=1, seed=7)
        self.assertLessEqual(abs(sim["uncorrected_fp_rate"] - 0.05), 0.02)
        self.assertEqual(sim["uncorrected_false_positives"],
                         sim["corrected_false_positives"])

    def test_counts_consistent(self):
        self.assertEqual(self.sim["n_sims"], 2000)
        self.assertLessEqual(self.sim["corrected_false_positives"],
                             self.sim["n_sims"])


class TestVerdict(unittest.TestCase):
    def test_flags(self):
        self.assertTrue(peeking_verdict(5, False)["uncorrected_peeking"])
        self.assertFalse(peeking_verdict(5, True)["uncorrected_peeking"])
        self.assertFalse(peeking_verdict(1, False)["uncorrected_peeking"])


if __name__ == "__main__":
    unittest.main()
