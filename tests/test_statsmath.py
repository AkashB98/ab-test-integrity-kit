import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from statsmath import (chi2_sf, gammainc_reg, gammaincc_reg,  # noqa: E402
                       normal_cdf, two_sided_p_from_z, z_value)


class TestGamma(unittest.TestCase):
    def test_p_plus_q_is_one(self):
        for a, x in ((0.5, 0.3), (1.0, 2.0), (2.5, 5.0), (5.0, 1.0), (0.5, 8.0)):
            self.assertAlmostEqual(gammainc_reg(a, x) + gammaincc_reg(a, x),
                                   1.0, places=12)

    def test_q_at_zero_is_one(self):
        self.assertEqual(gammaincc_reg(1.5, 0.0), 1.0)
        self.assertEqual(gammainc_reg(1.5, 0.0), 0.0)

    def test_gamma_a1_is_exponential(self):
        # Q(1, x) = exp(-x)
        self.assertAlmostEqual(gammaincc_reg(1.0, 2.0), math.exp(-2.0), places=12)
        self.assertAlmostEqual(gammainc_reg(1.0, 0.7), 1 - math.exp(-0.7),
                               places=12)

    def test_gamma_half_matches_erfc(self):
        # Q(1/2, x) = erfc(sqrt(x))
        self.assertAlmostEqual(gammaincc_reg(0.5, 2.0), math.erfc(math.sqrt(2.0)),
                               places=12)

    def test_invalid_args(self):
        with self.assertRaises(ValueError):
            gammainc_reg(0.0, 1.0)
        with self.assertRaises(ValueError):
            gammaincc_reg(1.0, -1.0)


class TestChi2(unittest.TestCase):
    def test_critical_values_hit_005(self):
        # Published 95th-percentile critical values.
        for x, df in ((3.841458820694124, 1), (5.991464547107979, 2),
                      (7.814727903251179, 3), (18.307038053275146, 10)):
            self.assertAlmostEqual(chi2_sf(x, df), 0.05, places=9)

    def test_df2_is_exponential(self):
        for x in (0.5, 2.0, 9.0):
            self.assertAlmostEqual(chi2_sf(x, 2), math.exp(-x / 2), places=12)

    def test_sf_at_zero_and_monotone(self):
        self.assertEqual(chi2_sf(0.0, 3), 1.0)
        vals = [chi2_sf(x, 3) for x in (1.0, 3.0, 6.0, 12.0)]
        self.assertTrue(all(a > b for a, b in zip(vals, vals[1:])))

    def test_extreme_tail(self):
        self.assertLess(chi2_sf(611.524, 1), 1e-100)

    def test_invalid(self):
        with self.assertRaises(ValueError):
            chi2_sf(1.0, 0)
        with self.assertRaises(ValueError):
            chi2_sf(-1.0, 2)


class TestNormal(unittest.TestCase):
    def test_quantiles(self):
        self.assertAlmostEqual(z_value(0.975), 1.9599639845400536, places=12)
        self.assertAlmostEqual(z_value(0.5), 0.0, places=12)

    def test_cdf(self):
        self.assertEqual(normal_cdf(0.0), 0.5)

    def test_two_sided_p(self):
        self.assertAlmostEqual(two_sided_p_from_z(1.9599639845400536), 0.05,
                               places=9)
        self.assertEqual(two_sided_p_from_z(0.0), 1.0)


if __name__ == "__main__":
    unittest.main()
