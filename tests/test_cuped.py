import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cuped import (apply_cuped, correlation, covariance, cuped_theta,  # noqa: E402
                   variance)


class TestMoments(unittest.TestCase):
    def test_variance_hand(self):
        self.assertEqual(variance([1.0, 2.0, 3.0, 4.0]), 1.25)

    def test_covariance_hand(self):
        self.assertEqual(covariance([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]), 4.0 / 3.0)

    def test_correlation_perfect(self):
        self.assertAlmostEqual(correlation([1, 2, 3], [2, 4, 6]), 1.0)
        self.assertAlmostEqual(correlation([1, 2, 3], [6, 4, 2]), -1.0)

    def test_constant_sample_errors(self):
        with self.assertRaises(ValueError):
            correlation([1.0, 1.0], [1.0, 2.0])
        with self.assertRaises(ValueError):
            cuped_theta([5.0, 5.0, 5.0], [1.0, 2.0, 3.0])
        with self.assertRaises(ValueError):
            variance([1.0])


class TestCuped(unittest.TestCase):
    def test_theta_hand_computed(self):
        # Y = 2X exactly: theta = Cov/Var = 2, adjusted Y is constant.
        xs = [1.0, 2.0, 3.0, 4.0, 5.0]
        ys = [2.0, 4.0, 6.0, 8.0, 10.0]
        self.assertEqual(cuped_theta(xs, ys), 2.0)
        out = apply_cuped(xs, ys)
        self.assertEqual(out["variance_after"], 0.0)
        self.assertEqual(out["variance_reduction"], 1.0)
        self.assertEqual(out["rho_squared"], 1.0)

    def test_zero_correlation_does_nothing(self):
        xs = [1.0, 2.0, 3.0, 4.0]
        ys = [1.0, -1.0, -1.0, 1.0]  # constructed uncorrelated with xs
        out = apply_cuped(xs, ys)
        self.assertAlmostEqual(out["theta"], 0.0, places=9)
        self.assertAlmostEqual(out["variance_reduction"], 0.0, places=9)

    def test_reduction_equals_rho_squared_identity(self):
        xs = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]
        ys = [2.7, 0.9, 4.2, 1.4, 4.8, 8.1, 2.2, 5.5]
        out = apply_cuped(xs, ys)
        self.assertAlmostEqual(out["variance_reduction"],
                               out["rho_squared"], places=5)

    def test_adjustment_preserves_mean(self):
        xs = [10.0, 20.0, 30.0, 40.0]
        ys = [12.0, 19.0, 33.0, 38.0]
        out = apply_cuped(xs, ys)
        self.assertAlmostEqual(sum(out["adjusted"]) / len(ys),
                               sum(ys) / len(ys), places=9)

    def test_explicit_theta(self):
        out = apply_cuped([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], theta=0.0)
        self.assertEqual(out["adjusted"], [1.0, 2.0, 3.0])
        self.assertEqual(out["theta"], 0.0)

    def test_seeded_generated_sample(self):
        from generator import generate_scenario
        rows = generate_scenario("clean")
        out = apply_cuped([r["covariate_pre"] for r in rows],
                          [r["outcome"] for r in rows])
        self.assertEqual(out["theta"], 0.70491)
        self.assertAlmostEqual(out["variance_reduction"], 0.49, delta=0.05)
        self.assertAlmostEqual(out["variance_reduction"],
                               out["rho_squared"], places=5)

    def test_length_mismatch(self):
        with self.assertRaises(ValueError):
            apply_cuped([1.0, 2.0], [1.0])


if __name__ == "__main__":
    unittest.main()
