import hashlib
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generator import (arm_counts, generate_experiment,  # noqa: E402
                       generate_scenario, rows_to_csv, rows_to_jsonl)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestGenerator(unittest.TestCase):
    def test_deterministic(self):
        self.assertEqual(generate_scenario("clean"), generate_scenario("clean"))

    def test_seed_changes_data(self):
        self.assertNotEqual(generate_scenario("clean", seed=1),
                            generate_scenario("clean", seed=2))

    def test_shape_and_fields(self):
        rows = generate_scenario("clean")
        self.assertEqual(len(rows), 4000)
        for r in rows[:50]:
            self.assertEqual(set(r), {"user_id", "arm", "covariate_pre",
                                      "outcome", "guardrail_error"})
            self.assertIn(r["guardrail_error"], (0, 1))
            self.assertIn(r["arm"], ("control", "treatment"))

    def test_seeded_counts_pinned(self):
        self.assertEqual(arm_counts(generate_scenario("clean")),
                         {"treatment": 2015, "control": 1985})

    def test_jsonl_md5_pinned(self):
        md5 = hashlib.md5(rows_to_jsonl(generate_scenario("clean"))
                          .encode()).hexdigest()
        self.assertEqual(md5, "b5c211706907cac595835dcf392cb87f")

    def test_srm_scenario_skewed(self):
        counts = arm_counts(generate_scenario("srm"))
        self.assertEqual(counts, {"control": 2782, "treatment": 1218})

    def test_guardrail_scenario_rates(self):
        from readout import evaluate_readout
        out = evaluate_readout(generate_scenario("guardrail"))
        self.assertGreater(out["guardrail"]["worst_excess"], 0.03)

    def test_effect_direction(self):
        rows = generate_experiment(n_users=2000, effect=10.0)
        by = {}
        for r in rows:
            by.setdefault(r["arm"], []).append(r["outcome"])
        self.assertGreater(sum(by["treatment"]) / len(by["treatment"]),
                           sum(by["control"]) / len(by["control"]) + 5)

    def test_multi_arm_generation(self):
        rows = generate_experiment(n_users=3000,
                                   arms=("a", "b", "c"),
                                   expected_proportions=(1 / 3, 1 / 3, 1 / 3))
        self.assertEqual(set(arm_counts(rows)), {"a", "b", "c"})

    def test_csv_round_trip_shape(self):
        text = rows_to_csv(generate_scenario("clean", n_users=20))
        lines = text.strip().splitlines()
        self.assertEqual(len(lines), 21)
        self.assertTrue(lines[0].startswith("user_id,arm,"))

    def test_user_ids_unique(self):
        rows = generate_scenario("clean")
        self.assertEqual(len({r["user_id"] for r in rows}), 4000)

    def test_invalid(self):
        with self.assertRaises(ValueError):
            generate_scenario("nope")
        with self.assertRaises(ValueError):
            generate_experiment(n_users=5)
        with self.assertRaises(ValueError):
            generate_experiment(rho=1.5)


if __name__ == "__main__":
    unittest.main()
