import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers import row  # noqa: E402
from validation import validate_dataset, validate_row  # noqa: E402


class TestValidation(unittest.TestCase):
    def test_clean_row(self):
        self.assertEqual(validate_row(row()), [])

    def test_generated_dataset_clean(self):
        from generator import generate_scenario
        self.assertEqual(validate_dataset(generate_scenario("clean")), [])

    def test_missing_field(self):
        r = row()
        del r["outcome"]
        self.assertTrue(any("outcome" in i for i in validate_row(r)))

    def test_bool_is_not_a_number(self):
        self.assertTrue(validate_row(row(outcome=True)))
        self.assertTrue(validate_row(row(guardrail_error=True)))

    def test_guardrail_domain(self):
        self.assertTrue(validate_row(row(guardrail_error=2)))
        self.assertEqual(validate_row(row(guardrail_error=1)), [])

    def test_empty_arm_and_user(self):
        self.assertTrue(validate_row(row(arm="")))
        self.assertTrue(validate_row(row(user_id="")))

    def test_duplicate_user_flagged(self):
        rows = [row("user_00001", "control"), row("user_00001", "treatment")]
        self.assertTrue(any("duplicate" in i
                            for i in validate_dataset(rows)))

    def test_single_arm_flagged(self):
        rows = [row(f"user_{i:05d}", "control") for i in range(5)]
        self.assertTrue(any("fewer than 2 arms" in i
                            for i in validate_dataset(rows)))

    def test_empty_dataset(self):
        self.assertEqual(validate_dataset([]), ["dataset: no rows"])

    def test_non_dict_row(self):
        self.assertTrue(validate_row("nope"))


if __name__ == "__main__":
    unittest.main()
