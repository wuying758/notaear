import unittest

from notaear.validate_abc import validate_abc


class ABCValidationTests(unittest.TestCase):
    def test_full_bars_and_tonic_ending_are_accepted(self):
        report = validate_abc(
            "X:1\nM:4/4\nL:1/8\nK:C\n"
            "C2 D2 E2 F2 | G2 A2 B2 c2 |]\n"
        )
        self.assertTrue(report.accepted)
        self.assertTrue(report.rhythm_valid)
        self.assertEqual([bar.status for bar in report.bars], ["full", "full"])
        self.assertTrue(report.tonal["final_is_tonic"])

    def test_triplet_uses_performed_duration(self):
        report = validate_abc(
            "X:1\nM:4/4\nL:1/8\nK:C\n(3CDE F2 G2 A2 |]\n"
        )
        self.assertTrue(report.accepted)
        self.assertEqual(report.bars[0].actual_quarters, "4")

    def test_missing_unit_length_uses_abc_meter_default(self):
        report = validate_abc("X:1\nM:2/4\nK:C\nCDEFGABC |]\n")
        self.assertTrue(report.accepted)
        self.assertEqual(report.bars[0].actual_quarters, "2")

    def test_underfull_and_overfull_bars_are_rejected(self):
        under = validate_abc("X:1\nM:4/4\nL:1/8\nK:C\nC2 D2 |]\n")
        over = validate_abc("X:1\nM:2/4\nL:1/8\nK:C\nC8 |]\n")
        self.assertFalse(under.accepted)
        self.assertFalse(over.accepted)
        self.assertIn("bar_underfull", {issue.code for issue in under.issues})
        self.assertIn("bar_overfull", {issue.code for issue in over.issues})

    def test_complementary_pickup_is_accepted(self):
        report = validate_abc(
            "X:1\nM:4/4\nL:1/8\nK:G\n"
            "D2 | G2 A2 B2 c2 | d6 |]\n"
        )
        self.assertTrue(report.accepted)
        self.assertEqual(
            [bar.status for bar in report.bars],
            ["pickup", "full", "pickup_completion"],
        )

    def test_inline_key_change_updates_final_tonic(self):
        report = validate_abc(
            "X:1\nM:4/4\nL:1/8\nK:C\n"
            "C2D2E2F2|[K:G]G2A2B2G2|]\n"
        )
        self.assertTrue(report.accepted)
        self.assertEqual(report.tonal["key_change_count"], 1)
        self.assertEqual(report.tonal["final_key"], "G major")
        self.assertTrue(report.tonal["final_is_tonic"])

    def test_non_tonic_ending_is_warning_unless_strict(self):
        abc = "X:1\nM:2/4\nL:1/8\nK:C\nC2 D2 |]\n"
        normal = validate_abc(abc)
        strict = validate_abc(abc, require_final_tonic=True)
        self.assertTrue(normal.accepted)
        self.assertFalse(strict.accepted)
        self.assertIn("final_not_tonic", {issue.code for issue in normal.issues})

    def test_chords_are_rejected_for_single_voice_generation(self):
        report = validate_abc("X:1\nM:2/4\nL:1/8\nK:C\n[CEG]2 z2 |]\n")
        self.assertFalse(report.accepted)
        self.assertFalse(report.single_voice)
        self.assertIn("chord_events", {issue.code for issue in report.issues})

    def test_parser_warning_is_a_hard_failure(self):
        report = validate_abc(
            "X:1\nM:4/4\nL:1/8\nK:C\nSOSOO PMTuCO |]\n"
        )
        self.assertFalse(report.accepted)
        self.assertIn("abc_parser_output", {issue.code for issue in report.issues})


if __name__ == "__main__":
    unittest.main()
