import unittest

from notaear.difficulty import analyze_abc


class DifficultyTests(unittest.TestCase):
    def test_reports_dimensions_without_collapsing_to_level(self):
        abc = "X:1\nM:4/4\nL:1/8\nK:C\nC2 D2 E2 F2 | G2 A2 B2 c2 |]\n"
        report = analyze_abc(abc).to_dict()
        self.assertEqual(
            set(report["dimensions"]),
            {
                "tonal_language", "melodic_motion", "rhythm_and_meter",
                "notation_and_reading", "form_memory_and_breath",
            },
        )
        self.assertNotIn("level", report)
        self.assertEqual(report["dimensions"]["melodic_motion"]["pitch_range_semitones"], 12)

    def test_triplet_is_a_rhythm_feature_not_a_control_label(self):
        abc = "X:1\nM:4/4\nL:1/8\nK:C\n(3CDE F2 G2 A2 | C8 |]\n"
        report = analyze_abc(abc)
        rhythm = report.dimensions["rhythm_and_meter"]
        self.assertEqual(rhythm["tuplet_group_count"], 1)
        self.assertGreater(rhythm["tuplet_note_rate"], 0)

    def test_legacy_bare_inline_meter_is_repaired_only_for_parsing(self):
        abc = (
            "X:1\nM:3/4\nL:1/4\nK:C\na|:\n"
            "|:g2e|d3/2e/2f|e2c|d2a|g2e|d3/2e/2f|"
            "M:2/4e_B|M:3/4c2a:||:c'2a|_b2g|f2d|_B2a|"
            "c'2a|_b2g|M:2/4fd|M:3/4 [1 c2a:|:|[2c2z||\n"
        )
        report = analyze_abc(abc)
        self.assertTrue(any("裸内联 M:" in warning for warning in report.warnings))
        self.assertGreater(report.dimensions["rhythm_and_meter"]["meter_change_count"], 0)

    def test_legacy_bare_title_before_key_is_ignored_only_for_parsing(self):
        abc = (
            "X:1\ndAo Maria\nL:1/8\nM:4/4\nK:C\n"
            "C2 D2 E2 F2 | G2 A2 B2 c2 |]\n"
        )
        report = analyze_abc(abc)
        self.assertTrue(any("裸标题文本" in warning for warning in report.warnings))
        self.assertEqual(report.dimensions["form_memory_and_breath"]["note_count"], 8)


if __name__ == "__main__":
    unittest.main()
