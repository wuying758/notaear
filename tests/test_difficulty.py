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


if __name__ == "__main__":
    unittest.main()

