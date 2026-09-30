import unittest

from notaear.difficulty import analyze_abc
from notaear.tonality import TonalEvent, analyze_tonality


class TonalityTests(unittest.TestCase):
    def test_declared_dorian_is_not_counted_as_chromatic(self):
        abc = (
            "X:1\nM:4/4\nL:1/8\nK:Ddor\n"
            "D2 E2 F2 G2 | A2 B2 c2 d2 | d2 c2 B2 A2 | G2 F2 E2 D2 |]\n"
        )
        report = analyze_abc(abc).to_dict()
        self.assertEqual(report["schema_version"], 2)
        self.assertEqual(report["tonality"]["primary"]["tonic"], "D")
        self.assertEqual(report["tonality"]["primary"]["mode"], "dorian")
        self.assertEqual(
            report["dimensions"]["tonal_language"]["chromatic_note_rate"], 0.0
        )

    def test_declared_key_can_be_hidden_for_independent_evaluation(self):
        abc = (
            "X:1\nM:4/4\nL:1/8\nK:Ddor\n"
            "D2 E2 F2 G2 | A2 B2 c2 d2 | d2 c2 B2 A2 | G2 F2 E2 D2 |]\n"
        )
        report = analyze_abc(abc, use_declared_tonality=False).tonality
        self.assertNotEqual(report["status"], "declared")
        self.assertEqual((report["primary"]["tonic"], report["primary"]["mode"]),
                         ("D", "dorian"))

    def test_minor_variants_use_pitch_and_direction_evidence(self):
        harmonic = (
            "X:1\nM:4/4\nL:1/8\nK:Am\n"
            "A2 B2 c2 d2 | e2 f2 ^g2 a2 | a2 ^g2 f2 e2 | d2 c2 B2 A2 |]\n"
        )
        melodic = (
            "X:1\nM:4/4\nL:1/8\nK:Am\n"
            "A2 B2 c2 d2 | e2 ^f2 ^g2 a2 | a2 g2 f2 e2 | d2 c2 B2 A2 |]\n"
        )
        self.assertEqual(
            analyze_abc(harmonic).tonality["primary"]["mode"], "harmonic_minor"
        )
        self.assertEqual(
            analyze_abc(melodic).tonality["primary"]["mode"], "melodic_minor"
        )

    def test_pentatonic_mode_can_be_inferred_but_keeps_candidates(self):
        abc = "X:1\nM:4/4\nL:1/4\nK:none\nC D E G | A G E D |]\n"
        pitches = [60, 62, 64, 67, 69, 67, 64, 62, 60, 64, 67, 69, 67, 64, 62, 60]
        events = [
            TonalEvent(midi, 1.0, index // 4, 1.0 if index % 4 == 0 else 0.5)
            for index, midi in enumerate(pitches)
        ]
        report = analyze_tonality(events, abc, 4)
        self.assertEqual((report.primary.tonic, report.primary.mode), ("C", "gong"))
        self.assertGreaterEqual(len(report.candidates), 2)
        self.assertLess(report.candidates[0].support, 1.0)
        self.assertFalse(report.confidence_calibrated)
        self.assertEqual(report.confidence_kind, "heuristic_fit_margin")

    def test_extended_key_fields_use_the_correct_parent_signature(self):
        shang = (
            "X:1\nM:4/4\nL:1/4\nK:Dshang\n"
            "D E G A | c A G E | D G A c | A G E D |]\n"
        )
        harmonic = (
            "X:1\nM:4/4\nL:1/4\nK:Aharmonicminor\n"
            "A B c d | e f ^g a | a ^g f e | d c B A |]\n"
        )
        shang_report = analyze_abc(shang)
        harmonic_report = analyze_abc(harmonic)
        self.assertEqual(shang_report.tonality["primary"]["mode"], "shang")
        self.assertEqual(harmonic_report.tonality["primary"]["mode"], "harmonic_minor")
        self.assertEqual(
            shang_report.dimensions["tonal_language"]["chromatic_note_rate"], 0.0
        )
        self.assertTrue(any("母调调号" in item for item in shang_report.warnings))

    def test_inline_key_change_produces_segments(self):
        abc = (
            "X:1\nM:4/4\nL:1/8\nK:C\n"
            "C2 D2 E2 F2 | G2 A2 B2 c2 | c2 B2 A2 G2 | F2 E2 D2 C2 |"
            "[K:Ddor] D2 E2 F2 G2 | A2 B2 c2 d2 | d2 c2 B2 A2 | G2 F2 E2 D2 |]\n"
        )
        report = analyze_abc(abc).tonality
        self.assertEqual(report["modulation_count"], 1)
        self.assertEqual(len(report["segments"]), 2)
        self.assertEqual(report["segments"][0]["primary"]["mode"], "major")
        self.assertEqual(report["segments"][1]["primary"]["mode"], "dorian")
        self.assertEqual(report["out_of_scale_note_rate"], 0.0)

    def test_short_material_is_rejected_as_insufficient(self):
        abc = "X:1\nM:4/4\nL:1/4\nK:none\nC D E2 |]\n"
        events = [TonalEvent(60, 1.0, 0), TonalEvent(62, 1.0, 0), TonalEvent(64, 2.0, 0)]
        report = analyze_tonality(events, abc, 1)
        self.assertEqual(report.status, "insufficient")
        self.assertIsNone(report.primary)


if __name__ == "__main__":
    unittest.main()
