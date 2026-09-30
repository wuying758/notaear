import unittest
from xml.etree import ElementTree as ET

from notaear.train_difficulty import (
    FEATURE_NAMES, _percentile, _pitch_signature, extract_musicxml_features,
    public_level,
)


def score(pitches: list[tuple[str, int]], triplet: bool = False) -> bytes:
    notes = []
    for step, octave in pitches:
        modification = (
            "<time-modification><actual-notes>3</actual-notes>"
            "<normal-notes>2</normal-notes></time-modification>"
            if triplet else ""
        )
        notes.append(
            f"<note><pitch><step>{step}</step><octave>{octave}</octave></pitch>"
            f"<duration>1</duration><voice>1</voice><type>eighth</type>{modification}</note>"
        )
    return (
        "<score-partwise><part-list><score-part id='P1'><part-name>P1</part-name>"
        "</score-part></part-list><part id='P1'><measure number='1'><attributes>"
        "<divisions>2</divisions><key><fifths>0</fifths><mode>major</mode></key>"
        "<time><beats>4</beats><beat-type>4</beat-type></time></attributes>"
        + "".join(notes)
        + "</measure></part></score-partwise>"
    ).encode()


class DifficultyTrainingTests(unittest.TestCase):
    def test_feature_vector_is_complete_and_detects_triplets(self):
        features, parts, _ = extract_musicxml_features(
            score([("C", 4), ("E", 4), ("G", 4)], triplet=True)
        )
        self.assertEqual(parts, 1)
        self.assertEqual(set(features), set(FEATURE_NAMES))
        self.assertGreater(features["melodic_motion.leap_rate_gt_major_second"], 0)
        self.assertEqual(features["rhythm_and_meter.tuplet_note_rate"], 1.0)

    def test_unpitched_rhythm_exercise_keeps_rhythm_features(self):
        xml = (
            "<score-partwise><part-list><score-part id='P1'><part-name>P1</part-name>"
            "</score-part></part-list><part id='P1'><measure number='1'><attributes>"
            "<divisions>2</divisions><time><beats>4</beats><beat-type>4</beat-type></time>"
            "</attributes><note><unpitched><display-step>C</display-step><display-octave>5</display-octave>"
            "</unpitched><duration>1</duration><voice>1</voice><type>eighth</type>"
            "<time-modification><actual-notes>3</actual-notes><normal-notes>2</normal-notes>"
            "</time-modification></note></measure></part></score-partwise>"
        ).encode()
        features, parts, _ = extract_musicxml_features(xml)
        self.assertEqual(parts, 1)
        self.assertEqual(features["melodic_motion.pitch_range_semitones"], 0.0)
        self.assertEqual(features["rhythm_and_meter.tuplet_note_rate"], 1.0)
        self.assertEqual(features["form_memory_and_breath.note_count"], 1.0)

    def test_duplicate_group_is_transposition_invariant(self):
        first = ET.fromstring(score([("C", 4), ("D", 4), ("E", 4)]))
        second = ET.fromstring(score([("F", 4), ("G", 4), ("A", 4)]))
        different = ET.fromstring(score([("C", 4), ("E", 4), ("G", 4)]))
        self.assertEqual(_pitch_signature(first), _pitch_signature(second))
        self.assertNotEqual(_pitch_signature(first), _pitch_signature(different))

    def test_zero_complexity_is_at_low_end_of_percentile(self):
        quantiles = [0.0] * 80 + [1.0] * 21
        self.assertEqual(_percentile(0.0, quantiles), 0.0)
        self.assertEqual(_percentile(1.0, quantiles), 80.0)
        self.assertEqual(_percentile(1.0, quantiles, invert=True), 0.0)

    def test_public_api_is_l1_to_l5_and_reports_borderline(self):
        clear = public_level([0.02, 0.08, 0.75, 0.10, 0.05])
        self.assertEqual(clear["level"], "L3")
        self.assertEqual(clear["alternative"], "L4")
        self.assertFalse(clear["borderline"])
        self.assertFalse(clear["confidence_calibrated"])
        self.assertEqual(set(clear["level_support"]), {"L1", "L2", "L3", "L4", "L5"})

        boundary = public_level([0.02, 0.08, 0.44, 0.40, 0.06])
        self.assertEqual(boundary["level"], "L3")
        self.assertEqual(boundary["alternative"], "L4")
        self.assertTrue(boundary["borderline"])


if __name__ == "__main__":
    unittest.main()
