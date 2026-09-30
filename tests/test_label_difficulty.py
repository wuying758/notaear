import unittest

import numpy as np

from notaear.label_difficulty import Candidate, _public_pair, _select_pairs
from notaear.train_pairwise_difficulty import (
    _baseline_expected_series,
    _best_tie_threshold,
)


class FixedProbabilityModel:
    def predict_proba(self, matrix):
        return np.asarray([[0.1, 0.2, 0.4, 0.2, 0.1] for _ in matrix])


def candidate(index, source, expected, uncertainty):
    dimensions = {
        "tonal_language": float(index % 100),
        "melodic_motion": float((index * 3) % 100),
        "rhythm_and_meter": float((index * 7) % 100),
        "notation_and_reading": float((index * 11) % 100),
        "form_memory_and_breath": float((index * 13) % 100),
    }
    return Candidate(str(index), source, "book.zip", f"{index}.xml", str(index), {},
                     expected, uncertainty, 3, dimensions)


class LabelDifficultyTest(unittest.TestCase):
    def test_pairwise_baseline_is_recomputed_from_selected_model(self):
        from notaear.train_difficulty import FEATURE_NAMES

        bundle = {
            "feature_names": FEATURE_NAMES,
            "model": FixedProbabilityModel(),
        }
        features = {name: 0.0 for name in FEATURE_NAMES}
        self.assertEqual(_baseline_expected_series(bundle, features), 5.0)

    def test_queue_is_deterministic_and_has_unique_pairs(self):
        candidates = []
        for index in range(160):
            source = "eyes_and_ears" if index < 80 else "melodia"
            candidates.append(candidate(index, source, 1.0 + (index % 40) / 5, 0.3 + (index % 20) / 30))
        first = _select_pairs(candidates, 20, 42)
        second = _select_pairs(candidates, 20, 42)
        self.assertEqual(first, second)
        self.assertEqual(20, len({pair["pair_id"] for pair in first}))
        self.assertTrue(all(pair["left"]["candidate_id"] != pair["right"]["candidate_id"] for pair in first))

    def test_public_pair_hides_model_predictions_and_sources(self):
        left = candidate(1, "melodia", 5.1, 0.7)
        right = candidate(2, "eyes_and_ears", 5.2, 0.8)
        pair = {"pair_id": "p", "left": left.__dict__, "right": right.__dict__}
        public = _public_pair(pair, 0, 1)
        self.assertEqual({"candidate_id": "1"}, public["left"])
        self.assertNotIn("expected_series", public["left"])
        self.assertNotIn("source", public["left"])

    def test_tie_threshold_uses_equal_comparisons(self):
        scores = np.asarray([-2.0, -0.1, 0.05, 1.5])
        labels = np.asarray([-1, 0, 0, 1])
        weights = np.ones(4)
        threshold = _best_tie_threshold(scores, labels, weights)
        self.assertGreaterEqual(threshold, 0.1)
        self.assertLess(threshold, 1.5)


if __name__ == "__main__":
    unittest.main()
