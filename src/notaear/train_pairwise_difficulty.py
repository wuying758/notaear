"""Train a human pairwise correction head on top of the Melodia baseline."""
from __future__ import annotations

import argparse
import json
import math
import platform
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .label_difficulty import _load_candidates
from .train_difficulty import FEATURE_NAMES, _correlation


def _connected_groups(pairs: dict[str, dict]) -> dict[str, str]:
    parent: dict[str, str] = {}

    def find(item: str) -> str:
        parent.setdefault(item, item)
        if parent[item] != item:
            parent[item] = find(parent[item])
        return parent[item]

    def union(left: str, right: str) -> None:
        left, right = find(left), find(right)
        if left != right:
            parent[right] = left

    for pair in pairs.values():
        union(pair["left"]["candidate_id"], pair["right"]["candidate_id"])
    return {item: find(item) for item in parent}


def _best_tie_threshold(scores, labels, weights) -> float:
    import numpy as np

    absolute = np.unique(np.abs(scores))
    candidates = np.r_[0.0, absolute, (absolute[:-1] + absolute[1:]) / 2.0]
    best = (float("-inf"), 0.0)
    for threshold in candidates:
        predicted = np.where(np.abs(scores) <= threshold, 0, np.where(scores > 0, 1, -1))
        accuracy = float(np.average(predicted == labels, weights=weights))
        if accuracy > best[0] or (accuracy == best[0] and threshold < best[1]):
            best = accuracy, float(threshold)
    return best[1]


def _make_model(c: float):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), LogisticRegression(
        C=c, fit_intercept=False, max_iter=2000, random_state=20260928))


def _fit(model, x, y, weights):
    import numpy as np

    model.fit(np.r_[x, -x], np.r_[y, 1 - y],
              logisticregression__sample_weight=np.r_[weights, weights])
    return model


def _metrics(labels, predicted, weights) -> dict[str, float]:
    import numpy as np

    return {
        "accuracy": float(np.mean(predicted == labels)),
        "confidence_weighted_accuracy": float(np.average(predicted == labels, weights=weights)),
    }


def _baseline_expected_series(bundle, features: dict[str, float]) -> float:
    import numpy as np

    names = bundle["feature_names"]
    vector = np.asarray([[features[name] for name in names]], dtype=np.float32)
    probabilities = bundle["model"].predict_proba(vector)[0]
    return 1.0 + 2.0 * float(probabilities @ np.arange(len(probabilities)))


def _load_rows(queue_path: Path, annotation_path: Path, baseline_bundle):
    import numpy as np

    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    pairs = {pair["pair_id"]: pair for pair in queue["pairs"]}
    annotations = [json.loads(line) for line in annotation_path.read_text(encoding="utf-8").splitlines()
                   if line.strip()]
    groups = _connected_groups(pairs)
    records = []
    for annotation in annotations:
        pair = pairs.get(annotation["pair_id"])
        if pair is None or annotation["judgement"] == "skip":
            continue
        label = {"right_harder": -1, "equal": 0, "left_harder": 1}[annotation["judgement"]]
        difference = np.asarray([
            pair["left"]["features"][name] - pair["right"]["features"][name]
            for name in FEATURE_NAMES
        ], dtype=np.float64)
        records.append({
            "pair": pair,
            "difference": difference,
            "label": label,
            "weight": float(annotation["confidence"]),
            "group": groups[pair["left"]["candidate_id"]],
            "baseline_difference": (
                _baseline_expected_series(baseline_bundle, pair["left"]["features"])
                - _baseline_expected_series(baseline_bundle, pair["right"]["features"])
            ),
            "listened": annotation.get("listened", {"left": False, "right": False}),
        })
    return queue, annotations, records


def _cross_validate(records, folds: int, c: float, alpha: float, seed: int) -> dict[str, Any]:
    import numpy as np
    from sklearn.metrics import log_loss
    from sklearn.model_selection import StratifiedGroupKFold

    x = np.asarray([record["difference"] for record in records])
    labels = np.asarray([record["label"] for record in records], dtype=np.int64)
    weights = np.asarray([record["weight"] for record in records])
    groups = np.asarray([record["group"] for record in records])
    baseline = np.asarray([record["baseline_difference"] for record in records])
    head_scores = np.zeros(len(records), dtype=np.float64)
    hybrid_scores = np.zeros(len(records), dtype=np.float64)
    three_way = np.zeros(len(records), dtype=np.int64)
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    for train_indices, validation_indices in splitter.split(x, labels, groups):
        train_directional = train_indices[labels[train_indices] != 0]
        model = _fit(_make_model(c), x[train_directional],
                     (labels[train_directional] > 0).astype(int), weights[train_directional])
        head_scores[validation_indices] = model.decision_function(x[validation_indices])
        train_scores = baseline[train_indices] + alpha * model.decision_function(x[train_indices])
        hybrid_scores[validation_indices] = (baseline[validation_indices]
                                             + alpha * head_scores[validation_indices])
        threshold = _best_tie_threshold(train_scores, labels[train_indices], weights[train_indices])
        scores = hybrid_scores[validation_indices]
        three_way[validation_indices] = np.where(
            np.abs(scores) <= threshold, 0, np.where(scores > 0, 1, -1))

    directional = labels != 0
    binary_labels = (labels[directional] > 0).astype(int)
    head_probability = 1.0 / (1.0 + np.exp(-head_scores[directional]))
    report = {
        "folds": folds,
        "group": "connected score-comparison component",
        "directional_examples": int(directional.sum()),
        "equal_examples": int((labels == 0).sum()),
        "baseline_directional": _metrics(
            binary_labels, (baseline[directional] >= 0).astype(int), weights[directional]),
        "pairwise_head_directional": {
            **_metrics(binary_labels, (head_scores[directional] >= 0).astype(int), weights[directional]),
            "confidence_weighted_log_loss": float(log_loss(
                binary_labels, head_probability, sample_weight=weights[directional])),
        },
        "hybrid_directional": _metrics(
            binary_labels, (hybrid_scores[directional] >= 0).astype(int), weights[directional]),
        "hybrid_three_way": {
            **_metrics(labels, three_way, weights),
            "equal_recall": float(np.mean(three_way[labels == 0] == 0)) if np.any(labels == 0) else 0.0,
        },
    }
    return report


def _external_order_validation(model, baseline_bundle, queue: dict, alpha: float,
                               melodia: Path, eyes: Path) -> dict[str, Any]:
    import numpy as np

    used = {side["candidate_id"] for pair in queue["pairs"]
            for side in (pair["left"], pair["right"])}
    result = {}
    sources = (("melodia", melodia), ("eyes_and_ears", eyes))
    for source, path in sources:
        candidates = [candidate for candidate in _load_candidates(source, path, baseline_bundle)
                      if candidate.candidate_id not in used]
        matrix = np.asarray([[candidate.features[name] for name in FEATURE_NAMES]
                             for candidate in candidates], dtype=np.float64)
        baseline = np.asarray([candidate.expected_series for candidate in candidates])
        hybrid = baseline + alpha * model.decision_function(matrix)
        if source == "melodia":
            target = [int(re.search(r"_(\d+)-(\d+)-(\d+)", item.member).group(2))
                      for item in candidates]
        else:
            target = [int(re.search(r"_(\d+)\.xml$", item.member).group(1))
                      for item in candidates]
        result[source] = {
            "unseen_examples": len(candidates),
            "baseline_spearman": float(_correlation(target, baseline.tolist())),
            "hybrid_spearman": float(_correlation(target, hybrid.tolist())),
        }
    return result


def train(queue_path: Path, annotation_path: Path, baseline_path: Path, output_dir: Path,
          melodia: Path, eyes: Path, folds: int = 5, c: float = 0.1,
          alpha: float = 0.3, seed: int = 20260928) -> dict[str, Any]:
    import joblib
    import numpy as np
    import sklearn

    baseline_bundle = joblib.load(baseline_path)
    queue, annotations, records = _load_rows(
        queue_path, annotation_path, baseline_bundle
    )
    directional = [record for record in records if record["label"] != 0]
    if len(directional) < 20:
        raise ValueError("at least 20 directional human comparisons are required")
    x = np.asarray([record["difference"] for record in directional])
    y = np.asarray([record["label"] > 0 for record in directional], dtype=np.int64)
    weights = np.asarray([record["weight"] for record in directional])
    model = _fit(_make_model(c), x, y, weights)
    all_x = np.asarray([record["difference"] for record in records])
    all_labels = np.asarray([record["label"] for record in records], dtype=np.int64)
    all_weights = np.asarray([record["weight"] for record in records])
    baseline_differences = np.asarray([record["baseline_difference"] for record in records])
    hybrid_scores = baseline_differences + alpha * model.decision_function(all_x)
    tie_threshold = _best_tie_threshold(hybrid_scores, all_labels, all_weights)
    cross_validation = _cross_validate(records, folds, c, alpha, seed)
    external = _external_order_validation(model, baseline_bundle, queue, alpha, melodia, eyes)
    coefficients = model.named_steps["logisticregression"].coef_[0]
    top = sorted(zip(FEATURE_NAMES, coefficients), key=lambda item: abs(item[1]), reverse=True)
    report = {
        "schema_version": 2,
        "model": "Melodia baseline + human pairwise logistic correction",
        "artifact_environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "baseline_model": str(baseline_path),
        "annotation_queue_model_version": queue["model_version"],
        "annotations": {
            "total": len(annotations),
            "directional": len(directional),
            "equal": sum(record["label"] == 0 for record in records),
            "skipped": sum(item["judgement"] == "skip" for item in annotations),
            "judgements": dict(Counter(item["judgement"] for item in annotations)),
            "both_listened": sum(all(record["listened"].values()) for record in records),
        },
        "hyperparameters": {"C": c, "alpha": alpha, "tie_threshold": tie_threshold, "seed": seed},
        "cross_validation": cross_validation,
        "unseen_curriculum_order": external,
        "top_standardized_coefficients": [
            {"feature": name, "coefficient": float(value)} for name, value in top[:15]
        ],
        "decision": "candidate_correction_head_not_full_baseline_replacement",
        "limitations": [
            "Only one annotator and 49 directional comparisons are available.",
            "Most comparisons were made after listening to both examples.",
            "Alpha was selected on the current human and held-out curriculum diagnostics; future evaluation must be independent.",
        ],
    }
    bundle = {
        "schema_version": 2,
        "model": model,
        "feature_names": FEATURE_NAMES,
        "alpha": alpha,
        "tie_threshold": tie_threshold,
        "baseline_model": str(baseline_path),
        "report": report,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output_dir / "pairwise_correction.joblib", compress=3)
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=Path("out/active-label-v0/queue.json"))
    parser.add_argument("--annotations", type=Path,
                        default=Path("data/annotations/difficulty_pairs.jsonl"))
    parser.add_argument("--baseline", type=Path,
                        default=Path("out/difficulty-melodia-v1/difficulty_model.joblib"))
    parser.add_argument("--melodia", type=Path,
                        default=Path("data/reference_books/raw/melodia/Cole-Lewis-1904-xml.zip"))
    parser.add_argument("--eyes", type=Path,
                        default=Path("data/reference_books/raw/eyes_and_ears/Crowell-2004-xml.zip"))
    parser.add_argument("--output", type=Path, default=Path("out/difficulty-human-v2"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--c", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=20260928)
    args = parser.parse_args(argv)
    report = train(args.queue, args.annotations, args.baseline, args.output,
                   args.melodia, args.eyes, args.folds, args.c, args.alpha, args.seed)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
