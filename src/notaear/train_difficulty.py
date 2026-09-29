"""Train and apply a weakly supervised Melodia difficulty baseline.

Only the one-part Series 1, 3, 5, 7, and 9 are used. The alternating two-part
series measure ensemble independence and are deliberately kept out of this
single-line sight-singing model.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
import math
import re
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from io import BytesIO
from pathlib import Path
from statistics import mean
from typing import Any
from xml.etree import ElementTree as ET
from zipfile import ZipFile


STAGE_SERIES = (1, 3, 5, 7, 9)
FILENAME_RE = re.compile(r"_(\d+)-(\d+)-(\d+)(?:-(\d+))?\.xml$")
PITCH_CLASS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

FEATURE_NAMES = (
    "tonal_language.key_signature_accidentals",
    "tonal_language.key_change_count",
    "tonal_language.chromatic_note_rate",
    "tonal_language.distinct_pitch_classes",
    "tonal_language.final_is_tonic",
    "melodic_motion.pitch_range_semitones",
    "melodic_motion.mean_abs_interval_semitones",
    "melodic_motion.max_abs_interval_semitones",
    "melodic_motion.leap_rate_gt_major_second",
    "melodic_motion.large_leap_rate_gt_perfect_fifth",
    "melodic_motion.consecutive_leap_rate",
    "melodic_motion.direction_change_rate",
    "melodic_motion.repeated_note_rate",
    "rhythm_and_meter.meter_change_count",
    "rhythm_and_meter.shortest_note_quarters",
    "rhythm_and_meter.duration_class_count",
    "rhythm_and_meter.duration_entropy_bits",
    "rhythm_and_meter.tuplet_note_rate",
    "rhythm_and_meter.dotted_note_rate",
    "rhythm_and_meter.offbeat_attack_rate",
    "rhythm_and_meter.syncopated_attack_rate",
    "rhythm_and_meter.rest_event_rate",
    "notation_and_reading.accidental_marks_per_note",
    "notation_and_reading.ornament_count",
    "notation_and_reading.grace_group_count",
    "notation_and_reading.repeat_barline_count",
    "notation_and_reading.volta_count",
    "notation_and_reading.slur_start_count",
    "notation_and_reading.bar_patch_count",
    "form_memory_and_breath.note_count",
    "form_memory_and_breath.notes_per_bar",
    "form_memory_and_breath.motif_interval_3gram_repetition_rate",
    "form_memory_and_breath.max_notes_without_long_rest",
    "form_memory_and_breath.max_quarters_without_long_rest",
    "form_memory_and_breath.tie_attack_rate",
)

DIMENSION_FEATURES = {
    "tonal_language": (
        "tonal_language.key_signature_accidentals",
        "tonal_language.key_change_count",
        "tonal_language.chromatic_note_rate",
        "tonal_language.distinct_pitch_classes",
    ),
    "melodic_motion": (
        "melodic_motion.pitch_range_semitones",
        "melodic_motion.mean_abs_interval_semitones",
        "melodic_motion.max_abs_interval_semitones",
        "melodic_motion.leap_rate_gt_major_second",
        "melodic_motion.large_leap_rate_gt_perfect_fifth",
        "melodic_motion.consecutive_leap_rate",
    ),
    "rhythm_and_meter": (
        "rhythm_and_meter.shortest_note_quarters",
        "rhythm_and_meter.duration_class_count",
        "rhythm_and_meter.duration_entropy_bits",
        "rhythm_and_meter.tuplet_note_rate",
        "rhythm_and_meter.dotted_note_rate",
        "rhythm_and_meter.offbeat_attack_rate",
        "rhythm_and_meter.syncopated_attack_rate",
    ),
    "notation_and_reading": (
        "notation_and_reading.accidental_marks_per_note",
        "notation_and_reading.ornament_count",
        "notation_and_reading.grace_group_count",
        "notation_and_reading.repeat_barline_count",
        "notation_and_reading.volta_count",
        "notation_and_reading.slur_start_count",
    ),
    "form_memory_and_breath": (
        "form_memory_and_breath.note_count",
        "form_memory_and_breath.notes_per_bar",
        "form_memory_and_breath.max_notes_without_long_rest",
        "form_memory_and_breath.max_quarters_without_long_rest",
    ),
}

INVERTED_PERCENTILES = {"rhythm_and_meter.shortest_note_quarters"}


@dataclass(frozen=True)
class MelodiaExample:
    name: str
    book: int
    series: int
    exercise: int
    subexercise: int | None
    duplicate_group: str
    features: dict[str, float]


def _rate(count: int | float, total: int) -> float:
    return float(count) / total if total else 0.0


def _entropy(values: list[Fraction]) -> float:
    counts = Counter(values)
    total = len(values)
    return -sum((count / total) * math.log2(count / total) for count in counts.values()) if total else 0.0


def _motif_repetition_rate(pitches: list[int], order: int = 3) -> float:
    intervals = [right - left for left, right in zip(pitches, pitches[1:])]
    if len(intervals) < order:
        return 0.0
    motifs = [tuple(intervals[index:index + order]) for index in range(len(intervals) - order + 1)]
    counts = Counter(motifs)
    return _rate(sum(count for count in counts.values() if count > 1), len(motifs))


def _beat_quarters(beats: int, beat_type: int) -> Fraction:
    if beats > 3 and beats % 3 == 0 and beat_type in (8, 16):
        return Fraction(12, beat_type)
    return Fraction(4, beat_type)


def _scale_pitch_classes(fifths: int, mode: str) -> tuple[int, set[int]]:
    if mode == "minor":
        tonic, offsets = (9 + 7 * fifths) % 12, {0, 2, 3, 5, 7, 8, 10}
    else:
        tonic, offsets = (7 * fifths) % 12, {0, 2, 4, 5, 7, 9, 11}
    return tonic, {(tonic + offset) % 12 for offset in offsets}


def _pitch_signature(root: ET.Element) -> str:
    """Conservative transposition-invariant group for leakage-safe splitting."""
    parts: list[tuple[int | str, ...]] = []
    for part in root.findall("part"):
        first: int | None = None
        events: list[int | str] = []
        for note in part.findall("./measure/note"):
            if note.find("rest") is not None:
                events.append("r")
                continue
            pitch = note.find("pitch")
            if pitch is None:
                continue
            midi = ((int(pitch.findtext("octave", "0")) + 1) * 12
                    + PITCH_CLASS[pitch.findtext("step", "C")]
                    + int(float(pitch.findtext("alter", "0"))))
            if first is None:
                first = midi
            events.append(midi - first)
        parts.append(tuple(events))
    return hashlib.sha256(repr(tuple(parts)).encode()).hexdigest()


def extract_musicxml_features(xml: bytes) -> tuple[dict[str, float], int, str]:
    """Extract features compatible with :func:`difficulty.analyze_abc`."""
    root = ET.fromstring(xml)
    parts = root.findall("part")
    pitches: list[int] = []
    durations: list[Fraction] = []
    note_durations: list[Fraction] = []
    phrase_notes: list[int] = []
    phrase_quarters: list[Fraction] = []
    intervals: list[int] = []
    key_events: list[tuple[int, str]] = []
    meter_events: list[tuple[int, int]] = []
    rest_count = dotted_count = tuplet_count = accidental_count = 0
    ornament_count = grace_count = repeat_count = volta_count = slur_count = tie_count = 0
    attack_count = bar_count = offbeat = syncopated = 0

    for part in parts:
        divisions, beats, beat_type = 1, 4, 4
        previous_by_voice: dict[str, int] = {}
        running_notes, running_quarters = 0, Fraction(0)
        for measure in part.findall("measure"):
            bar_count += 1
            cursor = last_onset = Fraction(0)
            for element in list(measure):
                if element.tag == "attributes":
                    if element.findtext("divisions"):
                        divisions = int(element.findtext("divisions", "1"))
                    key = element.find("key")
                    if key is not None and key.findtext("fifths") is not None:
                        key_events.append((int(key.findtext("fifths", "0")), key.findtext("mode", "major")))
                    time = element.find("time")
                    if time is not None and time.findtext("beats") and time.findtext("beat-type"):
                        beats = int(time.findtext("beats", "4"))
                        beat_type = int(time.findtext("beat-type", "4"))
                        meter_events.append((beats, beat_type))
                    continue
                if element.tag in {"backup", "forward"}:
                    amount = Fraction(int(element.findtext("duration", "0")), divisions)
                    cursor += amount if element.tag == "forward" else -amount
                    continue
                if element.tag == "barline":
                    repeat_count += len(element.findall("repeat"))
                    volta_count += len(element.findall("ending"))
                    continue
                if element.tag != "note":
                    continue
                duration = Fraction(int(element.findtext("duration", "0")), divisions)
                durations.append(duration)
                is_chord = element.find("chord") is not None
                onset = last_onset if is_chord else cursor
                if not is_chord:
                    last_onset = onset
                    cursor += duration
                beat = _beat_quarters(beats, beat_type)
                if element.find("rest") is not None:
                    rest_count += 1
                    running_quarters += duration
                    if duration >= beat and running_notes:
                        phrase_notes.append(running_notes)
                        phrase_quarters.append(running_quarters - duration)
                        running_notes, running_quarters = 0, Fraction(0)
                    continue
                tie_types = {tie.get("type") for tie in element.findall("tie")}
                tie_count += int("start" in tie_types)
                if "continue" in tie_types or "stop" in tie_types:
                    continue
                attack_count += 1
                note_durations.append(duration)
                running_notes += 1
                running_quarters += duration
                dotted_count += int(element.find("dot") is not None)
                tuplet_count += int(element.find("time-modification") is not None)
                accidental_count += int(element.find("accidental") is not None)
                grace_count += int(element.find("grace") is not None)
                slur_count += sum(1 for slur in element.findall("./notations/slur") if slur.get("type") == "start")
                ornament_count += len(list(element.findall("./notations/ornaments/*")))
                position = onset % beat if beat else Fraction(0)
                if position:
                    offbeat += 1
                    syncopated += int(position + duration > beat)
                pitch = element.find("pitch")
                if pitch is None:
                    continue
                midi = ((int(pitch.findtext("octave", "0")) + 1) * 12
                        + PITCH_CLASS[pitch.findtext("step", "C")]
                        + int(float(pitch.findtext("alter", "0"))))
                voice = element.findtext("voice", "1")
                if not is_chord and voice in previous_by_voice:
                    intervals.append(midi - previous_by_voice[voice])
                if not is_chord:
                    previous_by_voice[voice] = midi
                pitches.append(midi)
        if running_notes:
            phrase_notes.append(running_notes)
            phrase_quarters.append(running_quarters)

    if not attack_count:
        raise ValueError("MusicXML has no note attacks")
    absolute = [abs(value) for value in intervals]
    leap_flags = [value > 2 for value in absolute]
    consecutive = sum(left and right for left, right in zip(leap_flags, leap_flags[1:]))
    direction_changes = sum(left * right < 0 for left, right in zip(intervals, intervals[1:]) if left and right)
    repeated = sum(value == 0 for value in intervals)
    first_fifths, first_mode = key_events[0] if key_events else (0, "major")
    tonic, scale = _scale_pitch_classes(first_fifths, first_mode)
    chromatic = sum(pitch % 12 not in scale for pitch in pitches)
    event_count, note_count = len(durations), attack_count
    features = {
        "tonal_language.key_signature_accidentals": float(abs(first_fifths)),
        "tonal_language.key_change_count": float(max(0, len(key_events) - 1)),
        "tonal_language.chromatic_note_rate": _rate(chromatic, note_count),
        "tonal_language.distinct_pitch_classes": float(len({pitch % 12 for pitch in pitches})),
        "tonal_language.final_is_tonic": float(bool(pitches) and pitches[-1] % 12 == tonic),
        "melodic_motion.pitch_range_semitones": float(max(pitches) - min(pitches)) if pitches else 0.0,
        "melodic_motion.mean_abs_interval_semitones": mean(absolute) if absolute else 0.0,
        "melodic_motion.max_abs_interval_semitones": float(max(absolute, default=0)),
        "melodic_motion.leap_rate_gt_major_second": _rate(sum(leap_flags), len(absolute)),
        "melodic_motion.large_leap_rate_gt_perfect_fifth": _rate(sum(value > 7 for value in absolute), len(absolute)),
        "melodic_motion.consecutive_leap_rate": _rate(consecutive, max(0, len(leap_flags) - 1)),
        "melodic_motion.direction_change_rate": _rate(direction_changes, max(0, len(intervals) - 1)),
        "melodic_motion.repeated_note_rate": _rate(repeated, len(intervals)),
        "rhythm_and_meter.meter_change_count": float(max(0, len(meter_events) - 1)),
        "rhythm_and_meter.shortest_note_quarters": float(min(note_durations, default=Fraction(0))),
        "rhythm_and_meter.duration_class_count": float(len(set(durations))),
        "rhythm_and_meter.duration_entropy_bits": _entropy(durations),
        "rhythm_and_meter.tuplet_note_rate": _rate(tuplet_count, note_count),
        "rhythm_and_meter.dotted_note_rate": _rate(dotted_count, note_count),
        "rhythm_and_meter.offbeat_attack_rate": _rate(offbeat, note_count),
        "rhythm_and_meter.syncopated_attack_rate": _rate(syncopated, note_count),
        "rhythm_and_meter.rest_event_rate": _rate(rest_count, event_count),
        "notation_and_reading.accidental_marks_per_note": _rate(accidental_count, note_count),
        "notation_and_reading.ornament_count": float(ornament_count),
        "notation_and_reading.grace_group_count": float(grace_count),
        "notation_and_reading.repeat_barline_count": float(repeat_count),
        "notation_and_reading.volta_count": float(volta_count),
        "notation_and_reading.slur_start_count": float(slur_count),
        "notation_and_reading.bar_patch_count": float(bar_count),
        "form_memory_and_breath.note_count": float(note_count),
        "form_memory_and_breath.notes_per_bar": _rate(note_count, bar_count),
        "form_memory_and_breath.motif_interval_3gram_repetition_rate": _motif_repetition_rate(pitches),
        "form_memory_and_breath.max_notes_without_long_rest": float(max(phrase_notes, default=note_count)),
        "form_memory_and_breath.max_quarters_without_long_rest": float(max(phrase_quarters, default=sum(note_durations))),
        "form_memory_and_breath.tie_attack_rate": _rate(tie_count, note_count + tie_count),
    }
    return features, len(parts), _pitch_signature(root)


def load_melodia(zip_path: Path) -> tuple[list[MelodiaExample], dict[str, int]]:
    examples: list[MelodiaExample] = []
    counts = Counter()
    with ZipFile(zip_path) as archive:
        for name in archive.namelist():
            match = FILENAME_RE.search(name)
            if not match:
                continue
            book, series, exercise = map(int, match.groups()[:3])
            subexercise = int(match.group(4)) if match.group(4) else None
            features, part_count, group = extract_musicxml_features(archive.read(name))
            counts["musicxml_files"] += 1
            if part_count != 1:
                counts["excluded_not_one_part"] += 1
                continue
            if series not in STAGE_SERIES:
                counts["excluded_non_target_series"] += 1
                continue
            examples.append(MelodiaExample(name, book, series, exercise, subexercise, group, features))
    counts["training_examples"] = len(examples)
    counts["duplicate_groups"] = len({example.duplicate_group for example in examples})
    return examples, dict(counts)


def _flatten_report(report: dict[str, Any]) -> dict[str, float]:
    dimensions = report.get("dimensions", report)
    values: dict[str, float] = {}
    for name in FEATURE_NAMES:
        dimension, feature = name.split(".", 1)
        raw = dimensions.get(dimension, {}).get(feature, 0.0)
        values[name] = 0.0 if raw is None or isinstance(raw, str) else float(raw)
    return values


def _rank(values):
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        value = (cursor + end - 1) / 2 + 1
        for index in order[cursor:end]:
            ranks[index] = value
        cursor = end
    return ranks


def _correlation(left, right) -> float:
    left, right = _rank(list(left)), _rank(list(right))
    left_mean, right_mean = mean(left), mean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    left_sum = sum((x - left_mean) ** 2 for x in left)
    right_sum = sum((y - right_mean) ** 2 for y in right)
    return numerator / math.sqrt(left_sum * right_sum) if left_sum and right_sum else 0.0


def _calibration(matrix) -> dict[str, list[float]]:
    import numpy as np
    probabilities = np.linspace(0.0, 1.0, 101)
    return {name: [float(value) for value in np.quantile(matrix[:, index], probabilities)]
            for index, name in enumerate(FEATURE_NAMES)}


def _percentile(value: float, quantiles: list[float], invert: bool = False) -> float:
    if not quantiles or quantiles[0] == quantiles[-1]:
        return 0.0
    if invert:
        rank = 101 - bisect.bisect_right(quantiles, value)
    else:
        rank = bisect.bisect_left(quantiles, value)
    return min(100.0, max(0.0, float(rank)))


def score_dimensions(features: dict[str, float], calibration: dict[str, list[float]]) -> dict[str, float]:
    return {dimension: round(mean(
                _percentile(features.get(name, 0.0), calibration[name], name in INVERTED_PERCENTILES)
                for name in names
            ), 2)
            for dimension, names in DIMENSION_FEATURES.items()}


def train(zip_path: Path, output_dir: Path, folds: int, trees: int, seed: int) -> dict[str, Any]:
    import joblib
    import numpy as np
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import StratifiedGroupKFold

    examples, inventory = load_melodia(zip_path)
    if not examples:
        raise ValueError("no eligible one-part Melodia examples found")
    x = np.asarray([[example.features[name] for name in FEATURE_NAMES] for example in examples], dtype=np.float32)
    series_to_stage = {series: index for index, series in enumerate(STAGE_SERIES)}
    y = np.asarray([series_to_stage[example.series] for example in examples], dtype=np.int64)
    groups = np.asarray([example.duplicate_group for example in examples])
    class_counts = Counter(int(value) for value in y)
    weights = np.asarray([len(y) / (len(class_counts) * class_counts[int(value)]) for value in y])
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    probabilities = np.zeros((len(examples), len(STAGE_SERIES)), dtype=np.float64)
    fold_reports = []
    for fold, (train_indices, validation_indices) in enumerate(splitter.split(x, y, groups), start=1):
        model = ExtraTreesClassifier(n_estimators=trees, min_samples_leaf=2, max_features=0.8,
                                     class_weight="balanced", n_jobs=-1, random_state=seed + fold)
        model.fit(x[train_indices], y[train_indices], sample_weight=weights[train_indices])
        fold_probabilities = model.predict_proba(x[validation_indices])
        for column, class_id in enumerate(model.classes_):
            probabilities[validation_indices, class_id] = fold_probabilities[:, column]
        fold_predicted = probabilities[validation_indices].argmax(axis=1)
        fold_reports.append({
            "fold": fold,
            "examples": int(len(validation_indices)),
            "exact_accuracy": float(np.mean(fold_predicted == y[validation_indices])),
            "within_one_stage_accuracy": float(np.mean(np.abs(fold_predicted - y[validation_indices]) <= 1)),
            "mean_absolute_stage_error": float(np.mean(np.abs(fold_predicted - y[validation_indices]))),
        })
    predicted = probabilities.argmax(axis=1)
    expected = probabilities @ np.arange(len(STAGE_SERIES), dtype=np.float64)
    entropy = -np.sum(np.where(probabilities > 0, probabilities * np.log(probabilities + 1e-12), 0.0), axis=1) / math.log(len(STAGE_SERIES))
    final_model = ExtraTreesClassifier(n_estimators=trees, min_samples_leaf=2, max_features=0.8,
                                       class_weight="balanced", n_jobs=-1, random_state=seed)
    final_model.fit(x, y, sample_weight=weights)
    importance = sorted(zip(FEATURE_NAMES, final_model.feature_importances_), key=lambda item: item[1], reverse=True)
    metrics = {
        "exact_accuracy": float(np.mean(predicted == y)),
        "within_one_stage_accuracy": float(np.mean(np.abs(predicted - y) <= 1)),
        "mean_absolute_stage_error": float(np.mean(np.abs(predicted - y))),
        "mean_absolute_series_error": float(2 * np.mean(np.abs(predicted - y))),
        "spearman_expected_stage": float(_correlation(y.tolist(), expected.tolist())),
        "mean_uncertainty": float(np.mean(entropy)),
    }
    report = {
        "schema_version": 1,
        "model": "ExtraTreesClassifier",
        "weak_supervision": "Melodia one-part Series 1/3/5/7/9",
        "target_series": list(STAGE_SERIES),
        "inventory": inventory,
        "class_counts": {str(STAGE_SERIES[key]): value for key, value in sorted(class_counts.items())},
        "cross_validation": {"folds": folds, "group": "transposition-invariant pitch contour", **metrics},
        "folds": fold_reports,
        "confusion_matrix": confusion_matrix(y, predicted, labels=range(len(STAGE_SERIES))).tolist(),
        "top_feature_importance": [{"feature": name, "importance": float(value)} for name, value in importance[:15]],
        "limitations": [
            "Labels are curriculum stages, not teacher or student performance scores.",
            "Validation measures within-Melodia generalization only.",
            "Two-part series and the three one-part Series 6 files are excluded.",
            "Triplets are sparse and require another curriculum source.",
        ],
    }
    bundle = {"schema_version": 1, "model": final_model, "feature_names": FEATURE_NAMES,
              "stage_series": STAGE_SERIES, "calibration": _calibration(x), "report": report}
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output_dir / "difficulty_model.joblib", compress=3)
    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (output_dir / "features.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["name", "book", "series", "exercise", "subexercise", "duplicate_group", *FEATURE_NAMES])
        for example in examples:
            writer.writerow([example.name, example.book, example.series, example.exercise,
                             example.subexercise or "", example.duplicate_group,
                             *(example.features[name] for name in FEATURE_NAMES)])
    return report


def score_abc(model_path: Path, abc_path: Path) -> dict[str, Any]:
    import joblib
    import numpy as np
    from .difficulty import analyze_abc

    bundle = joblib.load(model_path)
    raw_report = analyze_abc(abc_path.read_text(encoding="utf-8-sig")).to_dict()
    features = _flatten_report(raw_report)
    vector = np.asarray([[features[name] for name in bundle["feature_names"]]], dtype=np.float32)
    probabilities = bundle["model"].predict_proba(vector)[0]
    stages = list(bundle["stage_series"])
    expected_series = 1.0 + 2.0 * float(probabilities @ np.arange(len(stages)))
    entropy = -sum(float(p) * math.log(float(p) + 1e-12) for p in probabilities) / math.log(len(stages))
    return {
        "predicted_series": stages[int(probabilities.argmax())],
        "expected_series": round(expected_series, 3),
        "series_probabilities": {str(stage): round(float(p), 6) for stage, p in zip(stages, probabilities)},
        "uncertainty": round(entropy, 6),
        "dimension_percentiles": score_dimensions(features, bundle["calibration"]),
        "raw_features": raw_report,
        "warning": "Melodia curriculum estimate, not a certified examination grade",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train_parser = commands.add_parser("train")
    train_parser.add_argument("musicxml_zip", type=Path)
    train_parser.add_argument("--output", type=Path, default=Path("out/difficulty-melodia-v0"))
    train_parser.add_argument("--folds", type=int, default=5)
    train_parser.add_argument("--trees", type=int, default=500)
    train_parser.add_argument("--seed", type=int, default=20260928)
    score_parser = commands.add_parser("score")
    score_parser.add_argument("model", type=Path)
    score_parser.add_argument("abc", type=Path)
    args = parser.parse_args(argv)
    report = (train(args.musicxml_zip, args.output, args.folds, args.trees, args.seed)
              if args.command == "train" else score_abc(args.model, args.abc))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
