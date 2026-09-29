"""External validation for the Melodia weak-supervision difficulty model."""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any
from xml.etree import ElementTree as ET
from zipfile import ZipFile

from .train_difficulty import (
    FEATURE_NAMES,
    _correlation,
    extract_musicxml_features,
    score_dimensions,
)


EYES_CHAPTERS = (
    ("I Rhythm", 1, 26),
    ("II Major", 27, 169),
    ("III Minor", 170, 229),
    ("IV Other diatonic", 230, 359),
    ("V Nondiatonic", 360, 411),
    ("VI C clef", 412, 427),
)

EYES_SUBSECTIONS = (
    ("I-1 basic rhythm", 1, 5),
    ("I-3 eighth notes", 8, 11),
    ("I-5 sixteenth notes", 17, 17),
    ("I-7 compound time", 21, 24),
    ("I-8 syncopation", 25, 25),
    ("I-9 triplets", 26, 26),
    ("II-2 steps", 27, 51),
    ("II-5 tonic-triad leaps", 59, 79),
    ("II-6 fifth", 80, 88),
    ("II-7 sixth", 89, 101),
    ("II-8 dominant", 102, 134),
    ("II-9 wider dominant leaps", 135, 169),
    ("III-2 minor steps", 170, 181),
    ("III-3 minor tonic leaps", 182, 197),
    ("III-4 minor dominant leaps", 198, 229),
    ("V-1 chromatic", 360, 398),
    ("V-2 blue notes", 399, 403),
    ("V-3 relative keys", 404, 411),
    ("VI C clef", 412, 427),
)


def _first_part_xml(xml: bytes) -> bytes:
    root = ET.fromstring(xml)
    parts = root.findall("part")
    if not parts:
        raise ValueError("MusicXML has no parts")
    for part in parts[1:]:
        root.remove(part)
    return ET.tostring(root)


def _predict(bundle: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    import numpy as np

    matrix = np.asarray(
        [[row["features"][name] for name in FEATURE_NAMES] for row in rows],
        dtype=np.float32,
    )
    probabilities = bundle["model"].predict_proba(matrix)
    stages = np.asarray(bundle["stage_series"])
    for row, probability in zip(rows, probabilities):
        row["expected_series"] = float(probability @ stages)
        row["predicted_series"] = int(stages[probability.argmax()])
        row["uncertainty"] = -sum(
            float(value) * math.log(float(value) + 1e-12) for value in probability
        ) / math.log(len(probability))
        row["dimensions"] = score_dimensions(row["features"], bundle["calibration"])


def _range_summary(rows: list[dict[str, Any]], start: int, end: int) -> dict[str, Any]:
    selected = [row for row in rows if start <= row["number"] <= end]
    if not selected:
        return {"examples": 0}
    result = {
        "examples": len(selected),
        "mean_expected_series": round(mean(row["expected_series"] for row in selected), 6),
        "mean_uncertainty": round(mean(row["uncertainty"] for row in selected), 6),
        "mean_dimensions": {
            name: round(mean(row["dimensions"][name] for row in selected), 3)
            for name in selected[0]["dimensions"]
        },
    }
    if len(selected) > 2:
        result["spearman_order"] = round(_correlation(
            [row["number"] for row in selected],
            [row["expected_series"] for row in selected],
        ), 6)
    return result


def _quintiles(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for index in range(5):
        selected = rows[index * len(rows) // 5:(index + 1) * len(rows) // 5]
        result.append({
            "quintile": index + 1,
            "first_order": list(selected[0]["order"]),
            "last_order": list(selected[-1]["order"]),
            "examples": len(selected),
            "mean_expected_series": round(mean(row["expected_series"] for row in selected), 6),
            "mean_uncertainty": round(mean(row["uncertainty"] for row in selected), 6),
        })
    return result


def _overall(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "examples": len(rows),
        "spearman_order": round(_correlation(
            list(range(len(rows))), [row["expected_series"] for row in rows]
        ), 6),
        "mean_expected_series": round(mean(row["expected_series"] for row in rows), 6),
        "mean_uncertainty": round(mean(row["uncertainty"] for row in rows), 6),
        "predicted_series_counts": dict(sorted(Counter(
            str(row["predicted_series"]) for row in rows
        ).items())),
        "quintiles": _quintiles(rows),
    }


def _load_eyes(path: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows = []
    excluded = Counter()
    with ZipFile(path) as archive:
        for name in archive.namelist():
            match = re.search(r"_(\d+)\.xml$", name)
            if not match:
                continue
            features, parts, _ = extract_musicxml_features(archive.read(name))
            if parts != 1:
                excluded["multi_part"] += 1
                continue
            number = int(match.group(1))
            rows.append({"name": name, "number": number, "order": (number,), "features": features})
    rows.sort(key=lambda row: row["order"])
    return rows, dict(excluded)


def _load_concone_voice(path: Path) -> list[dict[str, Any]]:
    rows = []
    pattern = re.compile(r"_(\d+)(?:-(\d+)(?:-(\d+))?)?\.xml$")
    with ZipFile(path) as archive:
        for name in archive.namelist():
            match = pattern.search(name)
            if not match:
                continue
            order = tuple(int(value) for value in match.groups() if value is not None)
            features, _, _ = extract_musicxml_features(_first_part_xml(archive.read(name)))
            rows.append({"name": name, "number": order[0], "order": order, "features": features})
    rows.sort(key=lambda row: row["order"])
    return rows


def validate(model_path: Path, eyes_path: Path, concone_path: Path) -> dict[str, Any]:
    import joblib

    bundle = joblib.load(model_path)
    eyes, eyes_excluded = _load_eyes(eyes_path)
    concone = _load_concone_voice(concone_path)
    _predict(bundle, eyes)
    _predict(bundle, concone)
    eyes_overall = _overall(eyes)
    concone_overall = _overall(concone)
    chapters = {
        name: _range_summary(eyes, start, end) for name, start, end in EYES_CHAPTERS
    }
    subsections = {
        name: _range_summary(eyes, start, end) for name, start, end in EYES_SUBSECTIONS
    }
    internal_uncertainty = bundle["report"]["cross_validation"]["mean_uncertainty"]
    checks = {
        "eyes_global_spearman_at_least_0_4": eyes_overall["spearman_order"] >= 0.4,
        "eyes_major_spearman_at_least_0_4": chapters["II Major"]["spearman_order"] >= 0.4,
        "eyes_minor_spearman_at_least_0_4": chapters["III Minor"]["spearman_order"] >= 0.4,
        "concone_global_spearman_at_least_0_3": concone_overall["spearman_order"] >= 0.3,
        "external_uncertainty_exceeds_internal": (
            eyes_overall["mean_uncertainty"] > internal_uncertainty
            and concone_overall["mean_uncertainty"] > internal_uncertainty
        ),
    }
    return {
        "schema_version": 1,
        "model": str(model_path),
        "method": "external curriculum-order validation; no external Series labels assumed",
        "acceptance_checks": checks,
        "passed": all(checks.values()),
        "eyes_and_ears": {
            "overall": eyes_overall,
            "excluded": eyes_excluded,
            "chapters": chapters,
            "subsections": subsections,
        },
        "concone_voice_p1": {"overall": concone_overall},
        "limitations": [
            "Curriculum order is weak supervision, not student-performance ground truth.",
            "Eyes and Ears resets difficulty when a new chapter starts.",
            "C-clef reading cost is reader-profile dependent and is not represented by sounding-note features.",
            "Concone validation extracts only the P1 vocal melody and ignores piano accompaniment.",
            "Gilchrist has no local machine-readable score and is not used for numerical validation.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("eyes_xml_zip", type=Path)
    parser.add_argument("concone_xml_zip", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = validate(args.model, args.eyes_xml_zip, args.concone_xml_zip)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
