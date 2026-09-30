"""Build and evaluate the versioned NotaEar synthetic tonality benchmark."""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .difficulty import analyze_abc
from .tonality import parse_key_field


# This is deliberately a benchmark-owned theory table rather than an import
# from tonality.py. The committed JSONL remains fixed when the analyser changes.
REFERENCE_MODES: dict[str, tuple[int, ...]] = {
    "major": (0, 2, 4, 5, 7, 9, 11),
    "natural_minor": (0, 2, 3, 5, 7, 8, 10),
    "harmonic_minor": (0, 2, 3, 5, 7, 8, 11),
    "melodic_minor": (0, 2, 3, 5, 7, 9, 11),
    "dorian": (0, 2, 3, 5, 7, 9, 10),
    "phrygian": (0, 1, 3, 5, 7, 8, 10),
    "lydian": (0, 2, 4, 6, 7, 9, 11),
    "mixolydian": (0, 2, 4, 5, 7, 9, 10),
    "locrian": (0, 1, 3, 5, 6, 8, 10),
    "gong": (0, 2, 4, 7, 9),
    "shang": (0, 2, 5, 7, 10),
    "jue": (0, 3, 5, 8, 10),
    "zhi": (0, 2, 5, 7, 9),
    "yu": (0, 3, 5, 7, 10),
}

TONIC_NAMES = ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")
SPELLINGS = (
    ("C", False), ("C", True), ("D", False), ("D", True),
    ("E", False), ("F", False), ("F", True), ("G", False),
    ("G", True), ("A", False), ("A", True), ("B", False),
)

AMBIGUOUS_COLLECTIONS = {
    "major_pentatonic": (0, 2, 4, 7, 9),
    "minor_pentachord": (0, 2, 3, 5, 7),
    "open_fifth": (0, 2, 5, 7),
}


def _abc_note(midi: int) -> str:
    letter, sharp = SPELLINGS[midi % 12]
    octave = midi // 12 - 1
    if octave >= 5:
        written = letter.lower() + "'" * (octave - 5)
    else:
        written = letter + "," * (4 - octave)
    # Explicit accidentals make K:none independent of accidental carry and
    # key-signature interpretation.
    return ("^" if sharp else "=") + written + "2"


def _render_abc(identifier: int, pitches: Iterable[int], comment: str) -> str:
    tokens = [_abc_note(midi) for midi in pitches]
    if len(tokens) != 16:
        raise ValueError("benchmark melodies must contain exactly 16 notes")
    bars = [" ".join(tokens[index:index + 4]) for index in range(0, 16, 4)]
    return (
        f"X:{identifier}\n% NotaEar tonality benchmark: {comment}\n"
        "M:4/4\nL:1/8\nK:none\n" + " | ".join(bars) + " |]\n"
    )


def _complete_scale(mode: str) -> list[int]:
    if mode == "melodic_minor":
        return [0, 2, 3, 5, 7, 9, 11, 12, 12, 10, 8, 7, 5, 3, 2, 0]
    intervals = list(REFERENCE_MODES[mode])
    result = intervals + [12] + list(reversed(intervals[1:])) + [0]
    return (result + [0] * 16)[:16]


def _modal_motif(mode: str) -> list[int]:
    if mode == "melodic_minor":
        return [0, 3, 7, 2, 5, 9, 11, 12, 10, 8, 7, 5, 3, 2, 11, 0]
    intervals = REFERENCE_MODES[mode]
    if len(intervals) == 7:
        indices = (0, 2, 4, 1, 3, 5, 4, 2, 0, 3, 1, 4, 2, 5, 6, 0)
    else:
        indices = (0, 2, 4, 1, 3, 2, 1, 0, 4, 2, 3, 1, 4, 2, 1, 0)
    return [intervals[index] for index in indices]


def _repeat_to_sixteen(intervals: tuple[int, ...]) -> list[int]:
    pattern = list(intervals) + list(reversed(intervals[1:-1]))
    result = []
    while len(result) < 15:
        result.extend(pattern)
    return result[:15] + [0]


def build_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    identifier = 1
    for mode, intervals in REFERENCE_MODES.items():
        for tonic_pc in range(12):
            base = 60 + tonic_pc
            for template, relative in (
                ("complete_scale", _complete_scale(mode)),
                ("modal_motif", _modal_motif(mode)),
            ):
                row_id = f"synthetic-v1:{template}:{mode}:{tonic_pc}"
                rows.append({
                    "schema_version": 1,
                    "id": row_id,
                    "partition": "core",
                    "template": template,
                    "tonic_pc": tonic_pc,
                    "tonic": TONIC_NAMES[tonic_pc],
                    "expected_mode": mode,
                    "acceptable_modes": [mode],
                    "abc": _render_abc(
                        identifier,
                        (base + value for value in relative),
                        row_id,
                    ),
                })
                identifier += 1
    for collection, intervals in AMBIGUOUS_COLLECTIONS.items():
        acceptable = sorted(
            mode for mode, scale in REFERENCE_MODES.items()
            if set(intervals).issubset(scale)
        )
        for tonic_pc in range(12):
            base = 60 + tonic_pc
            row_id = f"synthetic-v1:ambiguous:{collection}:{tonic_pc}"
            rows.append({
                "schema_version": 1,
                "id": row_id,
                "partition": "challenge",
                "template": collection,
                "tonic_pc": tonic_pc,
                "tonic": TONIC_NAMES[tonic_pc],
                "expected_mode": None,
                "acceptable_modes": acceptable,
                "expected_status": "ambiguous",
                "abc": _render_abc(
                    identifier,
                    (base + value for value in _repeat_to_sixteen(intervals)),
                    row_id,
                ),
            })
            identifier += 1
    return rows


def write_benchmark(path: Path) -> dict[str, Any]:
    rows = build_rows()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    counts = Counter(row["partition"] for row in rows)
    return {"output": str(path), "rows": len(rows), "partitions": dict(counts)}


def _macro_f1(
    labels: list[str], predictions: list[str], modes: Iterable[str] = REFERENCE_MODES,
) -> float:
    values = []
    for mode in modes:
        true_positive = sum(a == mode and b == mode for a, b in zip(labels, predictions))
        false_positive = sum(a != mode and b == mode for a, b in zip(labels, predictions))
        false_negative = sum(a == mode and b != mode for a, b in zip(labels, predictions))
        denominator = 2 * true_positive + false_positive + false_negative
        values.append(2 * true_positive / denominator if denominator else 0.0)
    return sum(values) / len(values)


def _load_source_abcs(
    rows: Iterable[dict[str, Any]], data_root: Path,
) -> dict[str, str]:
    requests: dict[Path, dict[int, str]] = defaultdict(dict)
    for row in rows:
        path = data_root / row["target_key"] / row["shard"]
        requests[path][int(row["line_number"])] = row["id"]
    result: dict[str, str] = {}
    for path, by_line in requests.items():
        maximum = max(by_line)
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if line_number in by_line:
                    source = json.loads(line)
                    result[by_line[line_number]] = source["abc"]
                if line_number >= maximum:
                    break
    return result


def _data_root(manifest: Path) -> Path:
    summary = json.loads((manifest.parent / "summary.json").read_text(encoding="utf-8"))
    root = Path(summary["input_root"])
    if not root.is_dir():
        raise FileNotFoundError(root)
    return root


def build_declared_benchmark(
    manifest: Path, output: Path, *, per_mode: int = 50, seed: int = 20260930,
) -> dict[str, Any]:
    """Build a real-score weak-label set from the manifest test split."""
    records = []
    with manifest.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("split") != "test":
                continue
            target_key = row["representative_key"]
            shard, line_number = row["variants"][target_key]
            records.append({
                "id": row["example_id"],
                "score_id": row["score_id"],
                "source": row["source"],
                "work_fingerprint": row["work_fingerprint"],
                "target_key": target_key,
                "shard": shard,
                "line_number": line_number,
            })
    abcs = _load_source_abcs(records, _data_root(manifest))
    candidates = []
    for row in records:
        abc = abcs.get(row["id"])
        if abc is None:
            continue
        header_keys = re.findall(r"(?m)^\s*K:\s*([^\r\n]+)", abc)
        inline_keys = re.findall(r"\[K:([^\]]+)\]", abc)
        if len(header_keys) != 1 or inline_keys:
            continue
        declared = parse_key_field(header_keys[0])
        if declared is None or declared[1] not in REFERENCE_MODES:
            continue
        candidates.append({
            "schema_version": 1,
            **row,
            "expected_tonic_pc": declared[0],
            "expected_tonic": TONIC_NAMES[declared[0]],
            "expected_mode": declared[1],
            "label_kind": "abc_declared_key_weak_label",
        })
    random.Random(seed).shuffle(candidates)
    selected = []
    mode_counts = Counter()
    source_mode_counts = Counter()
    works = set()
    for row in candidates:
        mode = row["expected_mode"]
        key = (row["source"], mode)
        if mode_counts[mode] >= per_mode or source_mode_counts[key] >= 20:
            continue
        if row["work_fingerprint"] in works:
            continue
        selected.append(row)
        works.add(row["work_fingerprint"])
        mode_counts[mode] += 1
        source_mode_counts[key] += 1
    selected.sort(key=lambda row: (row["expected_mode"], row["id"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    return {
        "output": str(output),
        "manifest": str(manifest),
        "test_split_records": len(records),
        "eligible_declared_records": len(candidates),
        "selected": len(selected),
        "mode_counts": dict(sorted(mode_counts.items())),
        "seed": seed,
    }


def evaluate_declared(path: Path, manifest: Path) -> dict[str, Any]:
    """Evaluate against real ABC declarations while hiding them from inference."""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    abcs = _load_source_abcs(rows, _data_root(manifest))
    accepted_statuses = {"confident", "declared"}
    counts = Counter()
    status_counts = Counter()
    source_counts = Counter()
    labels: list[str] = []
    predictions: list[str] = []
    per_mode: dict[str, Counter] = defaultdict(Counter)
    confusion = Counter()
    failures = []
    for row in rows:
        try:
            result = analyze_abc(
                abcs[row["id"]], use_declared_tonality=False
            ).tonality
        except Exception as exc:
            failures.append({"id": row["id"], "error": f"{type(exc).__name__}: {exc}"})
            continue
        primary = result["primary"]
        predicted_mode = primary["mode"] if primary else "__abstain__"
        predicted_tonic = primary["tonic_pc"] if primary else None
        expected_mode = row["expected_mode"]
        accepted = result["status"] in accepted_statuses
        tonic_correct = predicted_tonic == row["expected_tonic_pc"]
        mode_correct = predicted_mode == expected_mode
        joint = tonic_correct and mode_correct
        counts.update(total=1, tonic_correct=tonic_correct, mode_correct=mode_correct,
                      joint_correct=joint, accepted=accepted,
                      accepted_joint_correct=accepted and joint)
        status_counts[result["status"]] += 1
        source_counts[row["source"]] += 1
        per_mode[expected_mode].update(total=1, joint_correct=joint, accepted=accepted,
                                       accepted_joint_correct=accepted and joint)
        labels.append(expected_mode)
        predictions.append(predicted_mode)
        confusion[(expected_mode, predicted_mode)] += 1

    def ratio(numerator: int, denominator: int) -> float:
        return round(numerator / denominator, 6) if denominator else 0.0

    total, accepted = counts["total"], counts["accepted"]
    observed_modes = sorted(set(labels))
    return {
        "schema_version": 1,
        "benchmark": str(path),
        "manifest": str(manifest),
        "label_kind": "abc_declared_key_weak_label",
        "examples": total,
        "failures": failures,
        "tonic_accuracy": ratio(counts["tonic_correct"], total),
        "mode_accuracy": ratio(counts["mode_correct"], total),
        "joint_accuracy": ratio(counts["joint_correct"], total),
        "mode_macro_f1": round(_macro_f1(labels, predictions, observed_modes), 6),
        "accepted_coverage": ratio(accepted, total),
        "accepted_joint_accuracy": ratio(counts["accepted_joint_correct"], accepted),
        "status_counts": dict(status_counts),
        "source_counts": dict(sorted(source_counts.items())),
        "per_mode": {
            mode: {
                "examples": values["total"],
                "joint_accuracy": ratio(values["joint_correct"], values["total"]),
                "accepted_coverage": ratio(values["accepted"], values["total"]),
                "accepted_joint_accuracy": ratio(
                    values["accepted_joint_correct"], values["accepted"]
                ),
            }
            for mode, values in sorted(per_mode.items())
        },
        "confusions": [
            {"expected": expected, "predicted": predicted, "count": count}
            for (expected, predicted), count in sorted(confusion.items())
            if expected != predicted
        ],
        "limitations": [
            "ABC K: declarations are weak metadata, not independently verified human truth.",
            "The declaration was hidden from tonality inference but retained for pitch parsing.",
            "A human-labelled real-score benchmark is still required for publishable accuracy.",
        ],
    }


def evaluate(path: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("benchmark contains duplicate ids")
    accepted_statuses = {"confident", "declared"}
    failures: list[dict[str, str]] = []
    core_labels: list[str] = []
    core_predictions: list[str] = []
    core = Counter()
    challenge = Counter()
    per_mode: dict[str, Counter] = defaultdict(Counter)
    confusion = Counter()
    for row in rows:
        try:
            result = analyze_abc(row["abc"]).tonality
        except Exception as exc:
            failures.append({"id": row["id"], "error": f"{type(exc).__name__}: {exc}"})
            continue
        primary = result["primary"]
        predicted_mode = primary["mode"] if primary else "__abstain__"
        predicted_tonic = primary["tonic_pc"] if primary else None
        accepted = result["status"] in accepted_statuses
        if row["partition"] == "core":
            expected = row["expected_mode"]
            tonic_correct = predicted_tonic == row["tonic_pc"]
            mode_correct = predicted_mode == expected
            joint = tonic_correct and mode_correct
            core.update(total=1, tonic_correct=tonic_correct, mode_correct=mode_correct,
                        joint_correct=joint, accepted=accepted,
                        accepted_joint_correct=accepted and joint)
            per_mode[expected].update(total=1, joint_correct=joint, accepted=accepted,
                                      accepted_joint_correct=accepted and joint)
            core_labels.append(expected)
            core_predictions.append(predicted_mode)
            confusion[(expected, predicted_mode)] += 1
        else:
            acceptable = predicted_mode in row["acceptable_modes"]
            abstained = result["status"] in {"ambiguous", "insufficient"}
            challenge.update(total=1, tonic_correct=predicted_tonic == row["tonic_pc"],
                             acceptable_top=acceptable, abstained=abstained,
                             overconfident=accepted)

    def ratio(numerator: int, denominator: int) -> float:
        return round(numerator / denominator, 6) if denominator else 0.0

    core_total = core["total"]
    accepted = core["accepted"]
    challenge_total = challenge["total"]
    return {
        "schema_version": 1,
        "benchmark": str(path),
        "rows": len(rows),
        "failures": failures,
        "core": {
            "examples": core_total,
            "tonic_accuracy": ratio(core["tonic_correct"], core_total),
            "mode_accuracy": ratio(core["mode_correct"], core_total),
            "joint_accuracy": ratio(core["joint_correct"], core_total),
            "mode_macro_f1": round(_macro_f1(core_labels, core_predictions), 6),
            "accepted_coverage": ratio(accepted, core_total),
            "accepted_joint_accuracy": ratio(core["accepted_joint_correct"], accepted),
            "per_mode": {
                mode: {
                    "examples": values["total"],
                    "joint_accuracy": ratio(values["joint_correct"], values["total"]),
                    "accepted_coverage": ratio(values["accepted"], values["total"]),
                    "accepted_joint_accuracy": ratio(
                        values["accepted_joint_correct"], values["accepted"]
                    ),
                }
                for mode, values in sorted(per_mode.items())
            },
            "confusions": [
                {"expected": expected, "predicted": predicted, "count": count}
                for (expected, predicted), count in sorted(confusion.items())
                if expected != predicted
            ],
        },
        "challenge": {
            "examples": challenge_total,
            "tonic_accuracy": ratio(challenge["tonic_correct"], challenge_total),
            "acceptable_top_rate": ratio(challenge["acceptable_top"], challenge_total),
            "abstention_rate": ratio(challenge["abstained"], challenge_total),
            "overconfidence_rate": ratio(challenge["overconfident"], challenge_total),
        },
        "limitations": [
            "Synthetic theory examples are regression tests, not real-score accuracy evidence.",
            "Core examples contain complete or characteristic modal evidence.",
            "Challenge examples are intentionally underdetermined and should be rejected.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument(
        "output", type=Path,
        default=Path("data/evaluation/tonality/synthetic-v1.jsonl"), nargs="?",
    )
    score = commands.add_parser("evaluate")
    score.add_argument("benchmark", type=Path)
    score.add_argument("--output", type=Path)
    declared = commands.add_parser("build-declared")
    declared.add_argument("manifest", type=Path)
    declared.add_argument(
        "output", type=Path, nargs="?",
        default=Path("data/evaluation/tonality/declared-v1.jsonl"),
    )
    declared.add_argument("--per-mode", type=int, default=50)
    declared.add_argument("--seed", type=int, default=20260930)
    declared_score = commands.add_parser("evaluate-declared")
    declared_score.add_argument("benchmark", type=Path)
    declared_score.add_argument("manifest", type=Path)
    declared_score.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.command == "build":
        result = write_benchmark(args.output)
    elif args.command == "evaluate":
        result = evaluate(args.benchmark)
    elif args.command == "build-declared":
        result = build_declared_benchmark(
            args.manifest, args.output, per_mode=args.per_mode, seed=args.seed
        )
    else:
        result = evaluate_declared(args.benchmark, args.manifest)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.command in {"evaluate", "evaluate-declared"} and args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
