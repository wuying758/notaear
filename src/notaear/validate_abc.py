"""Validate generated single-voice ABC structure, rhythm and tonal evidence."""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from music21 import chord, converter, key, meter, note, stream

from .abc_patches import AbcPatchError, BarPatch, split_abc


_HEADER_FIELD = re.compile(r"(?m)^\s*([KLM]):([^\r\n]+)")
_KEY_CHANGE = re.compile(
    r"\[K:([^\]]+)\]|^[ \t]*K:([^\r\n]+)\r?\n?", re.MULTILINE
)
_METER_CHANGE = re.compile(
    r"\[M:([^\]]+)\]|^[ \t]*M:([^\r\n]+)\r?\n?", re.MULTILINE
)


@dataclass(frozen=True)
class ValidationIssue:
    severity: str
    code: str
    message: str
    bar_index: int | None = None


@dataclass(frozen=True)
class BarTiming:
    index: int
    meter: str | None
    actual_quarters: str | None
    expected_quarters: str | None
    status: str


@dataclass(frozen=True)
class ABCValidationReport:
    accepted: bool
    structural_valid: bool
    parseable: bool
    rhythm_valid: bool
    single_voice: bool
    bar_count: int
    bars: tuple[BarTiming, ...]
    tonal: dict[str, Any]
    issues: tuple[ValidationIssue, ...]
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _fraction(value: Any) -> Fraction:
    return Fraction(float(value)).limit_denominator(1536)


def _fraction_text(value: Fraction | None) -> str | None:
    if value is None:
        return None
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def _fields(header: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for name, value in _HEADER_FIELD.findall(header):
        result[name] = value.strip()
    return result


def _field_changes(pattern: re.Pattern[str], text: str) -> list[tuple[int, int, str]]:
    changes = []
    for match in pattern.finditer(text):
        value = next(group for group in match.groups() if group is not None)
        changes.append((match.start(), match.end(), value.strip()))
    return changes


def _parse_abc(abc: str) -> tuple[stream.Stream, str]:
    captured = io.StringIO()
    with contextlib.redirect_stderr(captured), contextlib.redirect_stdout(captured):
        score = converter.parseData(abc, format="abc")
    return score, captured.getvalue().strip()


def _first_part(score: stream.Stream) -> tuple[stream.Stream, int]:
    parts = list(score.parts) if hasattr(score, "parts") else []
    return (parts[0], len(parts)) if parts else (score, 1)


def _synthetic_header(meter_spec: str, length_spec: str, key_spec: str) -> str:
    return f"X:1\nM:{meter_spec}\nL:{length_spec}\nK:{key_spec}\n"


def _expected_quarters(meter_spec: str) -> Fraction | None:
    normalized = meter_spec.strip().lower()
    if normalized in {"none", "free", ""}:
        return None
    try:
        signature = meter.TimeSignature(meter_spec.strip())
    except Exception:
        return None
    return _fraction(signature.barDuration.quarterLength)


def _default_length(meter_spec: str) -> str:
    expected = _expected_quarters(meter_spec)
    return "1/16" if expected is not None and expected < 3 else "1/8"


def _bar_duration(
    bar: BarPatch,
    meter_spec: str,
    length_spec: str,
    key_spec: str,
) -> tuple[Fraction | None, str, int, int]:
    score, warnings = _parse_abc(
        _synthetic_header(meter_spec, length_spec, key_spec) + bar.text
    )
    part, part_count = _first_part(score)
    voice_count = max(
        (len(measure.getElementsByClass(stream.Voice)) for measure in part.recurse().getElementsByClass(stream.Measure)),
        default=0,
    )
    actual = _fraction(part.highestTime)
    return actual, warnings, part_count, voice_count


def _scale_pitch_classes(active_key: key.Key) -> set[int]:
    return {pitch.pitchClass for pitch in active_key.getScale().getPitches()}


def _parse_tonal_segment(
    text: str,
    meter_spec: str,
    length_spec: str,
    key_spec: str,
) -> tuple[list[int], key.Key | None, str]:
    score, warnings = _parse_abc(
        _synthetic_header(meter_spec, length_spec, key_spec) + text
    )
    part, _ = _first_part(score)
    key_objects = list(part.recurse().getElementsByClass(key.Key))
    active_key = key_objects[0] if key_objects else None
    pitches: list[int] = []
    for element in part.recurse().notes:
        if isinstance(element, chord.Chord):
            pitches.extend(int(item.midi) for item in element.pitches)
        elif isinstance(element, note.Note):
            pitches.append(int(element.pitch.midi))
    return pitches, active_key, warnings


def _tonal_summary(
    body: str,
    meter_spec: str,
    length_spec: str,
    initial_key: str,
    issues: list[ValidationIssue],
    require_final_tonic: bool,
) -> dict[str, Any]:
    changes = _field_changes(_KEY_CHANGE, body)
    segments: list[tuple[str, str]] = []
    cursor = 0
    active_spec = initial_key
    for start, end, next_key in changes:
        segments.append((body[cursor:start], active_spec))
        active_spec = next_key
        cursor = end
    segments.append((body[cursor:], active_spec))

    total_notes = 0
    chromatic_notes = 0
    final_midi: int | None = None
    final_key: key.Key | None = None
    parsed_keys: list[str] = []
    for text, key_spec in segments:
        try:
            pitches, active_key, warnings = _parse_tonal_segment(
                text, meter_spec, length_spec, key_spec
            )
        except Exception as exc:
            issues.append(ValidationIssue(
                "warning",
                "tonal_segment_unparsed",
                f"could not parse tonal segment in K:{key_spec}: {exc}",
            ))
            continue
        if warnings:
            issues.append(ValidationIssue(
                "warning", "tonal_parser_warning", warnings.splitlines()[-1]
            ))
        if active_key is None:
            continue
        parsed_keys.append(active_key.name)
        scale = _scale_pitch_classes(active_key)
        total_notes += len(pitches)
        chromatic_notes += sum(pitch % 12 not in scale for pitch in pitches)
        if pitches:
            final_midi = pitches[-1]
            final_key = active_key

    final_is_tonic = None
    final_scale_degree = None
    if final_midi is not None and final_key is not None:
        final_is_tonic = final_midi % 12 == final_key.tonic.pitchClass
        final_scale_degree = final_key.getScale().getScaleDegreeFromPitch(
            note.Note(final_midi).pitch
        )
        if not final_is_tonic:
            severity = "error" if require_final_tonic else "warning"
            issues.append(ValidationIssue(
                severity,
                "final_not_tonic",
                f"final pitch {note.Note(final_midi).name} is not tonic of {final_key.name}",
            ))
    if not total_notes:
        issues.append(ValidationIssue("error", "no_pitched_notes", "score has no pitched notes"))

    return {
        "initial_key_spec": initial_key,
        "parsed_keys": parsed_keys,
        "key_change_count": len(changes),
        "note_count": total_notes,
        "chromatic_note_count": chromatic_notes,
        "chromatic_note_rate": round(chromatic_notes / total_notes, 6) if total_notes else None,
        "final_midi": final_midi,
        "final_pitch": note.Note(final_midi).nameWithOctave if final_midi is not None else None,
        "final_key": final_key.name if final_key is not None else None,
        "final_scale_degree": final_scale_degree,
        "final_is_tonic": final_is_tonic,
    }


def validate_abc(
    abc: str,
    *,
    require_final_tonic: bool = False,
) -> ABCValidationReport:
    """Validate one generated, nominally single-voice ABC tune."""
    issues: list[ValidationIssue] = []
    bars: list[BarTiming] = []
    tonal: dict[str, Any] = {}
    try:
        document = split_abc(abc)
    except (AbcPatchError, ValueError) as exc:
        issues.append(ValidationIssue("error", "structural_parse_failed", str(exc)))
        return ABCValidationReport(
            False, False, False, False, False, 0, (), {}, tuple(issues)
        )

    fields = _fields(document.header)
    meter_spec = fields.get("M", "4/4")
    length_spec = fields.get("L", _default_length(meter_spec))
    key_spec = fields.get("K")
    if key_spec is None:
        issues.append(ValidationIssue("error", "missing_key", "header has no K: field"))
        return ABCValidationReport(
            False, True, False, False, False, len(document.bars), (), {}, tuple(issues)
        )

    try:
        score, parser_warnings = _parse_abc(abc)
    except Exception as exc:
        issues.append(ValidationIssue("error", "abc_parse_failed", str(exc)))
        return ABCValidationReport(
            False, True, False, False, False, len(document.bars), (), {}, tuple(issues)
        )
    if parser_warnings:
        severity = "error" if "WARNING" in parser_warnings.upper() else "warning"
        issues.append(ValidationIssue(
            severity, "abc_parser_output", parser_warnings.splitlines()[-1]
        ))

    part, part_count = _first_part(score)
    single_voice = part_count == 1
    if part_count != 1:
        issues.append(ValidationIssue(
            "error", "multiple_parts", f"expected one part, found {part_count}"
        ))
    chord_count = len(list(part.recurse().getElementsByClass(chord.Chord)))
    if chord_count:
        single_voice = False
        issues.append(ValidationIssue(
            "error", "chord_events", f"single-voice score contains {chord_count} chords"
        ))

    actuals: list[Fraction | None] = []
    expecteds: list[Fraction | None] = []
    meters: list[str] = []
    rhythm_errors: set[int] = set()
    current_meter = meter_spec
    for bar in document.bars:
        changes = _field_changes(_METER_CHANGE, bar.text)
        if changes:
            current_meter = changes[-1][2]
            if changes[0][0] > len(bar.text) - len(bar.text.lstrip()):
                issues.append(ValidationIssue(
                    "warning",
                    "mid_bar_meter_change",
                    "meter change occurs after the start of a bar; expected duration uses the new meter",
                    bar.index,
                ))
        expected = _expected_quarters(current_meter)
        try:
            actual, warnings, local_parts, local_voices = _bar_duration(
                bar, current_meter, length_spec, key_spec
            )
        except Exception as exc:
            actual = None
            warnings = ""
            local_parts = local_voices = 0
            rhythm_errors.add(bar.index)
            issues.append(ValidationIssue(
                "error", "bar_parse_failed", str(exc), bar.index
            ))
        if warnings:
            rhythm_errors.add(bar.index)
            issues.append(ValidationIssue(
                "error", "bar_parser_warning", warnings.splitlines()[-1], bar.index
            ))
        if local_parts > 1 or local_voices > 1:
            single_voice = False
            rhythm_errors.add(bar.index)
            issues.append(ValidationIssue(
                "error", "multiple_voices", "bar contains multiple parts or voices", bar.index
            ))
        actuals.append(actual)
        expecteds.append(expected)
        meters.append(current_meter)

    statuses = ["unknown"] * len(document.bars)
    first_pickup = False
    for index, (actual, expected) in enumerate(zip(actuals, expecteds)):
        if actual is None or expected is None:
            statuses[index] = "unmetered" if expected is None else "parse_error"
            continue
        if actual == expected:
            statuses[index] = "full"
        elif actual > expected:
            statuses[index] = "overfull"
            rhythm_errors.add(index)
            issues.append(ValidationIssue(
                "error", "bar_overfull",
                f"bar duration {_fraction_text(actual)} exceeds {_fraction_text(expected)} quarters",
                index,
            ))
        elif actual == 0:
            statuses[index] = "empty"
            rhythm_errors.add(index)
            issues.append(ValidationIssue("error", "empty_bar", "bar has no duration", index))
        elif index == 0 and len(actuals) > 1:
            statuses[index] = "pickup"
            first_pickup = True
            issues.append(ValidationIssue(
                "warning", "pickup_bar",
                f"first bar is a pickup of {_fraction_text(actual)} quarters",
                index,
            ))
        elif index == len(actuals) - 1 and first_pickup:
            first_actual = actuals[0]
            first_expected = expecteds[0]
            if (
                first_actual is not None
                and first_expected == expected
                and first_actual + actual == expected
            ):
                statuses[index] = "pickup_completion"
            else:
                statuses[index] = "underfull"
                rhythm_errors.add(index)
        else:
            statuses[index] = "underfull"
            rhythm_errors.add(index)
        if statuses[index] == "underfull":
            issues.append(ValidationIssue(
                "error", "bar_underfull",
                f"bar duration {_fraction_text(actual)} is short of {_fraction_text(expected)} quarters",
                index,
            ))

    if first_pickup and statuses[-1] != "pickup_completion":
        rhythm_errors.add(len(statuses) - 1)
        issues.append(ValidationIssue(
            "error",
            "pickup_unmatched",
            "first-bar pickup is not complemented by the final bar",
            len(statuses) - 1,
        ))

    for index in range(len(document.bars)):
        bars.append(BarTiming(
            index=index,
            meter=meters[index],
            actual_quarters=_fraction_text(actuals[index]),
            expected_quarters=_fraction_text(expecteds[index]),
            status=statuses[index],
        ))

    tonal = _tonal_summary(
        "".join(bar.text for bar in document.bars),
        meter_spec,
        length_spec,
        key_spec,
        issues,
        require_final_tonic,
    )
    rhythm_valid = not rhythm_errors and bool(document.bars)
    if not document.bars:
        issues.append(ValidationIssue("error", "no_bars", "score has no music bars"))
    accepted = not any(issue.severity == "error" for issue in issues)
    return ABCValidationReport(
        accepted=accepted,
        structural_valid=True,
        parseable=True,
        rhythm_valid=rhythm_valid,
        single_voice=single_voice,
        bar_count=len(document.bars),
        bars=tuple(bars),
        tonal=tonal,
        issues=tuple(issues),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("abc", type=Path)
    parser.add_argument("--require-final-tonic", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    report = validate_abc(
        args.abc.read_text(encoding="utf-8-sig"),
        require_final_tonic=args.require_final_tonic,
    )
    rendered = json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report.accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
