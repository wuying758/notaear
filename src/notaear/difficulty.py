"""Extract raw, explainable sight-singing difficulty features from ABC.

No overall score or L1--L5 label is emitted.  Those require calibration against
ordered textbook material or human judgements.  The feature families are kept
separate so a later model can learn their weights without hiding the evidence.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from statistics import mean

from music21 import chord, converter, key, meter, note, stream

from .abc_patches import split_abc
from .tonality import (
    TonalEvent,
    analyze_tonality,
    parse_key_field,
    scale_pitch_classes,
)


@dataclass(frozen=True)
class DifficultyReport:
    dimensions: dict[str, dict[str, float | int | str | bool | None]]
    tonality: dict | None = None
    warnings: tuple[str, ...] = ()
    schema_version: int = 2

    def to_dict(self) -> dict:
        return asdict(self)


def _fraction(value) -> Fraction:
    return Fraction(float(value)).limit_denominator(384)


def _entropy(values: list[Fraction]) -> float:
    if not values:
        return 0.0
    counts = Counter(values)
    total = len(values)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def _rate(count: int | float, total: int) -> float:
    return round(float(count) / total, 4) if total else 0.0


def _quantile(values: list[int], proportion: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    index = round((len(ordered) - 1) * proportion)
    return ordered[index]


def _first_part(score: stream.Stream) -> stream.Stream:
    parts = list(score.parts) if hasattr(score, "parts") else []
    return parts[0] if parts else score


def _beat_quarters(ts: meter.TimeSignature | None) -> Fraction:
    if ts is None:
        return Fraction(1)
    if ts.numerator > 3 and ts.numerator % 3 == 0 and ts.denominator in (8, 16):
        return Fraction(12, ts.denominator)  # dotted beat
    return Fraction(4, ts.denominator)


def _motif_repetition_rate(pitches: list[int], order: int = 3) -> float:
    intervals = [right - left for left, right in zip(pitches, pitches[1:])]
    if len(intervals) < order:
        return 0.0
    motifs = [tuple(intervals[i:i + order]) for i in range(len(intervals) - order + 1)]
    counts = Counter(motifs)
    repeated = sum(count for motif, count in counts.items() if count > 1)
    return _rate(repeated, len(motifs))


def _music21_compatibility_copy(abc: str) -> tuple[str, tuple[str, ...]]:
    """Repair two known legacy forms in a private music21 parse copy."""
    repairs: list[str] = []
    output: list[str] = []
    saw_key = False
    for line in abc.splitlines(keepends=True):
        stripped = line.lstrip("\ufeff \t")
        if not saw_key and stripped.startswith("K:"):
            saw_key = True
        if (not saw_key and stripped.strip() and not stripped.startswith("%") and
                not re.match(r"^[A-Za-z+]:", stripped)):
            output.append("% notaear ignored legacy header text: " + line)
            repairs.append("bare_header_text")
        else:
            output.append(line)
    repaired = "".join(output)

    # music21 accepts some extended K: strings while silently applying the
    # wrong major key signature. Map only the parsing copy to the parent key;
    # tonality.py still reads the original declaration and retains its mode.
    def compatible_key(value: str) -> str | None:
        parsed = parse_key_field(value)
        if parsed is None:
            return None
        tonic, mode = parsed
        names = ("C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")
        if mode in {"harmonic_minor", "melodic_minor"}:
            return names[tonic] + "m"
        parent_offsets = {"gong": 0, "shang": -2, "jue": -4, "zhi": -7, "yu": -9,
                          "locrian": 1}
        if mode in parent_offsets:
            return names[(tonic + parent_offsets[mode]) % 12]
        return None

    def replace_line_key(match: re.Match) -> str:
        replacement = compatible_key(match.group(2))
        if replacement is None:
            return match.group(0)
        repairs.append("extended_key_signature")
        return match.group(1) + replacement

    def replace_inline_key(match: re.Match) -> str:
        replacement = compatible_key(match.group(1))
        if replacement is None:
            return match.group(0)
        repairs.append("extended_key_signature")
        return f"[K:{replacement}]"

    repaired = re.sub(r"(?m)^(\s*K:\s*)([^\r\n]+)$", replace_line_key, repaired)
    repaired = re.sub(r"\[K:([^\]]+)\]", replace_inline_key, repaired)
    repaired_meter = re.sub(
        r"(\|)M:(C\||C|none|\d+/\d+)", r"\1[M:\2]", repaired
    )
    if repaired_meter != repaired:
        repairs.append("bare_inline_meter")
    return repaired_meter, tuple(dict.fromkeys(repairs))


def analyze_abc(abc: str, *, use_declared_tonality: bool = True) -> DifficultyReport:
    """Return five groups of raw features for one cleaned, single-voice ABC tune."""
    document = split_abc(abc)
    warnings: list[str] = []
    compatibility_abc, repairs = _music21_compatibility_copy(abc)
    try:
        score = converter.parseData(compatibility_abc, format="abc")
    except Exception as exc:  # music21 errors are not stable across versions
        raise ValueError(
            f"ABC could not be parsed for difficulty analysis: {exc}"
        ) from exc
    if repairs:
        if "bare_header_text" in repairs:
            warnings.append(
                "music21 解析副本已忽略 K: 前的旧式裸标题文本；原始 ABC 未修改"
            )
        if "bare_inline_meter" in repairs:
            warnings.append(
                "music21 解析副本已将旧式裸内联 M: 字段规范为 [M:...]；原始 ABC 未修改"
            )
        if "extended_key_signature" in repairs:
            warnings.append(
                "music21 解析副本已将扩展调式 K: 映射到对应母调调号；原始调式声明未修改"
            )
    part = _first_part(score)

    pitches: list[int] = []
    durations: list[Fraction] = []
    note_durations: list[Fraction] = []
    attacks: list[tuple[int, Fraction, Fraction, Fraction]] = []
    tonal_events: list[TonalEvent] = []
    rest_count = chord_count = dotted_count = tuplet_note_count = tie_count = 0
    current_ts = next(iter(part.recurse().getElementsByClass(meter.TimeSignature)), None)
    measures = list(part.getElementsByClass(stream.Measure))
    if not measures:
        measures = list(part.makeMeasures(inPlace=False).getElementsByClass(stream.Measure))

    phrase_notes: list[int] = []
    phrase_quarters: list[Fraction] = []
    running_notes = 0
    running_quarters = Fraction(0)
    for bar_index, measure in enumerate(measures):
        local_ts = next(iter(measure.recurse().getElementsByClass(meter.TimeSignature)), None)
        if local_ts is not None:
            current_ts = local_ts
        beat = _beat_quarters(current_ts)
        elements = sorted(
            measure.recurse().notesAndRests,
            key=lambda item: (float(item.offset), getattr(item, "priority", 0)),
        )
        for element in elements:
            duration = _fraction(element.duration.quarterLength)
            durations.append(duration)
            running_quarters += duration
            if isinstance(element, note.Rest):
                rest_count += 1
                if duration >= beat and running_notes:
                    phrase_notes.append(running_notes)
                    phrase_quarters.append(running_quarters - duration)
                    running_notes = 0
                    running_quarters = Fraction(0)
                continue
            if isinstance(element, chord.Chord):
                chord_count += 1
                event_pitches = [int(item.midi) for item in element.pitches]
                representative = max(event_pitches)
            else:
                event_pitches = [int(element.pitch.midi)]
                representative = event_pitches[0]
            if getattr(element, "tie", None) is not None:
                tie_count += 1
                if element.tie.type in ("continue", "stop"):
                    continue
            pitches.append(representative)
            note_durations.append(duration)
            running_notes += 1
            dotted_count += int(bool(element.duration.dots))
            has_tuplet = bool(element.duration.tuplets)
            tuplet_note_count += int(has_tuplet)
            onset = _fraction(element.offset)
            attacks.append((bar_index, onset, duration, beat))
            tonal_events.append(TonalEvent(
                midi=representative,
                duration=float(duration),
                bar=bar_index,
                beat_strength=float(getattr(element, "beatStrength", 0.5) or 0.5),
            ))
    if running_notes:
        phrase_notes.append(running_notes)
        phrase_quarters.append(running_quarters)
    if not pitches:
        raise ValueError("ABC has no pitched note attacks")

    intervals = [right - left for left, right in zip(pitches, pitches[1:])]
    absolute_intervals = [abs(value) for value in intervals]
    leap_flags = [value > 2 for value in absolute_intervals]
    direction_changes = sum(
        left * right < 0 for left, right in zip(intervals, intervals[1:])
        if left and right
    )
    repeated_notes = sum(value == 0 for value in intervals)
    consecutive_leaps = sum(left and right for left, right in zip(leap_flags, leap_flags[1:]))

    offbeat = syncopated = 0
    for _, onset, duration, beat in attacks:
        if beat <= 0:
            continue
        position = onset % beat
        if position:
            offbeat += 1
            if position + duration > beat:
                syncopated += 1

    key_objects = list(part.recurse().getElementsByClass(key.Key))
    key_signatures = list(part.recurse().getElementsByClass(key.KeySignature))
    first_key = key_objects[0] if key_objects else None
    first_signature = key_signatures[0] if key_signatures else first_key
    tonality = analyze_tonality(
        tonal_events,
        abc,
        len(measures),
        use_declared_key=use_declared_tonality,
    )
    fallback_scale = (
        {pitch.pitchClass for pitch in first_key.getScale().getPitches()}
        if first_key is not None else set()
    )
    segment_scales: list[tuple[int, int, frozenset[int]]] = []
    for segment in tonality.segments:
        if segment.primary is not None and segment.status in {"confident", "declared"}:
            segment_scales.append((
                segment.start_bar - 1,
                segment.end_bar,
                scale_pitch_classes(segment.primary.tonic_pc, segment.primary.mode),
            ))
    chromatic = 0
    for event in tonal_events:
        local_scale = next(
            (scale for start, end, scale in segment_scales if start <= event.bar < end),
            fallback_scale,
        )
        chromatic += int(bool(local_scale) and event.midi % 12 not in local_scale)
    final_is_tonic = (
        pitches[-1] % 12 == first_key.tonic.pitchClass if first_key is not None else None
    )

    tuplet_markers = re.findall(r"\((\d+)(?::\d*)?(?::\d*)?", abc)
    accidental_marks = re.findall(r"(?:\^{1,2}|_{1,2}|=)(?=[A-Ga-g])", abc)
    ornament_pairs = len(re.findall(r"![^!\n]+!|\+[^+\n]+\+", abc))
    grace_groups = len(re.findall(r"\{[^}\n]*\}", abc))
    repeats = len(re.findall(r"\|:|:\|", abc))
    voltas = len(re.findall(r"(?:\[|\|)[1-9]", abc))
    key_fields = re.findall(r"(?m)^K:[^\n\r]*|\[K:[^\]]+\]", abc)
    meter_fields = re.findall(
        r"(?m)^M:[^\n\r]*|\[M:[^\]]+\]|\|M:(?:C\||C|none|\d+/\d+)", abc
    )
    event_count = len(durations)
    note_count = len(pitches)
    bar_count = len(document.bars)
    tessitura_low = _quantile(pitches, 0.10)
    tessitura_high = _quantile(pitches, 0.90)
    if chord_count:
        warnings.append("逻辑声部含和弦；旋律音程暂以和弦最高音计算")
    warnings.extend(tonality.warnings)
    warnings.append("这些是未校准原始指标，不是考试等级或总难度分数")

    dimensions = {
        "tonal_language": {
            "declared_key": first_key.name if first_key is not None else None,
            "declared_mode": first_key.mode if first_key is not None else None,
            "key_signature_accidentals": abs(first_signature.sharps) if first_signature is not None else None,
            "key_change_count": max(0, len(key_fields) - 1),
            "chromatic_note_rate": _rate(chromatic, note_count),
            "distinct_pitch_classes": len({pitch % 12 for pitch in pitches}),
            "final_is_tonic": final_is_tonic,
            "inferred_tonic": tonality.primary.tonic if tonality.primary else None,
            "inferred_mode": tonality.primary.mode if tonality.primary else None,
            "mode_status": tonality.status,
            "mode_confidence": tonality.confidence,
            "inferred_modulation_count": tonality.modulation_count,
        },
        "melodic_motion": {
            "lowest_midi": min(pitches),
            "highest_midi": max(pitches),
            "pitch_range_semitones": max(pitches) - min(pitches),
            "tessitura_10_midi": tessitura_low,
            "tessitura_90_midi": tessitura_high,
            "mean_abs_interval_semitones": round(mean(absolute_intervals), 4) if absolute_intervals else 0.0,
            "max_abs_interval_semitones": max(absolute_intervals, default=0),
            "leap_rate_gt_major_second": _rate(sum(leap_flags), len(absolute_intervals)),
            "large_leap_rate_gt_perfect_fifth": _rate(sum(value > 7 for value in absolute_intervals), len(absolute_intervals)),
            "consecutive_leap_rate": _rate(consecutive_leaps, max(len(leap_flags) - 1, 0)),
            "direction_change_rate": _rate(direction_changes, max(len(intervals) - 1, 0)),
            "repeated_note_rate": _rate(repeated_notes, len(intervals)),
        },
        "rhythm_and_meter": {
            "meter": current_ts.ratioString if current_ts is not None else None,
            "meter_change_count": max(0, len(meter_fields) - 1),
            "shortest_note_quarters": float(min(note_durations)) if note_durations else None,
            "duration_class_count": len(set(durations)),
            "duration_entropy_bits": round(_entropy(durations), 4),
            "tuplet_group_count": len(tuplet_markers),
            "tuplet_note_rate": _rate(tuplet_note_count, note_count),
            "dotted_note_rate": _rate(dotted_count, note_count),
            "offbeat_attack_rate": _rate(offbeat, len(attacks)),
            "syncopated_attack_rate": _rate(syncopated, len(attacks)),
            "rest_event_rate": _rate(rest_count, event_count),
            "tie_mark_count": len(re.findall(r"-(?=\s*[=_^]*[A-Ga-g])", abc)),
        },
        "notation_and_reading": {
            "accidental_mark_count": len(accidental_marks),
            "accidental_marks_per_note": round(len(accidental_marks) / note_count, 4),
            "ornament_count": ornament_pairs,
            "grace_group_count": grace_groups,
            "repeat_barline_count": repeats,
            "volta_count": voltas,
            "slur_start_count": len(re.findall(r"\((?!\d)", abc)),
            "chord_event_count": chord_count,
            "bar_patch_count": bar_count,
        },
        "form_memory_and_breath": {
            "note_count": note_count,
            "notes_per_bar": round(note_count / bar_count, 4) if bar_count else 0.0,
            "motif_interval_3gram_repetition_rate": _motif_repetition_rate(pitches),
            "max_notes_without_long_rest": max(phrase_notes, default=note_count),
            "max_quarters_without_long_rest": float(max(phrase_quarters, default=sum(note_durations))),
            "tie_attack_rate": _rate(tie_count, note_count + tie_count),
        },
    }
    return DifficultyReport(
        dimensions=dimensions,
        tonality=tonality.to_dict(),
        warnings=tuple(warnings),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("abc", type=Path)
    args = parser.parse_args(argv)
    raw = args.abc.read_text(encoding="utf-8-sig")
    print(json.dumps(analyze_abc(raw).to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
