"""Conservative tonal-centre and mode analysis for monophonic melodies.

The analyser deliberately returns ranked candidates, confidence and an
``ambiguous``/``insufficient`` state.  A pitch-class collection alone cannot
reliably distinguish every rotation of a diatonic or pentatonic collection,
so callers must not treat the primary candidate as ground truth when the
status is not ``confident`` or ``declared``.
"""
from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import Iterable

from .abc_patches import AbcDocument, split_abc


PC_NAMES = ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")

# Canonical modes. Ionian/Aeolian are represented as major/natural_minor to
# avoid duplicate candidates with identical pitch collections.
MODE_INTERVALS: dict[str, tuple[int, ...]] = {
    "major": (0, 2, 4, 5, 7, 9, 11),
    "natural_minor": (0, 2, 3, 5, 7, 8, 10),
    "harmonic_minor": (0, 2, 3, 5, 7, 8, 11),
    # Both directional forms are allowed. Directional evidence is added in
    # _candidate_score; this nine-note union is not interpreted as a scale.
    "melodic_minor": (0, 2, 3, 5, 7, 8, 9, 10, 11),
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

PENTATONIC_MODES = frozenset({"gong", "shang", "jue", "zhi", "yu"})
MINOR_MODES = frozenset({"natural_minor", "harmonic_minor", "melodic_minor"})


@dataclass(frozen=True)
class TonalEvent:
    midi: int
    duration: float
    bar: int
    beat_strength: float = 0.5


@dataclass(frozen=True)
class TonalityCandidate:
    tonic: str
    tonic_pc: int
    mode: str
    family: str
    score: float
    support: float


@dataclass(frozen=True)
class TonalitySegment:
    start_bar: int
    end_bar: int
    primary: TonalityCandidate | None
    candidates: tuple[TonalityCandidate, ...]
    status: str
    confidence: float
    out_of_scale_note_rate: float | None
    source: str


@dataclass(frozen=True)
class TonalityReport:
    primary: TonalityCandidate | None
    candidates: tuple[TonalityCandidate, ...]
    segments: tuple[TonalitySegment, ...]
    status: str
    confidence: float
    confidence_calibrated: bool
    confidence_kind: str
    uncertainty: float
    modulation_count: int
    out_of_scale_note_rate: float | None
    warnings: tuple[str, ...] = ()
    schema_version: int = 1

    def to_dict(self) -> dict:
        return asdict(self)


def _family(mode: str) -> str:
    if mode in PENTATONIC_MODES:
        return "chinese_pentatonic"
    if mode in MINOR_MODES:
        return "minor"
    return "western_mode"


def scale_pitch_classes(tonic_pc: int, mode: str) -> frozenset[int]:
    return frozenset((tonic_pc + interval) % 12 for interval in MODE_INTERVALS[mode])


def parse_key_field(raw: str) -> tuple[int, str] | None:
    value = raw.strip().split("%", 1)[0].strip()
    if not value or value.lower().startswith(("none", "perc", "hp")):
        return None
    match = re.match(r"^([A-Ga-g])([#b]?)(.*)$", value)
    if not match:
        return None
    letter, accidental, suffix = match.groups()
    base = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}[letter.upper()]
    tonic = (base + (1 if accidental == "#" else -1 if accidental == "b" else 0)) % 12
    compact = re.sub(r"[\s_-]+", "", suffix.lower())
    aliases = (
        (("harmonicminor", "harmminor", "harm"), "harmonic_minor"),
        (("melodicminor", "melminor", "mel"), "melodic_minor"),
        (("minor", "min", "aeolian", "aeo", "m"), "natural_minor"),
        (("dorian", "dor"), "dorian"),
        (("phrygian", "phr"), "phrygian"),
        (("lydian", "lyd"), "lydian"),
        (("mixolydian", "mix"), "mixolydian"),
        (("locrian", "loc"), "locrian"),
        (("gong", "宫"), "gong"),
        (("shang", "商"), "shang"),
        (("jue", "角"), "jue"),
        (("zhi", "徵", "征"), "zhi"),
        (("yu", "羽"), "yu"),
        (("major", "maj", "ionian", "ion", ""), "major"),
    )
    for prefixes, canonical in aliases:
        if compact in prefixes or any(compact.startswith(prefix) for prefix in prefixes if prefix):
            return tonic, canonical
    return tonic, "major"


def _declared_keys(abc: str, document: AbcDocument) -> list[tuple[int, tuple[int, str]]]:
    events: list[tuple[int, tuple[int, str]]] = []
    header_fields = re.findall(r"(?m)^\s*K:\s*([^\r\n]+)", document.header)
    if header_fields:
        parsed = parse_key_field(header_fields[-1])
        if parsed is not None:
            events.append((0, parsed))
    for bar in document.bars:
        for raw in bar.inline_keys:
            parsed = parse_key_field(raw)
            if parsed is not None and (not events or events[-1][1] != parsed):
                events.append((bar.index, parsed))
    return events


def _profile(mode: str) -> dict[int, float]:
    intervals = MODE_INTERVALS[mode]
    weights = {interval: 1.0 for interval in intervals}
    weights[0] = 3.0
    if 7 in weights:
        weights[7] = 1.65
    if 3 in weights or 4 in weights:
        third = 3 if 3 in weights else 4
        weights[third] = 1.3
    if mode == "melodic_minor":
        weights.update({8: 0.55, 9: 0.8, 10: 0.55, 11: 0.8})
    return weights


def _cosine(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    denominator = math.sqrt(sum(a * a for a in left) * sum(b * b for b in right))
    return numerator / denominator if denominator else 0.0


def _candidate_score(
    events: list[TonalEvent], tonic: int, mode: str, declared: tuple[int, str] | None
) -> tuple[float, float]:
    histogram = [0.0] * 12
    total = 0.0
    for event in events:
        weight = max(0.25, math.sqrt(max(event.duration, 0.0))) * (1.0 + 0.2 * event.beat_strength)
        histogram[event.midi % 12] += weight
        total += weight
    expected = [0.0] * 12
    for interval, weight in _profile(mode).items():
        expected[(tonic + interval) % 12] = weight
    scale = scale_pitch_classes(tonic, mode)
    coverage = sum(histogram[pc] for pc in scale) / total if total else 0.0
    fit = 0.58 * _cosine(histogram, expected) + 0.27 * coverage
    fit += 0.10 * float(events[-1].midi % 12 == tonic)
    fit += 0.05 * float(events[0].midi % 12 == tonic)

    # Melodic minor is directional: raised 6/7 ascending and natural 6/7
    # descending are positive evidence, not merely a permissive nine-note set.
    if mode == "melodic_minor" and len(events) > 1:
        evidence = matches = 0
        for left, right in zip(events, events[1:]):
            direction = (right.midi > left.midi) - (right.midi < left.midi)
            degree = (right.midi - tonic) % 12
            if degree in (8, 9, 10, 11) and direction:
                evidence += 1
                matches += int((direction > 0 and degree in (9, 11)) or
                               (direction < 0 and degree in (8, 10)))
        # The nine-note union is otherwise too permissive. Require actual
        # directional evidence before preferring melodic minor.
        fit -= 0.025
        if evidence >= 2:
            fit += 0.08 * (matches / evidence - 0.5)
        else:
            fit -= 0.04

    if declared is not None and tonic == declared[0]:
        fit += 0.025
        declared_mode = declared[1]
        if mode == declared_mode:
            fit += 0.025
        elif declared_mode == "natural_minor" and mode in MINOR_MODES:
            fit += 0.015
    return fit, coverage


def _rank_candidates(
    events: list[TonalEvent], declared: tuple[int, str] | None
) -> tuple[tuple[TonalityCandidate, ...], float, str, float | None]:
    if len(events) < 5 or len({event.midi % 12 for event in events}) < 3:
        return (), 0.0, "insufficient", None
    raw: list[tuple[float, float, int, str]] = []
    for tonic in range(12):
        for mode in MODE_INTERVALS:
            score, coverage = _candidate_score(events, tonic, mode, declared)
            raw.append((score, coverage, tonic, mode))
    raw.sort(reverse=True)
    maximum = raw[0][0]
    exponentials = [math.exp((item[0] - maximum) / 0.035) for item in raw]
    denominator = sum(exponentials)
    candidates = tuple(
        TonalityCandidate(
            tonic=PC_NAMES[tonic],
            tonic_pc=tonic,
            mode=mode,
            family=_family(mode),
            score=round(score, 6),
            support=round(exponential / denominator, 6),
        )
        for (score, _, tonic, mode), exponential in zip(raw[:3], exponentials[:3])
    )
    top_score, top_coverage = raw[0][0], raw[0][1]
    margin = top_score - raw[1][0]
    quality = min(1.0, max(0.0, (top_score - 0.45) / 0.45))
    confidence = 0.50 * quality + 0.35 * min(1.0, margin / 0.08)
    confidence += 0.15 * min(1.0, len(events) / 24.0)
    if declared is not None and (raw[0][2], raw[0][3]) == declared and top_coverage >= 0.75:
        confidence = max(confidence, 0.65)
        status = "declared"
    else:
        status = "confident" if confidence >= 0.58 and margin >= 0.018 else "ambiguous"
    out_rate = 1.0 - top_coverage
    return candidates, min(1.0, confidence), status, out_rate


def _segment(
    events: list[TonalEvent], start: int, end: int,
    declared: tuple[int, str] | None, source: str,
) -> TonalitySegment:
    selected = [event for event in events if start <= event.bar < end]
    candidates, confidence, status, out_rate = _rank_candidates(selected, declared)
    return TonalitySegment(
        start_bar=start + 1,
        end_bar=end,
        primary=candidates[0] if candidates else None,
        candidates=candidates,
        status=status,
        confidence=round(confidence, 6),
        out_of_scale_note_rate=round(out_rate, 6) if out_rate is not None else None,
        source=source,
    )


def _merge_segments(segments: list[TonalitySegment]) -> list[TonalitySegment]:
    merged: list[TonalitySegment] = []
    for segment in segments:
        if (merged and merged[-1].primary is not None and segment.primary is not None and
                (merged[-1].primary.tonic_pc, merged[-1].primary.mode) ==
                (segment.primary.tonic_pc, segment.primary.mode)):
            previous = merged.pop()
            merged.append(TonalitySegment(
                start_bar=previous.start_bar,
                end_bar=segment.end_bar,
                primary=previous.primary if previous.confidence >= segment.confidence else segment.primary,
                candidates=previous.candidates if previous.confidence >= segment.confidence else segment.candidates,
                status=previous.status if previous.confidence >= segment.confidence else segment.status,
                confidence=round((previous.confidence + segment.confidence) / 2, 6),
                out_of_scale_note_rate=round(
                    ((previous.out_of_scale_note_rate or 0.0) +
                     (segment.out_of_scale_note_rate or 0.0)) / 2, 6
                ),
                source=previous.source,
            ))
        else:
            merged.append(segment)
    return merged


def analyze_tonality(
    events: Iterable[TonalEvent], abc: str, bar_count: int | None = None,
    *, use_declared_key: bool = True,
) -> TonalityReport:
    """Analyse global and local tonal organisation of a monophonic ABC tune."""
    event_list = list(events)
    document = split_abc(abc)
    bars = bar_count if bar_count is not None else len(document.bars)
    bars = max(bars, max((event.bar + 1 for event in event_list), default=0))
    declarations = _declared_keys(abc, document) if use_declared_key else []
    initial = declarations[0][1] if declarations and declarations[0][0] == 0 else None
    candidates, confidence, status, global_out_rate = _rank_candidates(event_list, initial)

    segments: list[TonalitySegment]
    if len(declarations) > 1 or (declarations and declarations[0][0] > 0):
        segments = []
        boundaries: list[tuple[int, tuple[int, str] | None]] = list(declarations)
        if boundaries[0][0] > 0:
            boundaries.insert(0, (0, None))
        for index, (start, declared) in enumerate(boundaries):
            end = boundaries[index + 1][0] if index + 1 < len(boundaries) else bars
            if end > start:
                source = "declared" if declared is not None else "inferred"
                segments.append(_segment(event_list, start, end, declared, source))
        segments = _merge_segments(segments)
    elif bars >= 8:
        windows = []
        start = 0
        while start < bars:
            end = min(bars, start + 4)
            if end - start < 2 and windows:
                previous = windows.pop()
                start = previous.start_bar - 1
            windows.append(_segment(event_list, start, end, initial, "inferred"))
            start = end
        credible = all(item.status in {"confident", "declared"} for item in windows)
        identities = {
            (item.primary.tonic_pc, item.primary.mode)
            for item in windows if item.primary is not None
        }
        segments = _merge_segments(windows) if credible and len(identities) > 1 else [
            _segment(event_list, 0, bars, initial, "global")
        ]
    else:
        segments = [_segment(event_list, 0, bars, initial, "global")] if bars else []

    changes = 0
    for left, right in zip(segments, segments[1:]):
        if left.primary and right.primary and (
            left.primary.tonic_pc, left.primary.mode
        ) != (right.primary.tonic_pc, right.primary.mode):
            changes += 1
    if segments and all(item.out_of_scale_note_rate is not None for item in segments):
        lengths = [max(1, item.end_bar - item.start_bar + 1) for item in segments]
        out_rate = sum(
            float(item.out_of_scale_note_rate) * length
            for item, length in zip(segments, lengths)
        ) / sum(lengths)
    else:
        out_rate = global_out_rate

    warnings: list[str] = []
    if status == "ambiguous":
        warnings.append("调式证据不足以唯一判定；必须保留候选和置信度")
    elif status == "insufficient":
        warnings.append("有效音符或音级过少，无法可靠识别调式")
    if out_rate is not None and out_rate > 0.25:
        warnings.append("较多音符不属于最佳候选调式，可能存在半音化、转调或分布外材料")
    return TonalityReport(
        primary=candidates[0] if candidates else None,
        candidates=candidates,
        segments=tuple(segments),
        status=status,
        confidence=round(confidence, 6),
        confidence_calibrated=False,
        confidence_kind="heuristic_fit_margin",
        uncertainty=round(1.0 - confidence, 6),
        modulation_count=changes,
        out_of_scale_note_rate=round(out_rate, 6) if out_rate is not None else None,
        warnings=tuple(warnings),
    )
