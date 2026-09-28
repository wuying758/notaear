"""Losslessly split one ABC tune into a header and bar-sized patches.

This module deliberately does not normalize or re-serialize ABC.  Patch text is
cut directly from the input string, so joining a document must reproduce the
input byte-for-byte (for the decoded Python string).  The scanner recognizes
top-level ABC barlines while ignoring bar-like characters in chord symbols,
quoted annotations, decorations, grace groups, inline fields and comments.
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path


class AbcPatchError(ValueError):
    """Raised when a string is not a supported single-tune ABC document."""


_FIELD_LINE_RE = re.compile(r"^[A-Za-z+]:")
_KEY_LINE_RE = re.compile(r"^K:")
_TUPLET_RE = re.compile(r"\((\d+)(?::(\d*))?(?::(\d*))?")
_INLINE_KEY_RE = re.compile(r"\[K:([^\]]+)\]")
_INLINE_METER_RE = re.compile(r"\[M:([^\]]+)\]")

# Longest match wins.  These are structural barlines, not arbitrary pipes.
_BARLINES = (":|:", "|::", "::|", "[|", "|]", "||", "|:", ":|", "::", "|")


@dataclass(frozen=True)
class BarPatch:
    index: int
    text: str
    closing_barline: str | None
    tuplet_markers: tuple[str, ...]
    inline_keys: tuple[str, ...]
    inline_meters: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class AbcDocument:
    header: str
    bars: tuple[BarPatch, ...]
    trailer: str = ""

    def join(self) -> str:
        return self.header + "".join(bar.text for bar in self.bars) + self.trailer

    def to_dict(self) -> dict:
        return {
            "header": self.header,
            "bars": [bar.to_dict() for bar in self.bars],
            "trailer": self.trailer,
        }


def _header_end(abc: str) -> int:
    """Return the first music character after the initial ABC header."""
    offset = 0
    saw_key = False
    x_fields = 0
    for line in abc.splitlines(keepends=True):
        stripped = line.lstrip("\ufeff \t")
        if stripped.startswith("X:"):
            x_fields += 1
            if x_fields > 1:
                raise AbcPatchError("multiple X: fields; split the tune book first")
        is_field = bool(_FIELD_LINE_RE.match(stripped))
        is_header_aux = (
            not stripped.strip()
            or stripped.startswith("%")
            or stripped.startswith("%%")
            or is_field
        )
        if not saw_key:
            offset += len(line)
            if _KEY_LINE_RE.match(stripped):
                saw_key = True
            continue
        if is_header_aux:
            offset += len(line)
            continue
        return offset
    if not saw_key:
        raise AbcPatchError("missing K: field")
    return offset


def _music_visible(text: str) -> str:
    """Strip regions that may contain note-like letters but are annotations."""
    output: list[str] = []
    quote = False
    decoration: str | None = None
    comment = False
    escaped = False
    for char in text:
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if comment:
            if char in "\r\n":
                comment = False
            continue
        if quote:
            if char == '"':
                quote = False
            continue
        if decoration is not None:
            if char == decoration:
                decoration = None
            continue
        if char == "%":
            comment = True
        elif char == '"':
            quote = True
        elif char in "!+":
            decoration = char
        else:
            output.append(char)
    return "".join(output)


def _has_music(text: str) -> bool:
    visible = _music_visible(text)
    return bool(re.search(r"[A-Ga-gzZxX]", visible))


def _bar_metadata(index: int, text: str, closing_barline: str | None) -> BarPatch:
    return BarPatch(
        index=index,
        text=text,
        closing_barline=closing_barline,
        tuplet_markers=tuple(match.group(0) for match in _TUPLET_RE.finditer(text)),
        inline_keys=tuple(match.group(1).strip() for match in _INLINE_KEY_RE.finditer(text)),
        inline_meters=tuple(match.group(1).strip() for match in _INLINE_METER_RE.finditer(text)),
    )


def split_abc(abc: str) -> AbcDocument:
    """Split a single ABC tune and prove that no source character was lost."""
    if not isinstance(abc, str) or not abc:
        raise AbcPatchError("ABC input must be a non-empty string")
    header_end = _header_end(abc)
    header, body = abc[:header_end], abc[header_end:]
    if not body:
        return AbcDocument(header=header, bars=())

    pieces: list[tuple[str, str | None]] = []
    start = 0
    index = 0
    quote = False
    decoration: str | None = None
    comment = False
    escaped = False
    square_depth = 0
    brace_depth = 0

    while index < len(body):
        char = body[index]
        if escaped:
            escaped = False
            index += 1
            continue
        if char == "\\":
            escaped = True
            index += 1
            continue
        if comment:
            if char in "\r\n":
                comment = False
            index += 1
            continue
        if quote:
            if char == '"':
                quote = False
            index += 1
            continue
        if decoration is not None:
            if char == decoration:
                decoration = None
            index += 1
            continue
        if char == "%":
            comment = True
            index += 1
            continue
        if char == '"':
            quote = True
            index += 1
            continue
        if char in "!+":
            decoration = char
            index += 1
            continue
        if char == "{":
            brace_depth += 1
            index += 1
            continue
        if char == "}" and brace_depth:
            brace_depth -= 1
            index += 1
            continue
        if brace_depth:
            index += 1
            continue
        if char == "[":
            # [| is a barline. A volta such as [1 or [1,3 has no matching ],
            # so it must stay at top level; otherwise every later barline would
            # accidentally be hidden inside an unterminated square region.
            if body.startswith("[|", index) and square_depth == 0:
                pass
            elif square_depth == 0 and index + 1 < len(body) and body[index + 1].isdigit():
                index += 2
                while index < len(body) and (body[index].isdigit() or body[index] in ",- "):
                    index += 1
                continue
            else:
                square_depth += 1
                index += 1
                continue
        if char == "]" and square_depth:
            square_depth -= 1
            index += 1
            continue
        if square_depth:
            index += 1
            continue

        token = next((item for item in _BARLINES if body.startswith(item, index)), None)
        if token is None:
            index += 1
            continue
        end = index + len(token)
        segment = body[start:end]
        # A leading |: or [| opens a bar.  Keep it with the following music
        # instead of emitting an empty patch.
        if _has_music(segment):
            pieces.append((segment, token))
            start = end
        index = end

    tail = body[start:]
    trailer = ""
    if tail:
        if _has_music(tail) or not pieces:
            pieces.append((tail, None))
        else:
            trailer = tail
    bars = tuple(
        _bar_metadata(patch_index, text, closing)
        for patch_index, (text, closing) in enumerate(pieces)
    )
    document = AbcDocument(header=header, bars=bars, trailer=trailer)
    if document.join() != abc:
        raise AssertionError("internal error: ABC patching was not lossless")
    return document


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("abc", type=Path)
    parser.add_argument("--json", action="store_true", help="print complete patches as JSON")
    args = parser.parse_args(argv)
    raw = args.abc.read_text(encoding="utf-8-sig")
    document = split_abc(raw)
    if args.json:
        print(json.dumps(document.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(json.dumps({
            "path": str(args.abc),
            "header_characters": len(document.header),
            "bar_patches": len(document.bars),
            "trailer_characters": len(document.trailer),
            "tuplet_patches": sum(bool(bar.tuplet_markers) for bar in document.bars),
            "roundtrip_exact": document.join() == raw,
        }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

