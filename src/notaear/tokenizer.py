"""Lossless UTF-8 byte tokenizer with NotaGen-style fixed-size patches."""
from __future__ import annotations

from dataclasses import dataclass

from .abc_patches import split_abc


BYTE_VOCAB_SIZE = 256
PAD_ID = 256
DOC_BOS_ID = 257
DOC_EOS_ID = 258
PATCH_END_ID = 259
HEADER_ID = 260
BAR_ID = 261
CONT_ID = 262
TRAILER_ID = 263
VOCAB_SIZE = 264

_PATCH_MARKERS = {DOC_BOS_ID, DOC_EOS_ID, HEADER_ID, BAR_ID, CONT_ID, TRAILER_ID}


class TokenizerError(ValueError):
    """Raised when a patch matrix cannot be decoded safely."""


@dataclass(frozen=True)
class EncodedABC:
    token_ids: tuple[tuple[int, ...], ...]
    patch_size: int

    @property
    def patch_count(self) -> int:
        return len(self.token_ids)


class NotaEarTokenizer:
    """Encode ABC as marked, fixed-size UTF-8 byte patches.

    Every physical patch is marker + payload + PATCH_END + PAD. A logical
    header, bar or trailer may span multiple physical patches; later pieces use
    CONT_ID. Special ids live above the complete byte range, so arbitrary UTF-8
    text remains exactly reversible without an unknown token.
    """

    def __init__(self, patch_size: int = 32):
        if patch_size < 4:
            raise ValueError("patch_size must be at least 4")
        self.patch_size = patch_size
        self.payload_size = patch_size - 2

    def _special_patch(self, marker: int) -> tuple[int, ...]:
        return tuple([marker, PATCH_END_ID] + [PAD_ID] * (self.patch_size - 2))

    def _segment_patches(self, text: str, marker: int) -> list[tuple[int, ...]]:
        raw = text.encode("utf-8")
        if not raw:
            return []
        rows: list[tuple[int, ...]] = []
        for offset in range(0, len(raw), self.payload_size):
            chunk = list(raw[offset:offset + self.payload_size])
            row_marker = marker if offset == 0 else CONT_ID
            row = [row_marker, *chunk, PATCH_END_ID]
            row.extend([PAD_ID] * (self.patch_size - len(row)))
            rows.append(tuple(row))
        return rows

    def encode(self, abc: str) -> EncodedABC:
        document = split_abc(abc)
        rows = [self._special_patch(DOC_BOS_ID)]
        rows.extend(self._segment_patches(document.header, HEADER_ID))
        for bar in document.bars:
            rows.extend(self._segment_patches(bar.text, BAR_ID))
        rows.extend(self._segment_patches(document.trailer, TRAILER_ID))
        rows.append(self._special_patch(DOC_EOS_ID))
        encoded = EncodedABC(token_ids=tuple(rows), patch_size=self.patch_size)
        if self.decode(encoded) != abc:
            raise AssertionError("internal error: tokenizer round-trip was not lossless")
        return encoded

    def decode(
        self,
        encoded: EncodedABC | list[list[int]] | tuple[tuple[int, ...], ...],
    ) -> str:
        rows = encoded.token_ids if isinstance(encoded, EncodedABC) else encoded
        output = bytearray()
        for patch_index, row_value in enumerate(rows):
            row = list(row_value)
            if len(row) != self.patch_size:
                raise TokenizerError(
                    f"patch {patch_index} has length {len(row)}, expected {self.patch_size}"
                )
            marker = row[0]
            if marker not in _PATCH_MARKERS:
                raise TokenizerError(f"patch {patch_index} has invalid marker {marker}")
            try:
                end = row.index(PATCH_END_ID, 1)
            except ValueError as exc:
                raise TokenizerError(f"patch {patch_index} has no PATCH_END token") from exc
            if any(token != PAD_ID for token in row[end + 1:]):
                raise TokenizerError(
                    f"patch {patch_index} has non-padding tokens after PATCH_END"
                )
            if marker in (DOC_BOS_ID, DOC_EOS_ID):
                if end != 1:
                    raise TokenizerError(
                        f"document marker patch {patch_index} contains payload"
                    )
                continue
            payload = row[1:end]
            if any(not 0 <= token < BYTE_VOCAB_SIZE for token in payload):
                raise TokenizerError(
                    f"patch {patch_index} contains a non-byte payload token"
                )
            output.extend(payload)
        try:
            return output.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise TokenizerError("patch payload is not valid UTF-8") from exc
