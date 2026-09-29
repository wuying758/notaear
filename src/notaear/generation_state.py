"""Structural state machine for canonical NotaEar patch generation."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .abc_patches import AbcPatchError, split_abc
from .tokenizer import (
    BAR_ID,
    CONT_ID,
    DOC_BOS_ID,
    DOC_EOS_ID,
    HEADER_ID,
    PAD_ID,
    PATCH_END_ID,
    TRAILER_ID,
    NotaEarTokenizer,
)


class GenerationStateError(ValueError):
    """Raised when a prompt or sampled patch violates structural constraints."""


class Phase(str, Enum):
    HEADER = "header"
    BAR_COMPLETE = "bar_complete"
    BAR_CONTINUATION = "bar_continuation"
    TRAILER = "trailer"
    FINISHED = "finished"


def _row_payload(row: tuple[int, ...] | list[int]) -> bytes:
    try:
        end = row.index(PATCH_END_ID, 1)
    except ValueError as exc:
        raise GenerationStateError("patch is missing PATCH_END") from exc
    payload = row[1:end]
    if any(not 0 <= token < 256 for token in payload):
        raise GenerationStateError("patch payload contains a non-byte token")
    if any(token != PAD_ID for token in row[end + 1:]):
        raise GenerationStateError("patch has non-padding tokens after PATCH_END")
    return bytes(payload)


def _complete_single_bar(payload: bytes) -> bool:
    """Return whether payload is exactly one music bar ending in a barline."""
    try:
        text = payload.decode("ascii")
    except UnicodeDecodeError:
        return False
    try:
        document = split_abc("X:1\nK:C\n" + text)
    except AbcPatchError:
        return False
    return (
        len(document.bars) == 1
        and document.bars[0].closing_barline is not None
        and document.bars[0].text == text
        and not document.trailer
    )


@dataclass
class PatchGenerationState:
    """Track legal marker transitions and one logical bar across CONT patches."""

    tokenizer: NotaEarTokenizer
    phase: Phase
    bar_count: int
    bar_payload: bytearray
    active_marker: int | None = None

    @classmethod
    def from_prompt(
        cls,
        tokenizer: NotaEarTokenizer,
        rows: list[list[int]],
    ) -> "PatchGenerationState":
        if not rows or rows[0][0] != DOC_BOS_ID:
            raise GenerationStateError("prompt must start with DOC_BOS")
        logical_kind: str | None = None
        logical_payload = bytearray()
        bar_count = 0
        saw_header = False
        for index, row in enumerate(rows[1:], start=1):
            if len(row) != tokenizer.patch_size:
                raise GenerationStateError(
                    f"prompt patch {index} has length {len(row)}"
                )
            marker = row[0]
            payload = _row_payload(row)
            if marker == HEADER_ID:
                if saw_header or logical_kind is not None:
                    raise GenerationStateError("prompt has a misplaced HEADER patch")
                saw_header = True
                logical_kind = "header"
                logical_payload = bytearray(payload)
            elif marker == BAR_ID:
                if not saw_header or logical_kind == "trailer":
                    raise GenerationStateError("prompt has a misplaced BAR patch")
                if logical_kind == "bar" and not _complete_single_bar(bytes(logical_payload)):
                    raise GenerationStateError("prompt starts a new bar before the prior bar closes")
                logical_kind = "bar"
                logical_payload = bytearray(payload)
                bar_count += 1
            elif marker == CONT_ID:
                if logical_kind is None:
                    raise GenerationStateError("prompt CONT patch has nothing to continue")
                logical_payload.extend(payload)
            elif marker == TRAILER_ID:
                if logical_kind != "bar" or not _complete_single_bar(bytes(logical_payload)):
                    raise GenerationStateError("prompt trailer follows an incomplete bar")
                logical_kind = "trailer"
                logical_payload = bytearray(payload)
            else:
                raise GenerationStateError(f"prompt has invalid marker {marker}")

        if not saw_header or logical_kind is None:
            raise GenerationStateError("prompt must contain a header")
        if logical_kind == "header":
            phase = Phase.HEADER
            bar_payload = bytearray()
        elif logical_kind == "bar":
            if not _complete_single_bar(bytes(logical_payload)):
                raise GenerationStateError(
                    "prompt ends inside a bar; provide a header-only prompt or complete bars"
                )
            phase = Phase.BAR_COMPLETE
            bar_payload = logical_payload
        else:
            phase = Phase.TRAILER
            bar_payload = bytearray()
        return cls(
            tokenizer=tokenizer,
            phase=phase,
            bar_count=bar_count,
            bar_payload=bar_payload,
        )

    def allowed_markers(self) -> tuple[int, ...]:
        if self.active_marker is not None:
            raise GenerationStateError("cannot choose a marker while a patch is active")
        if self.phase == Phase.HEADER:
            return (BAR_ID,)
        if self.phase == Phase.BAR_CONTINUATION:
            return (CONT_ID,)
        if self.phase == Phase.BAR_COMPLETE:
            return (BAR_ID, DOC_EOS_ID)
        if self.phase == Phase.TRAILER:
            return (DOC_EOS_ID,)
        return ()

    def start_patch(self, marker: int) -> None:
        if marker not in self.allowed_markers():
            raise GenerationStateError(
                f"marker {marker} is not allowed in phase {self.phase.value}"
            )
        self.active_marker = marker
        if marker == BAR_ID:
            self.bar_count += 1
            self.bar_payload = bytearray()
            self.phase = Phase.BAR_CONTINUATION
        elif marker == DOC_EOS_ID:
            self.phase = Phase.FINISHED

    def allowed_payload_tokens(
        self,
        current_patch_payload: bytes,
        base_tokens: tuple[int, ...],
    ) -> tuple[int, ...]:
        if self.active_marker not in (BAR_ID, CONT_ID):
            raise GenerationStateError("active patch does not accept bar payload")
        combined = bytes(self.bar_payload) + current_patch_payload
        if not _complete_single_bar(combined):
            return base_tokens
        # Once a structural barline has appeared, allow only a longer valid
        # barline spelling (for example | -> |] or |:) or PATCH_END.
        extensions = tuple(
            token
            for token in base_tokens
            if _complete_single_bar(combined + bytes((token,)))
        )
        return (*extensions, PATCH_END_ID)

    def finish_patch(self, payload: bytes, *, full: bool) -> None:
        marker = self.active_marker
        if marker not in (BAR_ID, CONT_ID):
            raise GenerationStateError("no active content patch to finish")
        self.bar_payload.extend(payload)
        complete = _complete_single_bar(bytes(self.bar_payload))
        if not complete and not full:
            raise GenerationStateError("an incomplete bar ended before filling its patch")
        self.phase = Phase.BAR_COMPLETE if complete else Phase.BAR_CONTINUATION
        self.active_marker = None

    def finish_eos(self) -> None:
        if self.active_marker != DOC_EOS_ID or self.phase != Phase.FINISHED:
            raise GenerationStateError("EOS was not started legally")
        self.active_marker = None
