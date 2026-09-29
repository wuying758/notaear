"""Read NotaEar manifests and choose one deterministic key variant per epoch."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterator


def iter_manifest(path: Path, split: str | None = None) -> Iterator[dict[str, Any]]:
    """Yield manifest records, optionally restricted to one split."""
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid manifest JSON at line {line_number}: {exc}") from exc
            if split is None or record.get("split") == split:
                yield record


def choose_variant(
    record: dict[str, Any],
    epoch: int,
    *,
    seed: int = 0,
) -> tuple[str, str, int]:
    """Choose a stable variant; voices from one score receive the same key."""
    variants = record.get("variants")
    if not isinstance(variants, dict) or not variants:
        raise ValueError("manifest record has no variants")
    keys = sorted(variants)
    score_id = str(record.get("score_id", record.get("example_id", "")))
    digest = hashlib.blake2b(
        f"{seed}\0{epoch}\0{score_id}".encode(), digest_size=8
    ).digest()
    key = keys[int.from_bytes(digest, "big") % len(keys)]
    locator = variants[key]
    if not (
        isinstance(locator, list)
        and len(locator) == 2
        and isinstance(locator[0], str)
        and isinstance(locator[1], int)
        and locator[1] >= 1
    ):
        raise ValueError(f"invalid locator for key {key!r}: {locator!r}")
    return key, locator[0], locator[1]
