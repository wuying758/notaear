import json
from pathlib import Path

import pytest

from notaear.build_manifest import _source_weights, build_manifest
from notaear.manifest import choose_variant, iter_manifest


def _record(example_id: str, score_id: str, source: str, voice: str, key: str, body: str) -> dict:
    return {
        "id": example_id,
        "score_id": score_id,
        "source": source,
        "source_path": f"raw/{source}/{score_id}.xml",
        "voice": voice,
        "abc": f"X:1\nM:2/4\nL:1/8\nK:{key}\n{body}\n",
    }


def _write(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def _fixture(root: Path) -> None:
    for key, first, second in (("C", "CDEF |]", "GABc |]"), ("D", "DEFG |]", "ABcd |]")):
        _write(root / key / "pdmx-000.jsonl", [
            _record("pdmx:p:V:1", "pdmx:p", "pdmx", "V:1", key, first),
            _record("pdmx:p:V:2", "pdmx:p", "pdmx", "V:2", key, second),
        ])
        _write(root / key / "norbeck-000.jsonl", [
            _record("norbeck:n:V:1", "norbeck:n", "norbeck", "V:1", key, first),
            _record("norbeck:n:V:2", "norbeck:n", "norbeck", "V:2", key, second),
        ])
    _write(root / "C" / "bad-000.jsonl", [
        _record("bad:b:V:1", "bad:b", "bad", "V:1", "C", "[CEG]2 z2 |]"),
    ])


def test_build_manifest_deduplicates_by_work_and_resumes(tmp_path: Path) -> None:
    root = tmp_path / "input"
    output = tmp_path / "output"
    _fixture(root)

    summary = build_manifest(
        root, output, validation_rate=0.1, test_rate=0.1, workers=2
    )
    accepted = list(iter_manifest(output / "manifest.jsonl"))
    rejected = list(iter_manifest(output / "rejected.jsonl"))

    assert summary["accepted_examples"] == 2
    assert {record["score_id"] for record in accepted} == {"norbeck:n"}
    assert {record["variant_count"] for record in accepted} == {2}
    assert len({record["split"] for record in accepted}) == 1
    assert all(set(record["variants"]) == {"C", "D"} for record in accepted)
    assert sum("duplicate_work" in record["reasons"] for record in rejected) == 2
    assert any("chordal" in record["reasons"] for record in rejected)

    resumed = build_manifest(root, output, validation_rate=0.1, test_rate=0.1)
    assert resumed["processed_shards"] == 0
    assert resumed["resumed_shards"] == 3


def test_changed_shard_and_changed_index_config_fail_closed(tmp_path: Path) -> None:
    root = tmp_path / "input"
    output = tmp_path / "output"
    _fixture(root)
    build_manifest(root, output)

    with pytest.raises(RuntimeError, match="configuration changed"):
        build_manifest(root, output, patch_size=64)

    shard = root / "C" / "pdmx-000.jsonl"
    shard.write_text(shard.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="indexed shard changed"):
        build_manifest(root, output)


def test_source_weights_are_normalized_and_capped() -> None:
    counts = {"large": 1_000_000, "small": 1}
    weights = _source_weights(counts, alpha=0.5, max_weight=3.0)
    weighted_mean = sum(counts[key] * value for key, value in weights.items()) / sum(counts.values())
    assert weighted_mean == pytest.approx(1.0)
    assert max(weights.values()) <= 3.0
    assert weights["small"] > weights["large"]


def test_variant_selection_is_stable_and_shared_by_score() -> None:
    first = {
        "score_id": "same-score",
        "example_id": "same-score:V:1",
        "variants": {"C": ["x.jsonl", 1], "D": ["x.jsonl", 2]},
    }
    second = {
        "score_id": "same-score",
        "example_id": "same-score:V:2",
        "variants": {"C": ["x.jsonl", 3], "D": ["x.jsonl", 4]},
    }
    assert choose_variant(first, 7)[0] == choose_variant(second, 7)[0]
    assert choose_variant(first, 7) == choose_variant(first, 7)


def test_duplicate_variant_locator_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "input"
    output = tmp_path / "output"
    duplicate = _record("dup:s:V:1", "dup:s", "dup", "V:1", "C", "CDEF |]")
    _write(root / "C" / "first.jsonl", [duplicate])
    _write(root / "C" / "second.jsonl", [duplicate])

    build_manifest(root, output)
    assert list(iter_manifest(output / "manifest.jsonl")) == []
    rejected = list(iter_manifest(output / "rejected.jsonl"))
    assert rejected[0]["reasons"] == ["duplicate_variant_locator"]
