import json
from pathlib import Path

import pytest

from notaear.dataset import (
    ManifestABCDataset,
    ManifestBatchSampler,
    ManifestDataError,
    SampleRef,
    create_manifest_dataloader,
)


def _abc(key: str, body: str) -> str:
    return f"X:1\nM:2/4\nL:1/8\nK:{key}\n{body} |]\n"


def _write_fixture(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "corpus"
    records_by_key = {
        "C": [
            {"id": "s1:v1", "abc": _abc("C", "CDEF")},
            {"id": "s1:v2", "abc": _abc("C", "GABc")},
            {"id": "s2:v1", "abc": _abc("C", "C2 E2")},
            {"id": "validation:v1", "abc": _abc("C", "D2 F2")},
        ],
        "D": [
            {"id": "s1:v1", "abc": _abc("D", "DEFG")},
            {"id": "s1:v2", "abc": _abc("D", "ABcd")},
            {"id": "s2:v1", "abc": _abc("D", "D2 F2")},
            {"id": "validation:v1", "abc": _abc("D", "E2 G2")},
        ],
    }
    for key, records in records_by_key.items():
        directory = root / key
        directory.mkdir(parents=True)
        (directory / "source-000.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )

    manifest_dir = tmp_path / "manifest"
    manifest_dir.mkdir()
    common = {
        "schema_version": 1,
        "source": "test",
        "representative_key": "C",
        "variants": {
            "C": ["source-000.jsonl", 1],
            "D": ["source-000.jsonl", 1],
        },
        "variant_count": 2,
        "patch_count": 5,
        "bar_count": 1,
        "sampling_weight": 1.0,
    }
    manifest = [
        {**common, "example_id": "s1:v1", "score_id": "s1", "split": "train"},
        {
            **common,
            "example_id": "s1:v2",
            "score_id": "s1",
            "split": "train",
            "variants": {
                "C": ["source-000.jsonl", 2],
                "D": ["source-000.jsonl", 2],
            },
        },
        {
            **common,
            "example_id": "s2:v1",
            "score_id": "s2",
            "split": "train",
            "sampling_weight": 3.0,
            "variants": {
                "C": ["source-000.jsonl", 3],
                "D": ["source-000.jsonl", 3],
            },
        },
        {
            **common,
            "example_id": "validation:v1",
            "score_id": "validation",
            "split": "validation",
            "variants": {
                "C": ["source-000.jsonl", 4],
                "D": ["source-000.jsonl", 4],
            },
        },
    ]
    manifest_path = manifest_dir / "manifest.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(record) + "\n" for record in manifest),
        encoding="utf-8",
    )
    (manifest_dir / "summary.json").write_text(
        json.dumps({"input_root": str(root)}), encoding="utf-8"
    )
    return manifest_path, root


def _flatten(batches):
    return [reference for batch in batches for reference in batch]


def test_dataset_reads_split_and_verifies_source_locator(tmp_path: Path) -> None:
    manifest, _ = _write_fixture(tmp_path)
    dataset = ManifestABCDataset(manifest, split="train")
    assert len(dataset) == 3
    item = dataset[0]
    assert item["example_id"] == "s1:v1"
    assert item["target_key"] in {"C", "D"}
    assert len(item["token_ids"]) == 5

    invalid = SampleRef(0, "C", "source-000.jsonl", 2, 5)
    with pytest.raises(ManifestDataError, match="does not match manifest"):
        dataset[invalid]


def test_batch_sampler_keeps_score_voices_in_one_key_and_is_reproducible(
    tmp_path: Path,
) -> None:
    manifest, _ = _write_fixture(tmp_path)
    dataset = ManifestABCDataset(manifest, split="train")
    first = ManifestBatchSampler(
        dataset, weighted=False, shuffle=True, seed=19,
        max_examples_per_batch=8, max_patches_per_batch=64,
    )
    second = ManifestBatchSampler(
        dataset, weighted=False, shuffle=True, seed=19,
        max_examples_per_batch=8, max_patches_per_batch=64,
    )
    first.set_epoch(3)
    second.set_epoch(3)
    first_refs = _flatten(list(first))
    second_refs = _flatten(list(second))
    assert first_refs == second_refs
    assert len(first_refs) == len(dataset)
    s1_keys = {ref.target_key for ref in first_refs if ref.index in {0, 1}}
    assert len(s1_keys) == 1


def test_weighted_sampler_has_fixed_epoch_size(tmp_path: Path) -> None:
    manifest, _ = _write_fixture(tmp_path)
    dataset = ManifestABCDataset(manifest, split="train")
    sampler = ManifestBatchSampler(
        dataset, weighted=True, shuffle=False, seed=11,
        max_examples_per_batch=8, max_patches_per_batch=64,
    )
    sampler.set_epoch(2)
    first = _flatten(list(sampler))
    sampler.set_epoch(2)
    second = _flatten(list(sampler))
    assert first == second
    assert len(first) == len(dataset)


def test_small_shard_remainders_are_packed(tmp_path: Path) -> None:
    manifest, _ = _write_fixture(tmp_path)
    dataset = ManifestABCDataset(manifest, split="train")
    sampler = ManifestBatchSampler(
        dataset, weighted=False, shuffle=False, seed=11,
        max_examples_per_batch=8, max_patches_per_batch=64,
    )
    batches = list(sampler)
    assert len(batches) == 1
    assert len(batches[0]) == 3


def test_sampler_can_resume_at_an_epoch_batch_boundary(tmp_path: Path) -> None:
    manifest, _ = _write_fixture(tmp_path)
    dataset = ManifestABCDataset(manifest, split="train")
    sampler = ManifestBatchSampler(
        dataset, weighted=False, shuffle=False,
        max_examples_per_batch=1, max_patches_per_batch=64,
    )
    sampler.set_epoch(0)
    complete = list(sampler)
    sampler.set_epoch(0, start_batch=1)
    assert list(sampler) == complete[1:]
    assert len(sampler) == len(complete) - 1
    assert sampler.total_batches == len(complete)


def test_loader_collates_dynamic_batches_with_workers(tmp_path: Path) -> None:
    manifest, root = _write_fixture(tmp_path)
    dataset, sampler, loader = create_manifest_dataloader(
        manifest,
        data_root=root,
        split="train",
        weighted=False,
        shuffle=False,
        max_examples_per_batch=8,
        max_patches_per_batch=64,
        num_workers=2,
    )
    sampler.set_epoch(0)
    batches = list(loader)
    assert sum(batch["patches"].shape[0] for batch in batches) == len(dataset)
    keys_by_score = {}
    for batch in batches:
        assert batch["patches"].ndim == 3
        assert batch["patch_counts"].shape[0] == batch["patches"].shape[0]
        for score_id, target_key in zip(batch["score_ids"], batch["target_keys"]):
            keys_by_score.setdefault(score_id, set()).add(target_key)
    assert all(len(keys) == 1 for keys in keys_by_score.values())


def test_empty_split_is_rejected(tmp_path: Path) -> None:
    manifest, _ = _write_fixture(tmp_path)
    with pytest.raises(ManifestDataError, match="no records"):
        ManifestABCDataset(manifest, split="test")


def test_manifest_records_above_requested_limit_are_filtered(tmp_path: Path) -> None:
    manifest, _ = _write_fixture(tmp_path)
    lines = [json.loads(line) for line in manifest.read_text().splitlines()]
    lines[2]["patch_count"] = 8
    manifest.write_text(
        "".join(json.dumps(record) + "\n" for record in lines), encoding="utf-8"
    )
    dataset = ManifestABCDataset(manifest, split="train", max_patches=5)
    assert len(dataset) == 2
    assert dataset.index_stats == {
        "split_records": 3,
        "too_long": 1,
        "indexed": 2,
    }


def test_overlong_selected_variant_falls_back_deterministically(tmp_path: Path) -> None:
    manifest, root = _write_fixture(tmp_path)
    c_path = root / "C" / "source-000.jsonl"
    records = [json.loads(line) for line in c_path.read_text().splitlines()]
    records[0]["abc"] = _abc("C", "C" * 80)
    c_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    dataset = ManifestABCDataset(manifest, split="train", max_patches=5)
    item = dataset[SampleRef(0, "C", "source-000.jsonl", 1, 5)]
    assert item["requested_target_key"] == "C"
    assert item["target_key"] == "D"
    assert item["variant_fallback"] is True
