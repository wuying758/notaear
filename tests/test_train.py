import contextlib
import io
import json
from pathlib import Path

import pytest
import torch

from notaear.checkpoint import load_checkpoint_payload
from notaear.train import learning_rate_at_step, main as train_main


def _training_fixture(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "corpus"
    train_count = 8
    validation_count = 2
    total = train_count + validation_count
    for key, offset in (("C", 0), ("D", 1)):
        directory = root / key
        directory.mkdir(parents=True)
        records = []
        for index in range(total):
            notes = "CDEF" if (index + offset) % 2 == 0 else "GABc"
            records.append({
                "id": f"example:{index}",
                "abc": f"X:1\nT:{index}\nM:2/4\nL:1/8\nK:{key}\n{notes} |]\n",
            })
        (directory / "tiny.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )

    manifest_dir = tmp_path / "manifest"
    manifest_dir.mkdir()
    rows = []
    for index in range(total):
        rows.append({
            "schema_version": 1,
            "example_id": f"example:{index}",
            "score_id": f"score:{index}",
            "source": "tiny",
            "split": "train" if index < train_count else "validation",
            "sampling_weight": 1.0,
            "patch_count": 5,
            "bar_count": 1,
            "representative_key": "C",
            "variants": {
                "C": ["tiny.jsonl", index + 1],
                "D": ["tiny.jsonl", index + 1],
            },
        })
    manifest = manifest_dir / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    (manifest_dir / "summary.json").write_text(
        json.dumps({"input_root": str(root)}), encoding="utf-8"
    )
    return manifest, root


def _common_args(manifest: Path, root: Path, output: Path) -> list[str]:
    return [
        str(manifest),
        "--data-root", str(root),
        "--output-dir", str(output),
        "--preset", "micro",
        "--device", "cpu",
        "--precision", "fp32",
        "--threads", "1",
        "--workers", "0",
        "--batch-size", "2",
        "--batch-patches", "128",
        "--patch-size", "32",
        "--max-patches", "64",
        "--condition-dim", "0",
        "--schedule-steps", "2",
        "--warmup-steps", "0",
        "--eval-every", "0",
        "--checkpoint-every", "0",
        "--log-every", "1",
        "--max-validation-batches", "1",
        "--no-initial-validation",
        "--no-source-weighting",
    ]


def test_formal_training_resume_matches_uninterrupted(tmp_path: Path) -> None:
    manifest, root = _training_fixture(tmp_path)
    full_dir = tmp_path / "full"
    resumed_dir = tmp_path / "resumed"
    with contextlib.redirect_stdout(io.StringIO()):
        assert train_main([
            *_common_args(manifest, root, full_dir), "--max-steps", "2"
        ]) == 0
        assert train_main([
            *_common_args(manifest, root, resumed_dir), "--max-steps", "1"
        ]) == 0
        assert train_main([
            *_common_args(manifest, root, resumed_dir),
            "--max-steps", "2",
            "--resume", str(resumed_dir / "latest.pt"),
        ]) == 0

    full = load_checkpoint_payload(full_dir / "latest.pt")
    resumed = load_checkpoint_payload(resumed_dir / "latest.pt")
    assert full["step"] == resumed["step"] == 2
    assert full["metadata"]["position"]["epoch"] == resumed["metadata"]["position"]["epoch"]
    assert full["metadata"]["position"]["next_batch"] == resumed["metadata"]["position"]["next_batch"]
    for name, expected in full["model_state"].items():
        assert torch.equal(expected, resumed["model_state"][name]), name


def test_learning_rate_schedule() -> None:
    values = [
        learning_rate_at_step(
            step,
            peak=1.0,
            warmup_steps=2,
            schedule_steps=6,
            minimum_ratio=0.1,
        )
        for step in range(1, 7)
    ]
    assert values[0] == pytest.approx(0.5)
    assert values[1] == pytest.approx(1.0)
    assert values[-1] == pytest.approx(0.1)
    assert values[2:] == sorted(values[2:], reverse=True)
