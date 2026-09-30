import json
from pathlib import Path

from notaear.tonality_benchmark import (
    build_declared_benchmark,
    build_rows,
    evaluate,
    evaluate_declared,
    write_benchmark,
)


def test_benchmark_is_deterministic_and_has_no_label_leakage():
    first = build_rows()
    second = build_rows()
    assert first == second
    assert len(first) == 372
    assert len({row["id"] for row in first}) == len(first)
    assert sum(row["partition"] == "core" for row in first) == 336
    assert sum(row["partition"] == "challenge" for row in first) == 36
    assert all("K:none" in row["abc"] for row in first)


def test_small_benchmark_can_be_written_and_evaluated(tmp_path):
    path = tmp_path / "synthetic.jsonl"
    summary = write_benchmark(path)
    assert summary["rows"] == 372
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    path.write_text("\n".join(json.dumps(row) for row in rows[:2]) + "\n")
    report = evaluate(path)
    assert report["rows"] == 2
    assert not report["failures"]
    assert report["core"]["joint_accuracy"] == 1.0


def test_committed_synthetic_benchmark_matches_generator():
    path = Path(__file__).parents[1] / "data/evaluation/tonality/synthetic-v1.jsonl"
    committed = [json.loads(line) for line in path.read_text().splitlines()]
    assert committed == build_rows()


def test_declared_benchmark_uses_manifest_locators_and_hides_key(tmp_path):
    root = tmp_path / "source"
    shard = root / "C" / "sample.jsonl"
    shard.parent.mkdir(parents=True)
    abc = (
        "X:1\nM:4/4\nL:1/8\nK:C\n"
        "C2 D2 E2 F2 | G2 A2 B2 c2 | c2 B2 A2 G2 | F2 E2 D2 C2 |]\n"
    )
    shard.write_text(json.dumps({"id": "example", "abc": abc}) + "\n")
    manifest_dir = tmp_path / "manifest"
    manifest_dir.mkdir()
    (manifest_dir / "summary.json").write_text(json.dumps({"input_root": str(root)}))
    manifest = manifest_dir / "manifest.jsonl"
    manifest.write_text(json.dumps({
        "example_id": "example",
        "score_id": "score",
        "source": "fixture",
        "work_fingerprint": "work",
        "representative_key": "C",
        "variants": {"C": ["sample.jsonl", 1]},
        "split": "test",
    }) + "\n")
    benchmark = tmp_path / "declared.jsonl"
    summary = build_declared_benchmark(manifest, benchmark)
    assert summary["selected"] == 1
    report = evaluate_declared(benchmark, manifest)
    assert report["joint_accuracy"] == 1.0
    assert report["status_counts"] == {"confident": 1}
