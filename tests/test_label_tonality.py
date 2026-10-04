import json

import pytest

from notaear.label_tonality import (
    _easy_evidence,
    _html,
    _parse_byte_range,
    _public_task,
    _sanitized_musicxml,
    _validate_annotation,
    audit,
    evaluate_human,
    prepare_queue,
)


def _row(identifier="source:item:V:1"):
    return {
        "id": identifier,
        "score_id": "source:item",
        "source": "private-source",
        "work_fingerprint": "fingerprint",
        "target_key": "C",
        "shard": "private.jsonl",
        "line_number": 7,
        "expected_tonic_pc": 2,
        "expected_mode": "dorian",
        "label_kind": "abc_declared_key_weak_label",
    }


def test_prepare_queue_is_deterministic_and_public_task_is_blind(tmp_path):
    benchmark = tmp_path / "benchmark.jsonl"
    rows = [_row(f"source:item-{index}:V:1") for index in range(10)]
    benchmark.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    first = prepare_queue(benchmark, tmp_path / "first.json", repeat_fraction=.2, seed=4)
    second = prepare_queue(benchmark, tmp_path / "second.json", repeat_fraction=.2, seed=4)
    assert first["tasks"] == second["tasks"]
    assert first["item_count"] == 10
    assert first["repeat_count"] == 2
    assert len(first["tasks"]) == 12
    public = _public_task(first["tasks"][0], 0, 12)
    assert set(public) == {"task_id", "index", "total"}
    rendered = json.dumps(public)
    assert "private-source" not in rendered
    assert "dorian" not in rendered
    assert "expected" not in rendered


def test_annotation_validation_requires_complete_tonal_label():
    task = {"task_id": "task"}
    valid = {
        "task_id": "task", "judgement": "tonal", "tonic_pc": 2,
        "mode": "dorian", "modulation": "none", "confidence": 3,
        "listened": False, "notes": "clear final",
    }
    assert _validate_annotation(valid, task)["mode"] == "dorian"
    with pytest.raises(ValueError, match="tonic_pc"):
        _validate_annotation({**valid, "tonic_pc": None}, task)
    ambiguous = _validate_annotation(
        {**valid, "judgement": "ambiguous", "tonic_pc": 4, "mode": "major"}, task
    )
    assert ambiguous["tonic_pc"] is None
    assert ambiguous["mode"] is None
    partial = _validate_annotation(
        {**valid, "mode": "minor_unspecified", "uncertainty_reason": "mode_unclear"},
        task,
    )
    assert partial["mode"] == "minor_unspecified"
    assert partial["uncertainty_reason"] == "mode_unclear"


def test_audio_timeline_and_byte_range_support():
    page = _html()
    assert 'id="timeline"' in page
    assert "audio.currentTime=Number(timeline.value)" in page
    assert _parse_byte_range(None, 1000) is None
    assert _parse_byte_range("bytes=100-199", 1000) == (100, 199)
    assert _parse_byte_range("bytes=900-", 1000) == (900, 999)
    assert _parse_byte_range("bytes=-100", 1000) == (900, 999)
    with pytest.raises(ValueError):
        _parse_byte_range("bytes=1000-1001", 1000)
    with pytest.raises(ValueError):
        _parse_byte_range("bytes=1-2,4-5", 1000)


def test_sanitized_musicxml_keeps_notes_but_removes_answers():
    abc = (
        'X:1\nT:Secret Dorian Tune\nM:4/4\nL:1/8\nK:Ddor\n'
        '"Dm"D2 E2 F2 G2 | A2 B2 c2 d2 | d2 c2 B2 A2 | G2 F2 E2 D2 |]\n'
    )
    xml = _sanitized_musicxml(abc)
    lowered = xml.lower()
    assert xml.count(b"<pitch>") == 16
    assert b"<mode>" not in lowered
    assert b"<harmony" not in lowered
    assert b"secret dorian tune" not in lowered


def test_easy_evidence_rejects_parseable_but_rhythmically_broken_score():
    row = {"expected_tonic_pc": 0, "expected_mode": "major"}
    full = "C D E F G A B c"
    valid = "X:1\nM:4/4\nL:1/8\nK:C\n" + " | ".join([full] * 7 + ["c B A G F E D C"]) + " |]\n"
    assert _easy_evidence(row, valid)["strict_rhythm_valid"] is True
    broken = valid.replace(full, "C D E F G A B", 1)
    assert _easy_evidence(row, broken) is None


def test_audit_reports_hidden_repeat_consistency(tmp_path):
    benchmark = tmp_path / "benchmark.jsonl"
    benchmark.write_text(json.dumps(_row()) + "\n")
    queue_path = tmp_path / "queue.json"
    queue = prepare_queue(benchmark, queue_path, repeat_fraction=1.0, seed=2)
    annotations = tmp_path / "labels.jsonl"
    records = []
    for task in queue["tasks"]:
        records.append({
            "task_id": task["task_id"], "item_key": task["item_key"],
            "human": {"judgement": "tonal", "tonic_pc": 2, "mode": "dorian",
                      "modulation": "none", "listened": False},
            "weak_reference": {"tonic_pc": 2, "mode": "dorian"},
        })
    annotations.write_text("\n".join(json.dumps(row) for row in records) + "\n")
    report = audit(queue_path, annotations)
    assert report["task_completion"] == {"completed": 2, "total": 2}
    assert report["repeat_consistency"]["combined_label_agreement"] == 1.0
    assert report["human_vs_declared_weak_reference"]["joint_agreement"] == 1.0


def test_human_evaluation_hides_declared_key_from_inference(tmp_path):
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
    manifest.write_text("")
    queue_path = tmp_path / "queue.json"
    queue_path.write_text(json.dumps({
        "tasks": [{
            "task_id": "task", "item_key": "item", "occurrence": 0,
            "locator": {"id": "example", "target_key": "C",
                        "shard": "sample.jsonl", "line_number": 1},
        }],
    }))
    annotations = tmp_path / "annotations.jsonl"
    annotations.write_text(json.dumps({
        "task_id": "task", "item_key": "item", "occurrence": 0,
        "identity": {"id": "example", "target_key": "C",
                     "shard": "sample.jsonl", "line_number": 1},
        "human": {"judgement": "tonal", "tonic_pc": 0, "mode": "major"},
    }) + "\n")
    report = evaluate_human(queue_path, annotations, manifest)
    assert report["tonic_accuracy"] == 1.0
    assert report["joint_compatible_accuracy"] == 1.0
