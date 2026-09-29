"""Build a compact, leakage-safe training manifest from v-yidiao shards."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import multiprocessing
import os
import re
import sqlite3
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .abc_patches import AbcPatchError, split_abc
from .tokenizer import NotaEarTokenizer


SCHEMA_VERSION = 1
TARGET_KEYS = ("A", "Ab", "B", "Bb", "C", "C#", "Cb", "D", "Db", "E", "Eb", "F", "F#", "G", "Gb")
SOURCE_PRIORITY = {
    "chinese_folk_omr": 0,
    "guqin": 1,
    "traditional_chinese_midi": 1,
    "seils": 2,
    "essen": 3,
    "irishman_validation": 4,
    "irishman_train": 5,
    "thesession": 6,
    "norbeck": 7,
    "nottingham": 8,
    "mtd": 9,
    "lieder": 10,
    "bach370": 20,
    "jrp": 21,
    "project_1520s": 22,
    "dcml": 30,
    "asap": 31,
    "atepp": 32,
    "ossq": 33,
    "ossq_omr": 34,
    "humdrum": 40,
    "pdmx": 100,
}

_FIELD_RE = re.compile(r"(?m)^\s*([KLM]):([^\r\n]+)")
_COMMENT_RE = re.compile(r"%[^\r\n]*")
_QUOTE_RE = re.compile(r'"(?:\\.|[^"\\])*"')
_DECORATION_RE = re.compile(r"![^!\r\n]*!|\+[^+\r\n]*\+")
_WHITESPACE_RE = re.compile(r"\s+")
_CHORD_RE = re.compile(r"\[(?![A-Za-z]:)([^\]]+)\]")
_NOTE_RE = re.compile(r"(?:\^{1,2}|_{1,2}|=)?[A-Ga-g][,']*")


def _source_name(value: str) -> str:
    return value.removesuffix(".jsonl").removesuffix(".json")


def _has_chord(body: str) -> bool:
    for match in _CHORD_RE.finditer(body):
        if len(_NOTE_RE.findall(match.group(1))) > 1:
            return True
    return False


def _voice_fingerprint(abc: str) -> str:
    document = split_abc(abc)
    fields = {name: value.strip() for name, value in _FIELD_RE.findall(document.header)}
    body = "".join(bar.text for bar in document.bars)
    body = _COMMENT_RE.sub("", body)
    body = _QUOTE_RE.sub("", body)
    body = _DECORATION_RE.sub("", body)
    body = _WHITESPACE_RE.sub("", body)
    material = "\0".join((fields.get("M", ""), fields.get("L", ""), fields.get("K", ""), body))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _quality(abc: str, tokenizer: NotaEarTokenizer, max_patches: int) -> tuple[dict[str, Any], list[str]]:
    reasons: list[str] = []
    try:
        document = split_abc(abc)
        encoded = tokenizer.encode(abc)
    except (AbcPatchError, ValueError, UnicodeError) as exc:
        return {"quality_error": str(exc)}, ["structural_invalid"]
    body = "".join(bar.text for bar in document.bars)
    if not document.bars:
        reasons.append("no_bars")
    if _has_chord(body):
        reasons.append("chordal")
    if encoded.patch_count > max_patches:
        reasons.append("too_many_patches")
    return {
        "patch_count": encoded.patch_count,
        "bar_count": len(document.bars),
        "character_count": len(abc),
        "has_triplet": any(bar.tuplet_markers for bar in document.bars),
        "voice_fingerprint": _voice_fingerprint(abc),
    }, reasons


def _connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.executescript("""
        CREATE TABLE IF NOT EXISTS examples (
            example_id TEXT PRIMARY KEY,
            score_id TEXT NOT NULL,
            source TEXT NOT NULL,
            source_path TEXT,
            voice TEXT,
            variants_json TEXT NOT NULL,
            representative_key TEXT NOT NULL,
            patch_count INTEGER,
            bar_count INTEGER,
            character_count INTEGER,
            has_triplet INTEGER NOT NULL DEFAULT 0,
            voice_fingerprint TEXT,
            eligible INTEGER NOT NULL,
            reasons_json TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS examples_score_id ON examples(score_id);
        CREATE TABLE IF NOT EXISTS processed_shards (
            shard TEXT PRIMARY KEY,
            signature TEXT NOT NULL,
            completed_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS ingest_errors (
            shard TEXT NOT NULL,
            target_key TEXT,
            line_number INTEGER,
            error TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
    """)
    return connection


def _shard_signature(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        stat = path.stat()
        digest.update(str(path).encode())
        digest.update(f"\0{stat.st_size}\0{stat.st_mtime_ns}\0".encode())
    return digest.hexdigest()


def _prefer_key(candidate: str, current: str) -> bool:
    if candidate == "C":
        return current != "C"
    if current == "C":
        return False
    return TARGET_KEYS.index(candidate) < TARGET_KEYS.index(current)


def _merge_example(
    connection: sqlite3.Connection,
    candidate: dict[str, Any],
) -> None:
    existing = connection.execute(
        "SELECT score_id, source, variants_json, representative_key, reasons_json "
        "FROM examples WHERE example_id = ?",
        (candidate["example_id"],),
    ).fetchone()
    if existing is None:
        connection.execute(
            "INSERT INTO examples VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                candidate["example_id"], candidate["score_id"], candidate["source"],
                candidate.get("source_path"), candidate.get("voice"),
                json.dumps(candidate["variants"], sort_keys=True),
                candidate["representative_key"], candidate.get("patch_count"),
                candidate.get("bar_count"), candidate.get("character_count"),
                int(candidate.get("has_triplet", False)), candidate.get("voice_fingerprint"),
                int(not candidate["reasons"]), json.dumps(candidate["reasons"], sort_keys=True),
            ),
        )
        return

    score_id, source, variants_json, representative_key, reasons_json = existing
    variants = json.loads(variants_json)
    reasons = set(json.loads(reasons_json)) | set(candidate["reasons"])
    if score_id != candidate["score_id"] or source != candidate["source"]:
        reasons.add("metadata_inconsistent")
    for key_name, locator in candidate["variants"].items():
        if key_name in variants and variants[key_name] != locator:
            reasons.add("duplicate_variant_locator")
        else:
            variants[key_name] = locator
    update_rep = _prefer_key(candidate["representative_key"], representative_key)
    values = {
        "representative_key": candidate["representative_key"] if update_rep else representative_key,
        "patch_count": candidate.get("patch_count") if update_rep else None,
        "bar_count": candidate.get("bar_count") if update_rep else None,
        "character_count": candidate.get("character_count") if update_rep else None,
        "has_triplet": int(candidate.get("has_triplet", False)) if update_rep else None,
        "voice_fingerprint": candidate.get("voice_fingerprint") if update_rep else None,
    }
    connection.execute(
        "UPDATE examples SET variants_json=?, representative_key=?, "
        "patch_count=COALESCE(?,patch_count), bar_count=COALESCE(?,bar_count), "
        "character_count=COALESCE(?,character_count), has_triplet=COALESCE(?,has_triplet), "
        "voice_fingerprint=COALESCE(?,voice_fingerprint), eligible=?, reasons_json=? "
        "WHERE example_id=?",
        (
            json.dumps(variants, sort_keys=True), values["representative_key"],
            values["patch_count"], values["bar_count"], values["character_count"],
            values["has_triplet"], values["voice_fingerprint"], int(not reasons),
            json.dumps(sorted(reasons)), candidate["example_id"],
        ),
    )


def _read_shard_group(
    paths: list[Path],
    tokenizer: NotaEarTokenizer,
    max_patches: int,
) -> tuple[dict[str, dict[str, Any]], list[tuple[str, str, int, str]]]:
    candidates: dict[str, dict[str, Any]] = {}
    errors: list[tuple[str, str, int, str]] = []
    # Prefer C as the representative so quality/fingerprint work is performed
    # once per example rather than first for A and then again for C.
    for path in sorted(
        paths,
        key=lambda item: (item.parent.name != "C", TARGET_KEYS.index(item.parent.name)),
    ):
        target_key = path.parent.name
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    record = json.loads(line)
                    example_id = str(record["id"])
                    score_id = str(record["score_id"])
                    source = _source_name(str(record["source"]))
                    abc = record["abc"]
                    if not isinstance(abc, str):
                        raise ValueError("abc is not a string")
                except (KeyError, ValueError, json.JSONDecodeError) as exc:
                    errors.append((path.name, target_key, line_number, str(exc)))
                    continue
                current = candidates.get(example_id)
                locator = [path.name, line_number]
                if current is None:
                    metrics, reasons = _quality(abc, tokenizer, max_patches)
                    current = {
                        "example_id": example_id,
                        "score_id": score_id,
                        "source": source,
                        "source_path": record.get("source_path"),
                        "voice": record.get("voice"),
                        "variants": {target_key: locator},
                        "representative_key": target_key,
                        "reasons": reasons,
                        **metrics,
                    }
                    candidates[example_id] = current
                else:
                    if (
                        target_key in current["variants"]
                        and current["variants"][target_key] != locator
                    ):
                        current["reasons"] = sorted(
                            set(current["reasons"]) | {"duplicate_variant_locator"}
                        )
                    else:
                        current["variants"][target_key] = locator
                    if _prefer_key(target_key, current["representative_key"]):
                        metrics, reasons = _quality(abc, tokenizer, max_patches)
                        current.update(metrics)
                        current["reasons"] = sorted(set(current["reasons"]) | set(reasons))
                        current["representative_key"] = target_key
    return candidates, errors


def _read_shard_group_worker(
    task: tuple[str, list[str], int, int],
) -> tuple[str, dict[str, dict[str, Any]], list[tuple[str, str, int, str]]]:
    name, path_values, patch_size, max_patches = task
    tokenizer = NotaEarTokenizer(patch_size)
    candidates, errors = _read_shard_group(
        [Path(value) for value in path_values], tokenizer, max_patches
    )
    return name, candidates, errors


def _discover(root: Path) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = defaultdict(list)
    for target_key in TARGET_KEYS:
        directory = root / target_key
        if not directory.is_dir():
            continue
        for path in directory.glob("*.jsonl"):
            if path.stat().st_size:
                groups[path.name].append(path)
    return dict(groups)


def _stable_split(fingerprint: str, salt: str, validation_rate: float, test_rate: float) -> str:
    digest = hashlib.blake2b(
        f"{salt}\0{fingerprint}".encode(), digest_size=8
    ).digest()
    value = int.from_bytes(digest, "big") / 2**64
    if value < test_rate:
        return "test"
    if value < test_rate + validation_rate:
        return "validation"
    return "train"


def _build_works(connection: sqlite3.Connection) -> None:
    connection.executescript("""
        DROP TABLE IF EXISTS works;
        DROP TABLE IF EXISTS canonical_works;
        CREATE TABLE works (
            score_id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            work_fingerprint TEXT NOT NULL,
            eligible_voice_count INTEGER NOT NULL,
            total_voice_count INTEGER NOT NULL,
            source_priority INTEGER NOT NULL
        );
    """)
    cursor = connection.execute(
        "SELECT score_id, source, voice_fingerprint, eligible FROM examples "
        "ORDER BY score_id, example_id"
    )
    active_score: str | None = None
    active_source = ""
    fingerprints: list[str] = []
    eligible_count = total_count = 0

    def flush() -> None:
        if active_score is None:
            return
        material = "\0".join(sorted(fingerprints)) or f"missing:{active_score}"
        work_fingerprint = hashlib.sha256(material.encode()).hexdigest()
        connection.execute(
            "INSERT INTO works VALUES (?, ?, ?, ?, ?, ?)",
            (active_score, active_source, work_fingerprint, eligible_count, total_count,
             SOURCE_PRIORITY.get(active_source, 50)),
        )

    for score_id, source, voice_fingerprint, eligible in cursor:
        if score_id != active_score:
            flush()
            active_score = score_id
            active_source = source
            fingerprints = []
            eligible_count = total_count = 0
        total_count += 1
        eligible_count += int(eligible)
        fingerprints.append(voice_fingerprint or f"invalid:{score_id}:{total_count}")
    flush()
    connection.executescript("""
        CREATE TABLE canonical_works AS
        SELECT work_fingerprint, score_id FROM (
            SELECT work_fingerprint, score_id,
                   ROW_NUMBER() OVER (
                       PARTITION BY work_fingerprint
                       ORDER BY (eligible_voice_count * 1.0 / total_voice_count) DESC,
                                eligible_voice_count DESC, source_priority, score_id
                   ) AS rank
            FROM works
        ) WHERE rank = 1;
        CREATE UNIQUE INDEX canonical_fingerprint ON canonical_works(work_fingerprint);
    """)
    connection.commit()


def _source_counts(connection: sqlite3.Connection) -> dict[str, int]:
    rows = connection.execute("""
        SELECT e.source, COUNT(*)
        FROM examples e
        JOIN works w ON w.score_id = e.score_id
        JOIN canonical_works c ON c.work_fingerprint = w.work_fingerprint
                              AND c.score_id = w.score_id
        WHERE e.eligible = 1
        GROUP BY e.source
    """)
    return {source: int(count) for source, count in rows}


def _source_weights(counts: dict[str, int], alpha: float, max_weight: float) -> dict[str, float]:
    if max_weight < 1.0:
        raise ValueError("max source weight must be at least one")
    base = {
        source: count ** (alpha - 1.0)
        for source, count in counts.items()
        if count
    }
    if not base:
        return {}

    # Find a scale whose count-weighted mean is one while capping rare-source
    # per-example weights. This makes count**alpha the uncapped source mass.
    total = sum(counts.values())

    def weighted_sum(scale: float) -> float:
        return sum(
            counts[source] * min(scale * value, max_weight)
            for source, value in base.items()
        )

    low, high = 0.0, 1.0
    while weighted_sum(high) < total:
        high *= 2.0
    for _ in range(80):
        middle = (low + high) / 2.0
        if weighted_sum(middle) < total:
            low = middle
        else:
            high = middle
    return {
        source: min(high * value, max_weight)
        for source, value in base.items()
    }


def _check_index_config(
    connection: sqlite3.Connection,
    root: Path,
    patch_size: int,
    max_patches: int,
) -> None:
    expected = {
        "schema_version": str(SCHEMA_VERSION),
        "input_root": str(root.resolve()),
        "patch_size": str(patch_size),
        "max_patches": str(max_patches),
    }
    current = dict(connection.execute("SELECT key, value FROM metadata"))
    if current and current != expected:
        differences = {
            key: {"indexed": current.get(key), "requested": value}
            for key, value in expected.items()
            if current.get(key) != value
        }
        raise RuntimeError(
            "manifest index configuration changed; use a new output directory: "
            + json.dumps(differences, sort_keys=True)
        )
    if not current:
        connection.executemany(
            "INSERT INTO metadata(key, value) VALUES (?, ?)", expected.items()
        )
        connection.commit()


def _write_manifests(
    connection: sqlite3.Connection,
    output_dir: Path,
    root: Path,
    *,
    alpha: float,
    max_source_weight: float,
    split_salt: str,
    validation_rate: float,
    test_rate: float,
) -> dict[str, Any]:
    counts = _source_counts(connection)
    weights = _source_weights(counts, alpha, max_source_weight)
    split_counts = Counter()
    source_split_counts: dict[str, Counter] = defaultdict(Counter)
    rejection_counts = Counter()
    manifest_tmp = output_dir / "manifest.jsonl.tmp"
    rejected_tmp = output_dir / "rejected.jsonl.tmp"
    query = connection.execute("""
        SELECT e.example_id, e.score_id, e.source, e.source_path, e.voice,
               e.variants_json, e.representative_key, e.patch_count, e.bar_count,
               e.character_count, e.has_triplet, e.eligible, e.reasons_json,
               w.work_fingerprint, c.score_id
        FROM examples e
        JOIN works w ON w.score_id = e.score_id
        JOIN canonical_works c ON c.work_fingerprint = w.work_fingerprint
        ORDER BY e.source, e.score_id, e.example_id
    """)
    with manifest_tmp.open("w", encoding="utf-8") as accepted_handle, rejected_tmp.open(
        "w", encoding="utf-8"
    ) as rejected_handle:
        for row in query:
            (
                example_id, score_id, source, source_path, voice, variants_json,
                representative_key, patch_count, bar_count, character_count,
                has_triplet, eligible, reasons_json, work_fingerprint,
                canonical_score_id,
            ) = row
            reasons = json.loads(reasons_json)
            if score_id != canonical_score_id:
                reasons = [*reasons, "duplicate_work"]
            record = {
                "schema_version": SCHEMA_VERSION,
                "example_id": example_id,
                "score_id": score_id,
                "source": source,
                "source_path": source_path,
                "voice": voice,
                "work_fingerprint": work_fingerprint,
                "representative_key": representative_key,
                "variants": json.loads(variants_json),
                "variant_count": len(json.loads(variants_json)),
                "patch_count": patch_count,
                "bar_count": bar_count,
                "character_count": character_count,
                "has_triplet": bool(has_triplet),
            }
            if eligible and score_id == canonical_score_id:
                split = _stable_split(
                    work_fingerprint, split_salt, validation_rate, test_rate
                )
                record["split"] = split
                record["sampling_weight"] = round(weights[source], 8)
                accepted_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                split_counts[split] += 1
                source_split_counts[source][split] += 1
            else:
                record["reasons"] = sorted(set(reasons))
                record["canonical_score_id"] = canonical_score_id
                rejected_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                rejection_counts.update(record["reasons"])
    os.replace(manifest_tmp, output_dir / "manifest.jsonl")
    os.replace(rejected_tmp, output_dir / "rejected.jsonl")
    return {
        "schema_version": SCHEMA_VERSION,
        "input_root": str(root.resolve()),
        "source_sampling_alpha": alpha,
        "max_source_weight": max_source_weight,
        "split_salt": split_salt,
        "split_rates": {
            "train": 1.0 - validation_rate - test_rate,
            "validation": validation_rate,
            "test": test_rate,
        },
        "accepted_examples": sum(split_counts.values()),
        "split_counts": dict(split_counts),
        "source_counts": counts,
        "source_weights": weights,
        "source_split_counts": {
            source: dict(values) for source, values in sorted(source_split_counts.items())
        },
        "rejection_counts": dict(rejection_counts),
    }


def build_manifest(
    root: Path,
    output_dir: Path,
    *,
    patch_size: int = 32,
    max_patches: int = 512,
    alpha: float = 0.5,
    max_source_weight: float = 10.0,
    split_salt: str = "notaear-v1",
    validation_rate: float = 0.01,
    test_rate: float = 0.01,
    max_shards: int | None = None,
    log_every: int = 100,
    workers: int = 1,
) -> dict[str, Any]:
    if validation_rate < 0 or test_rate < 0 or validation_rate + test_rate >= 1:
        raise ValueError("validation and test rates must be non-negative and sum below one")
    if not 0 <= alpha <= 1:
        raise ValueError("source sampling alpha must be between zero and one")
    if workers < 1:
        raise ValueError("workers must be at least one")
    output_dir.mkdir(parents=True, exist_ok=True)
    groups = _discover(root)
    names = sorted(groups)
    if max_shards is not None:
        names = names[:max_shards]
    connection = _connect(output_dir / "index.sqlite")
    _check_index_config(connection, root, patch_size, max_patches)
    start = time.monotonic()
    processed = skipped = 0
    signatures: dict[str, str] = {}
    tasks: list[tuple[str, list[str], int, int]] = []
    for name in names:
        paths = groups[name]
        signature = _shard_signature(paths)
        signatures[name] = signature
        existing = connection.execute(
            "SELECT signature FROM processed_shards WHERE shard=?", (name,)
        ).fetchone()
        if existing is not None and existing[0] == signature:
            skipped += 1
            continue
        if existing is not None:
            raise RuntimeError(
                f"indexed shard changed: {name}; use a new output directory to avoid stale locators"
            )
        tasks.append((name, [str(path) for path in paths], patch_size, max_patches))

    if workers == 1:
        results: Iterable[
            tuple[str, dict[str, dict[str, Any]], list[tuple[str, str, int, str]]]
        ] = map(_read_shard_group_worker, tasks)
        executor = None
    else:
        executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
        )
        results = executor.map(_read_shard_group_worker, tasks, chunksize=1)

    try:
        for name, candidates, errors in results:
            signature = signatures[name]
            with connection:
                for candidate in candidates.values():
                    _merge_example(connection, candidate)
                connection.executemany(
                    "INSERT INTO ingest_errors VALUES (?, ?, ?, ?)", errors
                )
                connection.execute(
                    "INSERT INTO processed_shards VALUES (?, ?, ?)",
                    (name, signature, time.time()),
                )
            processed += 1
            completed = processed + skipped
            if completed == 1 or completed % log_every == 0 or completed == len(names):
                print(json.dumps({
                    "shard": completed,
                    "shards_total": len(names),
                    "examples_indexed": connection.execute(
                        "SELECT COUNT(*) FROM examples"
                    ).fetchone()[0],
                    "elapsed_seconds": round(time.monotonic() - start, 1),
                }), flush=True)
    finally:
        if executor is not None:
            executor.shutdown(cancel_futures=True)

    _build_works(connection)
    summary = _write_manifests(
        connection, output_dir, root, alpha=alpha,
        max_source_weight=max_source_weight, split_salt=split_salt,
        validation_rate=validation_rate, test_rate=test_rate,
    )
    summary.update({
        "patch_size": patch_size,
        "max_patches": max_patches,
        "workers": workers,
        "discovered_shards": len(groups),
        "selected_shards": len(names),
        "processed_shards": processed,
        "resumed_shards": skipped,
        "indexed_examples": connection.execute("SELECT COUNT(*) FROM examples").fetchone()[0],
        "indexed_works": connection.execute("SELECT COUNT(*) FROM works").fetchone()[0],
        "unique_work_fingerprints": connection.execute(
            "SELECT COUNT(*) FROM canonical_works"
        ).fetchone()[0],
        "ingest_errors": connection.execute("SELECT COUNT(*) FROM ingest_errors").fetchone()[0],
        "elapsed_seconds": round(time.monotonic() - start, 3),
    })
    connection.close()
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--max-patches", type=int, default=512)
    parser.add_argument("--source-alpha", type=float, default=0.5)
    parser.add_argument("--max-source-weight", type=float, default=10.0)
    parser.add_argument("--split-salt", default="notaear-v1")
    parser.add_argument("--validation-rate", type=float, default=0.01)
    parser.add_argument("--test-rate", type=float, default=0.01)
    parser.add_argument("--max-shards", type=int)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    summary = build_manifest(
        args.input_root, args.output_dir,
        patch_size=args.patch_size, max_patches=args.max_patches,
        alpha=args.source_alpha, max_source_weight=args.max_source_weight,
        split_salt=args.split_salt, validation_rate=args.validation_rate,
        test_rate=args.test_rate, max_shards=args.max_shards,
        log_every=args.log_every, workers=args.workers,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
