"""Manifest-backed ABC dataset and shard-local PyTorch batch loader."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from array import array
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Sampler

from .manifest import choose_variant
from .tokenizer import PAD_ID, NotaEarTokenizer


TARGET_KEYS = frozenset(
    ("A", "Ab", "B", "Bb", "C", "C#", "Cb", "D", "Db", "E", "Eb", "F", "F#", "G", "Gb")
)


class ManifestDataError(RuntimeError):
    """Raised when a manifest or one of its source locators is inconsistent."""


@dataclass(frozen=True, slots=True)
class SampleRef:
    """Compact instruction passed from the batch sampler to a dataset worker."""

    index: int
    target_key: str
    shard: str
    line_number: int
    estimated_patches: int


class ManifestABCDataset(Dataset[dict[str, Any]]):
    """Random-access view over one split of a NotaEar JSONL manifest.

    Only byte offsets, sampling weights and patch counts are retained in memory.
    Source shards are loaded into a small process-local LRU cache. This keeps the
    600+ MiB manifest and 13 GiB augmented corpus out of Python object memory.
    """

    def __init__(
        self,
        manifest_path: Path | str,
        *,
        data_root: Path | str | None = None,
        split: str = "train",
        patch_size: int = 32,
        max_patches: int = 512,
        shard_cache_size: int = 2,
    ) -> None:
        if split not in {"train", "validation", "test"}:
            raise ValueError("split must be train, validation or test")
        if max_patches < 2:
            raise ValueError("max_patches must be at least two")
        if shard_cache_size < 1:
            raise ValueError("shard_cache_size must be positive")
        self.manifest_path = Path(manifest_path).resolve()
        if not self.manifest_path.is_file():
            raise FileNotFoundError(self.manifest_path)
        self.data_root = self._resolve_data_root(data_root)
        self.split = split
        self.tokenizer = NotaEarTokenizer(patch_size)
        self.max_patches = max_patches
        self.shard_cache_size = shard_cache_size
        self._offsets = array("Q")
        self._weights = array("f")
        self._patch_counts = array("H")
        self._manifest_handle = None
        self._manifest_pid: int | None = None
        self._shard_cache: OrderedDict[tuple[str, str], tuple[bytes, ...]] = OrderedDict()
        self._epoch = 0
        self.index_stats = {"split_records": 0, "too_long": 0, "indexed": 0}
        self._build_index()

    def _resolve_data_root(self, value: Path | str | None) -> Path:
        if value is not None:
            root = Path(value).resolve()
        else:
            summary_path = self.manifest_path.parent / "summary.json"
            if not summary_path.is_file():
                raise ManifestDataError(
                    "data_root was not supplied and summary.json is missing"
                )
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                root = Path(summary["input_root"]).resolve()
            except (KeyError, ValueError, json.JSONDecodeError) as exc:
                raise ManifestDataError(
                    f"cannot read input_root from {summary_path}: {exc}"
                ) from exc
        if not root.is_dir():
            raise FileNotFoundError(root)
        return root

    def _build_index(self) -> None:
        with self.manifest_path.open("rb") as handle:
            line_number = 0
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                line_number += 1
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ManifestDataError(
                        f"invalid manifest JSON at line {line_number}: {exc}"
                    ) from exc
                if record.get("split") != self.split:
                    continue
                self.index_stats["split_records"] += 1
                try:
                    weight = float(record["sampling_weight"])
                    patch_count = int(record["patch_count"])
                    variants = record["variants"]
                except (KeyError, TypeError, ValueError) as exc:
                    raise ManifestDataError(
                        f"invalid manifest fields at line {line_number}: {exc}"
                    ) from exc
                if not math.isfinite(weight) or weight <= 0:
                    raise ManifestDataError(
                        f"invalid sampling weight at manifest line {line_number}: {weight}"
                    )
                if patch_count < 1:
                    raise ManifestDataError(
                        f"invalid patch count at manifest line {line_number}: {patch_count}"
                    )
                if patch_count > self.max_patches:
                    self.index_stats["too_long"] += 1
                    continue
                if not isinstance(variants, dict) or not variants:
                    raise ManifestDataError(
                        f"missing variants at manifest line {line_number}"
                    )
                self._offsets.append(offset)
                self._weights.append(weight)
                self._patch_counts.append(patch_count)
                self.index_stats["indexed"] += 1
        if not self._offsets:
            raise ManifestDataError(
                f"manifest contains no records for split {self.split!r}"
            )

    def __len__(self) -> int:
        return len(self._offsets)

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch cannot be negative")
        self._epoch = epoch

    def sampling_weights(self) -> Tensor:
        """Return an independent float64 tensor for weighted sampling."""
        return torch.tensor(self._weights, dtype=torch.float64)

    def estimated_patch_count(self, index: int) -> int:
        return int(self._patch_counts[index])

    def _manifest_file(self):
        pid = os.getpid()
        if self._manifest_handle is None or self._manifest_pid != pid:
            if self._manifest_handle is not None:
                self._manifest_handle.close()
            self._manifest_handle = self.manifest_path.open("rb")
            self._manifest_pid = pid
            self._shard_cache.clear()
        return self._manifest_handle

    def manifest_record(self, index: int) -> dict[str, Any]:
        if not 0 <= index < len(self):
            raise IndexError(index)
        handle = self._manifest_file()
        handle.seek(self._offsets[index])
        line = handle.readline()
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ManifestDataError(
                f"indexed manifest record {index} is no longer valid JSON"
            ) from exc
        if record.get("split") != self.split:
            raise ManifestDataError(
                f"manifest changed after indexing at record {index}"
            )
        return record

    @staticmethod
    def _validate_locator(target_key: str, shard: str, line_number: int) -> None:
        if target_key not in TARGET_KEYS:
            raise ManifestDataError(f"unsupported target key: {target_key!r}")
        if Path(shard).name != shard:
            raise ManifestDataError(f"unsafe shard name: {shard!r}")
        if line_number < 1:
            raise ManifestDataError(f"invalid source line number: {line_number}")

    def _source_lines(self, target_key: str, shard: str) -> tuple[bytes, ...]:
        self._validate_locator(target_key, shard, 1)
        cache_key = (target_key, shard)
        cached = self._shard_cache.pop(cache_key, None)
        if cached is not None:
            self._shard_cache[cache_key] = cached
            return cached
        path = self.data_root / target_key / shard
        try:
            with path.open("rb") as handle:
                lines = tuple(handle)
        except OSError as exc:
            raise ManifestDataError(f"cannot read source shard {path}: {exc}") from exc
        self._shard_cache[cache_key] = lines
        while len(self._shard_cache) > self.shard_cache_size:
            self._shard_cache.popitem(last=False)
        return lines

    def _default_ref(self, index: int) -> SampleRef:
        record = self.manifest_record(index)
        target_key, shard, line_number = choose_variant(record, self._epoch)
        return SampleRef(
            index=index,
            target_key=target_key,
            shard=shard,
            line_number=line_number,
            estimated_patches=self.estimated_patch_count(index),
        )

    def __getitem__(self, value: int | SampleRef) -> dict[str, Any]:
        reference = self._default_ref(value) if isinstance(value, int) else value
        self._validate_locator(
            reference.target_key, reference.shard, reference.line_number
        )
        manifest = self.manifest_record(reference.index)
        locator = manifest.get("variants", {}).get(reference.target_key)
        if locator != [reference.shard, reference.line_number]:
            raise ManifestDataError(
                f"sample reference does not match manifest for {manifest.get('example_id')}"
            )
        available = sorted(manifest["variants"])
        selected_position = available.index(reference.target_key)
        candidate_keys = (
            available[selected_position:] + available[:selected_position]
        )
        encoded = None
        actual_key = reference.target_key
        attempted: list[tuple[str, int]] = []
        for target_key in candidate_keys:
            shard, line_number = manifest["variants"][target_key]
            self._validate_locator(target_key, shard, line_number)
            lines = self._source_lines(target_key, shard)
            if line_number > len(lines):
                raise ManifestDataError(
                    f"source locator is past EOF: {target_key}/{shard}:{line_number}"
                )
            try:
                source = json.loads(lines[line_number - 1])
                abc = source["abc"]
            except (KeyError, TypeError, json.JSONDecodeError) as exc:
                raise ManifestDataError(
                    f"invalid source record at {target_key}/{shard}:{line_number}: {exc}"
                ) from exc
            if source.get("id") != manifest.get("example_id"):
                raise ManifestDataError(
                    "source locator id mismatch: "
                    f"expected {manifest.get('example_id')!r}, got {source.get('id')!r}"
                )
            candidate = self.tokenizer.encode(abc)
            attempted.append((target_key, candidate.patch_count))
            if candidate.patch_count <= self.max_patches:
                encoded = candidate
                actual_key = target_key
                break
        if encoded is None:
            raise ManifestDataError(
                f"no variant fits the {self.max_patches}-patch limit for "
                f"{manifest['example_id']}: {attempted}"
            )
        return {
            "token_ids": encoded.token_ids,
            "example_id": manifest["example_id"],
            "score_id": manifest["score_id"],
            "source": manifest["source"],
            "target_key": actual_key,
            "requested_target_key": reference.target_key,
            "variant_fallback": actual_key != reference.target_key,
            "sampling_weight": float(manifest["sampling_weight"]),
        }

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_manifest_handle"] = None
        state["_manifest_pid"] = None
        state["_shard_cache"] = OrderedDict()
        return state

    def close(self) -> None:
        if self._manifest_handle is not None:
            self._manifest_handle.close()
            self._manifest_handle = None
            self._manifest_pid = None
        self._shard_cache.clear()

    def __del__(self) -> None:
        if hasattr(self, "_manifest_handle"):
            self.close()


def _shared_target_key(
    score_id: str,
    records: Sequence[dict[str, Any]],
    epoch: int,
    seed: int,
) -> str:
    common = set(records[0]["variants"])
    for record in records[1:]:
        common.intersection_update(record["variants"])
    if not common:
        raise ManifestDataError(
            f"voices in score {score_id!r} have no common target key"
        )
    keys = sorted(common)
    digest = hashlib.blake2b(
        f"{seed}\0{epoch}\0{score_id}".encode(), digest_size=8
    ).digest()
    return keys[int.from_bytes(digest, "big") % len(keys)]


class ManifestBatchSampler(Sampler[list[SampleRef]]):
    """Build deterministic, source-weighted, shard-local dynamic batches."""

    def __init__(
        self,
        dataset: ManifestABCDataset,
        *,
        max_examples_per_batch: int = 8,
        max_patches_per_batch: int = 2048,
        weighted: bool = True,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 7,
    ) -> None:
        if max_examples_per_batch < 1 or max_patches_per_batch < 1:
            raise ValueError("batch limits must be positive")
        self.dataset = dataset
        self.max_examples_per_batch = max_examples_per_batch
        self.max_patches_per_batch = max_patches_per_batch
        self.weighted = weighted
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0
        self.start_batch = 0
        self._cached_epoch: int | None = None
        self._cached_batches: list[list[SampleRef]] | None = None

    def set_epoch(self, epoch: int, *, start_batch: int = 0) -> None:
        if epoch < 0:
            raise ValueError("epoch cannot be negative")
        if start_batch < 0:
            raise ValueError("start_batch cannot be negative")
        self.epoch = epoch
        self.start_batch = start_batch
        self.dataset.set_epoch(epoch)
        self._cached_epoch = None
        self._cached_batches = None

    def _multiplicities(self) -> list[int]:
        if not self.weighted:
            return [1] * len(self.dataset)
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        selected = torch.multinomial(
            self.dataset.sampling_weights(),
            num_samples=len(self.dataset),
            replacement=True,
            generator=generator,
        )
        return torch.bincount(selected, minlength=len(self.dataset)).tolist()

    def _build_batches(self) -> list[list[SampleRef]]:
        multiplicities = self._multiplicities()
        groups: dict[tuple[str, str], list[SampleRef]] = defaultdict(list)
        active_score: str | None = None
        score_entries: list[tuple[int, int, dict[str, Any]]] = []

        def flush_score() -> None:
            if not score_entries or active_score is None:
                return
            records = [entry[2] for entry in score_entries]
            target_key = _shared_target_key(
                active_score, records, self.epoch, self.seed
            )
            for index, count, record in score_entries:
                shard, line_number = record["variants"][target_key]
                reference = SampleRef(
                    index=index,
                    target_key=target_key,
                    shard=shard,
                    line_number=line_number,
                    estimated_patches=(
                        min(
                            self.dataset.max_patches,
                            self.dataset.estimated_patch_count(index) + 64,
                        )
                        if self.dataset.estimated_patch_count(index) >= 400
                        else self.dataset.estimated_patch_count(index)
                    ),
                )
                groups[(target_key, shard)].extend([reference] * count)

        for index, count in enumerate(multiplicities):
            if count == 0:
                continue
            record = self.dataset.manifest_record(index)
            score_id = str(record["score_id"])
            if active_score is not None and score_id != active_score:
                flush_score()
                score_entries = []
            active_score = score_id
            score_entries.append((index, count, record))
        flush_score()

        rng = random.Random(self.seed + self.epoch)
        group_values = list(groups.values())
        if self.shuffle:
            rng.shuffle(group_values)
        batches: list[list[SampleRef]] = []
        remainders: list[list[SampleRef]] = []
        for references in group_values:
            if self.shuffle:
                rng.shuffle(references)
            batch: list[SampleRef] = []
            largest = 0
            for reference in references:
                next_largest = max(largest, reference.estimated_patches)
                exceeds_examples = len(batch) >= self.max_examples_per_batch
                exceeds_patches = (
                    next_largest * (len(batch) + 1) > self.max_patches_per_batch
                )
                if batch and (exceeds_examples or exceeds_patches):
                    # This is a complete dynamic batch even when the patch
                    # budget, rather than the example cap, closed it.
                    batches.append(batch)
                    batch = []
                    largest = 0
                batch.append(reference)
                largest = max(largest, reference.estimated_patches)
            if batch:
                remainders.append(batch)

        # Strict shard grouping creates thousands of one-item batches in the
        # smaller validation/test splits. Pack shard-local remainder chunks
        # together while preserving their internal order and both limits.
        carry: list[SampleRef] = []
        carry_largest = 0
        for chunk in remainders:
            for reference in chunk:
                next_largest = max(carry_largest, reference.estimated_patches)
                exceeds_examples = len(carry) >= self.max_examples_per_batch
                exceeds_patches = (
                    next_largest * (len(carry) + 1) > self.max_patches_per_batch
                )
                if carry and (exceeds_examples or exceeds_patches):
                    batches.append(carry)
                    carry = []
                    carry_largest = 0
                carry.append(reference)
                carry_largest = max(carry_largest, reference.estimated_patches)
        if carry and not self.drop_last:
            batches.append(carry)
        if self.shuffle:
            rng.shuffle(batches)
        return batches

    def _batches(self) -> list[list[SampleRef]]:
        if self._cached_batches is None or self._cached_epoch != self.epoch:
            self._cached_batches = self._build_batches()
            self._cached_epoch = self.epoch
        return self._cached_batches

    def __iter__(self) -> Iterator[list[SampleRef]]:
        batches = self._batches()
        if self.start_batch > len(batches):
            raise ManifestDataError(
                f"start batch {self.start_batch} exceeds epoch length {len(batches)}"
            )
        return iter(batches[self.start_batch:])

    def __len__(self) -> int:
        return max(0, len(self._batches()) - self.start_batch)

    @property
    def total_batches(self) -> int:
        return len(self._batches())


def collate_manifest_batch(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        raise ValueError("cannot collate an empty batch")
    patch_size = len(items[0]["token_ids"][0])
    patch_counts = torch.tensor(
        [len(item["token_ids"]) for item in items], dtype=torch.long
    )
    maximum = int(patch_counts.max().item())
    patches = torch.full(
        (len(items), maximum, patch_size), PAD_ID, dtype=torch.long
    )
    for index, item in enumerate(items):
        value = torch.tensor(item["token_ids"], dtype=torch.long)
        if value.ndim != 2 or value.shape[1] != patch_size:
            raise ManifestDataError("batch contains inconsistent patch shapes")
        patches[index, : value.shape[0]] = value
    return {
        "patches": patches,
        "patch_counts": patch_counts,
        "example_ids": [item["example_id"] for item in items],
        "score_ids": [item["score_id"] for item in items],
        "sources": [item["source"] for item in items],
        "target_keys": [item["target_key"] for item in items],
        "requested_target_keys": [item["requested_target_key"] for item in items],
        "variant_fallbacks": torch.tensor(
            [item["variant_fallback"] for item in items], dtype=torch.bool
        ),
    }


def create_manifest_dataloader(
    manifest_path: Path | str,
    *,
    data_root: Path | str | None = None,
    split: str = "train",
    patch_size: int = 32,
    max_patches: int = 512,
    max_examples_per_batch: int = 8,
    max_patches_per_batch: int = 2048,
    weighted: bool | None = None,
    shuffle: bool | None = None,
    seed: int = 7,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    shard_cache_size: int = 2,
) -> tuple[ManifestABCDataset, ManifestBatchSampler, DataLoader]:
    """Construct the dataset, its epoch-aware batch sampler and DataLoader."""
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")
    if persistent_workers and num_workers == 0:
        raise ValueError("persistent_workers requires num_workers > 0")
    is_train = split == "train"
    dataset = ManifestABCDataset(
        manifest_path,
        data_root=data_root,
        split=split,
        patch_size=patch_size,
        max_patches=max_patches,
        shard_cache_size=shard_cache_size,
    )
    batch_sampler = ManifestBatchSampler(
        dataset,
        max_examples_per_batch=max_examples_per_batch,
        max_patches_per_batch=max_patches_per_batch,
        weighted=is_train if weighted is None else weighted,
        shuffle=is_train if shuffle is None else shuffle,
        seed=seed,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=collate_manifest_batch,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        generator=torch.Generator().manual_seed(seed),
        multiprocessing_context="spawn" if num_workers else None,
    )
    return dataset, batch_sampler, loader


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="train")
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--max-patches", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--batch-patches", type=int, default=2048)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--batches", type=int, default=3)
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)
    if args.batches < 1:
        parser.error("--batches must be positive")
    start = time.monotonic()
    dataset, sampler, loader = create_manifest_dataloader(
        args.manifest,
        data_root=args.data_root,
        split=args.split,
        patch_size=args.patch_size,
        max_patches=args.max_patches,
        max_examples_per_batch=args.batch_size,
        max_patches_per_batch=args.batch_patches,
        num_workers=args.workers,
        seed=args.seed,
    )
    sampler.set_epoch(args.epoch)
    observed = []
    for batch_index, batch in enumerate(loader):
        observed.append({
            "batch": batch_index,
            "shape": list(batch["patches"].shape),
            "target_keys": sorted(set(batch["target_keys"])),
            "sources": sorted(set(batch["sources"])),
        })
        if batch_index + 1 >= args.batches:
            break
    print(json.dumps({
        "split": args.split,
        "examples": len(dataset),
        "index_stats": dataset.index_stats,
        "planned_batches": len(sampler),
        "epoch": args.epoch,
        "observed": observed,
        "elapsed_seconds": round(time.monotonic() - start, 3),
    }, ensure_ascii=False, indent=2))
    dataset.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
