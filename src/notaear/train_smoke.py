"""Minimal reproducible trainer for validating the NotaEar learning pipeline."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch import Tensor

from .checkpoint import (
    CheckpointError,
    load_checkpoint_payload,
    model_from_checkpoint,
    restore_training_state,
    save_checkpoint,
)
from .model import NotaEarConfig, NotaEarTransformer
from .tokenizer import PAD_ID, NotaEarTokenizer


def _stable_validation_group(score_id: str) -> bool:
    digest = hashlib.blake2b(score_id.encode("utf-8"), digest_size=2).digest()
    return int.from_bytes(digest, "big") % 10 == 0


def load_examples(
    path: Path,
    tokenizer: NotaEarTokenizer,
    max_examples: int,
    max_patches: int,
) -> tuple[list[tuple[tuple[int, ...], ...]], list[tuple[tuple[int, ...], ...]], dict]:
    train: list[tuple[tuple[int, ...], ...]] = []
    validation: list[tuple[tuple[int, ...], ...]] = []
    stats = {"rows_seen": 0, "too_long": 0, "invalid": 0}
    selection_hash = hashlib.sha256()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if len(train) + len(validation) >= max_examples:
                break
            stats["rows_seen"] += 1
            try:
                record = json.loads(line)
                encoded = tokenizer.encode(record["abc"])
            except (KeyError, ValueError, json.JSONDecodeError):
                stats["invalid"] += 1
                continue
            if encoded.patch_count > max_patches:
                stats["too_long"] += 1
                continue
            score_id = str(record.get("score_id") or record.get("id") or stats["rows_seen"])
            target = validation if _stable_validation_group(score_id) else train
            target.append(encoded.token_ids)
            selection_hash.update(score_id.encode("utf-8"))
            selection_hash.update(b"\0")
            selection_hash.update(record["abc"].encode("utf-8"))
            selection_hash.update(b"\0")
    if not train or not validation:
        raise RuntimeError(
            f"need non-empty train and validation groups, got {len(train)} and {len(validation)}"
        )
    stats.update({
        "train_examples": len(train),
        "validation_examples": len(validation),
        "selection_sha256": selection_hash.hexdigest(),
    })
    return train, validation, stats


def pad_batch(
    examples: list[tuple[tuple[int, ...], ...]],
    patch_size: int,
    device: torch.device,
) -> Tensor:
    patch_count = max(len(example) for example in examples)
    batch = torch.full(
        (len(examples), patch_count, patch_size),
        PAD_ID,
        dtype=torch.long,
        device=device,
    )
    for index, example in enumerate(examples):
        value = torch.tensor(example, dtype=torch.long, device=device)
        batch[index, :value.shape[0]] = value
    return batch


@torch.no_grad()
def evaluate(
    model: NotaEarTransformer,
    examples: list[tuple[tuple[int, ...], ...]],
    batch_size: int,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    batches = 0
    for offset in range(0, len(examples), batch_size):
        batch = pad_batch(
            examples[offset:offset + batch_size],
            model.config.patch_size,
            device,
        )
        output = model(batch)
        total += float(output.loss)
        batches += 1
    return total / batches


def micro_config(patch_size: int, max_patches: int) -> NotaEarConfig:
    return NotaEarConfig(
        patch_size=patch_size,
        max_patches=max_patches,
        d_model=64,
        patch_token_dim=16,
        n_heads=4,
        patch_layers=2,
        char_layers=1,
        ff_multiplier=2,
        dropout=0.0,
        condition_dim=5,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--preset", choices=("micro", "small"), default="micro")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-examples", type=int, default=96)
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--max-patches", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="atomically save the final checkpoint to this path",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="resume model, optimizer, step and RNG state from a checkpoint",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="also update --checkpoint every N total steps (0 disables)",
    )
    args = parser.parse_args(argv)
    if min(args.steps, args.batch_size, args.max_examples, args.threads) < 1:
        parser.error("steps, batch-size, max-examples and threads must be positive")
    if args.checkpoint_every < 0:
        parser.error("checkpoint-every cannot be negative")
    if args.checkpoint_every and not (args.checkpoint or args.resume):
        parser.error("checkpoint-every requires --checkpoint or --resume")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    device_name = (
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    device = torch.device(device_name)

    resume_payload = None
    if args.resume:
        try:
            resume_payload = load_checkpoint_payload(args.resume, map_location=device)
            saved_config = NotaEarConfig(**resume_payload["model_config"])
        except (CheckpointError, TypeError, ValueError) as exc:
            parser.error(str(exc))
        if args.patch_size != saved_config.patch_size:
            parser.error(
                f"--patch-size {args.patch_size} does not match checkpoint "
                f"value {saved_config.patch_size}"
            )
        if args.max_patches != saved_config.max_patches:
            parser.error(
                f"--max-patches {args.max_patches} does not match checkpoint "
                f"value {saved_config.max_patches}"
            )

    tokenizer = NotaEarTokenizer(args.patch_size)
    train, validation, data_stats = load_examples(
        args.jsonl,
        tokenizer,
        args.max_examples,
        args.max_patches,
    )
    config = (
        saved_config
        if resume_payload is not None
        else (
            micro_config(args.patch_size, args.max_patches)
            if args.preset == "micro"
            else NotaEarConfig(
                patch_size=args.patch_size,
                max_patches=args.max_patches,
            )
        )
    )
    model = (
        model_from_checkpoint(resume_payload, device=device)
        if resume_payload is not None
        else NotaEarTransformer(config).to(device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    starting_step = 0
    if resume_payload is not None:
        saved_data = resume_payload["metadata"].get("data", {})
        saved_training = resume_payload["metadata"].get("training", {})
        if saved_data.get("selection_sha256") != data_stats["selection_sha256"]:
            parser.error(
                "resume data selection does not match the checkpoint; "
                "use the same JSONL content and selection arguments"
            )
        if saved_training.get("batch_size", args.batch_size) != args.batch_size:
            parser.error(
                f"--batch-size {args.batch_size} does not match checkpoint value "
                f"{saved_training['batch_size']}"
            )
        saved_learning_rate = saved_training.get("learning_rate", args.learning_rate)
        if saved_learning_rate != args.learning_rate:
            parser.error(
                f"--learning-rate {args.learning_rate} does not match checkpoint "
                f"value {saved_learning_rate}"
            )
        try:
            starting_step = restore_training_state(resume_payload, optimizer)
        except CheckpointError as exc:
            parser.error(str(exc))
        if args.steps <= starting_step:
            parser.error(
                f"--steps is the total target step and must exceed checkpoint "
                f"step {starting_step}"
            )

    checkpoint_path = args.checkpoint or args.resume
    checkpoint_metadata = {
        "data": {
            **data_stats,
            "path": str(args.jsonl.resolve()),
            "max_examples": args.max_examples,
        },
        "training": {
            "preset": args.preset,
            "batch_size": args.batch_size,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "seed": args.seed,
        },
    }

    initial_validation_loss = evaluate(
        model, validation, args.batch_size, device
    )
    history = []
    start = time.monotonic()
    model.train()
    for step in range(starting_step + 1, args.steps + 1):
        selected = random.choices(train, k=args.batch_size)
        batch = pad_batch(selected, args.patch_size, device)
        optimizer.zero_grad(set_to_none=True)
        output = model(batch)
        output.loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        entry = {
            "step": step,
            "train_loss": round(output.loss.detach().item(), 6),
            "gradient_norm": round(gradient_norm.detach().item(), 6),
        }
        history.append(entry)
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            print(json.dumps(entry), flush=True)
        if (
            checkpoint_path is not None
            and args.checkpoint_every
            and step % args.checkpoint_every == 0
        ):
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer=optimizer,
                step=step,
                metadata=checkpoint_metadata,
            )

    final_validation_loss = evaluate(
        model, validation, args.batch_size, device
    )
    report = {
        "preset": args.preset,
        "device": str(device),
        "config": asdict(config),
        "parameters": model.parameter_count(),
        "data": data_stats,
        "starting_step": starting_step,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "learning_rate": optimizer.param_groups[0]["lr"],
        "initial_validation_loss": round(initial_validation_loss, 6),
        "final_validation_loss": round(final_validation_loss, 6),
        "validation_loss_change": round(
            final_validation_loss - initial_validation_loss, 6
        ),
        "elapsed_seconds": round(time.monotonic() - start, 3),
        "history": history,
    }
    if checkpoint_path is not None:
        save_checkpoint(
            checkpoint_path,
            model,
            optimizer=optimizer,
            step=args.steps,
            metadata=checkpoint_metadata,
        )
        report["checkpoint"] = str(checkpoint_path)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
