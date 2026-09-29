"""Train NotaEar on a versioned manifest with exact batch-boundary resume."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .checkpoint import (
    CheckpointError,
    load_checkpoint_payload,
    model_from_checkpoint,
    restore_training_state,
    save_checkpoint,
)
from .dataset import ManifestBatchSampler, create_manifest_dataloader
from .model import NotaEarConfig, NotaEarTransformer
from .tokenizer import PAD_ID


TRAINER_FORMAT = "notaear-manifest-trainer-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def learning_rate_at_step(
    step: int,
    *,
    peak: float,
    warmup_steps: int,
    schedule_steps: int,
    minimum_ratio: float,
) -> float:
    """Linear warmup followed by cosine decay to ``peak * minimum_ratio``."""
    if step < 1:
        raise ValueError("step must be positive")
    if not 0 <= warmup_steps < schedule_steps:
        raise ValueError("warmup_steps must be non-negative and below schedule_steps")
    if peak <= 0 or not 0 <= minimum_ratio <= 1:
        raise ValueError("invalid learning-rate schedule")
    if warmup_steps and step <= warmup_steps:
        return peak * step / warmup_steps
    progress = min(
        1.0,
        max(0.0, (step - warmup_steps) / (schedule_steps - warmup_steps)),
    )
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return peak * (minimum_ratio + (1.0 - minimum_ratio) * cosine)


def _micro_config(patch_size: int, max_patches: int, condition_dim: int) -> NotaEarConfig:
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
        condition_dim=condition_dim,
    )


def _resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _resolve_precision(requested: str, device: torch.device) -> str:
    if requested == "auto":
        if device.type == "cuda":
            return "bf16" if torch.cuda.is_bf16_supported() else "fp16"
        return "fp32"
    if requested == "fp16" and device.type != "cuda":
        raise ValueError("fp16 training requires CUDA")
    if requested == "bf16" and device.type not in {"cuda", "cpu"}:
        raise ValueError(f"bf16 is unsupported on device type {device.type}")
    return requested


def _autocast(device: torch.device, precision: str):
    if precision == "fp32":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype)


def _new_optimizer(
    model: nn.Module,
    *,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
) -> torch.optim.Optimizer:
    kwargs: dict[str, Any] = {
        "lr": learning_rate,
        "betas": (0.9, 0.95),
        "weight_decay": weight_decay,
    }
    if device.type == "cuda":
        kwargs["fused"] = True
    try:
        return torch.optim.AdamW(model.parameters(), **kwargs)
    except TypeError:
        kwargs.pop("fused", None)
        return torch.optim.AdamW(model.parameters(), **kwargs)


@torch.inference_mode()
def evaluate(
    model: NotaEarTransformer,
    loader,
    *,
    device: torch.device,
    precision: str,
    max_batches: int = 0,
) -> dict[str, Any]:
    model.eval()
    weighted_loss = 0.0
    token_count = 0
    example_count = 0
    fallback_count = 0
    batch_count = 0
    started = time.monotonic()
    for batch in loader:
        patches = batch["patches"].to(device, non_blocking=True)
        with _autocast(device, precision):
            output = model(patches)
        valid_tokens = int((patches != PAD_ID).sum().item())
        weighted_loss += float(output.loss) * valid_tokens
        token_count += valid_tokens
        example_count += patches.shape[0]
        fallback_count += int(batch["variant_fallbacks"].sum().item())
        batch_count += 1
        if max_batches and batch_count >= max_batches:
            break
    model.train()
    return {
        "loss": weighted_loss / token_count if token_count else None,
        "batches": batch_count,
        "examples": example_count,
        "tokens": token_count,
        "variant_fallbacks": fallback_count,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _position_after_batch(
    epoch: int, batch_index: int, total_batches: int
) -> tuple[int, int]:
    next_batch = batch_index + 1
    return (epoch + 1, 0) if next_batch >= total_batches else (epoch, next_batch)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("out/generator-v1"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--preset", choices=("micro", "small"), default="small")
    parser.add_argument("--max-steps", type=int, default=100_000)
    parser.add_argument("--schedule-steps", type=int)
    parser.add_argument("--warmup-steps", type=int, default=2_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--minimum-lr-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--max-gradient-norm", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--batch-patches", type=int, default=2048)
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--max-patches", type=int, default=512)
    parser.add_argument("--condition-dim", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--precision", choices=("auto", "fp32", "bf16", "fp16"), default="auto")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=1_000)
    parser.add_argument("--checkpoint-every", type=int, default=1_000)
    parser.add_argument("--max-validation-batches", type=int, default=0)
    parser.add_argument("--no-initial-validation", action="store_true")
    parser.add_argument("--no-source-weighting", action="store_true")
    args = parser.parse_args(argv)

    positive = {
        "max_steps": args.max_steps,
        "gradient_accumulation": args.gradient_accumulation,
        "batch_size": args.batch_size,
        "batch_patches": args.batch_patches,
        "patch_size": args.patch_size,
        "max_patches": args.max_patches,
        "workers_plus_one": args.workers + 1,
        "threads": args.threads,
        "log_every": args.log_every,
    }
    invalid = [name for name, value in positive.items() if value < 1]
    if invalid:
        parser.error("positive values required for: " + ", ".join(invalid))
    if min(args.eval_every, args.checkpoint_every, args.max_validation_batches) < 0:
        parser.error("eval/checkpoint/max-validation intervals cannot be negative")
    if args.condition_dim < 0:
        parser.error("condition-dim cannot be negative")

    try:
        device = _resolve_device(args.device)
        precision = _resolve_precision(args.precision, device)
    except ValueError as exc:
        parser.error(str(exc))
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")

    resume_payload = None
    saved_training: dict[str, Any] = {}
    if args.resume:
        try:
            resume_payload = load_checkpoint_payload(args.resume, map_location=device)
        except CheckpointError as exc:
            parser.error(str(exc))
        metadata = resume_payload["metadata"]
        if metadata.get("trainer_format") != TRAINER_FORMAT:
            parser.error("checkpoint was not created by the formal manifest trainer")
        saved_training = metadata.get("training", {})

    schedule_steps = args.schedule_steps
    if schedule_steps is None:
        schedule_steps = (
            int(saved_training["schedule_steps"])
            if saved_training
            else args.max_steps
        )
    if schedule_steps < args.max_steps:
        parser.error("schedule-steps must be at least max-steps")
    if not 0 <= args.warmup_steps < schedule_steps:
        parser.error("warmup-steps must be non-negative and below schedule-steps")

    manifest_path = args.manifest.resolve()
    if not manifest_path.is_file():
        parser.error(f"manifest not found: {manifest_path}")
    manifest_sha256 = _sha256(manifest_path)
    weighting = not args.no_source_weighting
    immutable_training = {
        "preset": args.preset,
        "schedule_steps": schedule_steps,
        "warmup_steps": args.warmup_steps,
        "learning_rate": args.learning_rate,
        "minimum_lr_ratio": args.minimum_lr_ratio,
        "weight_decay": args.weight_decay,
        "gradient_accumulation": args.gradient_accumulation,
        "max_gradient_norm": args.max_gradient_norm,
        "batch_size": args.batch_size,
        "batch_patches": args.batch_patches,
        "seed": args.seed,
        "source_weighting": weighting,
        "precision": precision,
    }
    if saved_training:
        mismatches = {
            key: {"checkpoint": saved_training.get(key), "requested": value}
            for key, value in immutable_training.items()
            if saved_training.get(key) != value
        }
        if mismatches:
            parser.error(
                "resume training configuration changed: "
                + json.dumps(mismatches, sort_keys=True)
            )
        saved_hash = resume_payload["metadata"].get("data", {}).get("manifest_sha256")
        if saved_hash != manifest_sha256:
            parser.error("resume manifest content does not match checkpoint")

    persistent = args.workers > 0
    train_dataset, train_sampler, train_loader = create_manifest_dataloader(
        manifest_path,
        data_root=args.data_root,
        split="train",
        patch_size=args.patch_size,
        max_patches=args.max_patches,
        max_examples_per_batch=args.batch_size,
        max_patches_per_batch=args.batch_patches,
        weighted=weighting,
        shuffle=True,
        seed=args.seed,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=persistent,
    )
    validation_dataset, validation_sampler, validation_loader = create_manifest_dataloader(
        manifest_path,
        data_root=args.data_root,
        split="validation",
        patch_size=args.patch_size,
        max_patches=args.max_patches,
        max_examples_per_batch=args.batch_size,
        max_patches_per_batch=args.batch_patches,
        weighted=False,
        shuffle=False,
        seed=args.seed,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=persistent,
    )
    validation_sampler.set_epoch(0)

    if resume_payload is not None:
        try:
            model = model_from_checkpoint(resume_payload, device=device)
        except CheckpointError as exc:
            parser.error(str(exc))
        if model.config.patch_size != args.patch_size or model.config.max_patches != args.max_patches:
            parser.error("checkpoint model patch dimensions do not match loader arguments")
        if model.config.condition_dim != args.condition_dim:
            parser.error("checkpoint condition_dim does not match --condition-dim")
    else:
        config = (
            _micro_config(args.patch_size, args.max_patches, args.condition_dim)
            if args.preset == "micro"
            else NotaEarConfig(
                patch_size=args.patch_size,
                max_patches=args.max_patches,
                condition_dim=args.condition_dim,
            )
        )
        model = NotaEarTransformer(config).to(device)

    optimizer = _new_optimizer(
        model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=device,
    )
    scaler = torch.amp.GradScaler(device.type, enabled=precision == "fp16")
    global_step = 0
    epoch = 0
    next_batch = 0
    best_validation_loss: float | None = None
    if resume_payload is not None:
        try:
            global_step = restore_training_state(resume_payload, optimizer)
        except CheckpointError as exc:
            parser.error(str(exc))
        position = resume_payload["metadata"].get("position", {})
        if position.get("global_step") != global_step:
            parser.error("checkpoint step and trainer position disagree")
        epoch = int(position.get("epoch", 0))
        next_batch = int(position.get("next_batch", 0))
        best_validation_loss = position.get("best_validation_loss")
        scaler_state = position.get("scaler_state")
        if scaler_state:
            scaler.load_state_dict(scaler_state)
        if args.max_steps <= global_step:
            parser.error(
                f"max-steps {args.max_steps} must exceed checkpoint step {global_step}"
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    latest_path = args.output_dir / "latest.pt"
    best_path = args.output_dir / "best.pt"
    report_path = args.output_dir / "report.json"
    data_metadata = {
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "data_root": str(train_dataset.data_root),
        "train_examples": len(train_dataset),
        "validation_examples": len(validation_dataset),
        "train_index_stats": train_dataset.index_stats,
        "validation_index_stats": validation_dataset.index_stats,
    }

    def metadata_for(save_epoch: int, save_next_batch: int) -> dict[str, Any]:
        return {
            "trainer_format": TRAINER_FORMAT,
            "data": data_metadata,
            "training": immutable_training,
            "position": {
                "global_step": global_step,
                "epoch": save_epoch,
                "next_batch": save_next_batch,
                "best_validation_loss": best_validation_loss,
                "scaler_state": scaler.state_dict(),
            },
        }

    history: list[dict[str, Any]] = []
    started = time.monotonic()
    last_validation_step = -1
    if global_step == 0 and not args.no_initial_validation:
        baseline = evaluate(
            model,
            validation_loader,
            device=device,
            precision=precision,
            max_batches=args.max_validation_batches,
        )
        baseline["step"] = 0
        history.append({"validation": baseline})
        last_validation_step = 0
        print(json.dumps({"validation": baseline}), flush=True)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    running_loss = 0.0
    running_microbatches = 0
    running_examples = 0
    running_fallbacks = 0
    while global_step < args.max_steps:
        train_sampler.set_epoch(epoch, start_batch=next_batch)
        total_batches = train_sampler.total_batches
        if next_batch >= total_batches:
            epoch += 1
            next_batch = 0
            continue
        accumulated = 0
        step_loss = 0.0
        step_examples = 0
        step_fallbacks = 0
        for batch_index, batch in enumerate(train_loader, start=next_batch):
            if accumulated == 0:
                learning_rate = learning_rate_at_step(
                    global_step + 1,
                    peak=args.learning_rate,
                    warmup_steps=args.warmup_steps,
                    schedule_steps=schedule_steps,
                    minimum_ratio=args.minimum_lr_ratio,
                )
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate
            patches = batch["patches"].to(device, non_blocking=True)
            with _autocast(device, precision):
                output = model(patches)
                scaled_loss = output.loss / args.gradient_accumulation
            scaler.scale(scaled_loss).backward()
            accumulated += 1
            value = float(output.loss.detach())
            step_loss += value
            step_examples += patches.shape[0]
            step_fallbacks += int(batch["variant_fallbacks"].sum().item())
            at_epoch_end = batch_index + 1 >= total_batches
            if accumulated < args.gradient_accumulation and not at_epoch_end:
                continue
            if accumulated != args.gradient_accumulation:
                correction = args.gradient_accumulation / accumulated
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(correction)
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.max_gradient_norm
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            save_epoch, save_next_batch = _position_after_batch(
                epoch, batch_index, total_batches
            )
            next_batch = batch_index + 1
            running_loss += step_loss / accumulated
            running_microbatches += 1
            running_examples += step_examples
            running_fallbacks += step_fallbacks

            if global_step == 1 or global_step % args.log_every == 0:
                entry = {
                    "step": global_step,
                    "epoch": epoch,
                    "next_batch": save_next_batch,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "train_loss": running_loss / running_microbatches,
                    "gradient_norm": float(gradient_norm),
                    "examples": running_examples,
                    "variant_fallbacks": running_fallbacks,
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                }
                if device.type == "cuda":
                    entry["cuda_peak_memory_mib"] = round(
                        torch.cuda.max_memory_allocated(device) / 1024**2, 1
                    )
                history.append({"train": entry})
                print(json.dumps(entry), flush=True)
                running_loss = 0.0
                running_microbatches = 0
                running_examples = 0
                running_fallbacks = 0

            should_evaluate = bool(
                args.eval_every and global_step % args.eval_every == 0
            )
            if should_evaluate:
                validation = evaluate(
                    model,
                    validation_loader,
                    device=device,
                    precision=precision,
                    max_batches=args.max_validation_batches,
                )
                validation["step"] = global_step
                history.append({"validation": validation})
                last_validation_step = global_step
                print(json.dumps({"validation": validation}), flush=True)
                if (
                    validation["loss"] is not None
                    and (
                        best_validation_loss is None
                        or validation["loss"] < best_validation_loss
                    )
                ):
                    best_validation_loss = validation["loss"]
                    save_checkpoint(
                        best_path,
                        model,
                        optimizer=optimizer,
                        step=global_step,
                        metadata=metadata_for(save_epoch, save_next_batch),
                    )

            if args.checkpoint_every and global_step % args.checkpoint_every == 0:
                save_checkpoint(
                    latest_path,
                    model,
                    optimizer=optimizer,
                    step=global_step,
                    metadata=metadata_for(save_epoch, save_next_batch),
                )
            if global_step >= args.max_steps:
                epoch, next_batch = save_epoch, save_next_batch
                break
            accumulated = 0
            step_loss = 0.0
            step_examples = 0
            step_fallbacks = 0
        else:
            epoch += 1
            next_batch = 0
            continue
        if global_step >= args.max_steps:
            break

    if last_validation_step != global_step:
        validation = evaluate(
            model,
            validation_loader,
            device=device,
            precision=precision,
            max_batches=args.max_validation_batches,
        )
        validation["step"] = global_step
        history.append({"validation": validation})
        print(json.dumps({"validation": validation}), flush=True)
        if validation["loss"] is not None and (
            best_validation_loss is None or validation["loss"] < best_validation_loss
        ):
            best_validation_loss = validation["loss"]
            save_checkpoint(
                best_path,
                model,
                optimizer=optimizer,
                step=global_step,
                metadata=metadata_for(epoch, next_batch),
            )
    save_checkpoint(
        latest_path,
        model,
        optimizer=optimizer,
        step=global_step,
        metadata=metadata_for(epoch, next_batch),
    )
    report = {
        "trainer_format": TRAINER_FORMAT,
        "device": str(device),
        "precision": precision,
        "model_config": asdict(model.config),
        "parameters": model.parameter_count(),
        "data": data_metadata,
        "training": immutable_training,
        "final_step": global_step,
        "position": {"epoch": epoch, "next_batch": next_batch},
        "best_validation_loss": best_validation_loss,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "history": history,
        "latest_checkpoint": str(latest_path),
        "best_checkpoint": str(best_path) if best_path.exists() else None,
    }
    if device.type == "cuda":
        report["cuda_peak_memory_mib"] = round(
            torch.cuda.max_memory_allocated(device) / 1024**2, 1
        )
    _write_json_atomic(report_path, report)
    print(json.dumps({
        "completed": True,
        "step": global_step,
        "latest_checkpoint": str(latest_path),
        "best_validation_loss": best_validation_loss,
        "report": str(report_path),
    }), flush=True)
    train_dataset.close()
    validation_dataset.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
