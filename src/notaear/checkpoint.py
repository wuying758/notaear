"""Versioned checkpoints for NotaEar generator training and inference."""
from __future__ import annotations

import os
import random
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from .model import NotaEarConfig, NotaEarTransformer


CHECKPOINT_FORMAT = "notaear-generator"
CHECKPOINT_VERSION = 1


class CheckpointError(ValueError):
    """Raised when a checkpoint is incompatible or incomplete."""


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = handle.name
        torch.save(payload, temporary_path)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and os.path.exists(temporary_path):
            os.unlink(temporary_path)


def save_checkpoint(
    path: Path,
    model: NotaEarTransformer,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    step: int = 0,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Atomically save enough state for inference or exact training resume."""
    if step < 0:
        raise ValueError("checkpoint step cannot be negative")
    payload = {
        "format": CHECKPOINT_FORMAT,
        "version": CHECKPOINT_VERSION,
        "model_config": asdict(model.config),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "step": step,
        "rng_state": _rng_state(),
        "metadata": dict(metadata or {}),
    }
    _atomic_torch_save(payload, Path(path))


def load_checkpoint_payload(
    path: Path,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load and validate a trusted local NotaEar checkpoint.

    PyTorch checkpoints use pickle internally. Do not load checkpoint files from
    an untrusted source.
    """
    try:
        payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    except (OSError, RuntimeError, EOFError) as exc:
        raise CheckpointError(f"could not load checkpoint {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise CheckpointError("checkpoint root must be a dictionary")
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise CheckpointError("not a NotaEar generator checkpoint")
    if payload.get("version") != CHECKPOINT_VERSION:
        raise CheckpointError(
            f"unsupported checkpoint version {payload.get('version')!r}; "
            f"expected {CHECKPOINT_VERSION}"
        )
    required = {"model_config", "model_state", "step", "metadata"}
    missing = sorted(required.difference(payload))
    if missing:
        raise CheckpointError(f"checkpoint is missing fields: {', '.join(missing)}")
    if not isinstance(payload["metadata"], dict):
        raise CheckpointError("checkpoint metadata must be a dictionary")
    return payload


def model_from_checkpoint(
    payload: dict[str, Any],
    *,
    device: str | torch.device = "cpu",
) -> NotaEarTransformer:
    try:
        config = NotaEarConfig(**payload["model_config"])
        model = NotaEarTransformer(config)
        model.load_state_dict(payload["model_state"], strict=True)
    except (KeyError, TypeError, RuntimeError, ValueError) as exc:
        raise CheckpointError(f"checkpoint model is incompatible: {exc}") from exc
    return model.to(device)


def restore_training_state(
    payload: dict[str, Any],
    optimizer: torch.optim.Optimizer,
    *,
    restore_rng: bool = True,
) -> int:
    optimizer_state = payload.get("optimizer_state")
    if optimizer_state is None:
        raise CheckpointError("checkpoint has no optimizer state; cannot resume training")
    optimizer.load_state_dict(optimizer_state)
    if restore_rng:
        state = payload.get("rng_state")
        if not isinstance(state, dict) or "python" not in state or "torch_cpu" not in state:
            raise CheckpointError("checkpoint has no complete RNG state")
        random.setstate(state["python"])
        torch.set_rng_state(state["torch_cpu"].cpu())
        cuda_state = state.get("torch_cuda")
        if cuda_state is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(cuda_state)
    step = payload.get("step")
    if not isinstance(step, int) or step < 0:
        raise CheckpointError(f"invalid checkpoint step {step!r}")
    return step
