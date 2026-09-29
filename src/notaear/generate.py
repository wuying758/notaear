"""Generate an ABC continuation from a trained NotaEar checkpoint."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from .checkpoint import load_checkpoint_payload, model_from_checkpoint
from .generation_state import GenerationStateError, PatchGenerationState
from .model import NotaEarTransformer
from .tokenizer import (
    DOC_BOS_ID,
    DOC_EOS_ID,
    PAD_ID,
    PATCH_END_ID,
    EncodedABC,
    NotaEarTokenizer,
)
from .validate_abc import ABCValidationReport, validate_abc


_ASCII_PAYLOAD = (9, 10, 13, *range(32, 127))


@dataclass(frozen=True)
class GenerationResult:
    abc: str
    token_ids: tuple[tuple[int, ...], ...]
    generated_patches: int
    stopped_on_eos: bool
    validation: ABCValidationReport


def _sample_allowed(
    logits: Tensor,
    allowed: tuple[int, ...],
    *,
    temperature: float,
    top_k: int,
    generator: torch.Generator,
) -> int:
    indices = torch.tensor(allowed, dtype=torch.long, device=logits.device)
    values = logits.index_select(0, indices) / temperature
    if top_k > 0 and top_k < values.numel():
        values, positions = torch.topk(values, top_k)
        indices = indices.index_select(0, positions)
    probabilities = torch.softmax(values.float(), dim=0)
    selected = torch.multinomial(probabilities, 1, generator=generator)
    return int(indices[selected].item())


@torch.inference_mode()
def generate_continuation(
    model: NotaEarTransformer,
    tokenizer: NotaEarTokenizer,
    prompt_abc: str,
    *,
    max_new_patches: int = 32,
    temperature: float = 1.0,
    top_k: int = 50,
    seed: int = 7,
    require_final_tonic: bool = False,
    device: str | torch.device | None = None,
) -> GenerationResult:
    """Autoregressively extend a complete, tokenizer-accepted ABC prefix.

    Marker transitions and physical bar segmentation are constrained. The
    decoded ABC is then checked by the independent musical validator.
    """
    if max_new_patches < 1:
        raise ValueError("max_new_patches must be positive")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if top_k < 0:
        raise ValueError("top_k cannot be negative")
    if tokenizer.patch_size != model.config.patch_size:
        raise ValueError("tokenizer patch size does not match the model checkpoint")

    model_device = next(model.parameters()).device
    requested_device = torch.device(device) if device is not None else model_device
    if requested_device != model_device:
        model = model.to(requested_device)
    model.eval()

    encoded = tokenizer.encode(prompt_abc)
    rows = [list(row) for row in encoded.token_ids]
    if not rows or rows[0][0] != DOC_BOS_ID or rows[-1][0] != DOC_EOS_ID:
        raise ValueError("prompt encoding does not have document boundaries")
    rows.pop()  # The prompt EOS is replaced by generated continuation.
    state = PatchGenerationState.from_prompt(tokenizer, rows)
    if len(rows) + max_new_patches > model.config.max_patches:
        raise ValueError(
            f"prompt plus {max_new_patches} new patches exceeds model limit "
            f"{model.config.max_patches}"
        )

    generator_device = requested_device.type if requested_device.type == "cuda" else "cpu"
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(seed)
    stopped_on_eos = False
    generated = 0

    # Reserve the last requested patch for a forced EOS if sampling never emits it.
    for new_index in range(max_new_patches):
        if new_index == max_new_patches - 1:
            if DOC_EOS_ID not in state.allowed_markers():
                raise GenerationStateError(
                    "generation budget ended inside an unfinished bar; "
                    "increase max_new_patches"
                )
            state.start_patch(DOC_EOS_ID)
            rows.append(tokenizer._special_patch(DOC_EOS_ID))
            state.finish_eos()
            generated += 1
            break

        history = torch.tensor([rows], dtype=torch.long, device=requested_device)
        context = model.next_patch_context(history)
        current = [PAD_ID] * tokenizer.patch_size
        rows.append(current)
        prefix: list[int] = []
        logits = model.next_token_logits(
            context,
            torch.empty((1, 0), dtype=torch.long, device=requested_device),
        )[0]
        marker = _sample_allowed(
            logits,
            state.allowed_markers(),
            temperature=temperature,
            top_k=top_k,
            generator=generator,
        )
        current[0] = marker
        prefix.append(marker)
        state.start_patch(marker)
        if marker == DOC_EOS_ID:
            current[1] = PATCH_END_ID
            state.finish_eos()
            stopped_on_eos = True
            generated += 1
            break

        payload = bytearray()
        for position in range(1, tokenizer.patch_size):
            if position == tokenizer.patch_size - 1:
                token = PATCH_END_ID
            else:
                prefix_tensor = torch.tensor(
                    [prefix], dtype=torch.long, device=requested_device
                )
                logits = model.next_token_logits(context, prefix_tensor)[0]
                allowed = state.allowed_payload_tokens(bytes(payload), _ASCII_PAYLOAD)
                token = _sample_allowed(
                    logits,
                    allowed,
                    temperature=temperature,
                    top_k=top_k,
                    generator=generator,
                )
            current[position] = token
            if token == PATCH_END_ID:
                break
            prefix.append(token)
            payload.append(token)
        state.finish_patch(
            bytes(payload),
            full=len(payload) == tokenizer.payload_size,
        )
        generated += 1

    token_ids = tuple(tuple(row) for row in rows)
    abc = tokenizer.decode(EncodedABC(token_ids=token_ids, patch_size=tokenizer.patch_size))
    canonical = tokenizer.encode(abc)
    if canonical.token_ids != token_ids:
        raise GenerationStateError(
            "internal error: generated patches are not canonical after re-tokenization"
        )
    validation = validate_abc(abc, require_final_tonic=require_final_tonic)
    return GenerationResult(
        abc=abc,
        token_ids=token_ids,
        generated_patches=generated,
        stopped_on_eos=stopped_on_eos,
        validation=validation,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("prompt", type=Path, help="UTF-8 ABC prompt with a K: field")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-new-patches", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--require-final-tonic", action="store_true")
    parser.add_argument("--allow-invalid", action="store_true")
    parser.add_argument("--validation-report", type=Path)
    args = parser.parse_args(argv)
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    device = (
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    payload = load_checkpoint_payload(args.checkpoint, map_location=device)
    model = model_from_checkpoint(payload, device=device)
    tokenizer = NotaEarTokenizer(model.config.patch_size)
    try:
        result = generate_continuation(
            model,
            tokenizer,
            args.prompt.read_text(encoding="utf-8-sig"),
            max_new_patches=args.max_new_patches,
            temperature=args.temperature,
            top_k=args.top_k,
            seed=args.seed,
            require_final_tonic=args.require_final_tonic,
            device=device,
        )
    except (GenerationStateError, ValueError) as exc:
        parser.error(str(exc))
    if args.validation_report:
        args.validation_report.parent.mkdir(parents=True, exist_ok=True)
        args.validation_report.write_text(
            json.dumps(result.validation.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    if not result.validation.accepted and not args.allow_invalid:
        error_codes = [
            issue.code
            for issue in result.validation.issues
            if issue.severity == "error"
        ]
        parser.error(
            "generated ABC failed musical validation: " + ", ".join(error_codes)
        )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result.abc, encoding="utf-8")
    else:
        sys.stdout.write(result.abc)
    print(json.dumps({
        "checkpoint_step": payload["step"],
        "generated_patches": result.generated_patches,
        "stopped_on_eos": result.stopped_on_eos,
        "validation_accepted": result.validation.accepted,
        "rhythm_valid": result.validation.rhythm_valid,
        "final_is_tonic": result.validation.tonal.get("final_is_tonic"),
        "output": str(args.output) if args.output else None,
    }), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
