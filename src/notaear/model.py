"""A compact hierarchical causal Transformer for NotaEar byte patches."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .tokenizer import PAD_ID, VOCAB_SIZE


@dataclass(frozen=True)
class NotaEarConfig:
    vocab_size: int = VOCAB_SIZE
    patch_size: int = 32
    max_patches: int = 512
    d_model: int = 384
    patch_token_dim: int = 64
    n_heads: int = 6
    patch_layers: int = 6
    char_layers: int = 3
    ff_multiplier: int = 4
    dropout: float = 0.1
    condition_dim: int = 5

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.patch_size < 4 or self.max_patches < 2:
            raise ValueError("patch_size must be >= 4 and max_patches must be >= 2")


@dataclass
class NotaEarOutput:
    logits: Tensor
    loss: Tensor | None


def _causal_mask(length: int, device: torch.device) -> Tensor:
    return torch.ones(length, length, dtype=torch.bool, device=device).triu(1)


class NotaEarTransformer(nn.Module):
    """Patch-level causal Transformer plus an intra-patch causal decoder.

    The patch context for patch p is computed only from patches before p.
    The byte decoder then predicts patch p left-to-right. Explicit shifting
    prevents teacher-forcing leakage from the target patch.
    """

    def __init__(self, config: NotaEarConfig = NotaEarConfig()):
        super().__init__()
        self.config = config
        self.patch_token_embedding = nn.Embedding(
            config.vocab_size, config.patch_token_dim, padding_idx=PAD_ID
        )
        self.patch_projection = nn.Linear(
            config.patch_size * config.patch_token_dim, config.d_model
        )
        self.patch_start = nn.Parameter(torch.empty(1, 1, config.d_model))
        self.patch_position = nn.Embedding(config.max_patches, config.d_model)

        patch_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_model * config.ff_multiplier,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.patch_decoder = nn.TransformerEncoder(
            patch_layer,
            num_layers=config.patch_layers,
            enable_nested_tensor=False,
        )
        self.patch_norm = nn.LayerNorm(config.d_model)

        self.token_embedding = nn.Embedding(
            config.vocab_size, config.d_model, padding_idx=PAD_ID
        )
        self.char_position = nn.Embedding(config.patch_size, config.d_model)
        char_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_model * config.ff_multiplier,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.char_decoder = nn.TransformerEncoder(
            char_layer,
            num_layers=config.char_layers,
            enable_nested_tensor=False,
        )
        self.char_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.condition_projection = (
            nn.Linear(config.condition_dim, config.d_model, bias=False)
            if config.condition_dim
            else None
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.patch_token_embedding.weight, std=0.02)
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.patch_position.weight, std=0.02)
        nn.init.normal_(self.char_position.weight, std=0.02)
        nn.init.xavier_uniform_(self.patch_projection.weight)
        nn.init.zeros_(self.patch_projection.bias)
        nn.init.normal_(self.patch_start, std=0.02)
        with torch.no_grad():
            self.patch_token_embedding.weight[PAD_ID].zero_()
            self.token_embedding.weight[PAD_ID].zero_()

    def parameter_count(self, trainable_only: bool = True) -> int:
        parameters = (
            (parameter for parameter in self.parameters() if parameter.requires_grad)
            if trainable_only
            else self.parameters()
        )
        return sum(parameter.numel() for parameter in parameters)

    def _patch_context(
        self,
        patches: Tensor,
        conditioning: Tensor | None,
    ) -> Tensor:
        batch, patch_count, patch_size = patches.shape
        if patch_size != self.config.patch_size:
            raise ValueError(
                f"expected patch size {self.config.patch_size}, got {patch_size}"
            )
        if patch_count > self.config.max_patches:
            raise ValueError(
                f"patch count {patch_count} exceeds {self.config.max_patches}"
            )
        embedded = self.patch_token_embedding(patches).flatten(start_dim=2)
        embedded = self.patch_projection(embedded)
        history = torch.cat(
            (self.patch_start.expand(batch, -1, -1), embedded[:, :-1]),
            dim=1,
        )
        positions = torch.arange(patch_count, device=patches.device)
        history = history + self.patch_position(positions).unsqueeze(0)
        if conditioning is not None:
            if self.condition_projection is None:
                raise ValueError("conditioning was supplied but condition_dim is zero")
            expected = (batch, self.config.condition_dim)
            if tuple(conditioning.shape) != expected:
                raise ValueError(f"conditioning shape must be {expected}")
            history = history + self.condition_projection(conditioning).unsqueeze(1)
        context = self.patch_decoder(
            history,
            mask=_causal_mask(patch_count, patches.device),
        )
        return self.patch_norm(context)

    def forward(
        self,
        patches: Tensor,
        conditioning: Tensor | None = None,
        compute_loss: bool = True,
    ) -> NotaEarOutput:
        if patches.ndim != 3:
            raise ValueError("patches must have shape [batch, patches, patch_size]")
        if patches.dtype != torch.long:
            patches = patches.long()
        batch, patch_count, patch_size = patches.shape
        context = self._patch_context(patches, conditioning)

        targets = patches.reshape(batch * patch_count, patch_size)
        previous_tokens = self.token_embedding(targets[:, :-1])
        char_inputs = torch.cat(
            (context.reshape(batch * patch_count, 1, -1), previous_tokens),
            dim=1,
        )
        positions = torch.arange(patch_size, device=patches.device)
        char_inputs = char_inputs + self.char_position(positions).unsqueeze(0)
        decoded = self.char_decoder(
            char_inputs,
            mask=_causal_mask(patch_size, patches.device),
        )
        logits = self.lm_head(self.char_norm(decoded)).reshape(
            batch,
            patch_count,
            patch_size,
            self.config.vocab_size,
        )
        loss = None
        if compute_loss:
            loss = F.cross_entropy(
                logits.reshape(-1, self.config.vocab_size),
                patches.reshape(-1),
                ignore_index=PAD_ID,
            )
        return NotaEarOutput(logits=logits, loss=loss)

    def next_patch_context(
        self,
        previous_patches: Tensor,
        conditioning: Tensor | None = None,
    ) -> Tensor:
        """Return the patch-level context for one patch after the history."""
        if previous_patches.ndim != 3:
            raise ValueError(
                "previous_patches must have shape [batch, patches, patch_size]"
            )
        if previous_patches.shape[2] != self.config.patch_size:
            raise ValueError(
                f"expected patch size {self.config.patch_size}, "
                f"got {previous_patches.shape[2]}"
            )
        if previous_patches.shape[1] >= self.config.max_patches:
            raise ValueError("no model context remains for another patch")
        placeholder = torch.full(
            (previous_patches.shape[0], 1, self.config.patch_size),
            PAD_ID,
            dtype=torch.long,
            device=previous_patches.device,
        )
        patches = torch.cat((previous_patches.long(), placeholder), dim=1)
        return self._patch_context(patches, conditioning)[:, -1]

    def next_token_logits(self, context: Tensor, prefix: Tensor) -> Tensor:
        """Predict the token after a partial patch without recomputing history."""
        if context.ndim != 2 or context.shape[1] != self.config.d_model:
            raise ValueError(
                f"context must have shape [batch, {self.config.d_model}]"
            )
        if prefix.ndim != 2 or prefix.shape[0] != context.shape[0]:
            raise ValueError("prefix must have shape [batch, prefix_length]")
        if prefix.shape[1] >= self.config.patch_size:
            raise ValueError("prefix is already a complete patch")
        inputs = context.unsqueeze(1)
        if prefix.shape[1]:
            inputs = torch.cat((inputs, self.token_embedding(prefix.long())), dim=1)
        positions = torch.arange(inputs.shape[1], device=context.device)
        inputs = inputs + self.char_position(positions).unsqueeze(0)
        decoded = self.char_decoder(
            inputs,
            mask=_causal_mask(inputs.shape[1], context.device),
        )
        return self.lm_head(self.char_norm(decoded[:, -1]))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    config = NotaEarConfig()
    model = NotaEarTransformer(config)
    count = model.parameter_count()
    print(json.dumps({
        "config": asdict(config),
        "trainable_parameters": count,
        "trainable_parameters_millions": round(count / 1_000_000, 3),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
