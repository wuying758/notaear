import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from notaear.checkpoint import (
    load_checkpoint_payload,
    model_from_checkpoint,
    save_checkpoint,
)
from notaear.generate import generate_continuation
from notaear.model import NotaEarConfig, NotaEarTransformer
from notaear.tokenizer import (
    BAR_ID,
    DOC_EOS_ID,
    PATCH_END_ID,
    VOCAB_SIZE,
    NotaEarTokenizer,
)
from notaear.train_smoke import main as train_main


def tiny_config() -> NotaEarConfig:
    return NotaEarConfig(
        patch_size=16,
        max_patches=64,
        d_model=64,
        patch_token_dim=16,
        n_heads=4,
        patch_layers=2,
        char_layers=1,
        ff_multiplier=2,
        dropout=0.0,
        condition_dim=5,
    )


class ScriptedGeneratorModel(torch.nn.Module):
    """Emit one canonical C| bar and then EOS for deterministic inference tests."""

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(
            patch_size=16,
            max_patches=64,
            vocab_size=VOCAB_SIZE,
        )

    def next_patch_context(self, previous_patches, conditioning=None):
        has_bar = torch.any(previous_patches[:, :, 0] == BAR_ID, dim=1)
        return has_bar.to(dtype=torch.float32).unsqueeze(1)

    def next_token_logits(self, context, prefix):
        logits = torch.full(
            (context.shape[0], VOCAB_SIZE), -1000.0, device=context.device
        )
        position = prefix.shape[1]
        if position == 0:
            for index, has_bar in enumerate(context[:, 0]):
                logits[index, DOC_EOS_ID if bool(has_bar) else BAR_ID] = 1000.0
        elif position == 1:
            logits[:, ord("C")] = 1000.0
        elif position == 2:
            logits[:, ord("4")] = 1000.0
        elif position == 3:
            logits[:, ord("|")] = 1000.0
        else:
            logits[:, PATCH_END_ID] = 1000.0
        return logits


class CheckpointGenerationTests(unittest.TestCase):
    def test_checkpoint_roundtrip_preserves_logits_and_metadata(self):
        torch.manual_seed(11)
        tokenizer = NotaEarTokenizer(16)
        tokens = torch.tensor([
            tokenizer.encode("X:1\nM:2/4\nL:1/8\nK:C\nCDEF |]\n").token_ids
        ])
        model = NotaEarTransformer(tiny_config()).eval()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        expected = model(tokens, compute_loss=False).logits

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            save_checkpoint(
                path,
                model,
                optimizer=optimizer,
                step=17,
                metadata={"data": {"selection_sha256": "abc"}},
            )
            payload = load_checkpoint_payload(path)
            restored = model_from_checkpoint(payload).eval()

        actual = restored(tokens, compute_loss=False).logits
        self.assertTrue(torch.equal(expected, actual))
        self.assertEqual(payload["step"], 17)
        self.assertEqual(payload["metadata"]["data"]["selection_sha256"], "abc")

    def test_generation_is_seeded_decodable_and_keeps_prompt(self):
        tokenizer = NotaEarTokenizer(16)
        model = ScriptedGeneratorModel().eval()
        prompt = "X:1\nM:2/4\nL:1/8\nK:C\n"
        first = generate_continuation(
            model, tokenizer, prompt, max_new_patches=3, top_k=1, seed=19
        )
        second = generate_continuation(
            model, tokenizer, prompt, max_new_patches=3, top_k=1, seed=19
        )
        self.assertEqual(first.token_ids, second.token_ids)
        self.assertTrue(first.abc.startswith(prompt))
        self.assertTrue(first.abc.endswith("C4|"))
        self.assertTrue(first.validation.accepted)
        self.assertEqual(first.token_ids[-1][0], DOC_EOS_ID)
        self.assertEqual(tokenizer.decode(first.token_ids), first.abc)

    def test_resumed_smoke_training_matches_uninterrupted_training(self):
        records = [
            {
                "score_id": str(index),
                "abc": f"X:1\nT:{index}\nM:2/4\nL:1/8\nK:C\nCDEF |]\n",
            }
            for index in range(64)
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jsonl = root / "sample.jsonl"
            jsonl.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            full = root / "full.pt"
            resumed = root / "resumed.pt"
            common = [
                str(jsonl), "--device", "cpu", "--threads", "1",
                "--batch-size", "2", "--max-examples", "32",
                "--patch-size", "16", "--max-patches", "64",
            ]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(train_main([*common, "--steps", "2", "--checkpoint", str(full)]), 0)
                self.assertEqual(train_main([*common, "--steps", "1", "--checkpoint", str(resumed)]), 0)
                self.assertEqual(train_main([*common, "--steps", "2", "--resume", str(resumed)]), 0)
            full_payload = load_checkpoint_payload(full)
            resumed_payload = load_checkpoint_payload(resumed)

        self.assertEqual(full_payload["step"], resumed_payload["step"])
        self.assertEqual(full_payload["model_state"].keys(), resumed_payload["model_state"].keys())
        for name in full_payload["model_state"]:
            self.assertTrue(
                torch.equal(full_payload["model_state"][name], resumed_payload["model_state"][name]),
                name,
            )


if __name__ == "__main__":
    unittest.main()
