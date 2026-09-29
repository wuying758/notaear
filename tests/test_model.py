import unittest

import torch

from notaear.model import NotaEarConfig, NotaEarTransformer
from notaear.tokenizer import NotaEarTokenizer


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.tokenizer = NotaEarTokenizer(patch_size=16)
        self.config = NotaEarConfig(
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

    def _tokens(self, abc):
        encoded = self.tokenizer.encode(abc)
        return torch.tensor([encoded.token_ids], dtype=torch.long)

    def test_forward_shape_and_loss(self):
        tokens = self._tokens("X:1\nM:4/4\nL:1/8\nK:C\nC2 D2 E2 F2 |]\n")
        model = NotaEarTransformer(self.config)
        output = model(tokens, conditioning=torch.zeros(1, 5))
        self.assertEqual(
            output.logits.shape,
            (*tokens.shape, self.config.vocab_size),
        )
        self.assertTrue(torch.isfinite(output.loss))
        self.assertGreater(output.loss.item(), 4.0)
        self.assertLess(output.loss.item(), 7.0)

    def test_future_patch_cannot_change_earlier_logits(self):
        tokens = self._tokens("X:1\nM:2/4\nL:1/8\nK:C\nCDEF | GABc |]\n")
        model = NotaEarTransformer(self.config).eval()
        changed = tokens.clone()
        changed[:, -2, 1] = ord("z")
        with torch.no_grad():
            original_logits = model(tokens, compute_loss=False).logits
            changed_logits = model(changed, compute_loss=False).logits
        self.assertTrue(
            torch.equal(original_logits[:, :-2], changed_logits[:, :-2])
        )

    def test_incremental_logits_match_full_forward(self):
        tokens = self._tokens("X:1\nM:2/4\nL:1/8\nK:C\nCDEF | GABc |]\n")
        model = NotaEarTransformer(self.config).eval()
        history = tokens[:, :-1]
        target = tokens[:, -1]
        with torch.no_grad():
            context = model.next_patch_context(history)
            full = model(tokens, compute_loss=False).logits[:, -1]
            for position in range(self.config.patch_size):
                incremental = model.next_token_logits(context, target[:, :position])
                self.assertTrue(
                    torch.allclose(incremental, full[:, position], atol=1e-5),
                    position,
                )


if __name__ == "__main__":
    unittest.main()
