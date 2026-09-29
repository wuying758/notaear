import unittest

from notaear.tokenizer import BAR_ID, CONT_ID, NotaEarTokenizer


class TokenizerTests(unittest.TestCase):
    def test_utf8_abc_roundtrips_exactly(self):
        abc = (
            "X:1\nT:茉莉花\nM:4/4\nL:1/8\nK:C\n"
            "C2 D2 (3EFG A2 | [K:G] B2 c2 d4 |]\n"
        )
        tokenizer = NotaEarTokenizer(patch_size=16)
        encoded = tokenizer.encode(abc)
        self.assertEqual(tokenizer.decode(encoded), abc)
        self.assertTrue(all(len(row) == 16 for row in encoded.token_ids))

    def test_long_bar_uses_continuation_patch(self):
        abc = "X:1\nM:4/4\nL:1/16\nK:C\n" + "CDEFGABc" * 8 + " |]\n"
        encoded = NotaEarTokenizer(patch_size=16).encode(abc)
        markers = [row[0] for row in encoded.token_ids]
        self.assertIn(BAR_ID, markers)
        self.assertIn(CONT_ID, markers)


if __name__ == "__main__":
    unittest.main()
