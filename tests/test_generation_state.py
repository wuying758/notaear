import unittest

from notaear.generation_state import (
    GenerationStateError,
    Phase,
    PatchGenerationState,
)
from notaear.tokenizer import (
    BAR_ID,
    CONT_ID,
    DOC_EOS_ID,
    PATCH_END_ID,
    NotaEarTokenizer,
)


class GenerationStateTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = NotaEarTokenizer(16)

    def state_for(self, abc):
        rows = [list(row) for row in self.tokenizer.encode(abc).token_ids[:-1]]
        return PatchGenerationState.from_prompt(self.tokenizer, rows)

    def test_header_requires_a_bar_before_eos(self):
        state = self.state_for("X:1\nM:2/4\nL:1/8\nK:C\n")
        self.assertEqual(state.phase, Phase.HEADER)
        self.assertEqual(state.allowed_markers(), (BAR_ID,))

    def test_full_incomplete_patch_requires_continuation(self):
        state = self.state_for("X:1\nM:2/4\nL:1/8\nK:C\n")
        state.start_patch(BAR_ID)
        state.finish_patch(b"C" * self.tokenizer.payload_size, full=True)
        self.assertEqual(state.phase, Phase.BAR_CONTINUATION)
        self.assertEqual(state.allowed_markers(), (CONT_ID,))

        state.start_patch(CONT_ID)
        state.finish_patch(b"|", full=False)
        self.assertEqual(state.phase, Phase.BAR_COMPLETE)
        self.assertEqual(state.allowed_markers(), (BAR_ID, DOC_EOS_ID))

    def test_quoted_pipe_does_not_close_bar(self):
        state = self.state_for("X:1\nM:2/4\nL:1/8\nK:C\n")
        state.start_patch(BAR_ID)
        base = tuple(range(9, 127))
        self.assertNotIn(
            PATCH_END_ID,
            state.allowed_payload_tokens(b'"C|G" C', base),
        )
        self.assertIn(
            PATCH_END_ID,
            state.allowed_payload_tokens(b'"C|G" C|', base),
        )

    def test_partial_bar_prompt_is_rejected(self):
        with self.assertRaisesRegex(GenerationStateError, "ends inside a bar"):
            self.state_for("X:1\nM:2/4\nL:1/8\nK:C\nCDEF")


if __name__ == "__main__":
    unittest.main()
