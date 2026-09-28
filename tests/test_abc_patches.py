import unittest

from notaear.abc_patches import AbcPatchError, split_abc


class AbcPatchTests(unittest.TestCase):
    def test_simple_tune_roundtrips_exactly(self):
        abc = "X:1\nT:Demo\nM:4/4\nL:1/8\nK:C\nCDEF GABc | d2 c2 B2 A2 |]\n"
        document = split_abc(abc)
        self.assertEqual(document.join(), abc)
        self.assertEqual(len(document.bars), 2)
        self.assertEqual(document.bars[-1].closing_barline, "|]")

    def test_barlines_inside_annotations_comments_and_chords_do_not_split(self):
        abc = (
            "X:1\nM:4/4\nL:1/8\nK:C\n"
            '"C|G" [CEG]2 !text|still-decoration! z2 | % comment | ignored\n'
            "(3CDE F2 G2 A2 |\n"
        )
        document = split_abc(abc)
        self.assertEqual(document.join(), abc)
        self.assertEqual(len(document.bars), 2)
        self.assertEqual(document.bars[1].tuplet_markers, ("(3",))

    def test_leading_repeat_stays_with_first_sounding_bar(self):
        abc = "X:1\nM:2/4\nL:1/8\nK:Ddor\n|:DEFG|A2 A2:|\n"
        document = split_abc(abc)
        self.assertEqual(document.join(), abc)
        self.assertEqual(len(document.bars), 2)
        self.assertTrue(document.bars[0].text.startswith("|:"))

    def test_inline_key_and_meter_are_metadata_not_boundaries(self):
        abc = "X:1\nM:4/4\nL:1/8\nK:C\nC4 [K:G][M:2/4] D4 | E4 F4 |\n"
        document = split_abc(abc)
        self.assertEqual(document.join(), abc)
        self.assertEqual(document.bars[0].inline_keys, ("G",))
        self.assertEqual(document.bars[0].inline_meters, ("2/4",))

    def test_volta_marker_does_not_hide_later_barlines(self):
        abc = "X:1\nM:2/4\nL:1/8\nK:C\n|: CDEF |1 GABc :|2 cBAG |]\n"
        document = split_abc(abc)
        self.assertEqual(document.join(), abc)
        self.assertEqual(len(document.bars), 3)
        self.assertEqual([bar.closing_barline for bar in document.bars], ["|", ":|", "|]"])

    def test_missing_key_is_rejected(self):
        with self.assertRaisesRegex(AbcPatchError, "missing K"):
            split_abc("X:1\nM:4/4\nCDEF|\n")


if __name__ == "__main__":
    unittest.main()

