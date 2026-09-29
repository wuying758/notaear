# Sight-singing reference books

This directory records curriculum references for learning a multidimensional
sight-singing difficulty model. The downloaded facsimiles and MusicXML files
live under `raw/` and are intentionally ignored by Git. `manifest.json` is the
reproducible inventory for those local files.

## Current collection

| Book / collection | Local material | Structured exercises | Primary use |
| --- | --- | ---: | --- |
| *Melodia* (Cole and Lewis, 1904) | 205-page PDF + MusicXML | 1,511 | Long, graded curriculum; filenames retain book, chapter, and exercise order |
| *330 Exercises for Sight-Singing Classes* (Gilchrist, 1891) | 196-page PDF + official web index | 330 in the book | Explicit concept sequence: rhythm, tuplets, chromaticism, intervals, keys, and modulation |
| *Eyes and Ears* (Crowell, 2004) | 134-page PDF + MusicXML | 427 | Modern, systematically arranged first-year melodic material |
| *The School of Sight-Singing* (Concone, 1897 edition) | 59-page PDF + MusicXML | 90 | Longer vocal studies and phrase-level progression |

The MusicXML transcriptions were obtained from The Trained Ear. Its download
page and the Gilchrist exercise index are also retained locally so later
parsers can reconstruct source order and topic boundaries.

## Difficulty-label policy

Book order is useful weak supervision, not an absolute label. Do not convert
page number or exercise number directly into the old `L1`-`L5` classes.
Instead, retain at least:

- `book_id`, volume/chapter, exercise number, and original order;
- concept tags explicitly introduced by the book (for example triplets,
  chromatic tones, modulation, and compound meter);
- measurable score features such as pitch span, interval distribution,
  rhythmic density, syncopation, tuplet rate, accidentals, phrase length,
  and tonal stability;
- source and transcription confidence, including missing or ambiguous fields.

The first training target should be pairwise curriculum ranking within a book:
later material is *usually* harder than earlier material, with uncertainty and
local exceptions. Cross-book calibration should be learned from shared score
features rather than assumed from raw exercise numbers.

## Licensing notes

- *Melodia* is identified as public domain by San Jose State University. The
  Trained Ear transcription is distributed under CC BY-SA 4.0.
- The Trained Ear site and its downloadable notation are marked CC BY-SA 4.0.
- *Eyes and Ears* is distributed under CC BY-SA / GPL terms, but individual
  melodies retain source-specific provenance. Audit each melody before using
  it in a redistributable training release.
- Gilchrist and Concone are historical works, but redistribution status should
  still be checked for the intended jurisdiction and for any modern scan or
  transcription.

This inventory is therefore suitable for local research. A later export tool
must filter records by a reviewed, per-item license field before publication.

## Reproduction

Exact URLs, byte sizes, page counts, exercise counts, retrieval time, and
SHA-256 hashes are recorded in `manifest.json`. Re-downloading dynamic ZIPs
from The Trained Ear requires first requesting its `create_zip.php` endpoint,
then downloading the generated archive from `source-files/zip-archives/`.
