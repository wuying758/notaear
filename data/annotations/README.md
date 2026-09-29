# Difficulty annotations

`difficulty_pairs.jsonl` is the human-supervised part of the difficulty dataset.
Each line is an immutable comparison event, with the two score hashes, judgement,
primary difficulty dimension, confidence, annotator, and the hidden model snapshot.

Allowed judgements are `left_harder`, `equal`, `right_harder`, and `skip`.
Allowed dimensions are `overall`, `pitch`, `rhythm`, `tonality`, `notation`, and
`length`. Confidence is 1–3. The `listened.left` and `listened.right` flags record
whether the annotator played either example, so score-reading and assisted-listening
judgements can be analyzed separately.

The model snapshot is stored for audit and active-learning evaluation, but is not
shown before the annotator submits a judgement. Raw books and generated queues are
not committed; the compact annotation log is intended to be tracked by Git.
