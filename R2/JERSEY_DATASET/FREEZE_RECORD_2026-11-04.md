# Jersey truth freeze, v1 (2026-11-04)

- **Labels:** Nathan, from `~/Downloads/labels_incoming/jersey/` (15 files). The Mac and staged hashes are identical.
- **Validated** against `JERSEY_CLIP_REGISTRY.json`: 15/15 VALID. All 15 clip sha256s were re-verified on the Mac
  before freezing.
- **Frozen:** `R2/labels/jersey_dataset/<clip>.jersey.v1.FROZEN.json`, byte-for-byte copies (read-only), each with a
  `.FROZEN.record.json` holding its sha256.
- **Dataset record:** `JERSEY_DATASET_RECORD.json`.
  - **dataset_sha256 `1d9669dd9bc6f58b7b7cc6aa11ecd54060cc6c6ba347076143a73807bf784e15`.**
- **Coverage:** 240 keyframes; READABLE 167, UNREADABLE 46, target not visible 27; HUMAN_UNCERTAIN 0; rejected
  clips 0.
  - Unreadable reasons: SIDE_ON 29, FRONT_ON 12, OCCLUDED_BY_PERSON 3, OCCLUDED_OTHER 1, FOLDED_SHIRT 1.
  - Every clip has a target number (none NEVER_VISIBLE). Numbers: 7, 17, 7, 5, 6, 3, 2, 47, 6, 12, 9, 23, 18, 14, 11.
  - J01 and J03 are both #7; J05 and J09 are both #6, on different clips.
- **Gaps (as labelled, not changed):**
  - No nearby-player / competitor numbers were entered on any clip.
  - No distractors were entered.
  - So the dataset can measure target-number reading (SUPPORT and false readings) on human target boxes. It cannot
    yet measure same-kit VETO behaviour on the competitors' own crops.
  - Adding competitors would be a v2 freeze with a reason. v1 stays.
- **No reader has been run.** EVAL is untouched.
