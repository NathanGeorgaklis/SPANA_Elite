# GOLD_HUMAN_LABELS_v1: provenance and corrections

**Labeller:** Nathan Georgaklis. Labelled in Apple Numbers and uploaded to the Claude session on 2026-10-24 at 14:19 (Sydney).

## Source
- `GOLD_HUMAN_LABELS_v1_SOURCE.numbers`: Nathan's file, byte-identical to the upload. SHA-256 `1d6e7737ca1b486955bff1f99f83fab0b9997404f432ad270db555fade70e490`.
- It has one sheet ("Sheet 1") with one table ("GOLD_HUMAN_LABELS_TEMPLATE"): 42 rows (a header plus 41 frames) by 10 columns.
- Conversion: `tools_rb/gold_labels_numbers_to_csv_v1.py`, using numbers-parser 4.22.0 and protobuf 7.36.2. Cell text is copied exactly as Numbers formats it, including Nathan's lower/upper case.

## First validation of the raw export (failed)
The raw export had SHA-256 `94251f2eef859687605863db63c048b85ba391dd099bef6690f68db2e73a0e76`. It is not kept.
- Frames 266–268: `merged_or_hidden` was `hidden`, which is not in the vocabulary.
- `first_clean_reappearance` was marked Y on 30 frames: 250–257 and 269–290.

## Corrections, chosen by Nathan (2026-10-24 ~14:20 Sydney)
1. Frames 266, 267 and 268: `merged_or_hidden` changed from `hidden` to `T23_HIDDEN`.
2. `first_clean_reappearance` set to `Y` on **frame 273 only**. It was cleared on the 29 other frames.

These are the only 32 cells that differ from the raw export (checked by a cell-by-cell diff). No other label was changed. No historical tracker output was consulted.

## Noted, not changed (Nathan's labels stand)
- Frames 264–265: #23 is `hidden`, yet `t23_identity_resolvable` is `yes` and the pair is `merged`.
- Frames 269–272: #23 is `visible` while the pair is still `merged`.
- No frame uses HUMAN_UNCERTAIN.

Any further correction must go in `GOLD_HUMAN_LABELS_v2.csv`.
