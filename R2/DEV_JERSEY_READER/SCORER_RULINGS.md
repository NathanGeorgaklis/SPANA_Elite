# Scorer implementation record (written and committed before any reader run)

- **Pre-registration:** `R2/JERSEY_DATASET/READER_EVAL_PREREGISTRATION.md`, unchanged.
- **Code:** `tools_rb/r2_jersey_reader_eval.py` (units / score) and `tools_rb/r2_jersey_ocr_easyocr.py` (pinned reader
  step). Tests: `tests_rb/test_r2_jersey_reader_eval.py`.

## Ruling on §3.2 vs §3.5

Pre-run review finding 1: OCR can split "10" into "1" and "0". Under §3.2 alone, "0" is invalid, so a lone "1" would
be read.

§3.5 states that the split "gives a conflict and an abstention". So every confident all-digit token in a region is
kept, valid or not:
- more than one distinct token in a region is a CONFLICT, and the region abstains;
- a lone invalid token ("0", "07", "123") is discarded.

## Other pre-run review fixes

- **Combined-pool readings:** counted as at most one J and one S reading per keyframe (§6). This applies to
  precision, the Wilson bound and the ≥ 100 minimum.
- **Q1:**
  - any wrong S CONFIRMED counts as new;
  - coverage = person-keyframes with at least one CORRECT region reading;
  - SUPPORT counts as established only for selected players; competitors need a correct CONFIRMED.
- **Reporting:**
  - V1_CONFLICT notes;
  - the distractor-leak check (not hard-coded);
  - breakdowns by condition, clip, unreadable reason and PARTIAL;
  - shirt vs shorts number sharing;
  - Q0 per clip;
  - RESULTS.md, run_record.json and SHA256SUMS (§8).
- **Reader:** reads the crop by file path (EasyOCR's canonical RGB loading). An empty box fails closed.

## Units (truth + crops, built before commit, no reader)

- 276 person-keyframes: 213 selected player, 63 competitor.
- 0 policy abstentions: every box is ≥ 80 px.
- All 15 clip sha256s verified against the registry, and the dataset v2 record re-verified.
