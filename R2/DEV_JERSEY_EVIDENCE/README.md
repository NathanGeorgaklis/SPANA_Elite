# R2 DEV: jersey-number evidence aggregator (2026-11-04)

Instruction: Nathan, 2026-11-04 08:13. Diagnostic only.

- `player_tracker_rb/jersey.py` (NEW_R2): crop eligibility (clean, geometrically valid, box >= `jersey_min_box_h_px`),
  reading parser (1-2 digits, no leading zero, conf >= `jersey_min_read_conf`, disagreeing results -> UNREADABLE),
  per-tracklet consensus (CONFIRMED / CONFLICT / UNKNOWN) and SUPPORT / VETO / UNKNOWN against a user-provided
  expected number. A number never proves TARGET.
- Not called by `run_rb`; the tracker (state, trust, bank, handoff safety) is unchanged. Handoff safety stays ON.
- **No OCR reader has been run on any footage.** EasyOCR 1.7.2 weights are pinned outside the repo (sha256 in the
  session record) but unused. Reader evaluation waits for the frozen jersey validation dataset (`R2/JERSEY_DATASET`).
- Jersey truth for the focus DEV clips (`R2/labels/jersey/JERSEY_TRUTH_v1.json`): C006, C034, SYO C016 target number
  never visible, so number evidence cannot resolve the current DEV false suspensions.
- Tests: `tests_rb/test_jersey_evidence.py` (11, scripted reader). Full suite: 801 passed, 9 skipped, 11 xfailed
  (`SUITE_OUTPUT_jersey_aggregator.txt`). No EVAL run.
- Config: `jersey_min_box_h_px` 80 px, `jersey_min_read_conf` 0.5, `jersey_confirm_span_s` 0.5 s (uncalibrated:
  CALIBRATION UNKNOWN until the reader is evaluated on frozen truth). Registry r2jn-1, version 0.18.0-r2jn.
