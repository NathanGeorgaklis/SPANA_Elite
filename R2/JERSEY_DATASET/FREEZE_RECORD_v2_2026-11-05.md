# Jersey truth v2 (region layer) freeze (2026-11-05)

- **Labels:** Nathan, `~/Downloads/labels_incoming/jersey_v2/` (15 files, re-saved after his fixes, 2026-11-05).
  - Received copies, unchanged: `R2/labels/jersey_dataset_v2_received/`.
- **Correction (Nathan's instruction 2026-11-05: "Yes, clear those 8"):** 8 leftover shorts-number fields had no
  READABLE shorts region. Nathan had marked those shorts NO_NUMBER / UNREADABLE / NOT_VISIBLE. They were set to
  null:
  - J01, J02 and J06 targets;
  - J04 target, C1 and C2;
  - J10 C1;
  - J14 C1.
  - The corrected copies are in `R2/labels/jersey_dataset_v2_corrected/`, with `CORRECTION_RECORD.json` (received
    and corrected sha256 per file, and a programmatic check that only those fields differ).
  - Nothing else was changed.
- **Validated:** 15/15 against frozen v1 + the registry.
- **Frozen:** `R2/labels/jersey_dataset_v2/<clip>.regions.v2.FROZEN.json` (read-only byte copies with freeze records).
- **Dataset v2 record:** `JERSEY_DATASET_RECORD_v2.json`, **dataset_sha256
  `b322b8f77ea81ad7fe39d14e7a66b4bb5a4da398f6a7e377405452e32a2d9ece`**.
- **Pre-inference coverage (truth only):** see `PRE_INFERENCE_COVERAGE_v2.md`.
- **No reader has been run.**
