# R1_PRECODING_GATE.md

SPANA engine v2-rebuild, R1 pre-coding gate. Recorded 2026-10-24 (Australia/Sydney).

## Actions performed

1. **R0 evidence.**
   - The R0 documents were authored and cross-checked in the preceding R0 session.
   - At this gate, the Mac R0 tree was re-verified against its manifest (0 mismatches) and the R0 archive (16/16 OK).
   - The SPANA_RECOVERY_INVENTORY and the event_engine_v1 status doc had already been read in R0.
2. **Source tree.** `~/Documents/SPANA_Engine_Rebuild/` exists.
3. **Git.**
   - The shell could not delete files, which git needs for its lock files. Nathan granted delete permission for `~/Documents` for this session. It is used only for git and scratch cleanup inside `SPANA_Engine_Rebuild`.
   - `git init` (branch `main`) succeeded, and `git status` works.
4. **R0 baseline.**
   - Contents: R0 docs; recovered originals unpacked read-only under `recovered/SPANA_recovered_originals/` (317/317 matched the zip's SHA256SUMS; `task3r/` excluded); README; .gitignore; PROJECT_METADATA.
   - Commit `fb90702f3ba2114725933190d83642c8538b5399`, tag `r0-recovery-map`. The working tree was clean.
5. **Pre-R1 snapshot.**
   - Path: `~/Documents/SPANA_Engine_Archive/R1_PRE_2026-10-24_1328/`. It holds the full tree including `.git`: 657 files.
   - Manifest sha256 `1827ccafad36a82a71e09b1557280fcd94889f34e2d6e32048b9675573edc6a8`.
   - Hashed from the archive copy: 657/657 OK. **PASS.**
6. **Model and backend options inspected.**
   - Environment: Python 3.10.12, numpy 2.2.6, OpenCV 5.0.0, onnxruntime 1.23.2. Torch, ultralytics and scipy are absent. 4 CPU, about 3 GB RAM. PyPI, GitHub and Hugging Face are reachable.
   - Licences were verified at the primary sources (see R1_MODEL_AND_LICENSE_RECORD.md).
7. **GOLD label sheets.**
   - `tools_rb/make_gold_label_sheet.py` (NEW) produced `R1/gold_label_sheet/` for extract frames 250–290, plus a seed reference page.
   - No model output is shown on any sheet.
   - The source was verified as 164541…; fixture 7929bb28…; decode was sequential.
   - The GOLD fixture, label, sign-off and hash files were copied into `fixtures/GOLD_DUEL_01/` (hashes unchanged).
8. **`R1/R1_DESIGN_DECISIONS.md`** written (D01–D22, plus recommended architecture changes).
9. **`R1/R1_MODEL_AND_LICENSE_RECORD.md`** written.

## Not done (by design)

- No tracker source written.
- No model downloaded.
- No event-engine change.
- Task 3R and the holdout were not touched.
- No labels were filled in by Claude.

## Open risks

- **No second independent backup.** Git history and every archive are on the same Mac disk.
- Delete permission on `~/Documents` is active for this session only.
