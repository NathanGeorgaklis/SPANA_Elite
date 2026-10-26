# R2 stage record: independent real-footage tracker benchmark (creation)

**Stage:** R2. Requested by Nathan on 2026-10-26 at 07:53 Sydney.

Nathan's answers the same morning:
- Task 3R: read the time ranges only.
- Annotation: Nathan alone.
- Boxes: 5 Hz plus every frame near events.

**Status: benchmark built; human truth NOT yet drawn.**
- No tracker logic was changed: `player_tracker_rb/` is identical to R1-G `30f2e39`.
- No tracker, detector or model was run on any R2 clip.
- No label exists, so nothing is frozen and nothing is scored.
- Stopped before inference, as instructed.

## Why the labels are not in this stage

Nathan's rules require human truth drawn by a person and never derived from model output (R0 Part C, charter). Claude is a model, so any box or state Claude drew would be model output. R2 therefore delivers everything except the drawing:
- the material, selected and extracted;
- the split, fixed before labelling;
- the schema, annotation page, validator, freeze tool and scorer adapter.

Nathan draws the truth with `R2/R2_LABELLING_GUIDE.md`. The labels are then frozen, hashed and committed before any inference.

## Benchmark composition

**Material:**
- Two allowed sources, with hashes re-verified on the Mac:
  - SYO v MAR: `videoplayback.mp4`, 30 fps, multi-camera broadcast, daylight;
  - MCFC v MIT: `videoplayback (1).mp4`, 60 fps, single low camera, dusk and floodlit.
- Excluded:
  - the holdout, never opened;
  - Task 3R R01–R05 and R10, plus a 60 s buffer (the youtube_game sources of R06–R09 are not R2 sources);
  - the GENERAL_BENCHMARK_V2 windows;
  - GOLD, plus its buffer.

**Selection.**
- Candidates were 80 seeded random 10 s windows, 40 per source, taken from allowed regions.
- Claude chose 25 clips by eye from ffmpeg frame strips, with no model used.
- Classes and targets are proposals (pointers), not labels.

**Split.** Fixed by a hash of each 300 s source block, pre-registered in commit `a55c181` before any footage was viewed: SYO 15/29 EVAL blocks, MCFC 18/30.

**Extraction.**
- Lossless libx264 qp0 of the 8-bit 4:2:0 sources.
- The md5 of every frame matches the source decode.
- A second seek path agrees.
- The absolute timestamp of each clip's first frame equals `round(start_s × fps)`.

| Split | Clips | SYO 30 fps | MCFC 60 fps | Frames |
|---|---|---|---|---|
| DEV | 12 | 6 | 6 | 5,400 |
| EVAL | 13 | 7 | 6 | 5,700 |
| **Total** | **25** | 13 | 12 | **11,100** (≈ 250 s) |

**Proposed scenario coverage** (clips per class; a clip can be in several classes):

| Class | DEV | EVAL |
|---|---|---|
| Clean tracking | 4 | 2 |
| Same-kit crossing | 2 | 2 |
| Opponent crossing | 1 | 4 |
| Partial occlusion | 2 | 3 |
| Full occlusion | 1 | 1 |
| Merged bodies | 2 | 2 |
| Target leaving / re-entering | 2 | 1 |
| Camera pan / zoom | 6 | 5 |
| Camera cut | 3 | 7 |
| Ambiguous reacquisition | 1 | 1 |

Every class is present in both splits. The clip list, windows and proposed targets are in `R2/R2_CLIP_SELECTION.json` and `R2/R2_CLIP_REGISTRY.json`.

**Disclosed.**
- Splits were visible while clips were chosen by eye, and the classes are unevenly balanced (Addendum B).
- Classes are Claude's proposals from thumbnails; Nathan's frozen labels decide what each clip actually contains.

## Human truth (to be drawn): schema `r2_labels_v1`

| Required by Nathan | Where |
|---|---|
| Target bounding box where identifiable | `target_boxes`: 5 Hz from the seed plus every frame within ±0.5 s of events. Required where VISIBLE / PARTIAL, resolvable and not uncertain; forbidden when HIDDEN / OUT_OF_FRAME. |
| Target visibility | `per_frame.visibility`: VISIBLE / PARTIAL / HIDDEN / OUT_OF_FRAME. Every frame from the seed; frames start UNLABELLED. |
| Identity resolvable YES / NO | `per_frame.resolvable` |
| Other-person boxes where needed to verify swaps | `people.*.boxes`. Every contact partner on every frame within ±0.5 s of each contact (or marked absent). A merged target, or one occluded by a person, must name that person. |
| Occlusion / merge state | `per_frame.occlusion` (NONE, PARTIAL/FULL × BY_PERSON/BY_OTHER), `merged`, `contact_with` |
| First clean reappearance | `first_clean_reappearance`: every return to a clean target must be marked. |
| HUMAN_UNCERTAIN | `per_frame.human_uncertain`. Excluded from error counts and coverage, and reported. |

Plus the human seed (frame and box), camera cuts, notes, clip rejection with a reason, and provenance of copied boxes.

**Tooling.**
- `tools_rb/r2_annotator.html`: offline page that shows exact per-frame JPEGs of the lossless clip and no model output. The pack is at `~/Downloads/_spana_r2_work/annotate/`.
- `tools_rb/r2_labels.py`: validator; `freeze` writes the frozen copy, the FROZEN record (sha256) and the run fixture with the human seed, atomically and never overwriting.

## Downstream actor rule (recorded)

Nathan's wording, 2026-10-26: *"Only frames used to establish target actor identity must be trusted. If any essential actor-evidence frame is untrusted, actor = UNKNOWN and target stats do not count the event."*

It is recorded in `R1/R1H/R1H_DOWNSTREAM_IDENTITY_RULE.md`, which replaced the earlier "every frame of the actor window" interpretation, and in pre-registration §6.

## Scoring (pre-registered; not run)

- `tools_rb/tracker_eval.py` with the `r2_v1` adapter.
- Priorities in Nathan's order:
  1. false trusted identity;
  2. swaps;
  3. wrong reacquisition;
  4. trusted on unresolvable / hidden / merged;
  5. coverage;
  6. reacquisition delay;
  7. box quality.
- UNKNOWN, never 0, where the truth cannot check.
- Labels, fixture, freeze record and results must describe the same clip.
- **EVAL is refused without Nathan's approval record naming the clean tracker commit.**

## Independent review

**Code review: 2 HIGH, 5 MEDIUM, 5 LOW, all validated and fixed** (commit `fe8b3a1`):
- H1: swaps and contact integrity could report 0 / PASS unchecked. Fixed: partner boxes are required across the contact window; contact integrity is UNKNOWN otherwise; swaps state their scope.
- H2: no cross-check that labels, fixture and results belong to the same clip. Fixed.
- M3: COMPLETE → COMPLETE_ON_GRID, with the box-checked fraction.
- M4: a merge or occlusion by a person must name the person.
- M5: unlabelled defaults; frames no longer assumed visible.
- M6: range apply erased cuts.
- M7: EVAL gate added, and the selection disclosed.
- L8–L12: stricter states; a first clean reappearance for every return; JS/Python rounding parity; freeze atomic and split recomputed; frame-aligned windows; absolute timestamp check; R0 deviations recorded.

**Test audit (on `d8170c9`):** 57 mutants, 20 caught, 37 survived, plus 4 bugs.
- Bugs: freeze not atomic (already fixed); non-canonical box keys (fixed); JS/Python rounding (fixed); swap count 0 without people (scope flag `count_is_lower_bound`). The smaller items are also fixed: page seed check, fps on load, reacquisition from an uncertain frame.
- New tests: `tests_rb/test_r2_audit.py` (pinned split and exclusions, edges, exact padding per event type, validator, adapter and reacquisition gaps, freeze atomicity) and browser tests (UI behaviours; randomized JS/Python parity at 25, 30 and 60 fps).
- Re-run of the same 57 mutants on the fixed code: **55 caught, 2 equivalent, 0 surviving.**
  - The 5 that initially survived the fixes (L7, E13, A1, A8, A10) are caught by the new tests.
  - Equivalent: L5 (the fixture-exists guard still refuses) and E16 (empty truth is refused upstream).

## Persistence

- Source, docs and tests: committed on `main` and pushed to GitHub. The GitHub copy was verified by a fresh clone and a sorted per-file sha256 comparison.
- Clips (not in Git, 25 lossless files): `SPANA_Engine_Archive/R2_CLIPS_2026-10-26_0821`; manifest `5f66732e…376b`, `sha256sum -c` PASS.
- Full project snapshot: `SPANA_Engine_Archive/R2_BENCHMARK_<timestamp>`, with FULL_FILE_MANIFEST_SHA256.txt and ARCHIVE_VERIFICATION.txt. The final commit and snapshot hashes are in `00_READ_FIRST`.
- Mac test output: `R2/R2_TEST_OUTPUT_mac.txt`.
- **Not in Git:** models, snapshots, source videos, clips, annotation pack. There is no second physical copy of those yet, so this is not full disaster recovery.

## Known limits

1. **One annotator** and no adjudication. Labelling effort is substantial: about 25 clips × 60–200 boxes.
2. **Box checks are on the grid only.** False trusted identity and swaps are box-checked on grid frames: 5 Hz plus every frame within ±0.5 s of events. Trusted frames between grid samples are counted, not checked. Priority 4 is checked on every frame.
3. **Two sources** (two matches, two camera styles). Small for "unseen footage" generalisation; EVAL is new footage but from the same two matches as DEV.
4. **Other development windows** (blind_r2, SEG_A–E, pass_completed_01) have unknown source times. Overlap with R2 is UNKNOWN.
5. **Seed detection-backing** is checked at run time, not at registration (no model before truth).
6. The browser tests run in the cloud (Playwright); they are skipped on the Mac.
7. **iCloud eviction on the Mac.** The disk was 96–98% full, and macOS "Optimize Mac Storage" evicted 416 files (~274 MB) in the Rebuild tree to cloud-only:
   - 408 `.git` objects;
   - 6 model files;
   - 2 recovered `.pyc` files.
   
   This caused 7 DINOv2 test errors ("Resource deadlock avoided"). The files were re-downloaded. Afterwards `git fsck --full` was clean, all 1,419 files read back, and the full suite passes. Evicted snapshots, clips or models are a standing risk until disk space is freed or optimisation is turned off for these folders.
