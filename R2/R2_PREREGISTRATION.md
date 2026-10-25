# R2 pre-registration: independent real-footage tracker benchmark

**Status:** pre-registered on 2026-10-26, Sydney. Committed **before** any R2 footage was viewed for clip selection and before any label exists. Nothing here may be changed after labels or results exist; a change needs a new version (`r2-protocol-2`) with the reason recorded.

**Instruction:** Nathan, 2026-10-26 07:53 (R2). Answers given the same morning:
- Task 3R: read time ranges only.
- Annotators: Nathan, as the only annotator.
- Box density: 5 Hz, plus every frame near events.

**Design basis:** `R0/R0_TRACKER_ACCEPTANCE_TARGETS.md` Part C.

**Machine-readable form:** `tools_rb/r2_protocol.py` and the record it prints (`R2/R2_PROTOCOL_RECORD.json`). Tests: `tests_rb/test_r2_protocol.py`.

## 1. Material

| Item | Rule |
|---|---|
| Allowed sources | `videoplayback.mp4`: SYO v MAR, 30 fps, multi-camera broadcast, sha256 `09aecc9b…bfec`. `videoplayback (1).mp4`: MCFC v MIT, 60 fps, single low camera, floodlit, sha256 `e8bd4d5f…c519`. Both hashes were re-verified on the Mac on 2026-10-26. |
| Holdout | `videoplayback (2).mp4` (ESFC v BCFC) and anything `v2_esfc_bcfc`: **never opened.** |
| Task 3R | The source time ranges of R01–R05 and R10 are excluded with a 60 s buffer. They were read from `extract_log*.txt` (time-range lines only, with Nathan's approval). R06–R09 come from youtube_game files, which are not R2 sources. No other Task 3R file was opened. |
| GENERAL_BENCHMARK_V2 | SYO/MAR 2400 s and 6600 s; MCFC/MIT 2400 s and 4200 s. Their lengths are not recorded in the surviving docs, so each is excluded as 120 s plus a 60 s buffer on each side. |
| GOLD_DUEL_01 | SYO/MAR 3036–3048 s, plus a 60 s buffer. GOLD is development footage and is **never** R2 evaluation truth. |
| youtube_game_1/2/3 | Not used: their provenance has not been reviewed (R0). |

## 2. DEV / EVAL split (fixed now, before selection and labelling)

- Each source is divided into 300 s blocks. A block is **EVAL** when the first byte of `sha256("SPANA-R2-split-v1:<source sha256>:<block index>")` is odd; otherwise it is **DEV**.
- Result: SYO/MAR has 15 EVAL blocks of 29; MCFC/MIT has 18 EVAL blocks of 30.
- A clip takes the split of its block. Nobody chooses a clip's split, and a clip may not cross a block boundary.
- Goal: every scenario class has at least one DEV clip and one EVAL clip, and DEV and EVAL are roughly balanced in clip count. If a class cannot be found in one split, that gap is recorded; no clip is moved.

## 3. Clip selection (no tracker, no detector)

- Clip length is 5–15 s. Boundaries are chosen by viewing frames sampled from the source. **No tracker, detector or model output is run or shown** during selection.
- Clips are selected by Claude, by eye. Each clip gets a proposed target: a player to follow, described in words (kit, position, action) and with an approximate location on the first frame. That location is only a pointer for the annotator, **not** a label. Nathan draws the real seed box and may reject the clip.
- **Disclosure.** Claude, the same agent that develops the tracker, sees EVAL footage while selecting clips. It never sees tracker output on EVAL before EVAL is frozen and scored. **Rule:** no tracker change may be motivated by EVAL footage. Every later tracker change must cite DEV or synthetic evidence only.
- Scenario classes, from the R2 instruction (R0 C2 IDs in brackets):

| Class | What the clip must contain |
|---|---|
| CLEAN | Continuous visible target, no contact (S1) |
| SAME_KIT_CROSSING | Target crosses or passes close to a teammate (S3) |
| OPPONENT_CROSSING | Target crosses or passes close to an opponent (S4) |
| PARTIAL_OCCLUSION | Target partly hidden by another player (S2/S7) |
| FULL_OCCLUSION | Target fully hidden for a moment, then reappears (S2) |
| MERGED_BODIES | Target and another player form one blob / box (S5/S6) |
| LEAVE_REENTER | Target leaves the frame and comes back (S10/S11) |
| PAN_ZOOM | Strong camera pan and/or zoom while the target is followed (S8/S9) |
| CAMERA_CUT | A hard cut with the target before and after (S13) |
| AMBIGUOUS_REACQ | After loss, several similar candidates near where the target reappears (S12) |

  One clip may cover several classes; its primary class is recorded.
- Extraction follows the Task 3J method: lossless re-encode (libx264 `-qp 0`) from the source, a framemd5 match of every frame against the source decode of the same range, and the sha256 recorded in the clip registry.

## 4. Human truth (before any inference)

- Annotator: **Nathan** (one annotator; the single-annotator status is a recorded limitation). Labels are drawn in the R2 annotation page from the clip's own frames, **with no model output shown**.
- Per frame (every frame): target visibility, identity resolvable, occlusion/merge state, contact, camera cut, first clean reappearance, and HUMAN_UNCERTAIN where needed.
- Boxes (target, and other people where needed to verify swaps): every 0.2 s (5 Hz), and every frame from 0.5 s before to 0.5 s after every contact, occlusion, merge, exit, re-entry and cut.
- The label schema `r2_labels_v1` and its validator are committed before labelling starts.
- **Freeze:** after labelling, each clip's label file is validated, hashed and written with a FROZEN record, then committed and pushed. All of this happens **before** the tracker is run on any R2 clip. Labels are never edited after freezing; corrections go into a new version with the reason recorded.

## 5. Scoring (pre-registered; not run in this stage)

- Scorer: `tools_rb/tracker_eval.py`, through an `r2_v1` adapter added before any R2 inference.
- Priorities, in order:
  1. false trusted identity = 0;
  2. identity swaps = 0;
  3. wrong reacquisition = 0;
  4. trusted on unresolvable / hidden / merged target = 0;
  5. trusted coverage;
  6. reacquisition delay;
  7. box quality.
- An error count is UNKNOWN when the truth cannot check it, never 0.
- DEV may be scored and inspected during development. **EVAL is scored only for a tracker version fixed in advance** and approved by Nathan, and is reported once. After EVAL results are seen, no threshold or logic change may be reported against the same EVAL set as unbiased performance.
- Events are not evaluated.

## 6. Downstream actor rule (recorded with R2; locked by Nathan on 2026-10-26)

Only the frames used to establish target actor identity must be trusted. If any essential actor-evidence frame is untrusted, the actor is UNKNOWN and target stats do not count the event. Details: `R1/R1H/R1H_DOWNSTREAM_IDENTITY_RULE.md`.

## Addendum A (2026-10-26, before any label or result exists): scoring details of the `r2_v1` adapter

Recorded once the label schema existed (`tools_rb/r2_labels.py`), before any label was drawn and before any R2 inference. Implemented in `tools_rb/tracker_eval.py` (`_r2_v1`).

- **Frames scored:** clip frames after the human seed. The seed is the run's frame 0, so frame f maps to rel f − seed.
- **Box checks on the label's box grid only.** The grid is 5 Hz from the seed, plus every frame within 0.5 s of an event, plus any frame where Nathan drew a target box.
  - On grid frames: wrong-box false trusted identity (IoU < 0.3), swaps (IoU ≥ 0.5 with an annotated person) and box quality are checked.
  - Trusted frames between grid frames are counted and reported, but not box-checked. Priority 4 (trusted on hidden, merged or unresolvable) uses per-frame truth and is checked on **every** frame.
- **Reacquisition** is judged on the first trusted frame after a gap. If that frame is off the grid, it is judged on the first grid frame of the same trusted run; if there is none, the outcome is UNVERIFIED (UNKNOWN, never CORRECT).
- **HUMAN_UNCERTAIN frames** are excluded from error counts and coverage and reported separately (including trusted frames on them).
- **Rejected clips** are never scored. An invalid label file, or a seed/fps mismatch, refuses the evaluation (fail closed).
- **Annotation display:** the page shows every frame of the lossless clip as a JPEG (q2), with frame index = clip frame index (`tools_rb/r2_annotation_pack.py`). No model output is shown.
