# R2 DEV remediation F3c: baseline population safety (2026-10-31)

**Instruction (Nathan, 2026-10-31 07:09 Sydney):**
- Fix two identity-safety failures at the population layer:
  1. a genuine two-person union that persists beyond the merge hold can become one ordinary tracklet;
  2. a person hidden or crouched wholly inside the target's box can stand up in front and inherit the target's
     tracklet.
- Fail closed when physical identity cannot be distinguished, never fabricate people, keep the appearance bank pure,
  and keep ordinary single-person poses trackable.
- Do not solve F3b; keep its 5 reproducers as expected failures. Do not modify F5. DEV only. Do not run EVAL.
- Move `R2/_spana_r2_work/` out of the repository to `~/Documents/SPANA_R2_Work/` (do not delete, do not commit).

**Persistence change (Nathan, 2026-10-31 17:45):** "let's stop saving it to my mac and save it straight to
github". From commit `f65c233` on, work is committed in the cloud workspace and pushed directly to
`github.com/NathanGeorgaklis/SPANA_Elite`. No new Mac archive snapshot is made.

**EVAL was not run, opened or scored.** No frozen label, fixture, holdout or Task 3R material was touched. The review
subagents were told not to open the DEV clips, EVAL material, labels or `SPANA_R2_Work`.

## Commits

| Commit | What |
|---|---|
| `f65c233` | F3c implementation, 10 new tests, `.gitignore` rule for `R2/_spana_r2_work/` |
| `9869c5c` | Code-review round 1 fixes (H2, M1, H3), 2 new tests |
| `c67c738` | Test-audit fixes (A1, A2, minimum-duration measurement), stronger tests |
| (this record) | DEV results, review and audit repro scripts, raw DEV outputs |

Tracker files changed: `player_tracker_rb/population.py`, `config.py` (registry `r2c-2`), `__init__.py` (`0.8.4-r2c`).
`target.py` (F5 reacquisition) is unchanged.

## Root causes (reproduced at `8c798bc`, identical tracker to `37a109c`)

**Failure 1: a persistent union becomes one ordinary tracklet.**
- Two people a body-width or less apart give a union box about 1.3–1.4× one body, below `merged_area_ratio` (1.6).
- A merged region's resolution counted that union box as one "single-sized" body.
- A third person crossing after the merge formed was not an outsider (only bodies tracked before the merge were),
  so union plus passer-by counted 2 and the region "resolved".
- Lineage was cleared, and the union box became a clean tracklet holding two people. In the reproducer the target was
  TRACKING (untrusted) on it for 59 frames.
- The region was also forgotten 2 s after its last merge evidence even while the union persisted.

**Failure 2: a hidden person stands up and inherits the target's tracklet.**
- A crouching person wholly inside the target's box is NESTED, so it gets no tracklet and is treated as part of the
  host.
- When they stand up in front, the host tracklet continues on them: same kit gave 21–66 wrong trusted frames and a
  contaminated bank. With an opponent kit, appearance caught it.

## Generic fixes (`population.py`)

**Persisting merged box.**
- A detection containing ≥ `population_region_fill_min` (0.9) of a region's box *fills* the region.
- A filling box never counts as a resolved single body on its own. It counts as one only when:
  - another merged body has been seen emerging from the region; and
  - the filling box is back near one body's size (area and aspect ≤ `merged_anomaly_clear_area_ratio`, 1.25; review
    H2).
- The region stays remembered while it is filled.
- Tracklets observed on a filling box keep MERGE_LINEAGE.
- A body first seen outside the region after it formed is an outsider and never resolves it.
- A claimant carrying MERGE_LINEAGE is never the region's single-body size reference (audit A1).

**Vanished hidden body.**
- A feet-aligned body (bottom within `population_hidden_feet_gap_max`, 0.1 × host height) nested in a tracklet's box
  is recorded with the tracklet.
- Only one seen there continuously for ≥ `population_hidden_min_s` (0.2 s) counts (review H3). It is tracked frame to
  frame by moving it with the host's bottom-centre displacement (review M1).
- The body counts as still there only if another detection overlaps where it would be (IoU ≥
  `population_match_iou_min`) **and** stands on the same feet line (audit A2).
- If it is seen nowhere, the link is contested: the tracklet closes CONTACT_AMBIGUITY, and the box is reborn naming
  it as possible predecessor (`Observation.continuity_break = "CONTAINED_BODY_VANISHED"`, decision log only).
- The target then goes through the existing reacquisition (an unverified, untrusted episode until contrast-verified).

**Not done (by instruction):**
- no F3b change;
- no appearance logic;
- no F5 change;
- no person fabricated inside a union.

## Tests

`tests_rb/test_population_safety.py` has 13 tests, run at 30, 15 and 10 fps:
- **Persistent union:** a real two-person union in both kits, with and without a third person, and with a detection
  gap. A union reborn after a long persistence. A real split gives two people with no swap.
- **Hidden person rising:** a population-level crouch sweep (scale, offset, rise speed); a target-level test in
  same and opponent kit, checking no wrong trusted frame, a pure bank, and no silent carry-over.
- **Ordinary cases stay trackable:** an isolated player; a short pose change; a persistent wide single-person pose;
  near/far occlusion; an opponent passing in front and behind.
- **Review round 1:** a third person stepping out of a persisting union; a persistent part box on a running player
  at every frame rate.
- **Audit round 1:** a farther person masking the crouch spot; the minimum-duration boundary (0.1 s vs 0.3 s at
  every frame rate).

`tests_rb/test_population.py`, TestNestedPartBoxes (changed, see *Recorded cost*):
- upper-body part boxes and a *flickering* lower-body box stay one tracklet;
- a *persistent* feet-aligned box that then vanishes breaks continuity (contested, never a merge or a swap).

Each new test was checked against the code before its fix:
- the 4 targeted F3c tests failed at `8c798bc`;
- the review tests failed at `f65c233` (13 subtests);
- the audit tests failed at `9869c5c`.

Final suite (cloud, `c67c738` tree): **635 passed, 9 skipped, 5 xfailed** (F3b reproducers, kept). The 9 skips are
environmental: the GOLD video is not set, and Playwright is not installed. On the Mac (`f65c233` files, before the
switch to GitHub), unittest chunks 1–4 passed with 0 failures; the target and long-run chunks were not run.

**Synthetic tracker evaluation** (`tools_rb/r1h_synthetic_eval.py`): identical to the pre-F3c run at every commit.

**Crossing matrix:** 54/54 crossing cases end TRACKING with 0 silent swaps.
- An earlier F3c draft made the four hide-mode cases end LOST: the target's own box, left after the far person was
  hidden, filled the region forever.
- Fixed by the emergence rule; a negative result, recorded.

## Independent review and test audit

**Code review (f65c233), no new safety regression vs 8c798bc.** Findings, all reproduced:
- **H2** union resolved by a third person stepping out: fixed.
- **M1** frame-rate dependence of the vanish test: fixed.
- **H3** a single-frame part box cost an isolated target its trust for the rest of the clip; 33/180 trusted frames
  beside a neighbour: fixed. After the fix: 180/180, and 0 s untrusted after a blip.
- **H1** the person standing up also has a feet-aligned lower-body box: NOT fixable by geometry, identical to
  baseline.
- **M2** crouched person missed on the last frame: identical to baseline.
- **L1** fill test has no upper size bound: documented.
- **L2** `continuity_break` added only to the decision log (no output schema change).
- **L3** inverted test: now split and justified.

**Test audit (34 mutants on f65c233 and 9869c5c), with probes.**
- **A1** (union plus third person plus gap → union tracked) and **A2** (masking by a farther person): fixed.
- **A3** (crouched person missed 1–2 frames before rising), **A4** (crouched detection flickering), **A5** (camera
  cut during a union): all equal to or better than the `8c798bc` baseline at every fps (the baseline mixes in every
  case). Recorded as known limits.
- **Vacuous test** found: the split test never formed a region. Fixed; it now asserts a merged region formed.
- **Surviving mutants:** the test gaps are listed in the audit report, with suggested tests 1–9. Tests were added
  for D1, D2 and the duration boundary; the others remain open.

Repro scripts are in `review_repros/` and `audit_probes/`. They are records, not part of the suite.

## DEV results (clean `c67c738`; EVAL not run)

Runner: `tools_rb/r2_dev_run.py`, 11 scorable DEV clips (C019 is excluded as rejected), in the cloud workspace with
the pinned models. The baseline is `37a109c` (tracker byte-identical to `8c798bc`).

| | 37a109c | c67c738 |
|---|---|---|
| Initialised / refused | 8 / 3 | 8 / 3 (F2: C034, C038, SYO C032) |
| P1 false confident identity | 5 | 5 (same frames: C006 3, C033 1, C021 1) |
| P2 swaps | 0 | 0 |
| P3 wrong reacquisition | 1 | 1 (C033) |
| P4 trusted on unavailable | 1 | 1 (C033) |
| Trusted frames / available (run clips) | 818 / 2105 (0.389) | 818 / 2105 (0.389) |
| Bank accepted / IoU-diagnostic flags / unverified | 72 / 10 / 35 | 72 / 10 / 35 |
| Target frames with MERGED reason | 817 | 835 (C004 11 → 28, C012 5 → 6) |
| Tracklet births | 1558 | 1500 |
| MULTI_CLAIM frames | 223 | 204 |
| MERGED_CONTACT closures | 722 | 678 |
| Target merge closures | 7 | 7 |
| CONTAINED_BODY_VANISHED breaks | n/a | 15 events in 5 clips, **none on the target's tracklet** |

- **Every official metric is unchanged.** There is no new confirmed wrong-person trusted identity, no swap and no new
  bank flag.
- The only target-level change is 18 more frames abstaining with a merged reason. These are persisting merged boxes
  remembered longer, and none of them was a trusted frame.
- The F3c failure classes were not observed on DEV targets. DEV has no labelled merged frames, so this is not
  evidence that they are rare.
- The intermediate run of `9869c5c` gave the same official metrics. Its C021 record says `dirty True` because the
  tree was edited during that run, so it is not used as a result.

Raw outputs: `raw/R2_DEV_F3C_raw_c67c738.tar.gz`, sha256
`d89694909c3bc6615920d1c7e3d8ed0b35d96080bcab0ecfe4487a8d12cca1f1` (per-file hashes in
`raw/RAW_OUTPUTS_SHA256_c67c738.txt`). The `9869c5c` raw outputs were not kept; only their scores are in
`INTERMEDIATE_9869c5c_*.json`.

## Recorded cost and known limits

**Costs.**
- A persistent feet-aligned lower-body part box that disappears while the body goes on breaks continuity. It is
  geometrically identical to a hidden person standing up, so the tracker fails closed. The target then needs
  reacquisition, and without a confusable competitor it stays untrusted for the rest of the episode (an F5 limit). DEV
  had no such break on the target.
- Two people seen apart after a union keep MERGE_LINEAGE inherited from the union box's tracklet (8c798bc behaviour,
  conservative: never a bank sample).

**Not caught (no worse than baseline).**
- The person standing up produces their own feet-aligned lower-body box (H1).
- The crouched person is missed on the last 1–2 frames or flickers (A3, A4); or is detected for less than 0.2 s.
- A camera cut during a union (regions are cleared at a cut). The target is never trusted on it in the probes, but it
  does TRACK it.
- A union tighter than 1.25× one body after an emergence is indistinguishable from one body.

All of these need evidence beyond geometry, such as part-versus-body appearance or appearance-verified re-entry (the
F5 stage).

**Other limits.**
- L1: the fill test has no upper size bound. A larger box over a stale region keeps it alive, which errs toward
  caution; state is bounded by the region count.
- F3b (95 of 223 MULTI_CLAIM frames at baseline) remains; its 5 reproducers are still `expectedFailure`.

## Folder move

`~/Documents/SPANA_Engine_Rebuild/R2/_spana_r2_work/` (about 2.0 GB, 11,316 files) was moved with `mv` to
`~/Documents/SPANA_R2_Work/`. It was not opened, deleted or committed. `.gitignore` has a defensive
`R2/_spana_r2_work/` rule.

## Mac storage finding

iCloud "Optimize Mac Storage" had evicted 498 files in the Mac repo's `.git` (mostly objects) and the DINOv2 model
files, and Git on the Mac crashed reading them.
- 100 were restored by staging (99 objects and `description`), and 389 objects by writing the identical object bytes
  from the cloud repository. The 9 unused `hooks/*.sample` files remain cloud-only.
- `git fsck` is clean, and the model files were re-downloaded.
- The Mac working copy is still at `8c798bc`. It holds the F3c `f65c233` files uncommitted and
  `R2/DEV_REMEDIATION_F3C/mac_test_chunks/`, both from before the switch to GitHub-only.
