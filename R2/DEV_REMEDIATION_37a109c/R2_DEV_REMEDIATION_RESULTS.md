# R2 DEV remediation: F1 seed interface + F3 merged guard (2026-10-30/31)

**Instruction (Nathan, 2026-10-30 17:41 Sydney):**
- Part 1: replace the IoU-only seed association (F1, and F7 at the seed interface) with a generic rule.
- Part 2: diagnose and fix the merged-guard false positives (F3).
- DEV only: rerun the 11 scorable DEV clips; C019 stays excluded. Do not run EVAL.

**EVAL was not run, opened or scored. No frozen label, fixture or holdout was changed. No threshold was tuned on these
results:** every rule was fixed, reviewed and committed before this DEV run, and nothing was changed after seeing it.

## What was run

| Item | Value |
|---|---|
| Tracker commit | `37a109c` (clean tree; every eval records `results_git_dirty: false`) |
| Baseline (pre) | `ef6d635` run, `R2/DEV_RUN_ef6d635` (tracker identical to R1-G `30f2e39`) |
| Config | registry `r2a-1`, tracker `0.8.0-r2a` |
| Models | YOLOX-s `c5c2d13e…8063`, DINOv2 ViT-S/14 ONNX `0ab62713…efea` (same as baseline) |
| Clips | the 11 identity-scorable DEV clips, sha256 = registry = fixture (runner verifies) |
| Environment | cloud, Linux x86_64, 2 cores, Python 3.13.17, onnxruntime 1.23.2, numpy 2.2.6, opencv-headless 5.0.0.93 (same as the baseline run; `ENVIRONMENT*.txt`) |
| Scorer | `tools_rb/tracker_eval.py` (r2_v1), unchanged |
| Comparison | `tools_rb/r2_dev_compare.py` → `R2_DEV_PRE_POST_COMPARE.json` (diagnostics never replace an official count) |

## F1 — seed association

**Root cause.**
- Human seed boxes are systematically looser than detector body boxes.
- On the 4 refused clips, the detector box lay wholly inside the human box, with 0.62–0.82 of its height and 0.55–0.68
  of its width.
- That gives an IoU of 0.42–0.46, under `seed_min_iou` 0.5.
- The first failing layer is target-identity seeding: the interface between the human box and the detector box. It is
  not detection; the detector found the player.

**New generic rule** (`target.seed`, `_seed_geometry`; all ratios are relative to the seed box, so the rule is
scale-free):
1. **IoU path (unchanged):** IoU ≥ `seed_min_iou` 0.5.
2. **Otherwise, the CONTAINMENT path:** exactly one body detection must pass all of these:
   - containment ≥ 0.9;
   - |centre offset| ≤ 0.15 seed heights (x and y);
   - height ratio ≥ 0.5;
   - width ratio ≥ 0.45;
   - |feet-line gap| ≤ 0.2 seed heights.

   In addition, **no other body** may have ≥ 0.25 of its area inside the seed box.
3. **Otherwise:** SeedError, failing closed. Ambiguous boxes, two-person boxes, off-centre bodies, smaller or farther
   people and partly contained bodies are all refused.

The bank-anchor isolation check (F2) is unchanged and still applies after association.

**DEV seeds:**
- **Seeded by CONTAINMENT:** C006, C016, SYO C004, SYO C016.
- **Seeded by IOU, as before:** C020, C033, SYO C012, SYO C021.
- **Still refused at the bank anchor (F2, unchanged, out of scope):** C034 (PARTIAL, TOUCHES_IMAGE_EDGE), C038
  (ALTERNATIVE, NESTED, NOT_ISOLATED), SYO C032 (NOT_ISOLATED).

## F3 — merged-detection guard

**Root cause.** The first failing layer is the population merged-detection guard.
- The size-only tests (AREA_RATIO and ASPECT_WIDE against the 1 s median size) fired on a single body's pose-driven
  widening. In C020 the width swung 49–115 px at a constant height of about 222 px, with camera scale 1.0.
- Cascade: the tracklet closed with MERGED_CONTACT. A merged region needing two bodies to resolve then never resolved
  (UNRESOLVED_MERGE / PERSISTENT_MERGE). The body's next tracklet carried MERGE_LINEAGE for life, so the target was
  permanently lost.

**New rule (v5).** An uncorroborated size change on a body continuing its own tracklet becomes a **SIZE_ANOMALY**.
While it lasts:
- the box is never trusted or banked (both banks refuse it, and the target abstains);
- the tracklet is **not** closed;
- the size reference is held (camera-scaled);
- no velocity or size is learnt from it.

It becomes a **merge**, with the full conservative region and lineage logic unchanged, in any of these cases:
- corroboration: another detection touches it, another live tracklet's predicted box touches it, or a recently closed
  tracklet's last box touches it, provided that box's position is still valid (not across a cut);
- the area reaches ≥ 2.5× (no single body explains it);
- it stays merge-sized for `merged_anomaly_max_s` (1.0 s) from the anomaly start;
- a second body touches a box that is not near single size.

It **ends** only after `merged_anomaly_clear_s` (0.2 s) of *continuously observed* boxes that are near single size
(area and aspect ≤ 1.25×), not partial and not at the image edge.

Target side: when the target's own tracklet ends an anomaly, the target's continuity is broken. The target returns only
through reacquisition, as a new episode that is untrusted and never banked until contrast-verified. Reason: an unseen
same-kit body may have replaced the target inside the union, and geometry cannot tell.

**Not F3.** F3 was proven not to be upstream of F4: in SYO C012 only 5 of 253 untrusted frames carry the merged
reason, and COMPETING_CANDIDATE dominates.

## Pre/post DEV results (official scorer, unchanged)

| Clip | Pre (`ef6d635`) | Post (`37a109c`) |
|---|---|---|
| C006 | REFUSED (F1) | P1 3, P2 0, P3 0, P4 0, cov 0.201 (57/284) |
| C016 | REFUSED (F1) | P1 0, P2 0, P3 0, P4 0, cov 0.946 (158/167) |
| C020 | P1 0, cov 0.376 (214/569) | P1 0, cov 0.376 (214/569) |
| C033 | P1 1, P3 1, P4 1, cov 0.249 (104/417) | identical |
| C034 | REFUSED (F2) | REFUSED (F2) |
| C038 | REFUSED (F2) | REFUSED (F2) |
| SYO C004 | REFUSED (F1) | P1 0, P2 0, P3 0, P4 0, cov 0.764 (97/127) |
| SYO C012 | P1 0, cov 0.004 (1/254) | identical |
| SYO C016 | REFUSED (F1) | P1 0, P2 0, P3 0, P4 0, cov 0.651 (136/209) |
| SYO C021 | P1 1, cov 0.654 (51/78) | identical |
| SYO C032 | REFUSED (F2) | REFUSED (F2) |

| Aggregate (11 scorable DEV clips) | Pre | Post |
|---|---|---|
| Clips initialised | 4 | **8** |
| Refused at the seed | 7 (F1 ×4, F2 ×3) | 3 (F2 ×3) |
| P1 false confident identity (official) | 2 | **5** |
| P2 identity swaps (where measurable) | 0 | 0 |
| P3 wrong reacquisitions | 1 | 1 |
| P4 trusted on hidden / merged / unresolvable | 1 | 1 |
| Trusted coverage, initialised clips | 0.281 (370/1318) | 0.389 (818/2105) |
| Trusted coverage, all 11 scorable clips | 0.127 (370/2921) | **0.280 (818/2921)** |
| Correct reacquisitions | 0 | 0 |
| Target frames with MERGED_DETECTION_SUSPECTED | 985 | 817 (618 on the 4 clips run in both) |
| Target-bank samples accepted | 30 | 72 |
| Bank samples flagged by the IoU diagnostic | 2 | 10 |
| Bank samples confirmed contaminated | 0 | **0** |

### P1 (false confident identity): the official count rose from 2 to 5

All 3 new P1 frames are in C006 (frames 391–393), a clip that was refused before. The official count stays 5.

Diagnostics (`P1_AND_BANK_FLAG_GEOMETRY.txt`) plus a visual check of the DEV frames:
- At frames 391–393 the tracker box is on the **correct player**, the one the human box frames. It covers only his upper
  body (legs cut off): containment 1.0, centre offset ≤ 0.04 seed heights, IoU 0.27–0.29 < 0.3.
- So this is box-extent error (F7 class) on the right person, not a wrong person.
- The 2 pre-existing P1 frames (C033 f105 = F6; SYO C021 f36 = F7) are unchanged.

**Confirmed false trusted identity: 0 new.**

### Bank contamination

The 10 accepted samples flagged by the diagnostic (IoU < 0.5 with the human box) are all on the target:
- 9 are wholly contained (containment 1.0) and centred (≤ 0.06 heights), with no other detection in the human box.
- 1 (C016 f390, containment 0.52) was checked visually: the tracker is on the target (#5). The camera zoomed at that
  moment, and the human box for that frame is shorter than the player.

**Confirmed contamination: 0.**

**Trusted-frame containment** (all human-boxed trusted frames, `TRUSTED_FRAME_CONTAINMENT.txt`): every trusted frame
is centred on the human box. A few have containment of 0.5–0.89, explained by box extent. Beyond the counted P1
frames, no trusted frame is off the human box.

### Merged guard: frequency and effect

- **C020** (the measured F3 clip): merged-reason frames fell from 355 to 61, and the frames now carry SIZE_ANOMALY (an
  abstention). The target is **no longer lost**: 0 LOST frames, against 295 before.
  - It is followed (TRACKING) on 262 available frames, on the right person in 54 of 56 human-boxed frames.
  - Those frames stay **untrusted** with INSUFFICIENT_CONTRAST. The new episode after the anomaly is never
    contrast-verified, so trusted coverage is unchanged at 0.376.
  - This is the accepted safety cost of the v4 continuity break. The remaining blocker is post-reacquisition
    verification, not the merged guard.
- **SYO C021:** merged-reason frames fell from 137 to 64 (SIZE_ANOMALY abstentions). Coverage is unchanged.
- **C033** (F3b, NOT FIXED; a distinct mechanism, see below): unchanged, with 488 merged-reason frames.
- **C006:** a genuine overlap with a same-kit teammate from about f110. The human truth labels contact with P1 from
  f125. The abstention is correct; the failure afterwards (never re-trusted) is the reacquisition class F5.

## Safety gates

| Gate | Result |
|---|---|
| No increase in **confirmed** false trusted identity | **PASS**: 0 new confirmed. The official P1 count rose 2 → 5 (all 3 new frames are on the right person, partial-body box; disclosed, not erased). |
| No silent swaps | **PASS**: P2 = 0 where measurable. |
| No appearance-bank contamination | **PASS**: 0 confirmed. 10 flagged by the IoU diagnostic, all on the target. |
| No clip-specific production logic | **PASS**: independent review (rounds 1–4) found none; all rules are in seconds and seed- or body-relative ratios. |

## Remaining failures after this stage

| Class | Clips | Status |
|---|---|---|
| F2 seed crop not isolated / at edge | C034, C038, SYO C032 | unchanged (out of scope) |
| **F3b** MULTI_CLAIM with a duplicate tracklet of the target's own body | C033 | **new sub-class, not fixed** (detail below) |
| F4 competing-candidate abstention that never recovers | SYO C012 | unchanged (F3 is not upstream) |
| F5 no reacquisition / never re-verified | C033 (return after f295); C006 (after the genuine overlap); C020 (post-anomaly episode never contrast-verified) | open; 0 correct reacquisitions on DEV |
| F6 one-frame trust lag into an unresolvable frame | C033 f105 | unchanged |
| F7 box-extent convention | C006 f391–393 (P1), SYO C021 f36 (P1); box IoU overall | evaluation interface; counts stand |

**F3b (C033).**
- At f112 the target (T005, a partial box at the right image edge, leaving the frame) and T026, a tracklet born earlier
  on a nested part box of the same body, both claim one detection.
- MULTI_CLAIM fires and the target's tracklet closes with MERGED_CONTACT.
- The first failing layer is population tracklet birth on a NESTED / part box: the same body is tracked twice.
- F3b may be upstream of the C033 F5 failure, because the merged reason is present during the 70 REACQUIRING frames.
  **UNKNOWN until F3b is diagnosed.**

## Negative results (recorded; rule 12)

| Version | Commit | Hypothesis | Observed | Decision |
|---|---|---|---|---|
| v1 | `cf9d57c` | Size evidence counts as a merge only if corroborated or growth is abrupt | Review H1: an unseen second body growing gradually became trusted. H2: a nested second-person box was ignored. H3: the abrupt test depended on frame rate. | Replaced |
| v2 | `9313201` | SIZE_ANOMALY abstains; reference frozen; escalation at 1 s; nested-tall corroboration | Round-2 H1: one near-size or partial frame ended the anomaly. M1: a tight nested box of the same body caused merges (DEV logs: tall same-body nested boxes about 1 per frame). | Replaced |
| v3 | `d4fc666` | Sustained clearing (0.2 s), escalation from anomaly start, nested corroboration removed | Round-3 H1: after an anomaly ended, trust resumed on the same tracklet without re-verification; a same-kit body could be trusted and banked. M1: partial frames escalated to a merge. | Replaced |
| v4 | `656a272` | Continuity break at the end of an anomaly; escalation needs merge-sized evidence or a touching body | Round-4 M1: pre-cut memory boxes corroborated merges after cuts. L1: stale anomaly marker caused a second reacquisition. L2: clearing continued across a detection gap of 0.13–0.17 s. | Fixed in v5 |
| DEV runs stopped as obsolete | `cf9d57c`, `9313201`, `d4fc666`, `656a272` | n/a | Code changed after review; partial outputs archived (cf9d57c completed C006 and C016; 9313201, d4fc666 and 656a272 completed no clip). None is used as a result. | Not reported as results |

The reviewer's L2 repro (a 0.3 s gap) actually starts a new tracklet, which is conservative. The real L2 window is a gap
of 4–5 frames (0.13–0.17 s) on the same tracklet. This was established before the test was written.

## Tests

- Cloud full suite at `37a109c`: **601 passed, 4 skipped** (`TEST_OUTPUT_cloud_37a109c.txt`). Mac unittest chunks all
  OK.
- Synthetic tracker eval (`tools_rb/r1h_synthetic_eval.py`): identical to the R1H baseline (0 metric differences).
- Regression tests: each round-3 and round-4 finding has a test that fails on the previous commit and passes here.
- **Mutation testing** (`MUTATION_TEST_LOG.txt`): 40 single-point mutations of `seed()` and the anomaly logic.
  - **38 caught.** At `656a272`, before the round-4 audit tests, 21 of 37 were caught.
  - **2 survivors, both equivalent by construction:**
    - S4 (one-sided feet check): containment ≥ 0.9 bounds the feet below the box to ≤ 0.11 seed heights, inside the 0.2
      limit.
    - S9 (pick the last candidate): unreachable, because more than one candidate is refused before any choice is made.

## Known limits (carried forward)

- **Sub-threshold unions:** union frames below the area / aspect test (< 1.6×) are trusted. These are the same 8
  synthetic frames as at `9ae0ce2`, about 0.28 s at any frame rate.
- **Sustained wide pose:** a wide pose held longer than 1.0 s with merge-sized boxes still escalates to merge and
  lineage (a coverage cost).
- **Pre-existing P1, same-kit occluder:** an unseen same-kit occluder stepping exactly over the target with no union is
  trusted, identical at `9ae0ce2`. Also pre-existing: an unverified episode can follow a same-kit replacement *untrusted*
  (no trusted frame, bank pure; identical at `9ae0ce2`).
- **Merged blob at the seed frame:** undetectable by geometry; relies on the labelling rule that a seed frame has no
  contact.
- **Seed competitor count:** the seed competitor test counts the same body's own NESTED / ALTERNATIVE boxes, which can
  refuse a seed. This fails closed (coverage only), and no DEV seed was refused by it.
- **Decision log:** additive debug keys (`seed_rule`, `continuity_broken`, `second_body_possible`). Their version is
  given by the sibling provenance (`tracker_version`, `config_hash`); the log itself carries no format version.
- **Post-anomaly coverage:** after any pose anomaly the target is an unverified episode until contrast-verified, a
  coverage cost accepted for safety (C020: 262 followed-but-untrusted frames).
- **Uncalibrated rule bounds:** the clear and escalation bounds (0.2 s, 1.25×, 1.0 s) and the seed bounds are
  DESIGN_CHOICE, not calibrated.

## Raw outputs

The full per-clip outputs (results, provenance, decision logs; 83 MB) are archived with SHA-256 in
`RAW_OUTPUTS_SHA256.txt`. Location: `SPANA_Engine_Archive/_r2_work/` (see the READ FIRST document).
