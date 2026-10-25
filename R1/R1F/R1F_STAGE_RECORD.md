# R1-F stage record: target identity state machine + contrastive association

**Stage:** R1-F. Requested by Nathan on 2026-10-25 at 08:55 Sydney. `R1/R1_GENERIC_TRACKER_REQUIREMENT.md` is binding.

**Built** (generic only: TARGET / PERSON_A / PERSON_B / TWIN in tests; no team, number, colour, clip or benchmark logic):
- `player_tracker_rb/target.py` (NEW): `TargetTracker`, the identity layer for one human-seeded target among generic people. Its module docstring is the specification.
- `player_tracker_rb/population.py` (CHANGED): merged regions now resolve when a body emerges from them (R1-F synthetic finding), with the review fix H2 and the outsider rule (a body tracked before the merge that was not one of its claimants never resolves it).
- `tests_rb/_synth_scene.py` (NEW): rendered synthetic scenes with real, measurable appearance (kit colour + identity signature; an optional pose component).
- `tests_rb/test_target.py`, `tests_rb/test_target_audit.py` (NEW), `tests_rb/test_population_audit.py` (extended).
- `tools_rb/target_eval.py` (NEW): GOLD development-footage identity run (cached DINOv2 embeddings; labels used only after the run).
- `tools_rb/crossing_matrix.py` (NEW): the synthetic crossing matrix.

**Not built:** no events, no ball, no stats, no integration pipeline (R1-G). The event engine was not touched. Task 3R and the holdout were not touched.

**Contracts:**
- Output contract (`output_schema.py`): unchanged. Every target frame is checked against `V2_TARGET_FIELDS` and the D05/D06/D19 invariants in every synthetic scenario.
- Config registry `r1f-2`: 157 parameters (r1e-1 had 154). Added: `identity_contact_cover_min`, `target_bank_sample_interval_s`, `target_known_other_min_s`. `population_merge_emerge_pad_h` existed only in the pre-review commit 3f9dd51 (`r1f-1`) and was removed by the review fix H2. New cross-check: `contact_memory_s ≤ population_closed_memory_s`. Default config hash `8e3b9d6781c0f5b6bd4ddf5dd3afa051061972f567fc2c70a70f187a73c9c62c`. Full list: `R1F_CONFIG_REGISTRY.csv`.
- No new models. GOLD runs use DINOv2 embeddings cached by crop SHA-1 (`SPANA_Engine_Archive/_r1f_emb_cache/`); a cache miss fails closed.

## Association logic

For every candidate detection, every frame (no fast path; applied even when only one box exists):
1. **Merged:** MERGED_SUSPECTED, MERGE_LINEAGE, or merge-sized against the target's own recent median size → can never confirm identity.
2. **Motion (veto only):** with a valid position and a motion prior, the box must be physically plausible (speed and scale bounds in body heights per second). Motion never adds identity evidence.
3. **Appearance (support, never proof):** LAB + DINOv2 composite against the target bank must pass all floors (`identity_support`; contaminated crops fail). Reacquisition uses the stricter composite floor `reacq_appearance_floor`.
4. **Contrast against all competitors:** every population body that could be this box (observed, hidden, recently closed within `contact_memory_s`), using their own banks: `contrast = sim(crop, target bank) − max sim(crop, competitor bank)` must reach the margin.
   - Excluded (the target never competes with itself): the candidate's own tracklet; the candidate's nested group (part-body / alternative boxes); tracklets the target was confirmed on, **unless** such a tracklet is observed this frame on a separate box (then it is a body beside the candidate and competes; review H1).
5. **Unknown competitor in contact:** a touching body with no bank yet (e.g. provisional) whose size could be the target's blocks the candidate: contrast cannot be verified.
6. **Ambiguity:** another in-gate box that also looks like the target, within `ambiguity_margin`, blocks it.

## State-transition rules

| From | Condition | To |
|---|---|---|
| any | continuing tracklet observed and passes 1–6 | TRACKING |
| any | continuing tracklet observed but fails | AMBIGUOUS (reasons) |
| any | tracklet hidden under an occluder / no box in gate | OCCLUDED (OCCLUDER_OVERLAP with ids / NO_CANDIDATE_IN_GATE) |
| any | merged box over the predicted target | OCCLUDED if the predicted centre is inside it, else AMBIGUOUS (MERGED_DETECTION_SUSPECTED) |
| any | tracklet closed: CONTACT_AMBIGUITY / MERGED_CONTACT / LEFT_FRAME / UNMATCHED_TIMEOUT | AMBIGUOUS (COMPETING_CANDIDATE) / D11 / LOST (TARGET_OUT_OF_FRAME) / LOST (AGED_OUT) |
| OCCLUDED / AMBIGUOUS | longer than `max_occluded_s` / `max_ambiguous_s` since the last confirmation | LOST (AGED_OUT), tracklet released |
| any | camera continuity broken (CUT; UNKNOWN by default; estimator reset) | LOST (CAMERA_CUT); motion state, size history and tracklet discarded |
| not TRACKING | reacquisition candidate passing | REACQUIRING (untrusted, REACQUISITION_UNVERIFIED) |
| REACQUIRING | verification complete | TRACKING, `target_episode + 1`, reacquisition event |

Only TRACKING is trusted and only TRACKING has an observed box. Every other state writes a PREDICTED or HELD box and at least one vocabulary reason, with confidence ≤ 0.49.

## Reacquisition rules

- **Candidates:** non-merged, non-alternative boxes that are not KNOWN OTHERS. A KNOWN OTHER is a body seen as a separate box beside the target at the seed (human identification), or while the target was confirmed by continuation on one tracklet for ≥ `target_known_other_min_s` (0.2 s). Known others are cleared on an accepted reacquisition and pruned to tracklets the population still knows.
- **Where:** inside the search gate while position is valid and the absence is < `lost_long_s`; otherwise anywhere (appearance only). After a cut, always appearance only.
- **Stricter than ordinary association:** composite floor `reacq_appearance_floor` instead of the ordinary floor; margin `m(a) = min(reacq_contrastive_margin + reacq_margin_growth_per_s · a, reacq_margin_cap)` instead of `contrastive_margin`, where `a` = seconds since the last confirmation. **Long absence → stricter.**
- **Verification:** the best passing candidate is followed by its tracklet for ≥ `reacq_window_s`, must pass in ≥ `reacq_min_hit_fraction` of the window's frames and in the accepting frame, and no other box (including the target's own continuing box) may be within `ambiguity_margin`. Verification restarts if the tracklet closes or another candidate becomes clearly better; it is abandoned after a window with no hits.
- **After acceptance:** bank quarantine for `post_reacq_quarantine_s`.

## Identity-confidence logic (`identity_confidence_rb`, [0, 1], UNCALIBRATED)

- Seed: 1.0 (human identification).
- TRACKING: `0.5 + 0.5 · min(a, k)`. `a` = appearance margin above its floor, normalised. `k` = contrast margin above the required margin, normalised. With no banked competitor: `k = 1` for continuation (the tracklet is geometric evidence of one body), `k = 0` for an accepted reacquisition (appearance alone; evidence `contrast_available: False`). The weakest piece of evidence decides.
- Untrusted: `min(untrusted_identity_confidence_ceiling, last trusted · 0.5^(t / confidence_half_life_s))`.
- Never folded in: detector score, target similarity, best competitor similarity (reported separately in the decision log).

## Appearance-bank update rules

Only on a TRACKING frame; the population must also consider the box clean (not CONTESTED / PARTIAL / TOUCHES_IMAGE_EDGE / DUPLICATE_ABSORBED / IN_MERGED_REGION / MERGE_LINEAGE / merged / NESTED / ALTERNATIVE, containing no other body); isolated from every other body (IoU ≤ `bank_contact_iou`, covered ≤ `bank_max_covered_fraction`); camera-verified frame; outside the post-reacquisition quarantine; at most once per `target_bank_sample_interval_s`; then `AppearanceBank`'s own quality and self-consistency checks. The seed must anchor the bank or the run fails closed.

## Synthetic results

**Crossing matrix** (`crossing_matrix.json`, `tools_rb/crossing_matrix.py`; final code): 2 kits (same kit at embedding cosine 0.75 / opponent at 0.2) × 3 speeds (2, 5, 9 px/frame at 100 px body height, 30 fps) × 3 vertical offsets (0, 15, 45 px) × 3 detection modes (both visible / detected as ONE merged box / far person missed while ≥ half hidden) = 54 cases.

| Mode | Cases | Silent swaps | End TRACKING | Coverage min | Coverage median |
|---|---|---|---|---|---|
| visible | 18 | 0 | 18 | 0.811 | 0.923 |
| merge | 18 | 0 | 18 | 0.828 | 0.878 |
| hide | 18 | 0 | 18 | 0.812 | 0.867 |
| **all** | **54** | **0** | **54** | **0.811** | **0.878** |

Coverage = TRACKING frames / frames on which the target was detected as its own box. Every target bank stayed pure (only the target's identity) in all 54 cases.

**Scenario tests** (every frame of every scenario is also checked against the v2 contract and the D05/D06/D19 invariants: observed ⇔ TRACKING, exactly one bbox source, trusted ⇒ no reasons, untrusted ⇒ reasons and confidence ≤ 0.49, the target never its own competitor):

| Required scenario | Test(s) | Result |
|---|---|---|
| same-kit crossing / opponent crossing | `test_same_kit_crossing`, `test_opponent_crossing`, crossing matrix | 0 swaps |
| full occlusion | `test_full_occlusion_occluder_never_inherits_identity` | occluder never inherits identity |
| merged detection | `test_merged_detection_cannot_confirm`, H2 regressions | merged box never TRACKING; bank pure |
| target leaves frame | `test_target_leaves_frame_and_returns` | LOST (TARGET_OUT_OF_FRAME), reacquired |
| short / long gaps | `test_short_gap_…`, `test_long_gap_ages_to_lost_then_reacquires`, `test_long_absence_is_stricter` | stricter with absence |
| correct reacquisition | `test_correct_reacquisition_needs_the_full_window`, `test_true_target_chosen_over_look_alike`, verification switch | accepted only after the window |
| wrong reacquisition rejection | same-kit, opponent-at-last-position, body-seen-beside, newcomer-in-contact (H1), flickering, rival during window | rejected |
| camera cut | `test_cut_invalidates_position_and_rejects_impostor_at_same_place`, M1 scale test | position invalid; impostor at the old place rejected |
| ambiguous two-candidate case | `test_two_identical_candidates_stay_ambiguous` | stays AMBIGUOUS |
| provisional competitor | `test_provisional_competitor_blocks_from_birth`, larger unknown body in contact | blocks |
| self-competition prevention | `test_target_never_competes_against_itself`, nested part box, invariant on every frame | never |
| appearance-bank contamination | quarantine, CONTAINS_OTHER, isolation, edge, unverified camera, untrusted frames, pose-growth purity | pass |

## GOLD identity observations (development footage only; no events evaluated)

Run v4 on the final code (`gold_identity/gold_identity_run_v4.json`, contact sheet alongside; decision-identical to v3: same transitions, same states and boxes in the label window, same 10 bank samples). Config r1f-2. GOLD human labels v1 (frozen) were read **only after** the run, to classify states per label class. 180 frames; runtime ≈ 7.6 ms/frame with cached embeddings.

| | v2 (pre-review, r1f-1) | v3 = v4 (post-review, r1f-2) |
|---|---|---|
| States | TRACKING 103, AMBIGUOUS 44, REACQUIRING 28, LOST 5 | TRACKING 103, AMBIGUOUS 33, OCCLUDED 11, REACQUIRING 28, LOST 5 |
| Labelled HIDDEN / NOT_RESOLVABLE (5 frames) | 0 TRACKING | 0 TRACKING (all OCCLUDED) |
| Labelled MERGED_WITH_OPPONENT (6) | 0 TRACKING | 0 TRACKING (OCCLUDED 4, AMBIGUOUS 2) |
| Labelled PARTIAL (4) | 3 TRACKING | 3 TRACKING |
| Labelled VISIBLE (26) | 8 TRACKING | 8 TRACKING |
| Reacquisition | rel 123, hit fraction 0.625, uncontrasted | same frame, now flagged `contrast_available: False`, confidence 0.5 |

- **Zero TRACKING claims on frames the human labelled hidden, not resolvable or merged.**
- Visual check of the v3 contact sheet (v4 is identical): every TRACKING box is on the target; the REACQUIRING verification box is on the target. No wrong identity was seen.
- Coverage on frames labelled VISIBLE is low (8/26). The misses are honest abstentions: after the contact at rel 81 the opponent's own bank is about as similar to the target's crop as the target bank is (INSUFFICIENT_CONTRAST), so the target cannot be re-confirmed until verification completes at rel 123. This was **not tuned**.
- From rel 145 (outside the label window) the target goes AMBIGUOUS (BELOW_APPEARANCE_FLOOR, COMPETING_CANDIDATE), then LOST at 175. Unlabelled; not evaluated.
- One label window on one clip; this is development evidence, not a benchmark.

## Independent review (project subagent policy)

### Code reviewer (on 3f9dd51): 11 findings, all validated
| Id | Finding | Validation | Fix |
|---|---|---|---|
| H1 | A newcomer touching the visible target was reacquired AS the target (frame 35): the target's own observed tracklet was excluded from the newcomer's competitors and rivals. | Reproduced (silent swap). | Target tracklets observed this frame on a separate box compete; reacquisition rivals include the continuing box. Regression tests (cos 0.78 / 0.9 / 1.0, edge twin). |
| H2 | An unrelated newcomer beside a merged box "resolved" the merge; the target then TRACKED the two-person box (rel 101–149) and its bank became impure. | Reproduced. | Population: emergence needs real overlap with the region (pad removed); a still-merge-sized box in the region does not count as a resolved body. Target: D11 size test against the target's own size. Regression tests at both levels. |
| M1 | A cut did not reset scale history. | Confirmed by reading; test added. | Cut clears sizes, last height and tracklet time. |
| M2 | Known others recorded after one trusted frame. | Confirmed. | Gate `target_known_other_min_s` (seed exempt: human identification). |
| L1 | Seed refusal left a half-initialised tracker. | Confirmed. | `_failed` flag; `update()` refuses. |
| L2 | Unbounded `_known_others` / `_target_tracklets`. | Confirmed. | Pruned each frame to tracklets the population still knows. |
| L3 | Gate freeze keyed on state not on ambiguity start. | Confirmed. | Uses `_ambiguous_since_t`. |
| L4 | Stale reasons on some frames. | Minor; reasons are a superset of causes. | Left as is (documented). |
| L5 | Occluder ids could include target tracklets. | Confirmed. | Excluded. |
| L6 | Motion veto applied without a prior. | Confirmed (docstring mismatch). | Only with a prior. |
| L7 | `target_eval` could index labels outside the run (negative index wraps). | Confirmed. | Fails closed. |

### Test auditor (on 3f9dd51)
- 44 mutants: 19 caught by the original suite, 34 with the auditor's probes. Probes adopted into `test_target_audit.py` (gate probe adapted to the real freeze reference).
- A1 = H1. A2/A3: a lone same-kit look-alike with no competitor bank is accepted at embedding cosine ≥ 0.78 (see KNOWN LIMITS). A4: reason logging (minor).
- Vacuous tests fixed: quarantine (now must exercise the path), CONTAINS_OTHER bank refusal (now asserted), nested part box (now actually produces one).
- Post-fix mutation re-run: the auditor's 44 mutants were re-applied to the reviewed code (5 no longer applied because the code they targeted had been rewritten; 14 new mutants were written for the new code). 10 survived the first re-run. Each was examined:
  - 7 exposed rules that other layers also protect (defence in depth). Each now has its own test (listed below), and each of those mutants is now caught.
  - 1 (P02) exposed a **real defect** (outsider resolves a merge; see below), now fixed and caught.
  - 2 are equivalent (M33, M40; reasons below).
  Of the 14 new mutants, 5 survived at first: N04b and N07b (each H1 layer alone is sufficient; both are now pinned separately), N_known_other_ungated, N_cut_keeps_scale (both now caught) and N_rival_role (equivalent: the rival test does not read the margin). **Final: every non-equivalent mutant is caught; 3 equivalent mutants remain, with reasons recorded.**

### Found during the post-review mutation re-run (each reproduced before the fix)
- **Outsider resolves a merge (population; mutant P02 survived because the H2 fix had made its test vacuous).** A body tracked elsewhere BEFORE a merge formed, walking INTO the merged region, was counted as one of the region's bodies. The region resolved and the still-merged visible box lost MERGE_LINEAGE (so it could become a bank sample or confirm identity). Fix: an OUTSIDER (tracked before the merge, not a claimant) never counts toward resolution, inside or emerging. The test was rebuilt so that it fails without the rule.
- **Decision-log role:** rival boxes evaluated beside the continuation were logged with role CONTINUATION. They now have role RIVAL (ordinary criteria; behaviour unchanged, the log is now truthful).
- Rules pinned by new tests because other layers also protect them (defence in depth): the motion veto (applied and logged), the target-level bank isolation refusal, contact memory for closed competitors (equivalent under default config, where `contact_memory_s` = `population_closed_memory_s`), the immediate verification switch to a clearly better candidate, contact in both directions for a larger unknown body, a rival during the whole window prevents acceptance, and the `_looks_like_target` contrast condition (unit test; equivalent behaviourally because synthetic similarity is symmetric).
- Judged equivalent (no test can distinguish them in this design, reasoning recorded): M33 (the seed's merged-flag check is unreachable on the first frame, where the population has no history; the seed-box size test is the real guard), M40 (re-evaluating the continuing box as a reacquisition candidate cannot pass where continuation failed, because the reacquisition criteria are stricter).

### Found while writing the regression tests
- **Seed contamination:** on the first frame the population has no size history, so a two-person detection matching a human seed box at IoU 0.57 was accepted as the seed and anchored the bank with two people. Fix: the seed box is the single-body size reference; a merge-sized detection is refused. Test added.

## Negative results (project rule 12)

| Hypothesis / change | Observed | Decision |
|---|---|---|
| v1: any unknown box in contact blocks the target. | GOLD v1: a 46×31 fragment (ball/leg) inside the target box blocked it for many frames. | Unknown contacts must be able to be the target by scale (motion scale bound). |
| Merge lineage permanent. | 10–12 of 54 crossing cases ended LOST after "hide" crossings. | Regions resolve by emergence. |
| Emergence by touching a padded region (`population_merge_emerge_pad_h` 0.25 h). | Review H2: an unrelated newcomer resolved a real merge → the target tracked a two-person box. | Pad removed; overlap required; test hide mode changed to the realistic "missed only while ≥ half hidden". |
| Known others only after 0.2 s of trusted continuation (M2). | A same-kit twin beside the seed made the target AMBIGUOUS from frame 1, so the twin was never recorded and was later reacquired as the target (135 swap frames). | The seed frame records known others (human identification). |
| H2 fix ("merge-sized in-region boxes do not count"). | Made the old pre-existing-body test pass trivially; mutation re-run then exposed the outsider-walks-in defect. | Outsider rule + rebuilt test. |
| Behavioural test that a motion-vetoed rival is not a rival. | The twin's own bank made contrast fail first; the test could not isolate the rule. | Unit test of `_looks_like_target` + evidence-log test of the veto. |
| Uncontrasted reacquisition at full confidence. | A2: a lone look-alike could be accepted with high confidence. | Confidence capped at 0.5 and `contrast_available: False` exposed. Acceptance itself kept (see KNOWN LIMITS). |

## Tests

- **379 tests pass on the Mac** (models required, none skipped): `R1F_TEST_OUTPUT_mac.txt`. In the cloud: 379, of which 4 are skipped there because the models / GOLD video are only on the Mac.
- New in R1-F: 69 tests (R1-E: 310). `test_target.py` 33 (scenarios), `test_target_audit.py` 31 (review regressions, audit probes, layer pins, documented-limitation evidence), `test_population_audit.py` +5 (emergence / H2 / outsider).
- Package hygiene (no numeric literals outside `config.py`, no benchmark terms in `player_tracker_rb`, units) passes.
- Determinism (`test_deterministic`) and fps invariance (`test_fps_invariance_of_decisions`) pass.

## Known limits

1. **Uncontrasted reacquisition (A2).** When no other banked person is around, reacquisition rests on appearance + verification (+ position while valid). A same-kit look-alike whose composite similarity reaches `reacq_appearance_floor` can be accepted as the target. Synthetic: from embedding cosine ≈ 0.78 (composite ≈ 0.86). Real same-kit data (R1-D, n = 109 LAB-confusable pairs): P99 0.855, max 0.867. So on real footage this is a tail risk, not hypothetical. Such acceptances are flagged (`contrast_available: False`, confidence 0.5, reacquisition event evidence). After acceptance, continuation confidence rises normally. The GOLD reacquisition at rel 123 is of this kind (correct, by visual check).
2. **Interrupted verification blocks the real return** until the interrupted target-looking tracklet leaves closed memory (`population_closed_memory_s`): conservative coverage loss (pinned by a test).
3. **Low GOLD coverage** after contact (8/26 labelled-visible frames TRACKING): the opponent looks similar to DINOv2 at this resolution.
4. Identity confidence is **UNCALIBRATED**.
5. Synthetic appearance is idealised (exact signatures); real crops are noisier. One real clip only.
6. L4 (some reason lists can include stale causes) left as is.

## Persistence

- Source: `~/Documents/SPANA_Engine_Rebuild/` (the Mac source of truth). Commits: `3f9dd51` (implementation, before review), `5944bf9` (review fixes), `6cc5d56` (mutation-gap tests), and the final R1-F commit holding this record. The commit hash, snapshot manifest hash and GitHub verification are recorded in READ FIRST and in the snapshot's `ARCHIVE_VERIFICATION.txt` (a record cannot contain its own commit hash).
- Diff vs R1-E (`b60245e`): `R1F_DIFF_code.patch`. Config registry: `R1F_CONFIG_REGISTRY.csv`. Mac test output: `R1F_TEST_OUTPUT_mac.txt`.
- GOLD runs: `gold_identity/gold_identity_run_v1…v4.json` with contact sheets (v1 and v2 pre-review, kept as history). The embedding cache is in `SPANA_Engine_Archive/_r1f_emb_cache/` (not in Git).
