# R1-G stage record: full tracker integration + synthetic end-to-end tests

**Stage:** R1-G. Requested by Nathan on 2026-10-25 at 20:40 Sydney. `R1/R1_GENERIC_TRACKER_REQUIREMENT.md` is binding. No event logic; the event engine was not touched; Task 3R and the holdout were not touched.

## Integrated pipeline

`player_tracker_rb/run_rb.py` (NEW) is the R1 Pass-1 entry point. It is the historical `run_mo` signature minus `resume` (approved plan item 10):

```
run(video_path, start_frame, init_box, jersey_number=None, max_frames=None, out_dir=..., debug=False,
    config_overrides=None, *, detector=None, embedder=None, expected_sha256=None, ...) -> summary
```

| Step | Component | Notes |
|---|---|---|
| 1 | `video_io.SequentialVideoReader` | One positioning, then sequential decode. The source is hashed before the run and re-hashed at close. |
| 2 | `DetectorBackend` | YOLOX-s ONNX, pinned. |
| 3 | `camera.CameraMotionEstimator` | Pan / zoom / cut. |
| 4 | `population.Population` | Generic tracklets, merges, contacts, competitors. |
| 5 | `target.TargetTracker` | Contrastive identity of the one seeded target. |
| 6 | `output_schema` builders | TARGET record T001, POPULATION records, frame v2. |
| 7 | Per-frame contract validation | `require_v2=True`; any violation stops the run (fail closed). |
| 8 | Streaming writer | Frame records stream to a partial file. The canonical document is assembled at the end. |

Every file is written under a temporary name and published with an exclusive hard link, in this order:
1. the decision log (debug);
2. the results document;
3. the provenance sidecar, **last**. Its presence marks a complete run, and it holds the SHA-256 of the results file.

A failed or refused run deletes only what it created.

Outputs:
- `{base}_mo_tracking_results.json`;
- `{base}_rb_video_provenance.json`: backends and settings, model hashes, git commit, frame-alignment digest, results hash;
- `{base}_rb_decision_log.jsonl` (with `debug=True`): per-frame target decision, camera and population.

Field-by-field semantics: `R1/R1G/R1G_OUTPUT_MAPPING.md`.

**Backends are config-bound.** A detector or embedding backend may be passed in only if its name and every reported setting equal the run config:
- name, score/NMS, tiling, input size and model hash for the detector;
- model hash, input size and pooling for the embedder.

A production-named backend must also report its model record. Anything else is refused.

## Output contract result

- **Contract unchanged:** v2 schema `2.0.0-r1a` (approved in R1-A). No key, type, unit or vocabulary was added or redefined.
- **Serialisation fix:** `camera_shift` and `camera_shift_h` never serialise `-0.0`. This is behaviour-neutral.
- **Tracker version:** `0.7.0-r1g`.
- **Validation in every synthetic run and in GOLD:**
  - `validate_document(require_v2=True)` returns no errors;
  - `validate_document(strip_v2(doc))` (historical contract) returns no errors;
  - the streamed file is byte-identical to `dumps_canonical` of the whole document.
- **Surviving Pass-2 consumers.** The unmodified recovered `event_adapter` and `actor_attribution` (hash-pinned, run with test-only stubs) read the R1 output safely:
  - a snapshot is TRACKING exactly on R1 TRACKING frames;
  - every frame that is not R1-trusted has a snapshot confidence below 0.5;
  - `frame_identity` trusts only R1-trusted frames;
  - logged identity conflicts reach `conflicts_involve`;
  - a look-alike accepted without verification is never attributed as actor TARGET_PLAYER.

## Identity changes made in R1-G

### Uncontrasted / unverified reacquisition (your instruction, plus review H1 and L4)

**Definition.** An episode is contrast-VERIFIED in either case:
- from the seed (human identification);
- once the target beats, by the contrast margin, a competitor that could be confused with it. "Confusable" means the competitor's bank similarity to the crop reaches `appearance_floor` (0.80).

**Unverified episode.** A reacquisition accepted any other way opens an UNVERIFIED episode. Its observed frames are written as:
- `rb_state` TRACKING (the box really is observed: ACTIVE / CONFIRMED_DETECTION);
- `identity_trusted = false`, with reason `INSUFFICIENT_CONTRAST`;
- `identity_confidence_rb` and the historical `identity_confidence` / `overall_track_confidence` at most 0.49.

**Other effects of an unverified episode:**
- it never updates the target bank (refusal `EPISODE_UNVERIFIED`);
- it never records known others;
- the reacquisition event carries `contrast_available` (any banked competitor contrasted) and `contrast_verified` (a confusable one).

**Why not the first implementation's 0.5 cap.** A 0.5 cap sits exactly on the surviving `actor_attribution` threshold, so a possibly wrong identity was attributed as actor HIGH (review H1, reproduced).

**Why "confusable".** Beating an obviously different person (an opponent at similarity 0.13) says nothing about an unseen look-alike (review L4).

### Physical-entity exclusion (project rule 28)

See `run_rb.possibly_target`. A tracklet that is, or may physically be, the target is never written as a separate person:
- the TRACKING tracklet;
- while not TRACKING, every tracklet the target was confirmed on (a tracklet never survives contact with another body);
- the continuing tracklet;
- the candidate under verification;
- the target's coasting tracklets;
- while not TRACKING with a valid position, a tracklet born after the last confirmation that lies on T001's reported box (re-emergence; test audit C1).

A rejected look-alike elsewhere, and (while TRACKING) an old target tracklet observed as a separate box, are written as people (review M1). This is checked in every frame of every synthetic scenario, including coasting records.

## Determinism

Determinism was checked on synthetic clips (`tests_rb/test_pipeline_run.py`, `DeterminismTooling`) and on GOLD.

**GOLD.** Two runs from the clean commit `30f2e39` (cloud, onnxruntime 1.29, production YOLOX-s + DINOv2 backends):
- documents identical except the two documented wall-clock fields (`run_rb.WALL_CLOCK_FIELDS`);
- decision logs byte-identical;
- results document sha256 (run A) `9dde1440…91e7`; all output hashes are in `gold_pipeline/RUN_OUTPUT_HASHES.sha256`;
- see `gold_pipeline/determinism_compare.json`.

**Process note (recorded, not hidden).** My first run B was made after I had added an untracked file to the cloud tree. Its provenance correctly recorded `git_dirty: true`, and the compare reported that as the first difference (`gold_pipeline/determinism_compare_run_b_dirty_tree.json`). With `git_dirty` normalised, nothing else differed. Run B was then repeated from a clean tree, giving the result above. This shows the provenance block does detect an unclean tree.

## Synthetic identity results

15 scenarios, each a lossless synthetic clip run through the full pipeline (`tests_rb/test_pipeline_e2e.py`). Every scenario passes the universal checks:
- R1 contract, and the historical contract after `strip_v2`;
- 0 silent swaps;
- target never written twice; target body never written as another person (rule 28), checked in every frame;
- bank never takes a non-target sample;
- never trusted without its own detection (hidden / merged / dropped);
- untrusted confidence < 0.5;
- observed vs predicted/held flags consistent.

Tabulated on the Mac by `tools_rb/synth_scenario_summary.py` → `R1G_SYNTHETIC_SCENARIOS.json`. "cov" is the trusted share of frames on which the target's own detection exists (project rule 26: errors **and** coverage).

| Scenario | Visible | Trusted | cov | Observed but unverified | Swaps | Rule-28 dup | Bank violations | Reacq (verified) |
|---|---|---|---|---|---|---|---|---|
| same_kit_crossing | 114 | 94 | 0.825 | 0 | 0 | 0/0 | 0 | 1 (1) |
| same_kit_crossing_hidden | 112 | 94 | 0.839 | 0 | 0 | 0/0 | 0 | 1 (1) |
| same_kit_crossing_merged | 110 | 94 | 0.855 | 0 | 0 | 0/0 | 0 | 1 (1) |
| opponent_crossing | 114 | 54 | 0.474 | 40 | 0 | 0/0 | 0 | 1 (0) |
| opponent_crossing_hidden | 112 | 54 | 0.482 | 40 | 0 | 0/0 | 0 | 1 (0) |
| full_occlusion | 112 | 43 | 0.384 | 48 | 0 | 0/0 | 0 | 1 (0) |
| leave_and_return | 79 | 18 | 0.228 | 45 | 0 | 0/0 | 0 | 1 (0) |
| wrong_reacquisition | 19 | 18 | 0.947 | 0 | 0 | 0/0 | 0 | 0 |
| hard_cut_impostor | 127 | 37 | 0.291 | 75 | 0 | 0/0 | 0 | 1 (0) |
| pan_crossing | 114 | 94 | 0.825 | 0 | 0 | 0/0 | 0 | 1 (1) |
| gap_self_competition | 79 | 17 | 0.215 | 47 | 0 | 0/0 | 0 | 1 (0) |
| newcomer_in_contact | 87 | 17 | 0.195 | 0 | 0 | 0/0 | 0 | 0 |
| twin_appears | 87 | 17 | 0.195 | 0 | 0 | 0/0 | 0 | 0 |
| uncontrasted_return | 57 | 7 | 0.123 | 35 | 0 | 0/0 | 0 | 1 (0) |
| posed_crossing | 152 | 73 | 0.480 | 21 | 0 | 0/0 | 0 | 1 (0) |

**How this maps to your acceptance criteria:**
- **Wrong reacquisition rejected:** the look-alike is never accepted as the target (`wrong_reacquisition`: 0 reacquisitions). An above-floor impostor at the old place after a cut is never trusted (`ImpostorAfterCut`).
- **Camera cuts reset continuity:** `hard_cut_impostor` ends LOST or REACQUIRING with CAMERA_CUT; the target is regained only through appearance-only reacquisition.
- **Target never competes with itself:** `gap_self_competition`, plus the universal check that the target's tracklet is never its own competitor.
- **Bank uncontaminated:** 0 violations in every scenario. `posed_crossing` makes sure the bank really receives non-seed samples, so the check is not vacuous.
- **No hidden/merged frame trusted:** universal check, plus GOLD below.

**Coverage cost.** Where the target is regained only against an opponent (opponent_crossing, occlusion, leave/return, cut), the regained episode stays observed-but-unverified: 35–75 frames per scenario.

**Long run** (`test_pipeline_longrun.py`, 1,800 frames ≈ 60 s at 30 fps; repeated crossings, occlusions, exits, pans and cuts):
- structural state stays bounded (population, target, writer), with no growth trend between the first and last thirds;
- heap growth < 10 MB;
- the target is still followed: of 1,606 visible frames, 290 trusted, 471 observed-but-unverified, 150 REACQUIRING and 628 LOST;
- 0 swaps; bank pure.

## GOLD development observations (development footage only; no events evaluated; not benchmark scoring)

Development footage only. No events evaluated. The frozen human labels v1 (sha256 `490a8af1…`, verified against the freeze record) were read only **after** the run, to tabulate states per labelled frame. Results are from run A on commit `30f2e39`.

- **Contract:** R1 errors 0; historical contract after `strip_v2` errors 0. Provenance: commit `30f2e39`, `git_dirty: false`, config hash `8e3b9d67…`, both model hashes recorded.
- **States over 179 frames:** TRACKING 105, AMBIGUOUS 34, REACQUIRING 25, OCCLUDED 11, LOST 4.
- **Trusted:** 79 frames, all in episode 1 (the seeded episode).

**By human label class (label window rel 70–110):**

| Label class | Frames | Trusted | States |
|---|---|---|---|
| HIDDEN_OR_NOT_RESOLVABLE | 5 | **0** | OCCLUDED 5 |
| MERGED_WITH_OPPONENT | 6 | **0** | OCCLUDED 4, AMBIGUOUS 2 |
| PARTIAL | 4 | 2 | TRACKING 2, AMBIGUOUS 2 |
| VISIBLE | 26 | 8 | TRACKING 8, REACQUIRING 16, OCCLUDED 2 |

**Reading the result:**
- No hidden or merged frame is trusted.
- Coverage on visible frames after the contact is low (8/26): honest abstention, not error.
- The contact sheet shows every trusted box on the target.

**Episode 2.** Reacquired at frame 120 against the opponent T012 at bank similarity 0.752. That is below the 0.80 confusable floor, so `contrast_verified: false`.
- Under the R1-G rule the whole episode is untrusted: 26 observed-but-unverified, 30 AMBIGUOUS and 4 LOST frames (rel 120–179).
- These frames are **outside** the human label window, so whether the box is on the target there is UNKNOWN. A visual sheet shows it on a blue-kit player (`gold_pipeline/episode2_unlabelled_frames_120-176.jpg`).
- In the interim run on `c6d118b`, the same episode was trusted.

**Other:** identity conflicts: one frame, T001|T012. Run time ~250 s per run (cloud, 2 CPUs).

## Independent review (project subagent policy)

### Code reviewer (on c6d118b): 9 findings, all validated

| Id | Finding | Validation | Fix |
|---|---|---|---|
| H1 | The uncontrasted cap of 0.5 equals the consumer trust threshold, so a look-alike becomes actor TARGET_PLAYER, band HIGH. | Reproduced with the real `actor_attribution`. | Unverified episode = observed but untrusted (above). The repro now gives actor UNKNOWN. Consumer-level regression test added. |
| M1 | A rejected look-alike vanished from the output for 106 frames. | Reproduced. | Exclusion limited to bodies that may physically be the target. Test added. |
| M2 | Merged / tracklet-less bodies are never written, and their contact with the target is dropped from `overlap_contests`. | Reproduced. | **Partly fixed:** a tracklet-less box now stands for its written claimants in contests. The rest is a documented **known limit**, because a representation needs a contract vocabulary decision. The mapping doc is corrected. |
| M3 | The fail-closed cleanup deleted files this run did not create; rename could replace a file created meanwhile. | Reproduced. | Created-files registry; exclusive hard-link publish. Tests added. |
| L1 | Per-frame lists grew with the clip. | Confirmed. | Counters and an incremental fingerprint digest. |
| L2 | The decision log was written under its final name. | Reproduced (SIGKILL). | Partial file, published at the end. Test added. |
| L3 | Passed-in backends were checked by name only. | Confirmed. | Settings and model-record checks. Tests added. |
| L4 | One contrast against any banked competitor lifted the cap. | Confirmed (design). | Confusable-competitor rule. Tests added. |
| L5 | `camera_shift_h` could be `-0.0`. | Confirmed. | Fixed. |

### Test auditor (on c6d118b): 40 mutants, 18 caught, 3 equivalent or masked, 19 survived; 6 defects

| Id | Finding | Fix |
|---|---|---|
| C1 | The target's re-emerging body was written as another person (rule 28) in an untrusted frame. | Newborn-on-reported-box rule. The rule-28 check now runs in every frame, coasting records included. |
| C2 | A seed on the last frame crashed with a raw `KeyError`. | Now `RunRefusedError`. |
| C3 | A stale partial file was deleted. | Refused up front, and the stale file is preserved. |
| C4 | Same as H1. | — |
| C5 | Bad `max_frames` raised `ConfigError`. | Now `RunRefusedError`. |
| C6 | An existing decision log blocked a non-debug run. | Only checked when debug is on. |

**Weak or vacuous tests fixed:**
- The bank check only ever saw the seed sample. A posed scenario now has real non-seed samples.
- Rule 28 was checked only on trusted frames. Now every frame.
- The `bbox` / `last_seen_frame` contract was untested.
- Backend refusals masked each other. Each is now tested alone.
- `compare_documents` sensitivity was untested.
- Population coasting semantics were untested.
- Conflict accounting and reacquisition rivals in conflicts were untested.
- `max_frames` as an argument was untested.
- Episode-lift persistence was untested.
- `_DocumentChecks.finish` negative cases were untested.
- The decision-log seed line was untested.
- 25/50 fps and `start_frame` 0 had no regression tests.
- An above-floor impostor after a cut was untested.

**Post-fix mutation run (16 new mutants on the fixes):** 16 mutants. 14 were caught at first. The two survivors were real test gaps, and both are now caught:
- K15: merged-box claimants not mapped;
- K16: passing candidates and other boxes' competitors counted as conflicts.

The second also exposed a loose definition: conflicts counted the competitors of every non-passing candidate, including rival boxes. Conflicts are now limited to the target's own failing continuation box, its unverifiable contacts, and the rivals that blocked it (the target now logs `continuation_rivals`). Tests: `ConflictScope` in `test_pipeline_audit.py`.

## Negative results and trade-offs (project rule 12)

| Change | Observed | Decision |
|---|---|---|
| Episode cap of 0.5 (first implementation) | Consumers trust 0.5 (H1). | Replaced by "observed but untrusted". |
| Any banked competitor verifies | A look-alike beating a far opponent would be trusted (L4). | Confusable-competitor rule. |
| Conflicts counted every non-passing candidate's competitor | Rival boxes' own competitors were counted against the target (found by post-fix mutant K16). | Restricted to the target's own competitors and rivals. |
| Rule 28 via the target's confirmed tracklets only | A rejected look-alike disappeared (M1). The target's re-emerging body was written twice (C1). After an ambiguity-induced LOST, the target's own observed tracklet was written as another person (found by the new every-frame check). | `possibly_target` rules above. |
| **Coverage cost of H1 + L4** | On the 1,800-frame synthetic match, the trusted share of visible frames fell from about 0.3+ to 0.18 (290/1,606). 471 visible frames are now "followed but unverified". In GOLD, episode 2 (reacquired at frame 120 against the opponent at similarity 0.75 < 0.80) becomes untrusted: 0 trusted frames after frame 120 (it was trusted on `c6d118b`). | Kept: your rules rank "confidently wrong" first. **Decision for Nathan** (see Limitations). |

## Tests

**Mac** (source of truth; python 3.10.12, onnxruntime 1.23.2, cv2 5.0.0; `SPANA_REQUIRE_MODELS=1` and the GOLD video set, so nothing is skipped): **444 tests, OK**, in 4 chunks (device time limit):
- chunk 1, 13 modules: 384;
- e2e / run / consumers: 35;
- audit: 21;
- long run: 4.

The full output is in `R1G_TEST_OUTPUT_mac.txt`. All 18 test modules ran.

**Cloud** (onnxruntime 1.29): 440 passed, 4 skipped. The skips are the model- and GOLD-gated tests, which the Mac ran.

**New in R1-G:**
- `test_pipeline_e2e.py`, `test_pipeline_run.py`, `test_pipeline_longrun.py`, `test_surviving_consumers.py`, `test_pipeline_audit.py`;
- `_synth_pipeline.py` harness;
- updates to `test_target.py`, `test_target_audit.py` and `test_package_hygiene.py`.

## Known limits

1. **Coverage trade-off.** An identity regained without beating a confusable competitor stays untrusted until one is beaten. In sparse or opponent-only surroundings, a correctly reacquired target can stay untrusted for long spans (synthetic: trusted coverage 0.18 of visible frames; GOLD: episode 2). The alternative, letting any banked contrast verify, re-opens the look-alike risk.
2. **Untracked bodies are not represented.** Merged or alternative boxes have no tracklet, so they are not written. Their contact with the target appears in `overlap_contests` only through written claimants. Representing them is a contract vocabulary decision for R1-H.
3. **Pass 2 sees an unverified episode as present.** `event_adapter` maps it to TRACKING: events are detected, but the actor is UNKNOWN. R1-H should decide whether Pass 2 should read `v2.identity_trusted`.
4. **Synthetic appearance is idealised**, and GOLD is one development clip. Synthetic "zero silent swaps" holds for the tested scenarios, including above-floor impostors without verification. The lone-look-alike risk is now exposed as untrusted rather than prevented.
5. **No team or number output** (generic tracker): `team = UNKNOWN`. Pass-2 logic that needs teams is an R1-H question.
6. **Identity confidence is UNCALIBRATED.**

## Persistence

- R1-G commits: `c6d118b` (implementation, before review) and `30f2e39` (review and test-audit fixes).
- The records commit follows; its hash, the GitHub verification and the full snapshot with its manifest hash are recorded in READ FIRST and the stage report.
- Code diff against R1-F `a89053e`: `R1G_DIFF_code.patch` (sha256 `d932b6fe…f61`).
- Config registry `R1G_CONFIG_REGISTRY.csv` is byte-identical to R1-F (r1f-2, 157 parameters, hash `8e3b9d67…`): **R1-G changed no parameter.**
