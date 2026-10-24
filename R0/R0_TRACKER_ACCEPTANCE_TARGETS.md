# R0_TRACKER_ACCEPTANCE_TARGETS.md

SPANA engine v2-rebuild — R0. Contains **R0.9, tracker acceptance targets** and **R0.10, R2 tracker benchmark design**. The benchmark is designed here only. **It is not run.** Nothing here has been measured on rebuilt code.

**Principle.** Tracking is judged on visual / physical player identity evidence. Event correctness is never used, and no event label influences any tracker decision or tuning.

---

## Part A — what history allows us to recover as targets

| Historical fact | Evidence | Use in R2 |
|---|---|---|
| GOLD_DUEL_01: target SOL #23; occluder MAR #4. Identity risk at extract frames 262–274; no same-kit player within 10 m | DOCUMENTED + human-signed (GOLD label + sign-off) | Clip-level identity truth. A per-frame box truth still needs a human pass (see B4). |
| Correct historical behaviour on the GOLD collision (3N current code): ACTIVE on #23 to 265; **OCCLUDED** 266–268 with box held; **AMBIGUOUS** 269–271; re-confirmed on #23 at 272; never confirmed on #4 | VERIFIED_OUTPUT (88246efe) + Claude visual check (not human) | **Minimum bar** for R1/R2: never assert TRACKING on #4 anywhere in 262–274. Reference behaviour, **not** an exact-match target — a better tracker may hold #23 longer. |
| Historical swap failure (stale pre-3L run): confirmed on #4 for 266–273 | DOCUMENTED (TASK3M) | **Negative test**: this exact outcome must be FAIL. |
| pass_completed_01: single online id T001, ACTIVE 100%, min identity_confidence 0.647, no contacts | VERIFIED_OUTPUT (40fa0df1) | Output-contract reference only; source video missing, so not re-runnable. |
| Historical score weights, gates, and margins | VERIFIED_OUTPUT / DOCUMENTED (`R0_CONSTANTS_AND_THRESHOLDS.csv`) | Design inputs only. **Not** acceptance targets. Do not copy-tune. |
| No historical per-frame tracking metric existed beyond tracker-self-reported state percentages (`selected_state_pct`, ablation ACTIVE+REID %) | DOCUMENTED | Self-reported state % is **not** an accuracy metric (ablation report §3 caveat). R2 must use human truth. |

**Recovered conclusion.** History offers exactly one human-anchored identity scenario (GOLD) and zero human per-frame box annotations. All R2 numeric targets below are therefore **NEW** and are proposed, not recovered.

---

## Part B — measurable tracker acceptance checks (NEW)

### Truth model for R2 (NEW)

For each benchmark frame *f*, a human annotates:

- `target_visible ∈ {VISIBLE, PARTIAL, HIDDEN, OUT_OF_FRAME}`;
- `target_box`, when VISIBLE or PARTIAL;
- `identity_resolvable ∈ {YES, NO}` — could a careful human tell, from this frame and its neighbours, which body is the target?;
- `contact_with` — the ids of other annotated players overlapping the target.

Two annotators label every event-critical window. Disagreements are adjudicated before scoring. The truth is frozen and hashed before inference (historical rule: "benchmark truth must be frozen before inference").

### Tracker output mapped to a claim per frame

- **CONFIDENT_ON(box)**: `state` is TRACKING (legacy) / ACTIVE (online), the position source is an observed detection, and the identity state is trusted.
- **HELD**: OCCLUDED with a predicted or held box.
- **ABSTAIN_AMBIGUOUS**: AMBIGUOUS.
- **ABSTAIN_LOST**: LOST.

### B1. Target identity continuity

| Metric | Definition | Proposed R2 gate |
|---|---|---|
| **False confident identity (FCI)** | Frames where the claim is CONFIDENT_ON(box) and box IoU with the human target box is < 0.3, *or* the human marks the target HIDDEN / OUT_OF_FRAME. | **0 frames per clip** on the event-critical windows. ≤ 0.5 % of confident frames overall. This is the most serious failure. |
| **Physical identity swaps** | Count of maximal runs where CONFIDENT_ON overlaps (IoU ≥ 0.5) a *different* annotated player. | **0** across the benchmark. |
| Correct target when visible | Over frames with human VISIBLE and identity_resolvable = YES: CONFIDENT_ON with IoU ≥ 0.5. | Report per scenario. Proposed floor: ≥ 80 % for clean-continuous; no floor for hard scenarios (measured, not gated, in the first R2). |
| Uncertainty when ambiguous | Over frames with human identity_resolvable = NO: the claim is ABSTAIN_* or HELD. | ≥ 95 %. |

### B2. Tracking state quality

| Check | Definition | Gate |
|---|---|---|
| TRACKING only when observed and grounded | CONFIDENT_ON ⇒ an actual detection was associated this frame (position source observed, not held / predicted). | **100 %** (contract check, automatic). |
| OCCLUDED when hidden | Human HIDDEN or PARTIAL with identity propagated → tracker HELD or ABSTAIN. | Report. Target ≥ 80 %. |
| AMBIGUOUS when unresolved | Human identity_resolvable = NO → ABSTAIN_AMBIGUOUS or HELD. | Part of B1. |
| LOST when identity genuinely unavailable | Human OUT_OF_FRAME for more than the documented hold limit → ABSTAIN_LOST within that limit. | Report. Hold limit is in seconds (frame-rate independent). |
| Time-in-state distribution | % TRACKING / OCCLUDED / AMBIGUOUS / LOST per clip. | Reported, never gated (self-report). |

### B3. Reacquisition

| Metric | Definition | Gate |
|---|---|---|
| Correct reacquisitions | After each human-annotated HIDDEN or OUT_OF_FRAME span, the first CONFIDENT_ON frame is on the target (IoU ≥ 0.5). | **100 %** of the reacquisitions that do occur. |
| Wrong reacquisitions | First CONFIDENT_ON after a gap is on another player. | **0**. |
| Reacquisition latency | Seconds from the target becoming VISIBLE and resolvable to the first correct CONFIDENT_ON. | Reported. Lower is better. Never traded against wrong reacquisitions. |
| Missed reacquisition | Target VISIBLE and resolvable for ≥ 2 s after a gap, but the tracker is still LOST. | Reported (coverage cost, acceptable). |

### B4. Bbox quality

| Metric | Definition | Gate |
|---|---|---|
| Mean IoU on CONFIDENT_ON frames vs human box | — | Report. Proposed ≥ 0.6. |
| Held-box honesty | On HELD frames, the box is flagged held / predicted in output. | 100 % (contract check). |

### B5. Identity integrity during overlap

| Metric | Definition | Gate |
|---|---|---|
| No silent jump during overlap | In every human `contact_with ≠ ∅` window ± 0.5 s: no CONFIDENT_ON on the contact partner; no CONFIDENT_ON on a merged box where the human marks the target < 50 % visible. | **0 violations**. |
| Same-team / opponent split | Report B1–B5 separately for same-kit and opponent contacts. | Report. |
| GOLD collision regression | Extract 262–274: never CONFIDENT_ON #4. | **Hard FAIL** if violated. |

### B6. Output contract

| Check | Definition | Gate |
|---|---|---|
| Historical schema | The output JSON validates against the 29-field record, the 7-key frame record, and the summary keys in `R0_TRACKER_INTERFACE_MAP.md` §2–4. Includes: `overlap_contests` is a list; exactly ≤ 1 `selected_player` per frame; frame numbering 1..N relative to `start_frame`. | **100 %**, automatic. |
| Surviving consumer compatibility | The surviving `actor_attribution.frame_identity` and `event_adapter.build_integrated_event_snapshots` run unmodified on R1 output. The adapter needs a NEW `new_tracker_snapshots` shim — see the R1 plan. | Pass / fail. |
| New fields additive | Removing every NEW field leaves a historically valid record. | Pass / fail. |
| Determinism | Two runs on the same input with the same models and config give byte-identical output. | Pass / fail. |
| Provenance | Every tracking JSON records the source video sha256, the seed, model hashes, the code hash, and the config hash. | 100 %. |

---

## Part C — R2 tracker benchmark design (R0.10). DESIGN ONLY, NOT RUN.

### C1. Material rules

- **Holdout**: `videoplayback (2).mp4` (ESFC v BCFC) and anything else marked `v2_esfc_bcfc` are **excluded**. They are never opened.
- **Rejected fixtures** (duel_lost_01, src2_clip_003, shot_close_range_01): excluded as truth. They may be cited as negative evidence only.
- **Task 3R windows are excluded** from R1 development and R2 tuning. These are R01–R10, at the source time ranges in `_spana_task3r_work/extract_log*.txt`. Task 3R stays frozen. Using its footage for tracker tuning would contaminate it.
- **Also excluded**: GENERAL_BENCHMARK_V2 windows (SYO/MAR 2400 s and 6600 s; MCFC/MIT 2400 s and 4200 s), unless re-registered with fresh labels.
- **Allowed sources**: `videoplayback.mp4` (SYO v MAR, multi-camera broadcast, zooms, cuts) and `videoplayback (1).mp4` (MCFC v MIT, single low camera, floodlit). These give two matches, two camera styles, and day/floodlit lighting. `youtube_game_1/2/3.mp4` are allowed only after their provenance is reviewed. R0 did not review it.
- **Extraction**: use the Task 3J method. Lossless re-encode (libx264 qp0), a framemd5 match against the source, and the sha256 recorded. Clip boundaries are chosen **without running any tracker**.
- **Labels**: drawn by a human. They must not be derived from model output (charter rule). Model detections may only be shown to the annotator as optional pointers, and as Task 3J did, that use must be recorded.
- **Development / evaluation split**: GOLD_DUEL_01 is development (already seen). Everything new is split 50/50 by source-time block into DEV and EVAL **before** labelling. EVAL is labelled and frozen, but **not looked at during R1/R2 development**.

### C2. Scenario matrix (minimum one clip each; target two)

| ID | Scenario | What makes it hard | Suggested source (non-holdout) | Required annotation |
|---|---|---|---|---|
| S1 | Clean continuous target | Baseline | either source | per-frame box at ≥ 5 Hz |
| S2 | Brief occlusion (< 0.5 s) by any player | hold vs swap | SYO/MAR (GOLD is the dev exemplar) | per-frame, every frame in window ± 1 s |
| S3 | **Same-team crossing** | kit identical; only ReID, motion, and number separate them | MCFC/MIT or SYO/MAR | every frame; both players' boxes |
| S4 | **Opponent crossing** | different kits; occluder becomes the only box | SYO/MAR (GOLD = dev; need ≥ 1 new for EVAL) | every frame; both boxes |
| S5 | Player–player overlap / contact (duel, shielding) | sustained overlap | either | every frame |
| S6 | Merged detection | one box for two people | found during S3–S5 annotation (flag frames) | flag + both boxes |
| S7 | Partial body (truncated by the frame edge or another player) | fragment vs separate player | either | per-frame; visibility class |
| S8 | Camera pan | global motion | SYO/MAR | per-frame at ≥ 5 Hz |
| S9 | Zoom (in or out) | scale change; pixel gates break | SYO/MAR (zooms documented, e.g. GOLD 5–7 s) | per-frame at ≥ 5 Hz |
| S10 | Target leaving frame | must go LOST, not swap | MCFC/MIT (small players, low camera) | the OUT_OF_FRAME span |
| S11 | Target re-entry | correct reacquisition or honest LOST | same clip as S10 | the first frames after re-entry |
| S12 | **Prolonged ambiguity** (≥ 2 s cluster, e.g. set piece or crowded box) | must stay AMBIGUOUS/LOST, not guess | either | identity_resolvable per frame |
| S13 | Hard camera cut (optional) | identity must not carry across | SYO/MAR (multi-camera; cuts documented in TASK3J shortlist) | the cut frame |

**Size.** Roughly 13–26 clips of 5–15 s. At about 30 fps and ≥ 5 Hz box annotation, that is around 4–8 k annotated boxes.

### C3. Procedure (R2, later)

1. Register each clip with: source sha256, extract sha256, time range, and the seed (human-selected frame + box). The seed must be verified to overlap a raw detection; otherwise the fixture is refused (the F12 lesson).
2. Freeze labels (two annotators, adjudicated) and hash them **before** any R1 run.
3. Run R1 fresh from source, with no resume. Freeze and hash the outputs.
4. Score with B1–B6. Report per scenario and per source. **False confident identity first.**
5. Report the abstention cost, coverage, separately and openly.
6. No threshold change between seeing EVAL results and reporting them.

### C4. Things the benchmark explicitly does not do

- It does not score events, possession, or stats.
- It does not use GOLD's DUEL label for anything tracking-related.
- It does not tune on EVAL.
