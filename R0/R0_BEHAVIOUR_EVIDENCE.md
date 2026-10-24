# R0_BEHAVIOUR_EVIDENCE.md

SPANA engine v2-rebuild, R0 (R0.12). This file preserves the evidence about the **rest of the engine** (Pass 2 and the event layer) for R3 and R4.

**Nothing here is rebuilt in R0 or R1.** The evidence levels follow the charter. The constants cited are listed in full in `R0_CONSTANTS_AND_THRESHOLDS.csv`.

Abbreviations:
- **S** = survives. "S-3N" means the 3N bytes survive and the 3Q version can be patched in.
- **M** = missing.

---

## Ball tracking (`ball_tracker.py` S-exact; 7 `ball_*` helpers M)

- **State machine.** CONFIRMED / PREDICTED / OCCLUDED / REACQUIRING / LOST.
  - VERIFIED_CODE: `ball_tracker.py:265-267, 354, 365, 443`.
- **Staged search.** Coarse-to-fine re-detection: cheap tier A first, then ROI escalation, then a lazily loaded heavy model (`yolov8x.pt`).
  - DOCUMENTED: CURRENT_ARCHITECTURE.md (ball table).
  - VERIFIED_CODE: imports at `ball_tracker.py:49-57`.
- **Confidence caps.** REACQUIRING is capped at 0.79 and bridged positions at 0.55 (`ball_confidence.py`, which is missing).
  - DOCUMENTED: CURRENT_ARCHITECTURE.md.
- **Offline bridging.** Bidirectional interpolation over completed LOST runs (`ball_temporal_bridge.py`, M).
  - DOCUMENTED.
- **Known failure (3G).** Once the ball track has been LOST for more than 10 frames, distance gating stops. Low-threshold (0.05) upscaled crops around the feet then return the player's **shadow or boot**, and the "near the tracked player" bonus accepts them.
  - DOCUMENTED: TASK3G_FINAL_REPORT, root cause items 1–2.
- **Recall.** Ball-detection recall is "~10–20 % per frame".
  - DOCUMENTED: `confidence.py:4` (3N).

## Ball physical validity (`ball_interaction_credibility.py` S-exact; `ball_static_decoy.py` S-exact)

- **3G gate.** A gate sits between ball tracking and possession. Only CREDIBLE ball evidence may create possession, and the gate fails closed.
  - DOCUMENTED: TASK3G.
  - VERIFIED_CODE: `ball_interaction_credibility.py` (`CREDIBLE`, imported by `controlled_duel.py:136`).
- **Known recall gap.** A ball-sized candidate seen only by weak stages at the feet is held in REVIEW and creates no touch (TASK3G).
- **3E static decoy.** `PitchRegionClassifier` / `StaticDecoyTracker` suppress recurring static and off-pitch false positives. On duel_lost_01, the MITRE logo dropped from 91.2 % to 3.8 % CONFIRMED.
  - DOCUMENTED: TASK3E.
  - VERIFIED_CODE: `ball_static_decoy.py`.

## Team / opponent context (`possession_detector.TeamClassifier` S-3N; `opponent_detection.py` S-exact)

- **Classification method.** Per frame: TEAMMATE / OPPONENT / UNKNOWN from LAB kit colour, with no persistent identity for other players.
  - Constants: `MIN_CONFIDENT_SIM 0.55` and `MARGIN_THRESH 0.12`.
  - VERIFIED_CODE: `possession_detector.py:53-54` (3N).
- **3L changes.**
  - Truncated boxes are UNKNOWN.
  - The opponent reference is the colour cluster worn by at least 2 people in the same frame.
  - `TARGET_PLAYER_FRAGMENT` exclusion applies when a box is at least 85 % inside, at most 50 % of the area, and at most 80 % of the height.
  - DOCUMENTED: TASK3L.
  - VERIFIED_CODE: `opponent_detection.py:127` `FRAGMENT_MAX_HEIGHT_RATIO 0.8`; flags `TASK3L_OTHER_PLAYER_CONTEXT` and `TASK3L_FRAGMENT_EXCLUSION`.
- **Limits.**
  - Colour-only labels mark staff and spectators as TEAMMATE.
  - Truncated real players become UNKNOWN.
  - There is no other-player entity, so store ids can switch people.
  - DOCUMENTED: TASK3L limits; status doc.
- **Cross-pass coupling.** Pass 1 imports this classifier, and 3L changed Pass-1 tracking through it.
  - DOCUMENTED: TASK3M.
  - **R1 must not import it** (failure catalogue F14).

## Possession (`possession_detector.py` S-3N; `possession_segmentation.py` M; `interaction_detector.py` M)

- **Possession rule.** The ball counts as at the player's feet within `TOUCH_RADIUS 40.0` px of the foot point. Possession needs `MIN_POSSESSION_FRAMES 2` consecutive close, usable, TRACKING frames.
  - VERIFIED_CODE: `possession_detector.py:47-48` (3N).
- **Pauses when the target is not TRACKING.** Possession pauses rather than guesses (`possession_detector.py:424-429`). This is the main way tracker state quality reaches event logic.
- **Segmentation.** The possession timeline is cut into a hysteresis-segmented timeline (M; DOCUMENTED, 3A spec §1).

## TOUCH / RECEIVE / CARRY (`event_classifier.py` M)

- **TOUCH.** Proximity/contact against the TrackingSnapshot box and the ball (DOCUMENTED, 3A spec §5).
- **RECEIVE.** The first touch of a new possession segment, temporally confirmed (DOCUMENTED; `reception_logic_report.md`).
- **CARRY.** A possession segment of at least 8 frames with at least 60 px displacement (DOCUMENTED, 3A spec §2/§5).
  - Both limits are pixel- and frame-based, so they are neither scale-normalised nor frame-rate independent. **INFERRED risk.**

## Receiver identity (`shot_candidate.ReceiverEvidence` S-3N; `receiver_eligibility.py` S-HV)

- **3P rule.** A teammate or unknown box next to a non-TRACKING target cannot confirm a reception.
  - VERIFIED_CODE: `receiver_eligibility.py:96-101, 142, 149-158`.
  - DOCUMENTED: TASK3P_RECEIVER_ELIGIBILITY_SPEC.
- **3O finding.** The reception score is built only from ball signals (visible 0.257, decelerating 0.180, still 0.162). There is no receiver-independence check, so the target's own re-emerging body qualified as the receiver (DOCUMENTED, TASK3O finding 4).
  - The true receiver (pass_completed_01) is separable by independence signals: 56 frames of prior existence, at least 600 px from the target, and a full body.
- **Open risk.** The receiver rule uses a fixed radius, and abstention has a cost when a real teammate is right next to a hidden target (status doc).

## PASS (`event_candidates.py` S-exact; `event_arbitration.py` S-3N)

- A release becomes PASS through candidate scoring and arbitration.
- `success` means the next confirmed possession or reception belongs to a teammate (DOCUMENTED, 3A spec §5).
- **`RECEIVER_CONFIRMED` short-circuit.** A confirmed teammate reception ends arbitration before any DUEL, CROSS, SHOT or UNKNOWN candidate is built (DOCUMENTED: TASK3O finding 5; CURRENT_ARCHITECTURE "two decision authorities").
- **Historical result.** pass_completed_01: PASS COMPLETE 0.804 (VERIFIED_OUTPUT, 3Q).

## SHOT (`shot_candidate.py` S-3N, about 1,000+ lines)

- A persistent multi-frame state machine with constants driven by past incidents (DOCUMENTED: CURRENT_ARCHITECTURE "textbook profile of thresholds accumulated one bug-fix at a time").
- **No valid SHOT fixture survives.** shot_close_range_01 is REJECTED. The SHOT path is therefore **unvalidated** (DOCUMENTED, 3D).

## CROSS (`event_candidates.py` S-exact)

- **Status.** CROSS_EMISSION = ACTIVE (DOCUMENTED: task3a `CROSS_EMISSION_FINDING.md`).
- **Thresholds.** `CROSS_GSCORE_BAND_LOW 0.35`, `CROSS_GSCORE_BAND_HIGH 0.85`, `CROSS_MIN_SUPPORTIVE_WIDE_SCORE 0.45` and `MIN_CROSS_DISPLACEMENT_PX 150` (VERIFIED_CODE, `event_candidates.py`).
- **No valid CROSS truth fixture exists.** The CROSS events seen on pass_completed_01 came from a misregistered segment (TASK3N).

## DUEL (`event_candidates.py` S-exact path A; `controlled_duel.py` S-exact path B; `confidence.py`/`event_arbitration.py`/`possession_events.py`/`player_stats.py` 3Q via patch)

- **Path A, contested arrival.**
  - Triggers when the incoming ball is fast (at least 3000 px/s), the touch lasts at most 2 frames and an opponent is near.
  - It is structurally unreachable for dribble duels (DOCUMENTED, TASK3O finding 8).
- **Path B, controlled-possession challenge (3Q).**
  - Thresholds are in body heights and seconds:
    - control ≥ 0.5 s;
    - opponent closes ≥ 0.5 h, from engagement ≤ 1.5 h;
    - opponent's foot reaches the ball within ≤ 0.5 h;
    - contest ≥ 0.15 s (reach, or overlap with centres ≤ 0.75 h);
    - ball stays within ≤ 1 h of a player;
    - identity grounded: non-TRACKING gap ≤ 0.5 s.
  - Outcome is WON if the target regains first, LOST if the opponent gains sustained control first, otherwise UNKNOWN.
  - VERIFIED_CODE: `controlled_duel.py` (32 module constants) and the `TASK3Q_CODE_DIFF.patch` text. DOCUMENTED: TASK3Q_CONTROLLED_DUEL_SPEC.
- **Historical result.** GOLD DUEL WON 0.858 (VERIFIED_OUTPUT, 3Q).
- **Fragility.** 2 of 18 constants flip the verdict at ±30 %. There is only 1 positive and 2 negatives on real footage, and no camera-motion compensation (DOCUMENTED, status doc).

## Arbitration (`event_arbitration.py` S-3N, 3Q patch)

- **Candidates.** Five independently scored candidates (PASS, CROSS, SHOT, DUEL, UNKNOWN_RELEASE) with margin and floor logic.
- **Exception.** The `RECEIVER_CONFIRMED` short-circuit (DOCUMENTED, CURRENT_ARCHITECTURE).
- **3Q change.** The runner-up is now the best non-duel candidate (DOCUMENTED, status doc).
- **Tracking gate.** Pass-2 use of tracking: `pf.tracking_state == "TRACKING" and pf.ball_distance <= 40.0` (`event_arbitration.py:326`, 3N).

## Confidence (`confidence.py` S-3N)

- **Bands.** `HIGH_THRESH 0.75`, `MEDIUM_THRESH 0.5`. The precision-favouring rule "rounds DOWN when in doubt" (VERIFIED_CODE, `confidence.py:1-12`).
- **Tracking quality.** `(TRACKING + 0.5 × OCCLUDED) / total` (VERIFIED_CODE, `confidence.py:27-32`).
- **Ball quality.** `(confirmed + 0.8·reacquiring + 0.4·predicted) / total`; the team term defaults to 0.4 (VERIFIED_CODE, `confidence.py:45, 51`; 3N).
- **Raw score.** `raw = 0.35·tq + 0.45·bq + 0.20·teamq` (VERIFIED_CODE, `confidence.py:65`; 3N).
- **Review flag.** For outcome events, `review_required = band != "HIGH" or event.success is None`; for position events, `band == "LOW"` (VERIFIED_CODE, `confidence.py:71, 73`; 3N).
- **3Q.** Path-B duel confidence is duel-specific (DOCUMENTED).

## Actor attribution (`actor_attribution.py` S-exact)

- Fully described in `R0_TRACKER_INTERFACE_MAP.md` §5.5.
- Actor is TARGET_PLAYER only when the actor band is HIGH. Event confidence is never read (`actor_attribution.py:62-63, 163`).
- **Known blind spot.** It measures track self-health, not physical identity (F13).

## Stats (`player_stats.py` S-3N, 3Q patch)

- Stats are a pure reducer over the event log (`player_stats.py`).
- `stat_eligible = actor == TARGET_PLAYER and band ∈ {HIGH, MEDIUM}` (VERIFIED_CODE, `analyze_target_player.py:157-160`).
- An UNKNOWN duel outcome counts as neither won nor lost (VERIFIED_CODE, 3Q patch text; `player_stats.py:98, 164-165` in the patched scratch tree).
- **Open risk.** `stat_eligible` ignores `review_required` (DOCUMENTED, status doc; TASK3O finding 5).

## POSSESSION_WON / POSSESSION_LOST (`possession_events.py` S-HV)

- Derived from transitions of the stat-eligible TARGET_PLAYER events (VERIFIED_CODE, `analyze_target_player.py:202-206`).
- 3Q: a path-B DUEL WON keeps the possession episode open (DOCUMENTED).
- **Known artifact.** POSSESSION_WON at tracking start (GOLD 6.53 s) (DOCUMENTED; VERIFIED_OUTPUT).

## Event clips (`event_clip_exporter.py` S-exact)

- `export_event_clips(video_path, event_log, out_dir)` writes a ±2 s pre/post roll (`PRE_ROLL_S`/`POST_ROLL_S 2.0`) into `OUTPUT_FOLDER_BY_TYPE` folders (VERIFIED_CODE, lines 37-65).
- VERIFIED_OUTPUT: task3n `clips/` folders, e.g. `passes/001_PASS_COMPLETE_00m09s.mp4`.

## Pass-2 frame registration

- Pass 2 seeks to the **absolute** `start_frame`. It does not seek to snapshot frame 0. Seeking to snapshot frame 0 was a 3A bug, fixed in `analyze_target_player.py:284-307`.
- Event times are converted to clip-absolute by adding `start_frame_offset` (`analyze_target_player.py:162-173`).
- **Sequential reads only.** Random-access seeks on .mov files drift by 3+ frames (DOCUMENTED, TASK3F §2).

## What R3/R4 will need to decide (recorded only, not decided)

- Whether to apply the 3P/3Q patches to the 7 PATCHABLE files, and to accept that the resulting bytes cannot be hash-proven.
- Whether to rebuild the 46 RUNTIME_CRITICAL missing football_analysis and tracker-adjacent modules as NEW code, or to slim the event engine.
- The Task 3R scorer, runner and scoring rules are lost. Their scoring semantics must be re-specified **from the surviving labels and reconciliation docs**, not from memory.
