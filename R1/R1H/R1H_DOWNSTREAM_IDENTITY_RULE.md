# R1-H downstream identity rule: `identity_trusted` is the authority for target attribution

**Status:** locked by Nathan on 2026-10-26 (R1-H instruction). It binds every Pass-2 consumer of R1 Pass-1 output, including the future Pass-2 / event-engine rebuild. R1-H does **not** rebuild Pass 2. This document states the rule, what R1 guarantees, and how the surviving recovered consumers measure against it.

## The rule

1. **Physical events may exist while identity is untrusted.** A touch, pass, duel or shot can be detected from physical evidence (ball, bodies) in a span where the target's identity is not trusted. The event is kept. Only its attribution to the target is withheld.
2. **Target actor attribution requires `identity_trusted = true`.** The authority is the TARGET record's `v2.identity_trusted`, nothing else:
   - not the legacy `state` (an unverified episode is ACTIVE / CONFIRMED_DETECTION because its box is really observed);
   - not `identity_confidence` or any other score;
   - not `rb_state` alone (`TRACKING` can be untrusted).

   For an event that spans frames, **every frame of its actor window** (the frames that establish who acted: the control span, or the release / contact frames) must have `identity_trusted = true`. A trusted *fraction* of the window is not enough.

   *Interpretation note (R1-H):* the per-frame form is explicit in the instruction; the "every frame of the actor window" extension to multi-frame events is the strictest reading. It is recorded here for Nathan to confirm before the Pass-2 rebuild.
3. **Untrusted target attribution becomes UNKNOWN.** Never "probably the target", never a lowered confidence on a target attribution: the actor is UNKNOWN.
4. **Target stats must not count events whose target attribution is untrusted.** A stat counts an event only if rule 2 credited it to the target. UNKNOWN-actor events are reported separately (as `excluded_unknown_actor`) and never folded into the target's totals.

## What R1 Pass 1 guarantees (tested)

| Guarantee | Where tested |
|---|---|
| `identity_trusted` ⇒ `rb_state = TRACKING`, the box is an observed detection of this frame, and it is in a contrast-verified episode. | `test_target.py` frame invariants; `test_pipeline_e2e.py` UniversalChecks |
| Never trusted on a frame without the target's own detection (hidden, merged, dropped, absent). | `test_pipeline_e2e.py` (15 scenarios); `test_tracker_eval.py` SyntheticIntegration |
| Untrusted ⇒ `identity_confidence`, `overall_track_confidence` and `identity_confidence_rb` < 0.5. This is the compatibility bridge for legacy consumers that read scores (below). | `test_pipeline_e2e.py`, `test_surviving_consumers.py` |
| An unverified episode (reacquired without beating a confusable competitor) is `TRACKING` but untrusted, with reason `INSUFFICIENT_CONTRAST`. | `test_target_audit.py`, `test_pipeline_audit.py` |
| GOLD (development clip): 0 trusted frames on 11 human-labelled hidden, merged or unresolvable frames. | `R1/R1H/gold_eval/GOLD_R1H_TRACKER_EVAL.json` |

## The surviving recovered consumers, measured against the rule

These files are hash-pinned and unmodified (`tests_rb/test_surviving_consumers.py`).

| Consumer | Behaviour on R1 output | Meets the rule? |
|---|---|---|
| `event_adapter.build_integrated_event_snapshots` | Snapshot TRACKING ⇔ rb TRACKING. Every untrusted frame has `tracking_confidence` < 0.5. Events are still detected in untrusted spans (rule 1). | **Yes** for rule 1. It exposes presence, not attribution. |
| `actor_attribution.frame_identity` | Trusts a frame only if R1 trusted it. It does so through legacy fields and the < 0.5 confidence bridge, not by reading `v2.identity_trusted`. | **Yes**, indirectly (tested every frame). |
| `actor_attribution.assess_actor` | Credits TARGET_PLAYER when `trusted_fraction × resolver_factor ≥ 0.75`. A window with up to 25 % untrusted frames can therefore be credited. | **No** (rule 2, strict reading). See the measurement below. |
| `player_stats.compute_player_stats` (3N version) | Counts only `actor == TARGET_PLAYER` with band HIGH or MEDIUM; UNKNOWN never counts. | **Yes** for rule 4, but it inherits `assess_actor`'s fraction rule. |

**Measured gap** (`tools_rb/r1h_legacy_attribution_gap.py` → `R1H_LEGACY_ATTRIBUTION_GAP.json`; synthetic mixed scenario, CARRY windows of 5–30 frames, target status RESOLVED):

| Clip-level identity conflict | Windows credited TARGET_PLAYER | Of which contain untrusted frames |
|---|---|---|
| No | 83 | **15** (for example window 52–57 with one untrusted frame, trusted fraction 0.83) |
| Yes (score halved) | 0 | 0 |

The first failing layer is Pass-2 aggregation, not Pass 1: R1 marks those frames untrusted correctly. **Requirement for the Pass-2 rebuild:** replace the trusted-fraction rule with rule 2 (every actor-window frame `v2.identity_trusted = true`, otherwise UNKNOWN), and read `v2.identity_trusted` directly instead of reconstructing trust from legacy fields. The recovered originals are not modified.

## Out of scope here

- No Pass-2 or event-engine code was written or changed in R1-H.
- Untracked merged bodies are not fabricated as separate people (Nathan, R1-H). A merged box that is not separately observed stays unrepresented as individuals. Pass 2 must not infer a second person's identity from it.
