# R2 — RESIDUAL SAME-KIT CONTRAST DIAGNOSIS (2026-11-07)

Instruction: Nathan 2026-11-07 14:53. Commit `276f9dc`. Tracker `1.3.2-r2rc`, registry `r2str-1` (no parameter
change; three descriptions extended). The 0.08 contrast margin, the separation certificate, ReID, the appearance
floor and the crossing/contested-link rules are untouched. EVAL untouched.

## 1. Exact residual root cause per clip (native 1080p60 logs at `e179cce`, before any change)

| clip | frame | what the log shows | class |
|---|---|---|---|
| G1_A | f179 | the target's own leg box (score 0.10–0.30 = 0.13–0.37 x his, 0.77–0.93 of his width, feet gap ≈ 0) sat wholly inside his box f170–178 (0.3 s); the detector dropped it at f179 and rule 4b (vanished hidden body, F3c) ended T003 as `CONTACT_AMBIGUITY` / `CONTAINED_BODY_VANISHED`. The teammate (T043) was tracked apart on his own box throughout (2 % overlap; contrast 0.09–0.13 with the separation exclusion active). After the break the target returns on T132 as a REACQUISITION, where the certificate cannot apply and same-kit contrast (0.00–0.05 vs banked T006/T080) never reaches 0.08 → LOST f209. | **population / tracklet continuity issue** (detector artefact) |
| G2_D | f102–130 | the size anomaly's only "contact" was a merged two-player box (98 px) overlapping the 150-px target box by 1–7 % of its area with its feet line 0.92–0.98 of his height ABOVE his feet (near his head: behind him). That contact ended the anomaly as a continuity break (`SIZE_ANOMALY_ENDED`) → 30-frame REACQUIRING on his own T003 → accepted f157 as an UNVERIFIED episode (reason `INSUFFICIENT_CONTRAST`). Contrast there is 0.33–0.45, far above 0.08; the episode can only be *verified* against a confusable competitor (bank similarity ≥ appearance floor) and the best one is 0.45–0.55. | **stale / incorrect competitor evidence** (a body at another depth) + unverified-episode rule |
| G2_A | f121–126 | same far-body mechanism: T012 (0.84) overlapping the target box with feet gap 0.91, but through a different rule — a detector union box claimed by both tracklets (`MULTI_CLAIM`), after the f121 sliver counted as an `UNKNOWN_COMPETITOR_IN_CONTACT` in the identity layer. | **third clip, same mechanism, other rules** (not changed here) |

Neither clip's residual was genuinely ambiguous contact at the frame that lost trust; G2_D's later losses (f164 occlusion, f168/f173 real contacts) are genuine.

## 2. What changed (population.py only; existing constants)

- **Rule 4b credible contained body**: a feet-line nested box counts for the vanished-hidden-body rule only if it was
  CREDIBLE on some frame of its stay — scored above `population_weak_part_score_ratio` (0.5) x its host, or as tall
  as the body (`population_partial_height_ratio`, 0.75: an occluded same-kit player scores low → fails closed, review
  H1). Every feet-line nested body stays in the contained list so a score flicker does not reset its stay (review H2).
- **Depth-apart touch** (anomaly path only, `credible_touch`): a touching box is not a contact when |feet-line gap| >
  `separation_min_gap_h` (0.25) x the host's height — the project's one notion of "physically separate", applied to
  the depth direction (`population_hidden_feet_gap_max` 0.1 is the one-box nested criterion and would dismiss a
  player stepping forward, 0.1–0.2 h) — unless its feet are not seen: partial or edge-clipped box, or a box the host
  covers by ≥ `population_contact_cover_min` (0.1; legs hidden behind him). Partial-by-height of a touching detection
  is computed before the merge loop (review M2); memory boxes carry the last observed box's partial state
  (`ClosedTracklet.last_partial`, internal, not logged; review M3). Ordinary D11 corroboration and `MULTI_CLAIM` are
  unchanged.

## 3. Tests and audit

`tests_rb/test_residual_contact.py` (18 tests, 17 subtests): weak leg box vanishing keeps the tracklet (0.1 / 0.3 /
0.45 x host); credible crouched (0.51 / 0.6 / 0.8 x) and tall (0.1 / 0.3 x) contained bodies still end it; score
boundary; score flicker across the ratio; body behind on a far feet line (0.5 / 0.8 / 0.92 h, real touches of
6–7.5 % below the 10 % cover guard) not a contact; body in front (−0.5 h); compatible feet lines (0 / 0.15 / ±0.24 h)
and the 0.25 h boundary; edge-clipped, partial (`ref_partial`) and legs-hidden (covered) boxes still count; the
measured G2_D pair geometry (4.9 % covered, 5.3 % touch, 0.92 h) and its on-the-feet-line twin; predicted-box
(flickering tracked body) and memory-box (closed body) branches, far and near; camera pan. 8 fail on `5df45aa`.
Full suite: 1002 passed, 9 skipped, 11 xfailed, 1843 subtests (`SUITE_OUTPUT_r2rc.txt`); synthetic eval identical to
the F3C reference (`SYNTH_EVAL_r2rc.log`). Reviewer subagent audit: 3 HIGH (tall weak contained body dismissed; score
flicker; three vacuous touch tests), 4 MED (airborne/truncated bottoms; partial flag read before computed; memory
partial state; semantics/symmetry), 3 LOW — H1–H3, M2, M3 fixed with tests; M1 recorded in §6; descriptions updated.

## 4. 720p DEV (complete 11 clips, labels v1), `e179cce` → `276f9dc`

Identical on every clip: trusted 1152 / 2674 (0.431), P1 5 (C006 f91–93, C033 f105, C021 f36), P2 0, P3 1, P4 1,
bank 10 / 88, 2 seeds refused. Population-wide: `CONTAINED_BODY_VANISHED` breaks 17 → 16, merged observations,
duration escalations and hold frames unchanged (4419 / 3 / 65). `COMPARE_DEV_ind_e179cce_vs_rc_276f9dc.json`.

## 5. Native 1080p (G1_A, G2_D, G2_A), `e179cce` → `276f9dc`

| clip | trusted coverage | wrong-player frames | ambiguity (AMBIGUOUS frames; longest run) | first LOST | REACQUIRING / unverified TRACKING frames | what remains |
|---|---|---|---|---|---|---|
| G1_A | 175 → **179** (58.3 → 59.7 %) | 0 → 0 | 34 (30) → 35 (30, f184–213) | 209 → 214 | 17 → 0 / 0 → 0 | T003 now continuous through f179; from f184 the confusable competitors are **T006/T080, same-kit bodies whose tracklets closed at f154/f148** (merged / contested) 1.3 body heights away and that persist as banked memory competitors for `population_closed_memory_s` = 2 s (T006's window ends exactly at the LOST frame 214); contrast against them is noise (−0.03 … +0.08); LOST by ageing at 1.0 s |
| G2_D | 102 → 102 (17.0 %) | 0 → 0 | 47 → 47 | 254 → 254 | 98 → 68 / 7 → 37 | the anomaly is ISOLATED (held f102–126), ends isolated at f127 and T003 continues directly — the spurious reacquisition episode is gone (episode count −1); the continuation is an **UNVERIFIED episode** by the stride-stage rule (review round 3 H1) and no confusable competitor exists to verify it (best 0.45–0.55); genuine contacts follow at f164/f168/f173 |
| G2_A | 77 → 77 (12.8 %) | 0 → 0 | 60 → 60 | 137 → 137 | 52 → 52 | unchanged: `MULTI_CLAIM` union box with the far player T012 at f126 (feet gap 0.91), and the sliver as an identity-layer unknown contact at f121 |

Trusted frames are on the seeded tracklet only in every run (T003 / T003 / T002). LOST frames on G1_A rise 74 → 86
because the 17 unverified REACQUIRING frames of the old T132 attempt (f243–259) no longer exist (no break, no
reacquisition); nothing trusted was lost. `NATIVE_COMPARE_ind_e179cce_vs_rc_276f9dc.json`.

## 6. Substitution safety

- Rule 4b still ends the tracklet for a credible contained body (score > 0.5 x host, or as tall as the body), with
  the stay continuous across score flicker; crossing / contested-link rule 4 untouched (`test_population`,
  `test_separation_contrast` crossings unchanged); DEV P1–P4 identical.
- Depth-apart applies only to the anomaly path's contact flag; bodies on compatible feet lines (within 0.25 h) still
  escalate at once; partial, edge-clipped and host-covered (legs hidden) boxes fail closed; first-sight D11 merge
  corroboration and `MULTI_CLAIM` unchanged.
- Known limit (review M1): a touching box whose bottom is not its feet — an airborne opponent (header duel, 0.3–0.5 h
  for ~0.4 s), a newcomer first detected with legs cut by the detector (no tracklet → not partial-by-height) — in
  the 5–10 % cover window is dismissed on those frames; before and after the jump its feet are on the ground, so
  the contact is latched by the neighbouring frames in the header case. Accepted twin of rule 4b: a crouched person
  detected below 0.5 x the host who stands up is not caught by 4b (as before for a box poking out).

## 7. Can the existing separation logic be safely extended?

The certificate itself was never the failing layer: on both clips it worked where it applies (G1_A f40–f183 with the
teammate observed apart). Two residuals sit outside its scope and are decisions for Nathan, not implemented here:

1. **Memory (recently closed) same-kit competitors** (G1_A f184+): T006/T080 were last seen ≥ 1.3 h away, 30–40
   frames earlier, while the target's tracklet was continuous and its box touched by nobody since. A safe-looking
   extension: exclude a RECENTLY_CLOSED competitor whose last observed box was ≥ `separation_min_gap_h` from the
   target at closure, whose separation event is at/before the last trusted confirmation, while the target's tracklet
   is continuous and has had **no contact with any body** (observed, predicted, memory or unknown) since that
   closure — a hidden body can only enter the target's box through contact. It is NOT "continuously observed
   separate" (the body is hidden), so under the rule as stated it stays out; its premise is the same as the whole
   population's (the detector sees bodies that are in the open). Expected effect on G1_A: trust through f184–213
   (contrast vs the observed teammate stays 0.09–0.13) and no LOST at f214.
2. **Verification of an UNVERIFIED episode after an ISOLATED anomaly** (G2_D f127+): the stride stage made the
   continuation after an isolated anomaly unverified (an undetected body could have been inside); it can only be
   verified by beating a confusable competitor, which never appears on G2_D. A candidate: treat an isolated anomaly
   that ended by returning to single size, with no body of any kind seen within contact range for its whole
   duration, as continuity-verified. That relaxes a reviewer-validated fail-closed decision (round 3 H1: a same-kit
   body stepping into the union undetected) and should be tested on labelled two-person unions first.

Both are separate stages; nothing in them is needed for the fixes above.

## 8. RECOMMEND INTEGRATION: **YES**

Two proven artefacts — a dropped weak leg box ending the tracklet, and a body at another depth counting as a
contact — removed with existing constants, 0 wrong-player frames, DEV identity/coverage identical, synthetic identical,
one reviewer audit with every HIGH/MED finding fixed or recorded, no change to the contrast margin, the certificate,
ReID or the appearance floor. Trusted coverage: +4 frames on G1_A, unchanged elsewhere; the remaining losses are the
two scoped decisions in §7 and genuine contacts. EVAL not run.

Persistence: commit `276f9dc` (+ this record); raw outputs in `raw/` (`SHA256SUMS.txt`).
