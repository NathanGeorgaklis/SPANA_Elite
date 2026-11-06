# R2 — SAME-KIT CONTRAST: PHYSICAL-SEPARATION EXCLUSION (2026-11-07)

Instruction: Nathan 2026-11-07 08:22. Commits `6e000d4` (rule + tests), `ffb14a3` (review fixes). Tracker
`1.2.0-r2sep`, registry `r2sep-1`. No appearance / trust threshold lowered; contrastive_margin 0.08 unchanged.
EVAL untouched.

## 1. Exact root cause
The contrast test (target-bank similarity minus the best competitor-bank similarity on the same crop, must be
>= 0.08) cannot separate two same-kit bodies: with DINOv2 + colour the difference is 0.03–0.07 whichever body the
crop holds. Native 1080p, frozen tracker: G1_A spent 45 frames `INSUFFICIENT_CONTRAST` at contrast 0.074 with the
same-kit teammate observed on his own tracklet 0.57–0.59 body heights away, never overlapping, continuously since
the last trusted frame, then LOST at the 1 s ambiguity limit. The margin is not "too strict" — it is the wrong
instrument for same kits; what separates the two bodies there is physical continuity, which the contrast test
ignored. G3_C (rival touching, 0.03–0.07 h) and G2_F (rival overlapping, IoU 0.07–0.10) are genuinely ambiguous;
G1_C also fails the appearance floor (composite 0.77 < 0.80), so only part of it is a contrast case.

## 2. New rule (target.py, `separation_exclusion_enabled`, continuation candidates only)
A banked competitor is dropped from a CONTINUATION candidate's contrast when ALL hold:
- it is observed this frame on its own tracklet, on a box outside the candidate's nested group;
- it has been continuously observed and apart from the target since the target's last TRUSTED confirmation: never
  overlapping or within `separation_min_gap_h` = 0.25 body heights edge to edge, no unobserved frame, no camera cut;
- the target's own tracklet was observed on every frame since that confirmation (same tracklet id), never clipped
  (PARTIAL / image edge);
- the candidate already passes gate, motion, appearance floor and merge tests;
- the episode is contrast-VERIFIED (an unverified reacquisition episode always contrasts in full).
Two continuously and separately observed bodies cannot have exchanged identities. Hidden, newly seen, touching,
crossing, occluded or clipped competitors compete in full; a rival excluded on a frame is never recorded as a known
other on that frame; the deferred-seed window keeps the bookkeeping. Evaluations log `excluded_competitors`.

## 3. Tests and review
`tests_rb/test_separation_contrast.py` (13): teammate alongside at cos 0.85/0.9, 30/60 fps -> trusted (OFF -> LOST,
documented); camera pan; same-kit crossing fails closed after first approach; same-spot substitution identical to
OFF and still caught by the optional handoff rule; look-alike appearing after the last trusted frame -> never
trusted, nothing excluded; two equally plausible reacquisition candidates -> ambiguous; contact / occlusion competes
in full; review cases: unverified episode never excludes and never verifies on a rival, deferred-seed window,
one held target frame voids the certificate, exact gap counts as contact, rival tracklet-id change. Reviewer
subagent on `6e000d4`: 2 HIGH (unverified-episode self-verification; no bookkeeping in the deferred-seed window),
2 MED, 3 LOW — all reproduced/validated and fixed in `ffb14a3`. Full suite 957 passed; synthetic eval identical.

## 4. 720p DEV (complete, labels v1): `e093570` (weak-part) vs `ffb14a3`
| | pre | post |
|---|---|---|
| trusted / available | 1138 / 2674 (0.426) | **1152 / 2674 (0.431)** |
| P1 / P2 / P3 / P4 | 5 / 0 / 1 / 1 | **5 / 0 / 1 / 1** (identical frames) |
| bank contaminated / accepted | 10 / 86 | 10 / 88 |
| seeds refused | 2 | 2 |
Only SYO C016 changed: 136 -> 150 trusted frames (0.651 -> 0.718), 31 frames used the exclusion, all on the labelled
target. Every other clip identical.

## 5. Native 1080p (G1_A, G1_C, G2_F, G3_C), previous run vs `ffb14a3`
| clip | trusted coverage | same-kit ambiguity (AMBIGUOUS + INSUFFICIENT_CONTRAST frames) | first LOST | wrong-player frames | frames using exclusion |
|---|---|---|---|---|---|
| G1_A | 44.5 % -> **58.2 %** | 52 -> 32 | f190 -> f209 | 0 -> 0 | 176 |
| G1_C | 8.0 % -> 8.0 % | 22 -> 22 | f70 | 0 -> 0 | 1 |
| G2_F | 2.3 % -> 2.3 % | 55 -> 55 | f75 | 0 -> 0 | 27 (never the deciding competitor: the overlapping rival competes) |
| G3_C | 5.7 % -> 5.7 % | 14 -> 14 | f48 | 0 -> 0 | 9 (idem) |
G1_A now stays trusted through the teammate-alongside stretch (f40–f178, contrast 0.07–0.11); it ends at f179 when
the teammate actually walks into contact (gap 0, OVERLAPS_OTHER / CONTAINS_OTHER) and the target box carries a
size anomaly — the stride/size-merge mechanism reported earlier, fail-closed and out of this stage's scope. G1_C
ends on the appearance floor; G2_F and G3_C on genuine contact. Reacquisitions: 0 -> 0 everywhere (unchanged).

## 6. Substitution safety
Unchanged: same-spot substitution produces the same frames as with the rule OFF (the known production-default
limit, caught by the optional handoff rule, which the exclusion does not disturb); a crossing, an overlap, a hidden
rival, a new rival, an unverified episode and a clipped target box all void the certificate (tests). No wrong-player
frame appeared anywhere (DEV 5 unchanged, native 0).

## 7. Runtime
Native: −17 % to −32 % per clip (447 / 518 / 976 / 622 s vs 540 / 745 / 1436 / 734 s), DEV comparable; both
confounded by shared CPU; the rule adds one pass over banked competitors per continuation frame.

## 8. RECOMMEND INTEGRATION: **YES**
Removes a proven failure of the contrast test on same kits without touching any threshold, with a certificate that
is void whenever a swap is physically possible; DEV identity safety identical, coverage +0.5 pt on DEV and +13.7 pt on
the affected native clip; audited, with both HIGH findings fixed and tested. Next failing layers on this footage,
unchanged and for separate decisions: the stride size-anomaly merge test (G1_A end, G2_A, G2_D) and the appearance
floor on small bodies (G1_C).
