# R2 — RUNNING STRIDE / SIZE-ANOMALY STAGE (2026-11-07)

Instruction: Nathan 2026-11-07 10:34. Commits `deb0d00` (stride rule + tests), `deed6bc` (review-1 fixes),
`e20f421` (output contract), `e179cce` (credible second-person indication, review-2 fixes). Tracker `1.3.1-r2str`,
registry `r2str-1` (one new parameter, `merged_corroboration_min_cover`; the second stage reuses existing constants).
No identity, ReID, appearance-floor, contrast or reacquisition change. Merge thresholds unchanged. EVAL untouched.

## 1. Exact root cause (native 1080p60 decision logs, before any change)

A sprinting body's box widens with every stride (G2_A: width 55 → 97 px with the height constant; area 1.5–1.7 x
the 1 s median at the widest, back to 0.9–1.1 x for only 6–10 frames = 0.1–0.17 s per stride). Geometry cannot
tell that from a union with an unseen second body, so `SIZE_ANOMALY` (abstention) is correct. The failures were
in what turned the anomaly into a MERGE (tracklet closed → LOST) on an isolated body:

| clip | frame | mechanism | fixed by |
|---|---|---|---|
| G2_D | f102 | stride box (area ratio 1.6) corroborated into `MERGED_CONTACT` by a **5 x 5 px corner touch** (0.6 % of the smaller box) from an unrelated two-player box | `_touch` needs a real overlap: ≥ `merged_corroboration_min_cover` (0.05) of the smaller box's area (`deb0d00`) |
| G2_A | f77→f121 | isolated stride anomaly **escalated by duration alone** (`merged_anomaly_max_s`) | duration escalates only after a second-person indication (`deb0d00`) |
| G2_A | f101–114 | the target's **own leg/torso part boxes** (0.1–0.4 x his score, wholly inside, on his feet line) and **tall same-body alternative boxes** (51 x 124 px inside the 67 x 138 px host) counted as a credible nested second body → "indication" → escalation | credible-nested rule, §2 (`e179cce`) |
| G2_A | f121 | an **18 x 143 px sliver** (w/h 0.13 against the body's 0.36, score 0.45, 64 % inside the target box, on his trailing edge) counted as a touching second body → immediate escalation | sliver rule, §2 (`e179cce`) |
| G1_A | f179 | same-kit teammate alongside: `INSUFFICIENT_CONTRAST` → `CONTACT_AMBIGUITY` | genuine contact; out of scope, unchanged |

Verified on the frames (`G2_A_f101_f138_annotated.png`): nobody is near the G2_A target between f77 and f122; the
first real image-space overlap is a farther player passing behind him at f123–126.

## 2. The size / merge rule (population.py, anomaly path only)

- A box continuing its own tracklet whose area ≥ `merged_area_ratio` (1.6) x its single-body reference is a MERGE
  only with corroboration: a real overlap (≥ 0.05 of the smaller box's area) by another body, predicted box or
  memory box, or area ≥ `merged_uncorroborated_area_ratio` (2.5). Otherwise `SIZE_ANOMALY`: never trusted, never
  banked, reference and motion held, tracklet NOT closed (unchanged D11 semantics + the cover floor).
- An active anomaly escalates to a merge (`SUSTAINED_SIZE_ANOMALY`) only (a) immediately when a **credible** body
  touches it while it is not near single size, or (b) after `merged_anomaly_max_s` (1.0 s) once a credible
  second-person **indication** existed at any of its frames. An anomaly with no indication is ISOLATED: never a
  merge by duration; it ends after `merged_anomaly_clear_s` of near-single size and the tracklet continues as an
  UNVERIFIED episode (untrusted until contrast-verified; `target.py`, isolated hold: AMBIGUOUS with the ageing
  test suspended, `size_anomaly_held`).
- **Credible nested indication** (`credible_nested`): a nested box is a possible second person beside the host only
  if the host leaves at least `merged_aspect_band[0]` (0.6) x the single-body reference width beside it (a
  same-body alternative or part box never does: G2_A f101 left 16 px of a ~50 px body); then a box as tall as the
  body (`population_partial_height_ratio`) counts whatever its score (an occluded same-kit player scores low —
  fails closed, review-2 H1), and a short box on the feet line (`population_hidden_feet_gap_max`) counts only above
  `population_weak_part_score_ratio` x the host's score (a leg box is short and weak; a crouched person is not).
- **Credible touch** (`credible_touch`): a touching box is a SLIVER — not an indication — when its w/h <
  `merged_aspect_band[0]` x the host's held single-body w/h; never when it touches the image edge (clipped, not
  narrow); judged against the host's reference, never its anomalous box. A merge-lineage reference falls back to
  the pre-stage form (any touch or nested box).
- Ordinary corroboration of a merge-sized box at first sight (D11: size + any real overlap) is **unchanged**.
- All ratios are scale-free and relative to the body's own reference; seconds, not frames. No new constants in the
  second stage; `merged_corroboration_min_cover` is the only new parameter (documented, `_STRIDE_EV`).

Negative result (recorded): judging a touching tracklet's sliver-ness against its *own* size reference was tried to
answer review-2 M1 (a side-on second person narrower than 0.6 x the target's aspect) and rejected — a persisting
sliver's own reference is a sliver, so it judged itself a body (two tests failed). M1 stays a known limit (§6).

## 3. Tests

`tests_rb/test_stride_anomaly.py` (27 tests, 19 subtests): sprint widening at 30/60 fps never a merge, tracklet
survives; corner touch not corroboration; genuine overlap still corroborates; cover boundary (== cover counts);
credible nested tall / crouched-on-feet-line bodies still escalate; own part boxes wholly inside on the feet line
not an indication (0.1, 0.3, 0.45 x host); feet-line box at exactly the ratio is a part; tall nested box with room
beside it is a person at 0.1 / 0.3 / 0.45 x host; same-body tall alternative without room is not (0.3 / 0.9);
room boundary; sliver fragment mid-anomaly not an indication; body-shaped touch escalates; sliver judged against the
reference, not the stride box; persisting sliver tracklet; D11 first-sight sliver unchanged (known limit);
edge-clipped touching box never a sliver; contact through a hidden tracklet's predicted box; edge-clipped target;
held-not-lost + unverified continuation; one hidden frame trusts nobody; F5 flag path; crouch/jump/turn;
same-kit contact fails closed; camera zoom/pan. 5 of the 12 second-stage tests fail on `e20f421` (reproduction).
`test_merge_guard.py` / `test_population.py` rewritten where the old duration-only escalation was asserted.
Full suite: 984 passed, 9 skipped, 11 xfailed, 1824 subtests (`SUITE_OUTPUT_r2str_indication.txt`); synthetic eval
identical to the F3C reference (`SYNTH_EVAL_r2str_indication.log`).
Reviewer subagent audits: round 1 on `deb0d00` (1 HIGH: crouched nested body; MEDs: hold guards, episode timers —
fixed in `deed6bc`); round 2 on the indication change (1 HIGH + 3 MED validated and fixed, 1 MED documented, L2
registry text updated). `test_sliver_corroboration_of_a_merge_sized_box_at_first_sight_is_unchanged` pins M3.

## 4. 720p DEV (complete 11 clips, labels v1), `sep_ffb14a3` → `e20f421` → `e179cce`

Identical at both commits on every clip: trusted 1152 / 2674 (coverage 0.431), P1 5 frames (C006 f91–93,
C033 f105, C021 f36), P2 0, P3 1, P4 1, bank contamination 10 / 88, 2 seeds refused (C038, C032).
The changed path was exercised population-wide (all bodies, all clips): merged observations 4480 → 4419, duration
escalations 4 → 3, 65 target hold frames — with no change to any trusted frame, P1–P4 or bank sample.
`COMPARE_DEV_sep_ffb14a3_vs_str_e20f421.json`, `COMPARE_DEV_sep_ffb14a3_vs_ind_e179cce.json`.

## 5. Native 1080p (G2_A, G2_D, G1_A), baseline → `e20f421` → `e179cce`

| clip | trusted coverage | false merge events from the stride | first LOST | LOST % | wrong-player frames | hold frames | genuine merge events |
|---|---|---|---|---|---|---|---|
| G2_A | 12.8 → 12.8 → 12.8 % | 1 (f121 duration) → 1 (f121 sliver) → **0** | 137 → 137 → 137 | 68.5 → 68.5 → 68.5 | 0 → 0 → 0 | 0 → 24 → 39 | f126 `MULTI_CLAIM` (real farther player overlapping, kept) |
| G2_D | 17.0 → 17.0 → 17.0 % | 1 (f102 corner touch) → **0** → 0 | 162 → **254** → 254 | 73.0 → **57.7** → 57.7 | 0 → 0 → 0 | 0 → 0 → 21 | f173 `MULTI_CLAIM` (real contact, kept) |
| G1_A | 58.3 → 58.3 → 58.3 % | 0 → 0 → 0 | 209 → 209 → 209 | 24.7 → 24.7 → 24.7 | 0 → 0 → 0 | 0 | same-kit `CONTACT_AMBIGUITY` f179 (kept) |

Trusted frames are on the seeded tracklet only (T002 / T003 / T003) in every run: 0 wrong-player frames.
G2_A after `e179cce`: hold f77–f120 continuous except f105–109 (an `INSUFFICIENT_CONTRAST` blip, contrast layer);
the sliver at f121 is ignored; at f123–126 a farther player (T012, score 0.84) genuinely overlaps the target box in
image space (`MULTI_CLAIM`, union boxes 105 x 164 / 129 x 227 px) — a real merge, kept; the ambiguity clock, which
counts from the last confirmation (f76), then ages the target out at f137 as before. Coverage is therefore unchanged
on this clip by design: the stride is no longer a merge, but a real overlap follows at once. G2_D: the corner-touch
fix alone moved the first LOST from f162 to f254 (−15 pp LOST); what remains is a real two-player box touching at
f102, same-kit contrast at f157, occlusion at f164 and a real contact at f173. G3_C was not rerun: its real second
player is a tracked body touching the widened box (D11 corroboration), a branch this stage does not change.
Pre/post counts: `NATIVE_COMPARE_wp_sep_vs_ind_e179cce.json` (tool `tools_rb/r2_native_compare.py`).

## 6. Known limits / new unknowns

- Coverage during a sustained sprint stays abstained: the near-single phase of a sprint stride (0.1–0.17 s here)
  is shorter than `merged_anomaly_clear_s` (0.2 s), so an isolated anomaly does not end until the player slows.
  Honest abstention (untrusted, not LOST while held); not tuned on this clip.
- The hold suspends the ageing *test*, not the clock: a hold longer than `max_ambiguous_s` followed by any
  non-held frame (contrast blip, rival) goes to LOST at once. Changing it would not have altered G2_A (a real merge
  followed); left as is, documented in `target.py`.
- Review-2 M1: a side-on second person narrower than 0.6 x the target's aspect on the touch frame is not an
  indication; the target stays untrusted and continues UNVERIFIED if the anomaly ends, so identity is not trusted
  on it, but no merged region is created for that frame. M3: a sliver touching a box that is merge-sized on the
  frame it first appears still corroborates (D11 unchanged; pinned by a test).
- Merge-lineage reference fallback (M4) is a defensive branch, not synthetically exercised (the region machinery
  keeps such tracklets corroborated).
- Reported target box during a held anomaly is PREDICTED/HELD (existing AMBIGUOUS contract; the contested
  observation is logged separately) and can drift from the observation under camera pan.

## 7. RECOMMEND INTEGRATION: **YES**

Removes two proven false-merge mechanisms on isolated sprinting bodies (corner-touch corroboration; non-credible
nested/sliver indications) and the duration-only escalation, with: 0 wrong-player frames, 720p DEV identity and
coverage exactly preserved (P1–P4, bank identical), genuine merges on G2_A/G2_D/G1_A still fired, synthetic eval
identical, two reviewer audits, no new thresholds beyond one documented cover floor. It raises no trusted coverage
on these clips by itself: the next failing layers are same-kit contrast (G1_A, G2_D f157) and genuine image-space
overlaps (G2_A f123, G2_D f173), which are separate decisions. EVAL not run.

Persistence: commits above; raw outputs in `raw/` (`SHA256SUMS.txt`): DEV and native runs at `e20f421` and `e179cce`.
