# R2 — NESTED PART vs REAL SECOND PLAYER: WEAK-PART RULE (2026-11-07)

Instruction: Nathan 2026-11-07 03:24. Commits `8f3d094` (rule + tests), `6bf0aa4` (review fixes), `e093570`
(edge-clipped hosts excluded). Tracker `1.1.0-r2wp`, registry `r2wp-1`. No identity, appearance, reacquisition,
seed, handoff or merge-threshold change. EVAL untouched.

## 1. Diagnosis correction (what the four clips actually showed)

Frame-level tracing of the native 1080p decision logs, before any change:

| clip | first merge event | mechanism | verdict |
|---|---|---|---|
| G1_A f166–168 | 0.12-score leg box 0.71 inside the 0.66-score target body, 1.11 x its width, centre offset 0.14, missed `population_nested_containment` 0.75 → started tracklet T116 → it vanished into the host → MULTI_CLAIM merge → target closed, LOST for the rest of the clip | **weak nested part read as a second body** | the proven mechanism of this stage |
| G2_A f77 | isolated 0.91-score target box, nothing touching it, width 55 → 87 px with the running stride → `SIZE_ANOMALY` (area ≥ 1.6 x the established median) → target abstains 44 frames → LOST | size-only merge test on one body's stride | a different mechanism; not touched |
| G2_D f102 | target box width 50 → 87 px with the stride → `AREA_RATIO`, corroborated by a 5-px corner touch from an unrelated two-player box → MERGED_CONTACT | same stride mechanism + trivial corroboration | not touched |
| G3_C f25 | a red player directly behind the blue target (0.7-score box, T089) inside the widened target box → `AREA_RATIO` corroborated | **genuine second person** | correct fail-closed behaviour |

So the earlier "nested low-score duplicates → merge" diagnosis was right for G1_A only; counting `CONTAINS_OTHER`
frames had overstated it. The stride/size mechanism (G2_A, G2_D) is reported in §7 for a separate decision.

## 2. Exact suppression rule (population.py step 1, flag `WEAK_PART`)

A detection j is NESTED in a larger same-frame detection i (host) when ALL hold:
- `score_j <= population_weak_part_score_ratio (0.5) x score_i`;
- fraction of j inside i `>= population_weak_part_min_containment (0.6)` (plain nesting stays at 0.75);
- `height_j < height_i` and `height_j >= population_part_box_min_height_ratio (0.3) x height_i` (existing part rule);
- `width_j <= population_weak_part_max_width_ratio (1.25) x width_i`;
- `|centre_x_j - centre_x_i| <= population_weak_part_max_centre_offset (0.3) x width_i`;
- the host does not touch the image edge (a clipped box is not a whole body).

Effect: j starts no tracklet, is not second-body corroboration for its host, is not fed to the hidden-body rule 4b,
stays an observation and an overlap/isolation competitor (ALTERNATIVE, like any unassigned nested box); the host is
`CONTAINS_OTHER` and the target never banks its crop while a weak part is inside (`WEAK_PART_INSIDE` bank refusal)
but still confirms identity on it. All ratios of the host's own score/width/height: scale-free, no pixel or clip
constants. Known limit: a part box that scored above the ratio on the frame it was born already has a tracklet and
can still produce the old merge (fails closed). The score ratio depends on the pinned detector's calibration.

## 3. Tests

`tests_rb/test_weak_part.py` (16): upper-body duplicate inside (plain nesting unchanged); lower-body part poking out
below the feet; 720p and 1080p scale; crouched person with comparable score stays a body; same-kit player beside the
host stays a body; weak box mostly inside but off-centre stays a body; taller-but-smaller-area and 1.4x-wide boxes
stay bodies; genuine credible overlap still fails closed (two tracklets); part appearing/disappearing leaves no
phantom; part persisting beyond `population_hidden_min_s` does not end the host (review HIGH); host not
second-body-corroborated by its part; score/width boundaries; host at the image edge is never a weak-part host;
target bank refuses while a weak part is inside and resumes after; pre-rule behaviour documented (rule disabled →
phantom tracklet). Full suite: 944 passed, 9 skipped, 11 xfailed, 1785 subtests; synthetic eval identical to the
F3C reference. Reviewer subagent audit on `8f3d094`: 1 HIGH, 3 MED, 3 LOW — all validated and fixed or documented
(`6bf0aa4`).

## 4. 720p DEV (complete 11 clips, labels v1), production `4cc95cf` vs `e093570`

Identical on every clip: trusted 1138 / 2674 (coverage 0.426), P1 5 frames (C006 f91–93, C033 f105, C021 f36),
P2 0, P3 1, P4 1, bank contamination 10 / 86, 2 seeds refused (C038, C032). `COMPARE_DEV_prod_4cc95cf_vs_e093570.json`.
Intermediate `6bf0aa4` (before the edge exclusion) had added one wrong-identity frame on C033 (f106: the clipped
target box held the other player's upper body at score 0.38) — the reason for the edge rule; recorded in
`COMPARE_DEV_prod_4cc95cf_vs_6bf0aa4_intermediate.json`. Eight DEV clips of the final set are copied from the
`6bf0aa4` run (`COPIED_FROM_6bf0aa4.txt`: no edge-host weak part in their logs, so the pipeline is deterministic-identical);
C016, C033, C012 were rerun at `e093570`.

## 5. Native 1080p (G1_A, G2_A, G2_D, G3_C), `2743f5e` vs weak-part rule

| clip | trusted coverage pre → post | MERGED_DETECTION_SUSPECTED target frames pre → post | escaped weak parts (carrying a tracklet) pre → post | wrong-player frames | reacq |
|---|---|---|---|---|---|
| G1_A | 44.5 % → 44.5 % | **132 → 3** | 8 (6) → 0 | 0 → 0 | 0 → 0 |
| G2_A | 12.7 % → 12.7 % | 523 → 523 | 171 (138) → 0 | 0 → 0 | 0 → 0 |
| G2_D | 16.9 % → 16.9 % | 498 → 498 | 20 (14) → 0 | 0 → 0 | 0 → 0 |
| G3_C | 5.7 % → 5.7 % | 275 → 275 | 21 (17) → 0 | 0 → 0 | 0 → 0 |

G1_A: the false merge and the phantom tracklet are gone (no MERGED_CONTACT closure, the target box is observed with
composite similarity 0.86–0.89 and in gate from f168 on) — but the frame is AMBIGUOUS anyway for
`INSUFFICIENT_CONTRAST`: the same-kit teammate standing beside the target keeps the contrast at 0.02–0.06
(< `reacq_contrastive_margin` 0.08), so LOST arrives at the same second as before. The next failing layer is
same-kit contrast, outside this stage. G2_A / G2_D / G3_C: unchanged, as diagnosed.
Native weak-part census on the four clips: 220 escaped weak-part boxes (175 carrying a tracklet) → 0.

## 6. Other reported quantities

- Seed acceptance: 720p 9/11 → 9/11; 1080p (4 clips) 4/4 → 4/4. The seed-frame nesting check is unchanged.
- LAB / appearance usability: unchanged by design (no `FEATURE_NOT_USABLE` / `LOW_QUALITY_CROP` at 720p/1080p before
  or after). Appearance code untouched.
- False merge frames from nested duplicates: G1_A 132 → 3; DEV: none existed on the target (DEV identical).
- Genuine second-person misses: none observed. Credible second people (score ratio > 0.5, beside, taller, wider, or
  at the image edge) are untouched; G3_C's real second player still merges; synthetic crouched/beside cases pass.
  The weak-score crouched person inside a non-clipped host is the accepted twin: still a competitor and never
  banked, but no longer a tracked body.
- Phantom tracklets prevented: DEV 3–36 per clip → 0–4 (the remainder are edge-host boxes, excluded on purpose);
  native 8–171 per clip → 0.
- Runtime: DEV reruns −7 % to +28 % per clip, native −15 % to −22 %; both runs shared the 2 CPUs with other work, so
  runtime is confounded. The rule adds one O(n²) pass over same-frame bodies and removes phantom tracklets; no
  measurable cost.

## 7. RECOMMEND INTEGRATION: **YES** — with its effect stated honestly

Integrate: it removes a proven false-merge mechanism (phantom tracklets from weak part boxes), is generic and
scale-free, preserves 720p DEV identity safety and coverage exactly, adds no wrong-player frame, and was audited.
It does **not** raise coverage on any clip by itself, because the layer behind it (same-kit contrast, then the
strict reacquisition floor) fails next. Candidates for the next single-mechanism stage, in first-failing-layer
order on this footage: (a) the size-only merge test tripping on a running stride of an isolated body (G2_A, G2_D:
`merged_area_ratio` 1.6 vs stride area ratios 1.5–1.8; `SIZE_ANOMALY` treated as merged by the target), (b)
same-kit contrast with a teammate alongside (G1_A, G1_C, G2_F, G3_C). Both are separate decisions; nothing was
changed there.

Persistence: see commit messages; raw outputs in `raw/` (SHA256SUMS.txt).
