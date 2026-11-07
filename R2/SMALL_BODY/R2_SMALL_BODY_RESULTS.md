# R2 — SMALL-BODY APPEARANCE USABILITY (2026-11-07)

Instruction: Nathan 2026-11-07 17:17. Commit `8312f2f`. Tracker `1.3.3-r2sb`, registry `r2sb-1`
(`lab_min_band_pixels` 150 → 50; new `lab_min_band_columns` 6). Appearance floor (0.80), embedding floor (0.70),
contrast margin (0.08), reacquisition thresholds, the ReID model, the crop preprocessing, `crop_min_height_px` (24)
and all separation / stride / merge / handoff logic are unchanged. EVAL untouched.

## 1. Exact first failing layer

| case | what fails | class |
|---|---|---|
| **G1_C (native 1080p)** — target T018, 19–43 × 39–72 px | never a crop-usability gate (0 FEATURE_NOT_USABLE / LOW_RESOLUTION / LAB_MISSING on 131 target frames). `EMBEDDING_BELOW_FLOOR` on 94/131, `COMPOSITE_BELOW_FLOOR` 106/131: the generic DINOv2 similarity on a ~60 px *running* body swings 0.41–0.94 with pose against a 4-sample walking-pose bank (f0/16/24/32). LAB stays 0.89–0.99. | **embedding discriminability** (genuinely insufficient visual information for the generic embedding at this size), not crop size, not the pipeline |
| other native targets (G1_A, G2_A, G2_D, G2_F, G3_C; 104–319 px) | no usability failure of any kind on the target | — |
| **360p user test** — G1_A target T004, 32–54 px tall (above the 24 px floor) | `FEATURE_NOT_USABLE` on 117/171 target frames: the LAB **band pixel floor** (150) — a 45 px body gives ~16 rows × 9 central columns ≈ 144 px | **resolution-specific pixel assumption** (LAB band requirement), the instruction's suspected class |

Diagnosed separately: (a) fixed crop requirements — `crop_min_height_px` 24 never bit a target at 1080p or 360p;
(b) LAB band pixels — the 360p failing layer; (c) DINOv2 crop quality — measured on G1_C's own crops: INTER_CUBIC /
LANCZOS4 upscaling leave same-person similarity unchanged (median 0.687 → 0.682 / 0.680), 10 % context padding
lowers it (0.668) and raises the different-person tail; (d) body aspect — the 22–44 px width swing of the stride is
the pose variation the embedding fails on; (e) blur/compression vs information — at 55–70 px, population-wide,
80 % of same-tracklet pairs reach the embedding floor vs 92–100 % at 120+ px; at G1_C's target the same-person
p25 (0.59) lies below the different-person p95 (0.77). No preprocessing change is justified; abstention is correct.

## 2. The scale-normalised rule

The LAB bands are already fractions of the box (0.15–0.50 h, 0.50–0.80 h, central 60 % of the width); the only
pixel assumption was the 150-pixel band floor. It is now set from measurement, not resolution:

- `lab_min_band_pixels` = 50: the lower band (0.30 h) of a ~30 px tall, 10 px wide body (9 rows × 6 central
  columns) — the smallest band at which the per-clip measurement shows the colour cue behaving as on large bodies;
- `lab_min_band_columns` = 6: a one- or two-column strip is not a colour sample however tall, and the detector's
  ±1 px box error stays ≤ 1/6 of the band (review H2: on a 24 × 6 crop a 1 px offset plus one shared row lifted an
  incompatible kit pair to 0.315 ≥ `lab_floor`; `test_the_rejected_geometry_would_have_lost_the_veto` pins it);
- `crop_min_height_px` 24 stays the conservative absolute floor; crops 24–29 px tall or < 10 px wide remain
  LAB-less (unusable = UNKNOWN, as before).

Measurement (`tools_rb/r2_appearance_lab_band_probe_per_clip.py`, `LAB_BAND_PROBE_per_clip.json`): 8 clips (5 at
360p, 3 at 1080p), crops ≥ 24 px, same tracklet 8–48 frames apart vs other tracklets of similar height on the same
frame, binned by the lower band. In the 50–150 band-pixel bins the other-tracklet veto rate (< 0.3) and p95 match
the same clip's ≥ 150 bins (360p G2_D 0.077/0.089 vs 0.008/0.000; 1080p G3_C 0.54/0.57 vs 0.48/0.39/0.20; 1080p
G1_A 0.000/0.003 vs 0.008/0.008/0.039; 360p G1_A 0.006/0.000 vs 0.000); same-tracklet vetoes 0.000 in every bin of
every clip; same-kit (lifetime-median LAB ≥ 0.8) p5 0.67–0.90, never vetoed. The 28–50 bin rests on one clip
(n = 103) and is not admitted. The earlier pooled probe (`LAB_BAND_PROBE.json`, bins by the upper band, crops
≥ 12 px) is kept as the superseded first pass: its sparse-bin veto collapse came from sub-floor crops mixed across
clips (review H1/H3).

Side effects, documented: the old 150 was by accident also the bank's size floor (a 1:3–1:4 body needed ~50–58 px
for LAB); the designed bank floor `bank_min_crop_quality` × `crop_good_height_px` = 38.4 px now applies as designed
(review M2). A LAB cue on 30–50 px bodies also makes them competitors in the contrast test and possible rivals;
their embedding floor behaviour is the same as at larger sizes (same ≥ floor 0.87–0.93, different ≥ floor
0.27–0.43 at 30–45 px vs 0.92 / 0.43 at 120–160 px) — the floor is a weak filter at every size; identity rests on
contrast and physical reasoning, unchanged (review M1).

## 3. Tests and audit

`tests_rb/test_appearance.py` +13: smallest measured body usable (30 × 10, 36 × 12); floor geometry pinned to its
stated derivation; sparser bands, 1–3-column strips, the 24 × 6 geometry and 26 × 10 refused; column boundary;
the floor applies to the lower band; tiny kits (30 × 10 … 40 × 14) still discriminate; blurred crop matches its
sharp self; partial crop judged on what it shows; box offset + shared row at the smallest admitted crop still
vetoes an incompatible kit; the rejected geometry would not; constant crop finite; a 20 px crop stays unusable.
Full suite 1015 passed, 9 skipped, 11 xfailed, 1857 subtests (`SUITE_OUTPUT_r2sb.txt`). Synthetic eval
(`SYNTH_EVAL_r2sb.log`, diff vs F3C v5 in `SYNTH_EVAL_r2sb_vs_f3c_v5.diff`): two scenes change in the safe
direction — `leave_and_return` cov 0.228 → 0.241 (amb 0.013 → 0), `wrong_reacquisition` cov 0.947 → 1.0
(amb 0.053 → 0); swaps / wrong / fci 0 unchanged, the impostor still not reacquired. Cause: the synthetic 28 × 70 px
target's edge-clipped crops (≥ 60 % inside, ≥ 6 central columns) are now usable. New reference
`f3c_synth_v6_r2sb.log`; v5 preserved. Reviewer audit: 3 HIGH, 4 MED, 4 LOW — H1/H3 answered by the per-clip
re-measurement, H2/M3/M4 fixed (column floor, floors moved to the measured region, description), M1/M2 documented,
L2 tests added.

## 4. Results

| set | appearance usability (candidate features usable) | seed acceptance | trusted coverage | wrong-player frames | P1–P4 | bank contamination |
|---|---|---|---|---|---|---|
| 720p DEV (11 clips), `276f9dc` → `8312f2f` | unusable 3463 → 1258 of 17922 (19.3 → 7.0 %); candidates with a competitor 11554 → 13432 | 9/11 → 9/11 | 1152 / 2674 = 0.431 → identical on every clip | DEV P1 5 → 5 (same frames) | 5/0/1/1 → 5/0/1/1 | 10 / 88 → 10 / 88 |
| native 1080p, 6 clips | G1_C 600 → 64 of 2652; G1_A 137 → 3 of 1097 | 6/6 → 6/6 | identical on all six (G1_C 8.3 %, G1_A 59.7 %, G2_A 12.8 %, G2_D 17.0 %, G2_F 2.5 %, G3_C 6.0 %) | 0 → 0 | — | unchanged |
| 360p G1_A, control `f18dcb8` → `8312f2f` | target feature unusable 118/172 → **0/171**; all candidates 2793/2958 → 557/1575 | 1/1 | **30 → 153 trusted (10.0 → 51.0 %)**, first LOST f90 → f202, LOST 54.7 → 32.7 % | 0 observed (all trusted frames on the seeded tracklet T004; `G1_A_360p_trusted_stretch_check.png`) | — | bank samples 6 → 18 |

The control at `f18dcb8` reproduces the frozen 360p result exactly (30 trusted), so the whole 360p gain is this
change alone. False usable classifications on genuinely poor crops: crops below 24 px, < 10 px wide, mostly
outside the image or strips stay unusable (tests); on DEV and native no target gained or lost a trusted frame.

## 5. Known limits / new unknowns

- G1_C stays at 8.3 %: the generic embedding cannot separate a ~60 px running body from same-size others reliably
  (same-person p25 < different-person p95); only a ReID model change or a pose-diverse bank could move it (out of
  scope; the bank is rate-limited at `target_bank_sample_interval_s`).
- `crop_good_height_px` (64) remains an absolute pixel assumption inside the bank quality term (bank eligibility
  needs h ≥ 38.4 px); not changed, flagged for a later measurement.
- Probe "same" pairs are same-tracklet (tracker errors contaminate slightly; same-tracklet vetoes 0.000 suggests
  little); "different" pairs are dominated by different kits — same-kit separation is the contrast layer's job.
- A box bottom that is not the feet (airborne) or a sliver is still refused by the column floor only when narrower
  than 10 px.

## 6. RECOMMEND INTEGRATION: **YES**

A measured, scale-aware floor replaces an untested absolute one; no identity threshold, model or logic change;
DEV identity, coverage and bank identical; native identical; the one footage where the old floor bit (360p) gains
41 pp trusted coverage on the correct player; audited, with every finding fixed, re-measured or documented. EVAL not
run.

Persistence: commit `8312f2f` (+ this record); raw outputs in `raw/` (`SHA256SUMS.txt`): DEV, native (6), 360p
G1_A and its control.
