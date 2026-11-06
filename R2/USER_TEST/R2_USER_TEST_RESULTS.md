# R2 USER TRACKER TESTING — RESULTS (2026-11-06)

Instruction: Nathan 2026-11-06 12:55. Tracker: frozen `1.0.0-r2prod` (`4cc95cf` logic; evaluation tooling at
`fc18d71`), `TrackerConfig.default()`, no overrides, no logic/threshold/config change. Footage: 12 fresh 10 s clips
from three matches never used by R2 (`CLIP_REGISTRY.md`), 640x360 @ 30 fps. Targets picked by Nathan on the seed
stills (`USER_CHOICES.json`; "G4_D: 23" read as G3_D). Visual MP4s: Nathan's Mac `_spana_usertest_work/visual/`
(hashes in `raw/VISUAL_MP4_SHA256SUMS.txt`); raw tracker outputs in `raw/`.

## Per clip

| clip | scenario | target (px tall) | seed | trusted coverage | TRACKING / AMBIGUOUS / LOST % | wrong player | recovery |
|---|---|---|---|---|---|---|---|
| G1_A | open play, tiny players, static wide cam | 56 | accepted | 29/300 = **9.7%** (f1–59) | 9.7 / 20.1 / 54.8 (+15.4 REACQUIRING) | none | no (REACQUISITION_UNVERIFIED) |
| G1_C | pan, tiny players | 20 | **REFUSED** LOW_RESOLUTION (crop < 24 px) | – | – | – | – |
| G2_A | open play, wide | 49 | accepted | 25/300 = **8.4%** (f2–38) | 8.4 / 14.4 / 77.3 | none | no |
| G2_B | attack, closer cam | 34 (13 px wide) | **REFUSED** "not usable: ()" = LAB colour band < 150 px | – | – | – | – |
| G2_D | counter / sprint | 51 | accepted | 11/300 = **3.7%** (f1–27) | 3.7 / 12.7 / 80.9 | none | no |
| G2_E | touchline crowd | 141 | **REFUSED** NOT_ISOLATED / PARTIAL / TOUCHES_IMAGE_EDGE, no clean crop within seed_anchor_max_s | – | – | – | – |
| G2_F | box crowd before set piece | 103 | accepted | 9/300 = **3.0%** (f1–9) | 3.0 / 8.4 / 87.0 | none | no |
| G2_G | open play | 35 | **REFUSED** bank anchor LOW_QUALITY (crop quality < 0.6) | – | – | – | – |
| G3_A | open play, pan | 44 | accepted | 5/300 = **1.7%** (f1–16) | 1.7 / 4.0 / 84.6 (+9.7 OCCLUDED) | none | no |
| G3_B | corner, box crowd, same kits | 37 | **REFUSED** OCCLUDED_BY_OTHER | – | – | – | – |
| G3_C | open play, pan | 44 (17 px wide) | **REFUSED** "not usable: ()" = LAB band < 150 px | – | – | – | – |
| G3_D | counter, two-player contact seed | 51 | accepted | 3/300 = **1.0%** (f1–8) | 1.0 / 8.4 / 87.3 | none | no |

Aggregate (6 clips that ran): trusted 82 / 1800 frames = **4.6%**; every trusted frame is on the seeded tracklet in
the first 0.3–2.0 s; 0 wrong-player episodes; 0 reacquisitions; 6 / 12 seeds refused; runtime 9–12 min per 10 s clip
on 2 CPUs (~0.5 fps).

## Failure pattern — one, repeatable, in every clip that ran

Sequence in all 6 runs: TRACKING on the seed tracklet -> AMBIGUOUS -> LOST exactly `max_ambiguous_s` (1 s) after the
last confirmation, then LOST to the end (AGED_OUT), never reacquired. The target was still detected, in the motion
gate, with plausible motion (decision logs: `in_gate true`, `plausible true`, displacement < 0.3 body heights) — the
tracker abstains because its appearance evidence is declared unusable or contaminated, not because it lost the body.

First failing layer: **appearance crop quality at 640x360** (layer 6, physical evidence), two mechanisms:

1. **Fixed-pixel appearance minima** (`lab_min_band_pixels` 150, `crop_min_height_px` 24, `crop_good_height_px` 64,
   via `bank_min_crop_quality` 0.6). A 45–50 px body is ~13–17 px wide; the central 60 % of a 35 % band is
   8–10 x 16 px = 130–160 pixels, so the LAB descriptor is None -> `FEATURE_NOT_USABLE` -> no confirmation.
   G1_A: 38 AMBIGUOUS frames of `FEATURE_NOT_USABLE` on the correct, in-gate box, then LOST. G1_C, G2_B, G2_G,
   G3_C refused at the seed for the same minima. These are the pixel constants rule 15 flags; they were calibrated on
   720p DEV crops (52–112 px median).
2. **Nested low-score duplicate boxes read as a second person.** At this resolution the detector (2x2 tiles + full
   frame, min score 0.10, NMS 0.65) emits partial-body boxes inside the target box (same effect as the 15 % duplicate
   rate in `SEED_STILL_AUDIT.md`). The population flags the target observation `CONTAINS_OTHER` -> target
   `MERGED_DETECTION_SUSPECTED` -> no bank update, no confirmation -> AMBIGUOUS -> LOST. G2_A: 7 such frames with
   inner boxes scoring 0.10–0.25; G1_A 14 frames, inner scores 0.11–0.25 (duplicates of the target itself).
   In G2_F (box crowd) and G3_D (contact) the inner boxes score 0.6–0.85 and are real players: there the fail-closed
   behaviour is correct.

Both are generic resolution effects, not clip-specific: the same thresholds that keep 720p DEV safe make 360p
unusable. The handoff rule (OFF) is not involved; no wrong identity was produced anywhere.

## Successes
- 0 wrong-player frames in 1800; the tracker never trusted anyone but the seeded body.
- Seed refusals were honest and specific (crowd/edge/occlusion/low resolution); G3_B's refusal (same-kit corner
  crowd) and G2_E's (touchline melee) are the intended behaviour.
- Camera pans (G3_A, G3_D) and the touchline melee did not create a wrong seed or a wrong reacquisition.

## Scenario coverage
Covered by seeds that ran: normal movement, sprint (G2_D), pan (G3_A/D), crowd (G2_F), contact (G3_D). Not
assessable because the tracker was LOST within 2 s: turns/pose changes, same-kit crossings, occlusion, leaving and
re-entering the frame, recovery. Coverage of the scenario list is therefore incomplete at this resolution.

## TRACKER PRACTICALLY READY: **NO** (on 640x360 / 30 fps broadcast footage)
Not usable: 6 / 12 seeds refused and 4.6 % trusted coverage on the rest, although identity safety held (0 wrong
player). The DEV result (0.426 coverage at 1280x720 / 60 fps) does not transfer to this resolution.

## Diagnosis of the repeatable failure (no code changed)
The failure is the appearance layer's fixed-pixel minima and the duplicate-box "contains other" reading at low
resolution. Any change must be a general rule (rule 15: body-height-normalised crop/band requirements with a
documented floor; rule 33: simplest mechanism), tested at unit / synthetic / real levels on BOTH 720p DEV (no
regression on P1–P4 or coverage) and these 360p clips, and must not weaken the "second body inside the box" test
where the inner box is a confident detection (G2_F, G3_D). Before any change: confirm the same clips at a higher
source resolution are not available (these are 360p downloads; the DEV sources are 720p), since the honest answer
may be "minimum supported body height ~50 px, minimum width ~20 px" as a documented input requirement rather than a
threshold change.
