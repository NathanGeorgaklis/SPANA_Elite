# R2 USER TRACKER TESTING — NATIVE-RESOLUTION RERUN (2026-11-07)

Instruction: Nathan 2026-11-06 23:00. Same 12 windows, same targets, frozen `1.0.0-r2prod`, production defaults,
nothing tuned, no EVAL. Tooling commit `2743f5e`.

## Native sources
The 360p files were YouTube's muxed 360p stream. The same three matches were identified on YouTube by exact duration
(`NATIVE_SOURCES.txt`): Sydney United 58 v APIA (`43Z1ZlBdVTo`) and Sydney FC v St George City (`XOiNDpghkno`) at
**1920x1080, 30 fps, H.264 ~3.3 / 2.9 Mbps** (itag 137); NWS Spirit v SD Raiders (`dIJucho_IV8`) at
**1920x1080, 60 fps, ~4.2 Mbps** (itag 299). 360p files were ~0.45–0.53 Mbps. The 12 windows were downloaded as
22 s sections (stream copy), content-aligned to frame 0 of each 360p clip (`ALIGNMENT.json`: every window starts
exactly 8.0 s into its section; match error 1.9–4.3 grey levels vs 4.5–25 for the runner-up) and cut losslessly
(`-qp 0`); hashes in `NATIVE_CLIP_SHA256SUMS.txt`. Targets: the 360p picks mapped to the native seed boxes by IoU
(`TARGET_MAPPING_360_TO_NATIVE.json`; 11/12 unambiguous, IoU 0.77–0.99; G3_D is a two-player contact that the
1080p detector separates, re-confirmation requested from Nathan, not run yet).

## Per clip (native) vs 360p

| clip | target px (360p -> native) | 360p | native | native trusted span | native loss reasons (AMBIGUOUS frames) |
|---|---|---|---|---|---|
| G1_A | 56 -> 158 | 9.7 % | **44.5 %** | 0–5.3 s, LOST from 6.3 s | INSUFFICIENT_CONTRAST 52, MERGED_SUSPECTED 22 (inner boxes score 0.12–0.31) |
| G1_C | 20 -> 63 | refused (LOW_RESOLUTION) | 8.0 % | 0–1.3 s | BELOW_APPEARANCE_FLOOR 45, INSUFFICIENT_CONTRAST 22 |
| G2_A | 49 -> 143 (60 fps) | 8.4 % | 12.7 % | 0–1.3 s | MERGED_SUSPECTED 60 (inner boxes 0.10–0.26), REACQUIRING 9.8 % unverified |
| G2_B | 34 -> 102 (60 fps) | refused (LAB band) | **95.0 %** | whole clip | 30 AMBIGUOUS frames (floor), never LOST |
| G2_D | 51 -> 150 (60 fps) | 3.7 % | 16.9 % | 0–1.7 s | MERGED_SUSPECTED 60 (inner 0.10–0.47), COMPETING_CANDIDATE 14 |
| G2_E | 141 -> 426 (60 fps) | refused (NOT_ISOLATED) | refused (NOT_ISOLATED, no clean crop in seed_anchor_max_s) | – | touchline melee |
| G2_F | 103 -> 306 (60 fps) | 3.0 % | 2.3 % | 0–0.2 s | INSUFFICIENT_CONTRAST 55, COMPETING 38, OCCLUDER 21 (box crowd) |
| G2_G | 35 -> 121 (60 fps) | refused (LOW_QUALITY) | refused (PARTLY_COVERED_BY_OTHER) | – | |
| G3_A | 44 -> 122 | 1.7 % | refused (NOT_ISOLATED; seed tracklet not observed) | – | |
| G3_B | 37 -> 107 | refused (OCCLUDED) | refused (NOT_ISOLATED; composite 0.819 < 0.85) | – | corner crowd |
| G3_C | 44 -> 130 | refused (LAB band) | 5.7 % | 0–0.6 s | MERGED_SUSPECTED 23, INSUFFICIENT_CONTRAST 16 |
| G3_D | 51 -> 152 | 1.0 % | pending user pick | | |

## Aggregate comparison

| | 360p (12) | native (11 run so far) |
|---|---|---|
| seeds accepted | 6 / 12 | 7 / 11 |
| trusted coverage (run clips) | 82 / 1800 = 4.6 % | 1061 / 3295 = **32.2 %** (median clip 12.7 %; G2_B 95 %, G1_A 44.5 %) |
| wrong-player frames | 0 | **0** (every trusted frame on the seeded tracklet) |
| reacquisitions | 0 | 0 (G2_A entered REACQUIRING 59 frames, never verified) |
| appearance usability | LAB descriptor missing on 45–50 px bodies (`FEATURE_NOT_USABLE`), 4 seed refusals for crop size | **no `FEATURE_NOT_USABLE` / `LOW_QUALITY_CROP` anywhere**; no crop-size refusals |
| nested low-score duplicate -> MERGED_SUSPECTED | G2_A 7, G1_A 14 frames | **still present and larger**: G2_A 60, G2_D 60, G1_A 22, G3_C 23 frames (inner boxes score 0.10–0.3) |
| runtime (2 CPUs) | 9–12 min / 10 s clip | 11–15 min (1080p30), 16–27 min (1080p60) |

## Reading
1. The resolution-specific failure (unusable colour crops, fixed-pixel minima) is **gone at native resolution**.
2. Two failure mechanisms that are **not** resolution-specific remain and now dominate:
   a. **Nested low-score partial boxes read as a second person** -> `MERGED_DETECTION_SUSPECTED` -> no
      confirmation -> AMBIGUOUS -> LOST after `max_ambiguous_s`. Resolution-independent; the inner boxes score
      0.10–0.3 at 1080p too. This is the proven first failing layer on G2_A, G2_D, G3_C, and part of G1_A.
   b. **Same-kit contrast**: `INSUFFICIENT_CONTRAST` / `BELOW_APPEARANCE_FLOOR` with a teammate nearby (G1_A end,
      G1_C, G2_F, G3_C) and seed refusals for NOT_ISOLATED in crowds (G2_E, G3_A, G3_B). This is the intended
      fail-closed behaviour; it is what the strict reacquisition floor (0.85) then turns into permanent LOST.
3. Identity safety held in both runs: 0 wrong-player frames in 5095 frames.

## TRACKER PRACTICALLY READY AT NATIVE RESOLUTION: **NO**
Safe, but 4/11 seeds refused and 5 of the 7 that ran lose the target within 2 s and never recover. Only G2_B (95 %)
and G1_A (44 %) are usable. Coverage 32 % is far from the DEV 42.6 % and is carried by two clips.

## Recommendation (no code changed)
Not a 360p robustness stage first: the native rerun shows the dominant failure is the **nested low-score part box
read as a second body** (mechanism 2 of the earlier diagnosis), which is resolution-independent, followed by
same-kit contrast with no recovery. Next stage should fix mechanism 2 only — distinguish a weak nested part box from a
credible second body (score- and geometry-based, generic), keep a credible second body failing closed, validate on
720p DEV (P1–P4, coverage), these 1080p clips and synthetic nested-part vs second-person cases — and re-measure
before touching contrast, reacquisition or the LAB minima. The 360p-specific LAB/crop minima can be a separate,
later stage.
