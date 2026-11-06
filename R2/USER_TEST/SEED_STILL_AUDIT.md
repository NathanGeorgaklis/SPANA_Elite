# R2 USER TEST — seed-still duplicate-detection audit (2026-11-06 17:02 instruction)

Scope: UI only. Raw detections are unchanged (`*_seed_candidates.json`); the tracker is untouched and receives the
chosen group's representative RAW box. Grouping: `tools_rb/r2_user_test.py` `ui_groups`.

Rules (scale-normalised, no pixel constants):
- DUPLICATE full body: IoU >= 0.60 -> one group (the production NMS is 0.65, so survivors sit below it).
- NESTED upper-body part: inner box >= 90% inside a host, shorter, top within 15% of host height of the host top and
  >= 60% of the host width -> one group with the host.
- Kept separate and marked "?" (possible second person): heavy overlap IoU 0.40-0.60; a nested box that is not an
  upper body (low in the host, or narrow). Nothing is collapsed when two people may be present.
- Selectable box = tallest member scoring >= 50% of the group's best score (the fullest body box, which is also the
  box the tracker's seed validation expects).

| clip | raw | selectable | duplicate groups | nested groups | "?" pairs | median height px | score < 0.5 |
|---|---|---|---|---|---|---|---|
| UT_G1_SU_API_A | 28 | 24 | 1 | 2 | 4 | 28 | 32% |
| UT_G1_SU_API_C | 29 | 26 | 2 | 1 | 0 | 24 | 41% |
| UT_G2_NWS_SDR_A | 33 | 26 | 4 | 1 | 5 | 36 | 36% |
| UT_G2_NWS_SDR_B | 13 | 12 | 1 | 0 | 1 | 40 | 15% |
| UT_G2_NWS_SDR_D | 23 | 16 | 3 | 2 | 3 | 30 | 61% |
| UT_G2_NWS_SDR_E | 28 | 23 | 3 | 0 | 4 | 108 | 36% |
| UT_G2_NWS_SDR_F | 12 | 11 | 0 | 1 | 1 | 95 | 17% |
| UT_G2_NWS_SDR_G | 6 | 4 | 1 | 0 | 0 | 46 | 83% |
| UT_G3_SFC_SGC_A | 40 | 27 | 7 | 0 | 10 | 31 | 65% |
| UT_G3_SFC_SGC_B | 28 | 20 | 5 | 0 | 7 | 34 | 39% |
| UT_G3_SFC_SGC_C | 33 | 31 | 0 | 2 | 0 | 20 | 64% |
| UT_G3_SFC_SGC_D | 43 | 27 | 8 | 3 | 8 | 32 | 47% |
| **total** | **316** | **247** | **35** | **12** | **43** | | |

- 12 / 12 seed frames contain duplicate or nested boxes on one person; 9 / 12 contain "?" pairs that could be two
  people (crowds and contacts: G3_A 10, G3_D 8, G3_B 7).
- Same grouping on frame 0 of four DEV clips (1280x720): 62 raw -> 53 selectable, 6 collapsed groups (10% of raw),
  17 "?" pairs (14 of them in one crowded SYO frame). Fresh 640x360 frames: 47 collapsed groups (15% of raw) and
  ~26 raw candidates per frame against ~15 on DEV.
- Where the duplicates come from (checked on G3_D and G2_A): about half the IoU >= 0.5 pairs are tile-vs-full-frame
  double detections (the 2x2 tiles upscale a 320x180 region to 640 on this footage), the other half are same-pass
  survivors of NMS 0.65 and partial-body boxes kept on purpose by `detector_min_score` 0.10. So the low resolution
  raises the duplicate rate moderately (15% vs 10%) and adds many low-score candidates (spectators, ~20-30 px
  players); it does not break detection outright. The population layer handles nested/merged boxes internally;
  this audit changes only the selection screen.
