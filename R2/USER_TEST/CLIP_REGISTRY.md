# R2 USER TEST — fresh clip registry (2026-11-06)

Instruction: Nathan 2026-11-06 12:55 — test the frozen 1.0.0-r2prod tracker on completely new footage; no tracker
logic / threshold / config change; no DEV or EVAL footage.

Sources (Nathan's Mac, `~/Downloads`; chosen by Nathan from the options offered): `youtube_game_1.mp4`
(Sydney United 58 v APIA Leichhardt, day, very wide camera), `youtube_game_2.mp4` (NWS Spirit v SD Raiders, floodlit,
closer camera), `youtube_game_3.mp4` (Sydney FC v St George City, floodlit, wide). All three: 640x360, 30 fps —
a new resolution and frame rate for the tracker (DEV: 1280x720 / 60 fps). None of them is an R2 DEV or EVAL source.
Their provenance was not reviewed in R0 (R0_TESTS_AND_MODELS X20): acceptable for user testing, not for benchmark
truth. Task 3R windows (R06 g2 2381 s, R07 g3 1821 s, R08 g2 7148 s, R09 g1 1264 s) were avoided with a 60 s buffer
(only the time-range lines of the Task 3R extract log were read, as in R2 pre-registration).

Extraction (Mac ffmpeg): `-ss <t> -i <src> -frames:v 300 -an -c:v libx264 -qp 0 -preset veryfast -pix_fmt yuv420p -r 30`
(10 s, lossless); `CLIP_EXTRACT_LOG.txt` has the times, `CLIP_SHA256SUMS.txt` the hashes (verified after staging).
15 candidates were cut; 12 kept (G1 B/D/E dropped as duplicates of the tiny-player wide condition, G2 C was a stoppage).

| clip | source time | what the contact sheet shows |
|---|---|---|
| UT_G1_SU_API_A | g1 1140 s | open play, static very wide camera, players ~15 px tall |
| UT_G1_SU_API_C | g1 2460 s | camera pan midfield -> box, tiny players |
| UT_G2_NWS_SDR_A | g2 1140 s | open play, wide |
| UT_G2_NWS_SDR_B | g2 1440 s | attack, closer camera, players ~50 px |
| UT_G2_NWS_SDR_D | g2 3180 s | counter / sprint, wide |
| UT_G2_NWS_SDR_E | g2 3420 s | crowd (mass confrontation on the touchline), large players |
| UT_G2_NWS_SDR_F | g2 3840 s | penalty-box crowd before a set piece |
| UT_G2_NWS_SDR_G | g2 1620 s | open play |
| UT_G3_SFC_SGC_A | g3 1440 s | open play with pan |
| UT_G3_SFC_SGC_B | g3 2520 s | corner / set piece, box crowd, same-kit groups |
| UT_G3_SFC_SGC_C | g3 2940 s | open play, pan |
| UT_G3_SFC_SGC_D | g3 3360 s | counter-attack with pan / sprint |

Seeds: frame 0 of each clip; the production detector's candidates are numbered on `seed_stills/`; Nathan picks the
target number; the picked detector box is the seed, unchanged (`tools_rb/r2_user_test.py`).
