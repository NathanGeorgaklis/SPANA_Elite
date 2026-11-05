# Pre-inference coverage, dataset v2 (truth only; computed before any reader run)

Dataset v2 sha256 `b322b8f77ea81ad7fe39d14e7a66b4bb5a4da398f6a7e377405452e32a2d9ece`.
Person-keyframes: 276 = 213 selected player + 63 competitor.

| Region | READABLE | UNREADABLE | NO_NUMBER | NOT_VISIBLE |
|---|---|---|---|---|
| JERSEY_BACK | 220 | 30 | 0 | 26 |
| JERSEY_FRONT | 0 | 0 | 168 | 108 |
| SHORTS_LEFT | 0 | 0 | 166 | 110 |
| SHORTS_RIGHT | 13 | 56 | 84 | 123 |

- **Jersey readable:** selected player 167/213; competitors 53/63.
  - All of it is on the back. No front number was readable anywhere: the front is NO_NUMBER or NOT_VISIBLE.
- **Shorts readable:** selected player 4/213; competitors 9/63.
  - Shorts present and readable on 13 region-keyframes, all SHORTS_RIGHT.
  - Only 3 persons ever have readable shorts: J15 selected player #11 (4 keyframes), J15 C1 #10 (7) and J07 opponent C1 #7 (2). In every case shorts = shirt.
  - Shorts NO_NUMBER on 250 region-keyframes.
- **Frames that gain number evidence from shorts when the shirt is unavailable:** selected player 0; competitors 8 (J15 C1: 7, J07 C1: 1).
- **Competitors:** 6 labelled.
  - Same-kit: 3 (J04 C1 #10, J10 C1 #9, J15 C1 #10). Opponents: 3 (J04 C2 #6, J07 C1 #7, J14 C1 #7).
- **Duplicate / conflicting numbers:**
  - No competitor shares its own clip's selected-player number (0 clips).
  - Across clips, #7 is the J01 and J03 selected player and the J07 / J14 opponent C1; #6 is the J05 / J09 selected player and the J04 opponent C2; #10 is two competitors.
  - V1_CONFLICT notes: 0. Selected players with shorts ≠ shirt: 0.
- **Distractors labelled:** 0.
