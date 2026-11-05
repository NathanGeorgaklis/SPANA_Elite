# R2 DEV jersey / shorts number reader evaluation, run 1 (2026-11-05)

**Instruction:** Nathan, 2026-11-05 17:18.
- Run the committed plan exactly: `R2/JERSEY_DATASET/READER_EVAL_PREREGISTRATION.md`, scorer `358a294`.
- No tracker, no EVAL, no threshold change after the results.

**Truth:** dataset v2 `b322b8f7…9ece` (frozen v1 + v2 region layer).
- 276 person-keyframes: 213 selected player, 63 competitor (6 competitors, 3 same-kit).
- 0 policy abstentions.

## Reader

- **EasyOCR 1.7.2** (JaidedAI, PyPI), Apache-2.0.
- Detector `craft_mlt_25k.pth` sha256 `4a5efbfb…db17`; recogniser `english_g2.pth` sha256 `e2272681…29e8`. The
  recogniser's training data is UNKNOWN.
- CPU, torch 2.15. Digit allowlist; library defaults otherwise. Crops loaded by path. Human boxes only.
- Pins, parameters and pip freeze are in `run1_easyocr/raw_reads.json` `__reader__`; scorer env and commit in
  `run_record.json`.

## Results

| | correct | wrong | abstain | false readings | precision (Wilson 95 %) |
|---|---|---|---|---|---|
| Shirt-only (all readable shirts were BACK) | 47 | 3 | 170 | 0 of 56 unreadable | 94.0 % (83.8–97.9) |
| Shirt FRONT | — | — | — | (0 readable in truth) | n/a |
| Shorts LEFT | 0 | 0 | 0 | 1 of 276 (NOT_VISIBLE) | 0 % |
| Shorts RIGHT | 0 | 0 | 13 | 0 of 263 (incl. 84 NO_NUMBER, 56 UNREADABLE) | n/a |
| Shorts-only (pooled) | 0 | 0 | 13 | 1 | 0 % |
| Combined (person-keyframe) | 47 | 4 | 177 | 0 | 92.2 % (81.5–96.9) |

**Coverage:**
- Correct shirt readings on 47 of 220 readable shirt keyframes (21 %).
- Selected players 41 of 167 (25 %); competitors 6 of 53 (11 %).
- By condition: day 60 fps 21 of 106; night 60 fps 18 of 64; day 30 fps (SYO broadcast) 8 of 50.

**The 4 errors** (crops in `crops.tar.gz`; review never changes truth):
- J10 #12 read 13;
- J12 #23 read 2;
- J13 #18 read 48;
- J13 #18 read in the **shorts band**. The person box was cut at the waist, so the shirt number sat at 55–80 % of the
  box height: the geometry caveat foreseen in §3.6. It scores as a false shorts reading.

**Temporal (3 readings / 0.5 s span, one per pool per keyframe), per person-clip:**
- **J CONFIRMED 3, all correct:**
  - J04 selected #5 → SUPPORT;
  - J05 selected #6 → SUPPORT;
  - J04 opponent #6 → VETO.
- **J CONFLICT 3:** J10, J12, J13 selected players. Their single misreads blocked confirmation, so these are
  abstentions, not errors.
- **J UNKNOWN 15.**
- **S never confirmed.**

## Pre-registered questions

- **Q0, shorts availability (truth):**
  - shorts readable on 13 of 552 shorts region-keyframes, from 3 persons (always equal to the shirt);
  - NO_NUMBER on 250 region-keyframes;
  - 8 keyframes where shorts are readable but the shirt isn't, all competitors.
- **Q1, do shorts add useful coverage when present? NO.**
  - The reader read none of the 13 readable shorts.
  - Coverage gain 0 pp; 0 extra persons; 1 false shorts reading.
- **Q2, is combined evidence safe enough for identity support? NO (insufficient evidence).**
  - Minimum evidence not met: 2 selected-player decisions (need 10) and 51 pool readings (need 100); competitors 6
    (need 3, met).
  - Independently, precision 92.2 % (< 0.95) and Wilson lower bound 81.5 % (< 0.90) fail the bar.
  - Safety conditions that did hold: 0 FALSE SUPPORT on competitors, 0 false target VETO, 0 wrongly confirmed
    numbers, 0 shorts-originated VETO, 0 distractor leaks.

**RECOMMEND TRACKER INTEGRATION: NO.**
- The temporal rule turned every misread into an abstention, so nothing unsafe was confirmed.
- But this reader confirms a number for only 2 of 15 selected players and 1 of 6 competitors. Its per-reading
  precision is below the pre-registered bar.
- Shorts added nothing on this footage.

No threshold or rule was changed after seeing these results. Any change would be a new pre-registration and a new
DEV run, labelled as tuned.
