# GOLD_DUEL_01: human identity labels, extract frames 250–290

**Status: AWAITING NATHAN.** These labels become R1/R2 truth only after you fill them in and approve them. The sheets show **no tracker or model output**.

## Who's who

`sheet_00_reference_seed_frame180.jpg` shows frame 180. The yellow box is the fixture seed:

- **SOL #23** wears royal blue. He is the target.
- **MAR #4** wears sky-blue and white. He is the defender.

## How to fill in `GOLD_HUMAN_LABELS_TEMPLATE.csv`

Fill in one row per frame. Leave a cell blank only where these instructions allow it.

| Column | Values | Meaning |
|---|---|---|
| t23_visibility | VISIBLE / PARTIAL / HIDDEN / OUT_OF_FRAME / HUMAN_UNCERTAIN | How much of #23 can be seen. |
| t23_identity_resolvable | YES / NO | Could a careful person tell, from this frame and its neighbours, which body is #23? |
| t23_location_note | free text (optional) | For example, "behind #4", "left of #4", "upper body only". |
| mar4_visibility | VISIBLE / PARTIAL / HIDDEN / OUT_OF_FRAME / HUMAN_UNCERTAIN | How much of #4 can be seen. |
| overlap_23_4 | YES / NO / HUMAN_UNCERTAIN | Do the two players overlap in the image? |
| merged_or_hidden | NONE / MERGED / T23_HIDDEN / HUMAN_UNCERTAIN | MERGED means both bodies look like one shape. T23_HIDDEN means #23 is fully behind #4. |
| first_clean_reappearance | Y or blank | Put **Y** on exactly one frame: the first frame where #23 is clearly visible again after being hidden or overlapped. |
| notes | free text | Anything else. |

**If you can't tell, write HUMAN_UNCERTAIN.** Don't guess. HUMAN_UNCERTAIN frames are excluded from hard scoring.

## When you're done

1. Save the file as `GOLD_HUMAN_LABELS_v1.csv` in the same folder.
2. Tell Claude it's approved.

Claude will then:
- hash and freeze the file before any tracker run;
- record your sign-off.
