# Jersey truth v2: region labelling guide (schema `r2_jersey_regions_v2`)

**Open:** `~/Downloads/_spana_jersey_work/jersey_annotation_pack/regions.html` in Chrome or Safari. It works offline.

**What v2 is:** an *added* layer. Your frozen v1 labels are shown read-only and never change:
- the selected player's box and shirt number;
- whether the shirt was readable on each keyframe.

The same 16 keyframes per clip are used.

## The four regions (judge each one on its own, on this keyframe image only)

| Region | Where |
|---|---|
| JERSEY BACK | number on the back of the shirt |
| JERSEY FRONT | number on the front of the shirt (chest) |
| SHORTS LEFT | number on the shorts, on the **left side of the screen** |
| SHORTS RIGHT | number on the shorts, on the **right side of the screen** |

Left and right always mean the side of the image as you see it, not the player's own left or right.

For every region choose:
- **READABLE:** you can read the number in this region on this image. Type it.
- **UNREADABLE:** the region is in view but you can't read the number. Pick a reason: motion blur, too small,
  occluded by a person, occluded by something else, folded, angle, lighting, partial crop, other.
- **NOT VISIBLE:** the region is not in view (facing away, hidden, out of frame).
- **NO NUMBER:** the region is in view and has no number printed on it.

**Shorts are optional evidence.** Many players have no number on their shorts, or it can't be seen. That is fine and
never counts against anyone. Use NO NUMBER or NOT VISIBLE and move on. Mark shorts READABLE only when a number is
genuinely there and you can read it.

Don't carry a number over from another region or another keyframe. If the shorts show 7, type 7 on the shorts even
if the shirt is also 7. If you can't read the shorts, they are not READABLE, whatever the shirt says.

## Selected player (all 15 clips)

1. **TARGET** is selected. Set **Shorts number** if you can read it anywhere in the clip; leave it blank if you
   never can. It is recorded separately from the shirt number.
2. On every keyframe where v1 had the player visible, label all four regions.
   - The zoom panel (bottom right) shows the player enlarged.
   - **P** copies the previous keyframe's region states. Then check each region against **this** image.
3. **If v1 and today disagree:** the page flags any keyframe where your jersey regions disagree with v1's "shirt
   readable". For example, v1 said readable but you now can't read it.
   - If the disagreement is real, add a note to that jersey region starting with `V1_CONFLICT:` and say why.
   - v1 is not edited; the conflict is recorded and reported.

## Competitors (J04, J10, J14; others optional)

1. Under **Add competitor**, describe each nearby same-kit player whose number you can read (opponents are optional),
   and type their shirt number and shorts number separately. Leave a number blank if it is never readable.
2. Select the competitor (C1, C2, …). On each keyframe where they are near the selected player:
   - set the visibility;
   - drag their box;
   - label the four regions.
3. **HUMAN_UNCERTAIN:** tick it when you can't tell it is the same person. That keyframe then carries no regions.
4. **If there is genuinely no readable nearby same-kit player** in J04, J10 or J14, write why in **No competitors**.

## Distractors (non-player numbers)

- Turn **Distractor box mode ON**, choose the kind (board, bib, scoreboard, advertising, other), type the text, and
  drag a box around it on the keyframe.
- Readings inside these boxes never count for anyone.

## Save

**Check layer**, then **Download layer**. Put the `<clip>.regions_v2.json` files in
`~/Downloads/labels_incoming/jersey_v2/`.

They are then:
- validated against frozen v1 and the registry;
- frozen read-only with a hash (`R2/labels/jersey_dataset_v2/`);
- recorded as dataset v2, and committed **before** any reader runs.
