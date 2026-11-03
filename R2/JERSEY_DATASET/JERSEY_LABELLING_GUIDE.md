# Jersey-number truth: labelling guide (schema `r2_jersey_labels_v1`)

**Open:** `~/Downloads/_spana_jersey_work/jersey_annotation_pack/index.html` in Chrome or Safari. It works offline.

The page shows only the clips' own keyframes, one every 0.5 s (16 per clip, 240 in total). No tracker, detector or OCR
output is shown, and none exists for these clips.

## Per clip

1. **Selected player.** The right-hand panel shows Claude's proposal. It is a pointer from a thumbnail, not truth.
   - **You decide** who the selected player is. Pick someone who stays mostly in view.
   - Describe them in words, e.g. "dark kit, walks away from camera, left".
   - Set **Team / kit**: A or B are your own names for the two kits in this clip; OTHER is anyone else.
   - If no player suits, press **Reject clip…** and give a reason. A rejected clip is never scored.
2. **Expected number:** the selected player's shirt number, if you can read it anywhere in the clip (you may play the
   clip itself to confirm it).
   - **Number in clip** = READABLE IN SOME FRAMES, or
   - NEVER VISIBLE: leave the number empty.
3. **Every keyframe** (strip at the bottom; red = still to do, green = done):
   - **Target:**
     - VISIBLE or PARTIAL: draw the target's box by dragging on the frame, around the whole body you can see.
     - HIDDEN or OUT OF FRAME: no box.
   - **Number:**
     - READABLE: *you* can read the number on **this keyframe image** (the 3× zoom panel bottom-right helps).
     - UNREADABLE: give a reason (FRONT ON, SIDE ON, OCCLUDED BY PERSON, OCCLUDED OTHER, MOTION BLUR, TOO SMALL,
       PARTIAL CROP, LIGHTING, FOLDED SHIRT, OTHER).
     - Judge each keyframe on its own. "I know it's 7 from other frames" is **not** READABLE.
   - **HUMAN_UNCERTAIN:** you cannot tell whether this body is the selected player. The box is then optional and the
     number cannot be READABLE.
4. **Nearby people:** add each person near the selected player whose number matters.
   - Always add same-kit players nearby. Add opponents with the **same number** as the target.
   - Give a short description, the relation, and their number (empty if never readable).
   - On each keyframe where **their** number is readable, tick **readable here**.
   - Boxes for them are optional but useful: select their layer under **Box layer** and drag.
5. **Distractors:** anything that shows a number that is not the target's shirt, one per line. Examples: a substitution
   board, a bib, a scoreboard digit beside the player, a number on shorts or socks.
6. Press **Check labels**, then **Download labels**. The file is `<clip_id>.jersey_labels.json`. Put the files in
   `~/Downloads/labels_incoming/jersey/`.

## Keys

| Key | Action |
|---|---|
| ← / → | previous / next keyframe |
| C | copy the previous box of the current layer |
| Del | delete the box |
| R | READABLE |
| U | UNREADABLE |

**Copy this keyframe's states…** fills the following *unlabelled* keyframes with the same target and number states.
It does not copy boxes.

Labels autosave in the browser per clip. Download is what counts.

## What happens next (Claude)

1. Each file is validated by `tools_rb/r2_jersey_labels.py`. Problems come back to you; nothing is changed for you.
2. Valid files are frozen: `R2/labels/jersey_dataset/<clip_id>.jersey.FROZEN.json` with its sha256. A frozen file is
   never overwritten; a correction becomes a new version.
3. A dataset record gives a digest over all frozen files and the coverage (readable vs unreadable keyframes, reasons,
   never-visible clips, same-kit competitors, duplicate numbers).
4. All of this is committed and pushed **before** any jersey reader is run.
