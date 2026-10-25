# R2 labelling guide (for Nathan)

R2 human truth is drawn by **you**. Claude selected and extracted the clips but labels nothing. No model output is shown in the annotation page.

## Open the page

1. In Finder, go to `~/Downloads/_spana_r2_work/annotate/`.
2. Open `index.html` in Chrome or Safari. It works offline and uploads nothing.
3. Type your name in **Annotator**.
4. Choose a clip from the menu. Each clip shows its split: DEV or EVAL.

The right-hand panel shows Claude's **proposed target**: a description and a suggested seed time. It is only a pointer.
- You decide who the target is.
- If no player is a sensible target, use **Reject clip…** and give a reason. A rejected clip is never scored.

## What to label

| Item | How |
|---|---|
| **Seed** | Go to a frame where the target is clearly visible, identifiable, not merged and not in contact. Draw the target box, then press **Seed = this frame + target box**. |
| **Per-frame state** (every frame) | Visibility (VISIBLE / PARTIAL / HIDDEN / OUT OF FRAME), Identity resolvable (YES / NO), Occlusion, Merged blob, Contact with, Camera cut here, HUMAN_UNCERTAIN. To fill a stretch, set the state on one frame, press **Range start**, move to the other end, then press **Apply this frame's state to range**. |
| **Target box** | Drag on the frame on every *required* frame where the target is VISIBLE / PARTIAL and identifiable. Never box a HIDDEN or OUT-OF-FRAME target. |
| **Other people** | Add each person who touches or overlaps the target (description and relation: OPPONENT / SAME_TEAM / OTHER, where OTHER is a referee, coach or spectator). Tick them in **Contact with** on contact frames, select their layer, and box them on the required frames. |
| **First clean reappearance** | Tick it on the first frame where the target is clearly identifiable again after being hidden, merged, unresolvable or out of frame. |
| **HUMAN_UNCERTAIN** | Tick it whenever you are not sure. Those frames are excluded from error counts. Never guess. |

**Required box frames:**
- every 0.2 s (5 Hz) from the seed;
- **every frame** within 0.5 s of any contact, occlusion, merge, visibility change, out-of-frame stretch or camera cut.

The timeline shows them: a red tick means a box is still missing, green means done. **next missing** (or <kbd>N</kbd>) jumps to the next gap.

**Keys:**
- <kbd>←</kbd>/<kbd>→</kbd>: move one frame; <kbd>⇧</kbd> + arrow: 10 frames.
- <kbd>C</kbd>: copy the previous box of the current layer (then adjust by redrawing).
- <kbd>Del</kbd>: delete the box.
- <kbd>T</kbd>: switch to the target layer.

## Saving and freezing

- The page autosaves in your browser per clip. When a clip is done, press **Check labels**, then **Download labels**. This saves `<clip_id>.labels_v1.json`.
- Put all downloaded files into `~/Downloads/_spana_r2_work/labels_incoming/`.
- Then tell Claude. Claude runs `tools_rb/r2_labels.py freeze` for each clip, which:
  - validates the file;
  - copies it to `R2/labels/`;
  - writes a FROZEN record (sha256) and the run fixture with your seed.

  Claude then commits and pushes **before any tracker run**.
- After freezing, a label file is never edited. A correction becomes `labels_v2`, with the reason recorded.

## Effort estimate

For each clip, the main work is the boxes:
- about 50 target boxes on the 5 Hz grid;
- plus every frame around events;
- plus the contact partners' boxes.

This is usually 60–200 boxes per clip, for 25 clips. You can label DEV first. EVAL must still be labelled and frozen before any tracker run on EVAL.
