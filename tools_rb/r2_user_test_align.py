# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R2 native-resolution user-test rerun (Nathan, 2026-11-06 23:00): re-cut the exact same 12 time
#                 windows from the native 1080p sources. Evaluation tooling only.
"""Align a native-resolution section to a 360p user-test clip by content and cut the same 10 s window losslessly.

    python3 tools_rb/r2_user_test_align.py SECT_DIR CLIPS_360_DIR OUT_DIR

For each UT_<id>.mp4 in CLIPS_360_DIR: frame 0 (640x360) is matched against every frame of the first part of
UT_<id>_sect.mp4 downscaled to 640x360 (mean absolute luma difference); the best frame is the window start. The
window is then cut as exactly 10 s of frames at the section's native frame rate with libx264 -qp 0 (lossless,
yuv420p), like the 360p clips. ALIGNMENT.json records the chosen start frame, the match score and the runner-up gap.
"""
import json
import os
import subprocess
import sys

import cv2
import numpy as np


def main(sect_dir, clips_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    rec = {}
    for f in sorted(os.listdir(clips_dir)):
        if not (f.startswith("UT_") and f.endswith(".mp4")):
            continue
        cid = f[:-4]
        sp = os.path.join(sect_dir, cid + "_sect.mp4")
        ref = cv2.VideoCapture(os.path.join(clips_dir, f)).read()[1]
        refg = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY).astype(np.float32)
        cap = cv2.VideoCapture(sp)
        fps = cap.get(cv2.CAP_PROP_FPS)
        n_win = int(round(10 * fps))
        scores = []
        i = 0
        while True:
            ok, img = cap.read()
            if not ok:
                break
            g = cv2.cvtColor(cv2.resize(img, (640, 360), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
            scores.append(float(np.abs(g.astype(np.float32) - refg).mean()))
            i += 1
        cap.release()
        total = i
        order = np.argsort(scores)
        best = int(order[0])
        # runner-up outside the immediate neighbourhood (adjacent frames are naturally similar)
        runner = next((float(scores[j]) for j in order[1:] if abs(int(j) - best) > int(round(fps / 2))), None)
        if best + n_win > total:
            raise SystemExit(f"{cid}: window exceeds the section ({best}+{n_win} > {total})")
        out = os.path.join(out_dir, cid + ".mp4")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", sp, "-vf", f"select='gte(n\\,{best})'", "-vsync", "0",
                        "-frames:v", str(n_win), "-an", "-c:v", "libx264", "-qp", "0", "-preset", "veryfast",
                        "-pix_fmt", "yuv420p", out], check=True)
        got = int(cv2.VideoCapture(out).get(cv2.CAP_PROP_FRAME_COUNT))
        rec[cid] = {"section": os.path.basename(sp), "fps": fps, "start_frame_in_section": best,
                    "match_mad": round(scores[best], 3), "runner_up_mad": round(runner, 3) if runner else None,
                    "frames": got, "section_frames": total}
        print(cid, rec[cid], flush=True)
    with open(os.path.join(out_dir, "ALIGNMENT.json"), "w") as fh:
        json.dump(rec, fh, indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:4])
