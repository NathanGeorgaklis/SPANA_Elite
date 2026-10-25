# SPANA engine v2-rebuild — R1-G development-only visual check (not a tool, not benchmark scoring).
# Draws the T001 reported box on video frames rel 120..176 (step 8) of a run_rb results file. These frames are
# OUTSIDE the frozen human label window (rel 70..110), so identity there is UNKNOWN; the sheet only shows where the
# unverified episode-2 box sat. Usage: python3 episode2_sheet_script.py RESULTS_JSON VIDEO OUT_JPG
import json, sys
import cv2, numpy as np
res, video, out = sys.argv[1:4]
d = json.load(open(res))
st = d["summary"]["start_frame"]; want = set(range(120, 177, 8))
recs = {fr["frame"]: [t for t in fr["tracks"] if t.get("selected_player")][0] for fr in d["frame_log"]}
cap = cv2.VideoCapture(video); tiles = []; i = 0
while True:
    ok, im = cap.read()
    if not ok:
        break
    f = i - st; i += 1
    if f in want:
        t = recs[f]
        b = [int(x) for x in t["observed_position_box"]]
        col = (255, 0, 255) if t["v2"]["rb_state"] == "TRACKING" else (0, 255, 255)
        cv2.rectangle(im, (b[0], b[1]), (b[2], b[3]), col, 3)
        cv2.putText(im, f"f{f} {t['v2']['rb_state']} trusted={t['v2']['identity_trusted']}", (20, 60), 0, 1.5, (255, 255, 255), 3)
        tiles.append(cv2.resize(im, (640, 360)))
cv2.imwrite(out, np.vstack([np.hstack(tiles[k:k + 2]) for k in range(0, len(tiles), 2)]))
