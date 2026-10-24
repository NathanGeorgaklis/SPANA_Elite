"""Task 3F visual audit sheets for duel_lost_01. Frames are read SEQUENTIALLY after one
seek to start_frame (exactly how Pass 1 and Pass 2 read the file -- per-frame random-access
seeks on this .mov drift by several frames and must not be used). Yellow = Pass-1 selected
target box; red = ball position the Pass-2 ball tracker reported (from Task 3E's trace)."""
import sys, json, cv2, numpy as np
VIDEO = "/root/.claude/uploads/98d69dca-3149-5a43-9670-3c417f2af6d4/8a710009-dual_lost_01.mov"
mo = json.load(open("event_engine_v1/duel_lost_01/mo_tracking/8a710009-dual_lost_01_mo_tracking_results.json"))
S = mo["summary"]["start_frame"]; fl = {r["frame"]: r for r in mo["frame_log"]}
trace = {r["frame"]: r for r in json.load(open("event_engine_v1/duel_lost_01/_diag_ball_vs_player_trace_postfix.json"))}
cap = cv2.VideoCapture(VIDEO); cap.set(cv2.CAP_PROP_POS_FRAMES, S)
seq = []
while True:
    ok, img = cap.read()
    if not ok or len(seq) > 262: break
    seq.append(img)
def sheet(name, frames, half=450, halfh=260, zoom=1.0):
    tiles = []
    for f in frames:
        img = seq[f].copy()
        sel = [t for t in fl[f]["tracks"] if t["selected_player"]][0]; b = [int(v) for v in sel["bbox"]]
        cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0, 255, 255), 2)
        t = trace.get(f)
        if t and t["ball_cx"] is not None:
            cv2.circle(img, (int(t["ball_cx"]), int(t["ball_cy"])), 12, (0, 0, 255), 2)
        cx, cy = (b[0] + b[2]) // 2, (b[1] + b[3]) // 2
        x0, y0 = max(0, cx - half), max(0, cy - halfh)
        crop = img[y0:y0 + 2 * halfh, x0:x0 + 2 * half].copy()
        crop = cv2.resize(crop, (int(crop.shape[1] * zoom), int(crop.shape[0] * zoom)))
        cv2.putText(crop, f"rel {f} t={(S+f)/mo['summary']['fps']:.2f}s {t['ball_state'] if t else ''}", (8, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        tiles.append(crop)
    h = min(x.shape[0] for x in tiles); w = min(x.shape[1] for x in tiles)
    tiles = [x[:h, :w] for x in tiles]
    while len(tiles) % 3: tiles.append(np.zeros_like(tiles[0]))
    out = np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)])
    cv2.imwrite(f"event_engine_v1/task3f/visual/{name}.png", out)
sheet("seed_to_duel_wide", [1, 10, 20, 30, 40, 50], half=600, halfh=300, zoom=0.5)
sheet("duel_window_feet", [55, 57, 58, 59, 60, 61, 62, 64, 68], half=160, halfh=110, zoom=1.8)
sheet("pass_window", [140, 146, 150, 152, 154, 156, 158, 161, 165], half=300, halfh=170, zoom=0.9)
sheet("cross_window", [182, 186, 188, 190, 191, 192, 194, 197, 201], half=300, halfh=170, zoom=0.9)
print("ok", len(seq))
