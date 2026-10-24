"""Task 3H Step 3: where was the seeded duel_lost_01 player during the labelled
contest (video frames 0-53)? Fresh, association-independent person detections on
every frame (sequential read), chained BACKWARD from the Task 3D seed box at frame
53 by best IoU with the previous link. Read-only; no tracker state reused."""
import sys, json
sys.path.insert(0, ".")
import cv2
from player_tracker.detector import PlayerDetector
V = "/root/.claude/uploads/98d69dca-3149-5a43-9670-3c417f2af6d4/8a710009-dual_lost_01.mov"
SEED = (262.1, 503.4, 335.3, 658.3)
det = PlayerDetector(); cap = cv2.VideoCapture(V); frames = []
for i in range(54):
    ok, img = cap.read(); frames.append(img)
dets = {i: [tuple(d[:4]) for d in det.detect_persons(frames[i])] for i in range(54)}
def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - ix*iy
    return ix*iy / u if u else 0
chain = {53: SEED}; cur = SEED; out = []
for i in range(52, -1, -1):
    cands = sorted(((iou(cur, d), d) for d in dets[i]), reverse=True)
    best_iou, best = cands[0] if cands else (0, None)
    second = cands[1][0] if len(cands) > 1 else 0
    # nearest other person to the chained box (crowding), centre distance
    if best is not None:
        cx, cy = (best[0]+best[2])/2, (best[1]+best[3])/2
        others = sorted(((( (d[0]+d[2])/2-cx)**2+((d[1]+d[3])/2-cy)**2)**0.5) for d in dets[i] if d != best)
    out.append({"frame": i, "box": [round(v, 1) for v in best] if best else None, "iou_with_next": round(best_iou, 3),
                "second_best_iou": round(second, 3), "n_dets": len(dets[i]),
                "nearest_other_person_px": round(others[0], 1) if best is not None and others else None})
    if best is None or best_iou < 0.3:
        print("chain broken at frame", i, "best IoU", best_iou); break
    cur = best; chain[i] = best
json.dump(out, open("event_engine_v1/task3h/seed_backtrace.json", "w"), indent=1)
for r in out[::4] + [out[-1]]: print(r)
# visual: chained box drawn on frames 0, 8, 16, 24, 32, 40, 48, 53
import numpy as np
tiles = []
for f in [0, 6, 12, 18, 24, 30, 38, 46, 53]:
    img = frames[f].copy()
    if f in chain:
        b = [int(v) for v in chain[f]]; cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0, 255, 255), 3)
    t = cv2.resize(img[200:900, 0:1500], (750, 350)); cv2.putText(t, f"f{f}", (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    tiles.append(t)
cv2.imwrite("event_engine_v1/task3h/seed_backtrace.png", np.vstack([np.hstack(tiles[i:i+3]) for i in range(0, 9, 3)]))
