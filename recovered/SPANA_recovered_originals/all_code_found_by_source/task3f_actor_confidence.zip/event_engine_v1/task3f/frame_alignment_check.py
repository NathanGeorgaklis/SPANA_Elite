"""Task 3F: check frame registration between Pass-1 frame_log frame f and video frame
start_frame + f + k, by matching every CONFIRMED_DETECTION track box in frame_log[f]
against fresh, association-independent detections on candidate video frames. Read-only."""
import sys, json, cv2
sys.path.insert(0, ".")
from player_tracker.detector import PlayerDetector
clip, mo_path, video = sys.argv[1], sys.argv[2], sys.argv[3]
mo = json.load(open(mo_path)); S = mo["summary"]["start_frame"]
fl = {r["frame"]: r for r in mo["frame_log"]}
det = PlayerDetector(); cap = cv2.VideoCapture(video)
def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy; u = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i
    return i / u if u > 0 else 0
res = {}
for f in [int(x) for x in sys.argv[4].split(",")]:
    boxes = [t["bbox"] for t in fl[f]["tracks"] if t.get("position_source") == "CONFIRMED_DETECTION" and t.get("frames_since_observed_position") == 0]
    row = {}
    for k in range(-3, 4):
        cap.set(cv2.CAP_PROP_POS_FRAMES, S + f + k); ok, img = cap.read()
        dets = [d[:4] for d in det.detect_persons(img)]
        m = [max((iou(b, d) for d in dets), default=0) for b in boxes]
        row[k] = round(sum(m) / len(m), 3) if m else None
    best = max(row, key=lambda k: row[k] or 0)
    res[f] = {"mean_best_iou_by_offset": row, "best_offset": best}
    print(clip, "frame_log", f, "mean IoU vs video S+f+k:", row, "-> best k =", best, flush=True)
json.dump(res, open(f"event_engine_v1/task3f/{clip}_frame_alignment.json", "w"), indent=2)
