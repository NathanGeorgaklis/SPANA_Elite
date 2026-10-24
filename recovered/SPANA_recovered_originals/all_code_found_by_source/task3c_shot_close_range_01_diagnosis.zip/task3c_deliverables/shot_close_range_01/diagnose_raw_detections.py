"""Read-only diagnostic for Task 3C Step 4 (detection vs association). Runs the
frozen PlayerDetector directly on a handful of absolute frames around the seed
(322) to see whether ANY raw detection exists near Yellow #9's seed box,
independent of track_manager.py's spawn-guard/association logic entirely.
No frozen file is edited; this is a plain inference call, same class the
online tracker itself uses."""
import sys
from pathlib import Path
import cv2

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
from player_tracker.detector import PlayerDetector

VIDEO = "/mnt/user-data/uploads/shot_close_range_01.mov"
SEED_BOX = (1474.0, 457.2, 1547.1, 634.6)
SEED_CX, SEED_CY = (SEED_BOX[0] + SEED_BOX[2]) / 2, (SEED_BOX[1] + SEED_BOX[3]) / 2

det = PlayerDetector()
cap = cv2.VideoCapture(VIDEO)

for abs_frame in [321, 322, 323, 324, 325, 330, 335]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, abs_frame)
    ok, frame = cap.read()
    if not ok:
        print(f"frame {abs_frame}: could not read")
        continue
    boxes = det.detect_persons(frame)
    print(f"\n=== absolute frame {abs_frame} ({len(boxes)} raw person detections) ===")
    dists = []
    for b in boxes:
        x1, y1, x2, y2 = b[:4]
        conf = b[4] if len(b) > 4 else None
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        dist = ((cx - SEED_CX) ** 2 + (cy - SEED_CY) ** 2) ** 0.5
        dists.append((dist, (x1, y1, x2, y2), conf))
    dists.sort(key=lambda x: x[0])
    for dist, box, conf in dists[:5]:
        print(f"    dist_from_seed={dist:6.1f}px box={tuple(round(v,1) for v in box)} conf={conf}")

cap.release()
