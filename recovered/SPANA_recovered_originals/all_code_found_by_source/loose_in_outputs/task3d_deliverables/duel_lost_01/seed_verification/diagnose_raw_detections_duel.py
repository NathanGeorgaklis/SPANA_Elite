"""Task 3D Step 4 pre-check: confirm the detector itself (frozen, association-
independent) finds a person at/near the duel_lost_01 seed location before
trusting the seed for inference. Read-only."""
import sys
from pathlib import Path
import cv2

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
from player_tracker.detector import PlayerDetector

VIDEO = "/root/.claude/uploads/98d69dca-3149-5a43-9670-3c417f2af6d4/8a710009-dual_lost_01.mov"
SEED_BOX = (262.1, 503.4, 335.3, 658.3)
SEED_CX, SEED_CY = (SEED_BOX[0] + SEED_BOX[2]) / 2, (SEED_BOX[1] + SEED_BOX[3]) / 2

det = PlayerDetector()
cap = cv2.VideoCapture(VIDEO)

for abs_frame in [51, 52, 53, 54, 55, 58, 60]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, abs_frame)
    ok, frame = cap.read()
    if not ok:
        print(f"frame {abs_frame}: could not read")
        continue
    boxes = det.detect_persons(frame)
    dists = []
    for b in boxes:
        x1, y1, x2, y2 = b[:4]
        conf = b[4] if len(b) > 4 else None
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        dist = ((cx - SEED_CX) ** 2 + (cy - SEED_CY) ** 2) ** 0.5
        dists.append((dist, (x1, y1, x2, y2), conf))
    dists.sort(key=lambda x: x[0])
    print(f"\n=== absolute frame {abs_frame} ({len(boxes)} raw person detections) ===")
    for dist, box, conf in dists[:3]:
        print(f"    dist_from_seed={dist:6.1f}px box={tuple(round(v,1) for v in box)} conf={conf}")

cap.release()
