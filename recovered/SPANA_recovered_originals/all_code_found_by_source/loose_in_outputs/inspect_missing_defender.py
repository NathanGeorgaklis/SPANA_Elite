"""
inspect_missing_defender.py -- OPPONENT-DETECTION TASK, rule #1: read-only
diagnosis of exactly why the navy defender never gets a separate detection
box during the critical challenge frames in known_match_carry_0_20.

No football_analysis/player_tracker code is modified by this script. It
drives ultralytics YOLO directly at three settings to isolate the failure
stage:
  (a) PRODUCTION settings, exactly as engine.py's own combined person+ball
      pass uses them: yolov8s.pt, classes=[0,32], conf=0.15, default imgsz
      (640), default NMS iou (0.7, ultralytics' own default -- engine.py
      never overrides it).
  (b) PRE-FILTER: same model/imgsz, conf pushed down to 0.001 and NMS
      iou pushed up to 0.99 (i.e. NMS effectively disabled) -- this is as
      close as ultralytics' public API gets to "everything the network
      proposed", so we can see whether a defender-shaped box exists at ANY
      confidence before either filter removes it.
  (c) HIGH-RES LOCAL CROP: a crop around the tracked player at production
      confidence but a larger imgsz, to test rule #4's hypothesis that the
      defender is simply too small/merged at full-frame resolution.
"""
import sys
sys.path.insert(0, "/tmp/claude-0/-home-claude/98d69dca-3149-5a43-9670-3c417f2af6d4/scratchpad")

import cv2
from ultralytics import YOLO
from football_analysis.adapter import TrackingAdapter
from football_analysis.geometry import iou, foot_point, dist, center

VIDEO = "/root/.claude/uploads/98d69dca-3149-5a43-9670-3c417f2af6d4/b66e2cfc-ScreenRecording_03-14-2026_00-25-09_1.MOV"
TRACKING_JSON = "validation/tracking_cache/known_match_full_tracking.json"
FRAMES = list(range(14, 21))
OUT_DIR = "diagnostics/outputs/case1_known_match_carry"

snapshots = {s.frame_number: s for s in TrackingAdapter.load(TRACKING_JSON)}
model = YOLO("yolov8s.pt")

cap = cv2.VideoCapture(VIDEO)


def boxes_at(frame, conf, iou_thresh, imgsz=640, classes=(0,)):
    res = model.predict(frame, classes=list(classes), conf=conf, iou=iou_thresh, imgsz=imgsz, verbose=False)[0]
    out = []
    for b in res.boxes:
        c = float(b.conf[0])
        xyxy = tuple(float(v) for v in b.xyxy[0])
        out.append((xyxy, c))
    return out


for fno in FRAMES:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fno)
    ok, frame = cap.read()
    if not ok:
        print(f"frame {fno}: could not read")
        continue
    snap = snapshots.get(fno)
    if snap is None:
        print(f"frame {fno}: no tracking snapshot")
        continue
    tracked_bbox = tuple(snap.bbox)
    tx1, ty1, tx2, ty2 = tracked_bbox
    tw, th = tx2 - tx1, ty2 - ty1
    print(f"\n=== frame {fno} === tracked_bbox={[round(v,1) for v in tracked_bbox]} "
          f"size={tw:.0f}x{th:.0f}")

    prod = boxes_at(frame, conf=0.15, iou_thresh=0.7, imgsz=640, classes=(0,))
    prod_others = [(b, c) for b, c in prod if iou(b, tracked_bbox) < 0.3]
    print(f"  (a) PRODUCTION (conf=0.15, iou=0.7, imgsz=640): {len(prod)} person box(es) total, "
          f"{len(prod_others)} after excluding the tracked player")
    for b, c in prod:
        print(f"      box={[round(v,1) for v in b]} conf={c:.3f} iou_vs_tracked={iou(b, tracked_bbox):.3f}")

    raw = boxes_at(frame, conf=0.001, iou_thresh=0.99, imgsz=640, classes=(0,))
    print(f"  (b) PRE-FILTER (conf=0.001, iou=0.99 i.e. NMS ~disabled, imgsz=640): {len(raw)} person box(es)")
    # candidates near the tracked player's box but NOT the tracked player itself:
    # "near" = box centre within 1.5x the tracked box's own diagonal, so a genuine
    # adjacent challenger is caught even if its box only partially overlaps.
    diag = (tw ** 2 + th ** 2) ** 0.5
    tcx, tcy = center(tracked_bbox)
    near_candidates = []
    for b, c in raw:
        bcx, bcy = center(b)
        d = ((bcx - tcx) ** 2 + (bcy - tcy) ** 2) ** 0.5
        i = iou(b, tracked_bbox)
        if d < 1.5 * diag and i < 0.85:   # exclude near-perfect self-duplicates of the tracked box
            near_candidates.append((b, c, d, i))
    near_candidates.sort(key=lambda z: z[2])
    for b, c, d, i in near_candidates:
        print(f"      NEARBY candidate box={[round(v,1) for v in b]} conf={c:.3f} "
              f"dist_from_tracked_centre={d:.1f} iou_vs_tracked={i:.3f}"
              f"{'  <-- would be BELOW production conf 0.15' if c < 0.15 else ''}"
              f"{'  <-- ABOVE conf 0.15, so NMS (iou=0.7) must be removing it in production' if c >= 0.15 else ''}")
    if not near_candidates:
        print("      (no nearby candidate at ANY confidence -- YOLO never proposed anything here at all)")

    # (c) local high-res crop centred on the tracked player, expanded to include
    # a plausible challenger standing right next to them.
    pad_x, pad_y = tw * 1.5, th * 1.0
    cx1, cy1 = max(0, tx1 - pad_x), max(0, ty1 - pad_y)
    cx2, cy2 = min(frame.shape[1], tx2 + pad_x), min(frame.shape[0], ty2 + pad_y)
    crop = frame[int(cy1):int(cy2), int(cx1):int(cx2)]
    if crop.size == 0:
        print("  (c) local crop empty, skipping")
        continue
    crop_res = model.predict(crop, classes=[0], conf=0.15, iou=0.7, imgsz=960, verbose=False)[0]
    crop_boxes = []
    for b in crop_res.boxes:
        c = float(b.conf[0])
        xyxy = tuple(float(v) for v in b.xyxy[0])
        # map back to full-frame coordinates
        full = (xyxy[0] + cx1, xyxy[1] + cy1, xyxy[2] + cx1, xyxy[3] + cy1)
        crop_boxes.append((full, c))
    crop_others = [(b, c) for b, c in crop_boxes if iou(b, tracked_bbox) < 0.3]
    print(f"  (c) LOCAL HIGH-RES CROP (conf=0.15, imgsz=960, upscaled {(cx2-cx1):.0f}x{(cy2-cy1):.0f} crop): "
          f"{len(crop_boxes)} person box(es) total, {len(crop_others)} after excluding the tracked player")
    for b, c in crop_others:
        print(f"      box={[round(v,1) for v in b]} conf={c:.3f} iou_vs_tracked={iou(b, tracked_bbox):.3f}")

    cv2.imwrite(f"{OUT_DIR}/missing_defender_frame_{fno}_raw.png", frame)

cap.release()
