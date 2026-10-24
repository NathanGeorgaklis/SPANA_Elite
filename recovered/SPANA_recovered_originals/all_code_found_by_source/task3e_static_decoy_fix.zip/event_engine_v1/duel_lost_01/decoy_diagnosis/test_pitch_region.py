"""Task 3E, Section 5/6: directly test what pitch_region the StaticDecoyTracker's
PitchRegionClassifier assigns to the Mitre-logo screen location on real frames
of duel_lost_01.mov. Pure read-only instantiation of frozen classes; nothing
modified."""
import sys
sys.path.insert(0, ".")
import cv2
from football_analysis.ball_static_decoy import PitchRegionClassifier, StaticDecoyTracker, OFF_PITCH_WEIGHT
from football_analysis.visual_pitch_calibration import _grass_mask, _pitch_horizon

VIDEO = "/root/.claude/uploads/98d69dca-3149-5a43-9670-3c417f2af6d4/8a710009-dual_lost_01.mov"
MITRE_CX, MITRE_CY = 1888.0, 1049.0

cap = cv2.VideoCapture(VIDEO)
clf = PitchRegionClassifier()

for abs_frame in [53, 60, 70, 80, 100, 150, 200, 250, 300]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, abs_frame)
    ok, frame = cap.read()
    if not ok:
        print(f"frame {abs_frame}: could not read")
        continue
    clf.prepare(frame, abs_frame)
    region = clf.classify(MITRE_CX, MITRE_CY)
    weight = OFF_PITCH_WEIGHT.get(region, 0.3)
    print(f"abs_frame={abs_frame}: pitch_region={region} off_pitch_weight={weight}")

cap.release()

print("\n--- raw grass-mask / horizon inspection at frame 53 ---")
cap = cv2.VideoCapture(VIDEO)
cap.set(cv2.CAP_PROP_POS_FRAMES, 53)
ok, frame = cap.read()
h, w = frame.shape[:2]
print(f"full-res frame shape: {frame.shape}")
downsample_width = 240
scale = downsample_width / float(w)
small = cv2.resize(frame, (downsample_width, max(1, int(h * scale))))
mask = _grass_mask(small)
horizon = _pitch_horizon(mask)
sx, sy = MITRE_CX * scale, MITRE_CY * scale
ix, iy = int(round(sx)), int(round(sy))
print(f"downsample scale={scale:.4f}, small shape={small.shape}, (ix,iy)=({ix},{iy})")
print(f"horizon[ix]={horizon[ix]}  (small image height={small.shape[0]})")
print(f"mask value at (iy,ix): {mask[iy, ix] if 0<=iy<mask.shape[0] and 0<=ix<mask.shape[1] else 'OOB'}")
# local grass fraction window
GRASS_LOCAL_WINDOW = 18
win = max(1, int(GRASS_LOCAL_WINDOW * scale))
x1, y1 = max(0, ix - win), max(0, iy - win)
x2, y2 = min(mask.shape[1], ix + win), min(mask.shape[0], iy + win)
local_grass_fraction = float((mask[y1:y2, x1:x2] > 0).mean()) if (x2 > x1 and y2 > y1) else 0.0
print(f"local window [{x1}:{x2},{y1}:{y2}] grass_fraction={local_grass_fraction:.3f} (ON_PITCH needs >=0.30)")
cv2.imwrite("event_engine_v1/duel_lost_01/decoy_diagnosis/small_frame53.png", small)
cv2.imwrite("event_engine_v1/duel_lost_01/decoy_diagnosis/grass_mask_frame53.png", mask)
cap.release()
