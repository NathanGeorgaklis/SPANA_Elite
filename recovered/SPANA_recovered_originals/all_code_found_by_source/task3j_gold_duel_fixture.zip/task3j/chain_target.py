"""Identity continuity evidence: raw YOLOv8x person detections on every frame
from the seed (180) to 320, chained by best IoU / nearest centre from the seed
box. Not SPANA tracking; reference evidence only."""
import cv2, json, numpy as np
from ultralytics import YOLO
m = YOLO('/tmp/claude-0/-home-claude/98d69dca-3149-5a43-9670-3c417f2af6d4/scratchpad/yolov8s.pt')
def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy; return i / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i + 1e-9)
cur = [635.3, 304.1, 757.8, 438.3]; out = []
for f in range(180, 321):
    im = cv2.imread(f'gold_frames/g{f:03d}.png')
    r = m.predict(im, classes=[0], conf=0.25, imgsz=1280, verbose=False)[0]
    B = [list(map(float, b)) for b in r.boxes.xyxy.cpu().numpy()]
    B = [b for b in B if b[3] - b[1] > 30 and b[3] < 690]
    if not B: out.append(dict(f=f, box=None)); continue
    c = np.array([(cur[0]+cur[2])/2, (cur[1]+cur[3])/2]); h = cur[3]-cur[1]
    sc = [(iou(cur, b), -np.linalg.norm(np.array([(b[0]+b[2])/2, (b[1]+b[3])/2]) - c) / h) for b in B]
    k = max(range(len(B)), key=lambda i: (sc[i][0] > 0.2, sc[i][0], sc[i][1]))
    b = B[k]; ious = sorted([iou(cur, x) for x in B], reverse=True)
    fc = np.array([(b[0]+b[2])/2, b[3]])
    others = [np.linalg.norm(np.array([(x[0]+x[2])/2, x[3]]) - fc) / (b[3]-b[1]) for i, x in enumerate(B) if i != k]
    out.append(dict(f=f, box=[round(v, 1) for v in b], iou_prev=round(ious[0], 3), second_iou=round(ious[1], 3) if len(ious) > 1 else 0,
                    nearest_other_heights=round(min(others), 2) if others else None))
    cur = b
json.dump(out, open('gold_target_chain.json', 'w'), indent=0)
v = [o for o in out if o['box']]
print('frames', len(out), 'with box', len(v), 'min iou_prev', min(o['iou_prev'] for o in v), 'max second_iou', max(o['second_iou'] for o in v))
low = [(o['f'], o['iou_prev'], o['second_iou'], o['nearest_other_heights']) for o in v if o['iou_prev'] < 0.5 or o['second_iou'] > 0.2]
print('weak links', low)
