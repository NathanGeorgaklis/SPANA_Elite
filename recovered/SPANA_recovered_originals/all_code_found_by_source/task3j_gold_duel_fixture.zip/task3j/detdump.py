"""Triage only. Generic COCO detector (person + sports ball) at 1 fps; raw
boxes saved with a crude kit-colour reading. Used to point human review at
moments; never decides a label."""
import sys, glob, json, numpy as np, cv2
from ultralytics import YOLO
m = YOLO('/tmp/claude-0/-home-claude/98d69dca-3149-5a43-9670-3c417f2af6d4/scratchpad/yolov8s.pt')
T = sys.argv[1]; files = sorted(glob.glob(f'scan/{T}/s*.jpg'))[::2]
out = {}


def kit(im, b):
    x1, y1, x2, y2 = [int(v) for v in b]; h = y2 - y1
    t = im[y1 + int(0.15 * h):y1 + int(0.5 * h), x1 + int(0.2 * (x2 - x1)):x2 - int(0.2 * (x2 - x1))]
    if t.size == 0:
        return 'unk', 0, 0
    hsv = cv2.cvtColor(t, cv2.COLOR_BGR2HSV)
    blue = (hsv[..., 0] >= 95) & (hsv[..., 0] <= 130) & (hsv[..., 1] > 60)
    royal = blue & (hsv[..., 1] > 150) & (hsv[..., 2] < 230)
    sky = blue & (hsv[..., 1] <= 150) & (hsv[..., 2] > 150)
    r, s = float(royal.mean()), float(sky.mean())
    lab = 'SOL' if r > 0.25 and r > s else ('MAR' if s > 0.2 and s > r else 'unk')
    return lab, round(r, 3), round(s, 3)


for i in range(0, len(files), 16):
    batch = files[i:i + 16]
    for f, r in zip(batch, m.predict(batch, classes=[0, 32], conf=0.2, imgsz=1280, verbose=False)):
        im = cv2.imread(f); dets = []
        for b, c, k in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy()):
            d = dict(b=[round(float(v), 1) for v in b], c=round(float(c), 3), k=int(k))
            if int(k) == 0:
                d['kit'] = kit(im, b)
            dets.append(d)
        out[f] = dets
    json.dump(out, open(f'det_{T}.json', 'w'))
print(T, len(files), 'done', flush=True)
