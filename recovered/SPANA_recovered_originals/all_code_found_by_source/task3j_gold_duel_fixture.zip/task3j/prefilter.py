"""Triage only: flags frames where exactly two people are in close contact with
few others nearby. Generic COCO person detector (not SPANA). Candidates are
then judged by eye; nothing here decides a label."""
import sys, glob, json, numpy as np
from ultralytics import YOLO
m = YOLO('/tmp/claude-0/-home-claude/98d69dca-3149-5a43-9670-3c417f2af6d4/scratchpad/yolov8s.pt')
T = sys.argv[1]; files = sorted(glob.glob(f'scan/{T}/s*.jpg'))[::2]
out = []
for i in range(0, len(files), 16):
    batch = files[i:i+16]
    for f, r in zip(batch, m.predict(batch, classes=[0], conf=0.35, imgsz=960, verbose=False)):
        b = r.boxes.xyxy.cpu().numpy()
        b = b[(b[:, 3] - b[:, 1] > 38) & (b[:, 3] < 700)] if len(b) else b
        best = None
        for a in range(len(b)):
            for c in range(a+1, len(b)):
                ha = b[a,3]-b[a,1]; hc = b[c,3]-b[c,1]; h = (ha+hc)/2
                fa = np.array([(b[a,0]+b[a,2])/2, b[a,3]]); fc = np.array([(b[c,0]+b[c,2])/2, b[c,3]])
                d = np.linalg.norm(fa-fc)/h
                if d < 0.7:
                    mid = (fa+fc)/2
                    others = sum(1 for k in range(len(b)) if k not in (a,c) and np.linalg.norm(np.array([(b[k,0]+b[k,2])/2, b[k,3]])-mid)/h < 3.0)
                    score = (0.7-d) + 0.002*h - 0.25*others
                    if best is None or score > best[0]:
                        best = (float(score), float(d), int(others), float(h), [float(x) for x in b[a]], [float(x) for x in b[c]])
        if best: out.append(dict(frame=f, idx=int(f[-8:-4]), score=float(best[0]), dist=best[1], others=best[2], h=best[3], a=best[4], c=best[5]))
json.dump(out, open(f'prefilter_{T}.json','w'), default=float)
print(T, len(files), len(out))
