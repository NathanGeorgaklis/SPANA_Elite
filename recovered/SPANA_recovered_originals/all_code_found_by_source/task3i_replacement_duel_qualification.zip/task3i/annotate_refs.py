import cv2, numpy as np, json
from ultralytics import YOLO
m = YOLO('yolov8x.pt')
FR = 'event_engine_v1/task3i/frames/f{:03d}.png'
frames = [100,112,120,124,128,150,165,204,213,222,234,246,270,285,291]
out = {}
tiles = []
for f in frames:
    im = cv2.imread(FR.format(f))
    big = cv2.resize(im, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    r = m.predict(big, classes=[0], conf=0.25, verbose=False)[0]
    dets = []
    for b, c in zip(r.boxes.xyxy.cpu().numpy() / 2, r.boxes.conf.cpu().numpy()):
        x1, y1, x2, y2 = b
        crop = im[int(y1):int(y2), int(x1):int(x2)]
        if crop.size == 0: continue
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        torso = hsv[: max(1, crop.shape[0] // 2)]
        yellow = float(((torso[..., 0] > 20) & (torso[..., 0] < 38) & (torso[..., 1] > 110) & (torso[..., 2] > 120)).mean())
        dark = float(((torso[..., 2] < 90)).mean())
        team = 'NWS_yellow' if yellow > 0.12 else ('SDR_dark' if dark > 0.25 else 'other')
        dets.append(dict(bbox=[round(float(v), 1) for v in (x1, y1, x2, y2)], conf=round(float(c), 3), team_guess=team,
                         yellow_frac=round(yellow, 3), dark_frac=round(dark, 3), height_px=round(float(y2 - y1), 1)))
        col = (0, 220, 255) if team == 'NWS_yellow' else ((60, 60, 60) if team == 'SDR_dark' else (200, 200, 200))
        cv2.rectangle(big, (int(x1*2), int(y1*2)), (int(x2*2), int(y2*2)), col, 2)
        cv2.putText(big, f'{len(dets)-1}', (int(x1*2), int(y1*2) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
    cv2.rectangle(big, (0, 0), (190, 30), (0, 0, 0), -1)
    cv2.putText(big, f'f{f} t={f/30:.2f}s', (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imwrite(f'event_engine_v1/task3i/ref_f{f:03d}.png', big)
    out[f] = dets
json.dump(out, open('event_engine_v1/task3i/reference_detections.json', 'w'), indent=1)
for f, d in out.items():
    print(f, [(i, x['team_guess'], x['bbox'], x['conf']) for i, x in enumerate(d) if x['team_guess'] != 'other'])
