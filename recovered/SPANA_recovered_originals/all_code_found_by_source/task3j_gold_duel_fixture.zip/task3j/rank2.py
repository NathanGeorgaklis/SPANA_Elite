"""Triage ranking over det_<T>.json. Flags 1 fps frames with an opposing-kit
pair in contact, few others nearby, and (bonus) a detected ball near them.
Produces a contact sheet for HUMAN review; it decides nothing."""
import json, sys, cv2, numpy as np
T = int(sys.argv[1]); N = int(sys.argv[2]) if len(sys.argv) > 2 else 18
K = {1500: 1504.933333, 3000: 3000.933333, 5400: 5400.933333}[T]
det = json.load(open(f'det_{T}.json'))
rows = []
for f, ds in det.items():
    idx = int(f[-8:-4]); P = [d for d in ds if d['k'] == 0 and 40 < d['b'][3] - d['b'][1] < 260 and d['b'][3] < 700]
    B = [d for d in ds if d['k'] == 32]
    feet = lambda d: np.array([(d['b'][0] + d['b'][2]) / 2, d['b'][3]])
    best = None
    for i in range(len(P)):
        for j in range(i + 1, len(P)):
            ki, kj = P[i]['kit'][0], P[j]['kit'][0]
            if {ki, kj} != {'SOL', 'MAR'}:
                continue
            h = ((P[i]['b'][3] - P[i]['b'][1]) + (P[j]['b'][3] - P[j]['b'][1])) / 2
            dd = np.linalg.norm(feet(P[i]) - feet(P[j])) / h
            if dd > 0.8:
                continue
            mid = (feet(P[i]) + feet(P[j])) / 2
            others = sum(1 for k in range(len(P)) if k not in (i, j) and np.linalg.norm(feet(P[k]) - mid) / h < 2.5)
            ball = min([np.linalg.norm(np.array([(b['b'][0] + b['b'][2]) / 2, (b['b'][1] + b['b'][3]) / 2]) - mid) / h for b in B] or [9])
            s = (0.8 - dd) - 0.3 * others + (0.6 if ball < 1.2 else 0) + 0.003 * h
            if best is None or s > best['s']:
                best = dict(s=s, dd=dd, others=others, ball=ball, h=h, a=P[i]['b'], c=P[j]['b'])
    if best and best['others'] <= 1:
        best.update(f=f, idx=idx); rows.append(best)
rows.sort(key=lambda r: r['idx'])
eps, cur = [], []
for r in rows:
    if cur and r['idx'] - cur[-1]['idx'] > 6:
        eps.append(cur); cur = []
    cur.append(r)
if cur:
    eps.append(cur)
eps = sorted([max(e, key=lambda r: r['s']) for e in eps], key=lambda r: -r['s'])[:N]
tiles = []
for r in eps:
    im = cv2.imread(r['f']); a, c = r['a'], r['c']
    cx = (min(a[0], c[0]) + max(a[2], c[2])) / 2; cy = (min(a[1], c[1]) + max(a[3], c[3])) / 2
    w = max(300, int(r['h'] * 5)); h = int(w * 9 / 16)
    X0 = int(max(0, min(1280 - w, cx - w / 2))); Y0 = int(max(0, min(720 - h, cy - h / 2)))
    cr = cv2.resize(im[Y0:Y0 + h, X0:X0 + w], (400, 225))
    t = r['idx'] / 2.0
    cv2.rectangle(cr, (0, 0), (400, 20), (0, 0, 0), -1)
    cv2.putText(cr, f'{T} t={t:.0f} src={K + t:.1f} h={r["h"]:.0f} o={r["others"]} b={r["ball"]:.1f}', (3, 15), 0, 0.42, (255, 255, 255), 1)
    tiles.append(cr)
while len(tiles) % 3:
    tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(f'rank2_{T}.png', np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]))
for r in eps:
    print(T, r['idx'] / 2, round(K + r['idx'] / 2, 1), round(r['s'], 2), r['others'], round(r['ball'], 1), round(r['h']))
