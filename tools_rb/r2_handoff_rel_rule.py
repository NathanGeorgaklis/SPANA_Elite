# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R2 DEV same-frame identity handoff safety (Nathan, 2026-11-02 20:14): evaluate the relative-drop and
#                 absolute-floor step rules on the DEV handoff measurement (tools_rb/r2_handoff_diag.py output).
"""Read-only: step-rule rates (drop >= ambiguity_margin below the median of >= 3 recent steps, or step < 0.85) on the DEV
measurement, on all frames and on CLEAN frames only (whole isolated boxes; previous clean crop <= 2 frames back).
    python3 tools_rb/r2_handoff_rel_rule.py   (reads /home/claude/r2dev/out/handoff_diag_4a3fd9e; run from the repo)
"""
import collections
import json
import sys

import numpy as np

sys.path.insert(0,'.')
from tools_rb.r2_handoff_diag import _iou, _fixture
O="/home/claude/r2dev/out/handoff_diag_4a3fd9e"
C="R2_MCFC_MIT_C006 R2_MCFC_MIT_C016 R2_MCFC_MIT_C020 R2_MCFC_MIT_C033 R2_MCFC_MIT_C034 R2_SYO_MAR_C004 R2_SYO_MAR_C012 R2_SYO_MAR_C016 R2_SYO_MAR_C021".split()
UNCLEAN={"OVERLAPS_OTHER","CONTAINS_OTHER","NESTED","PARTIAL","TOUCHES_IMAGE_EDGE","CONTESTED","MERGED_SUSPECTED","IN_MERGED_REGION","SIZE_ANOMALY","DUPLICATE_ABSORBED","REASSOCIATED_AFTER_GAP","ALTERNATIVE"}
QUNC={"MERGED_SUSPECTED","OCCLUDED_BY_OTHER"}
for mode in ("all","clean"):
  G,I=collections.defaultdict(list),[]
  for cid in C:
    fx=_fixture('.',cid); lab=json.load(open('R2/labels/'+fx['labels_file'])); s=fx['seed_frame']
    recs=[json.loads(x) for x in open(f"{O}/{cid}.handoff.jsonl")]
    votes=collections.defaultdict(lambda:[0,0])
    for r in recs:
        hb=lab['target_boxes'].get(str(r['frame']+s))
        if hb is None: continue
        for b in r['bodies']:
            if b['tracklet']:
                v=_iou(b['bbox'],hb); votes[b['tracklet']][0 if v>=0.5 else 1]+= v>=0.5 or v<0.1
    tt={t for t,(m,e) in votes.items() if m>0 and e==0}
    hist=collections.defaultdict(list); lastclean={}
    for r in recs:
        f=r['frame']
        for b in r['bodies']:
            if b['tracklet'] not in tt: continue
            ok = not (set(b['flags'])&UNCLEAN) and not (set(b['qflags'])&QUNC)
            prev_ok = lastclean.get(b['tracklet'])==f-1 or lastclean.get(b['tracklet'])==f-2
            use = (mode=="all") or (ok and prev_ok)
            if b['self'] and b['self']['composite'] is not None and use:
                h=[x for x in hist[b['tracklet']] if f-x[0]<=15]
                if len(h)>=3:
                    base=float(np.median([x[1] for x in h]))
                    tr = r['target']['trusted'] and r['target']['tracklet']==b['tracklet']
                    G['trusted' if tr else 'untrusted'].append((base,b['self']['composite'],cid))
                hist[b['tracklet']].append((f,b['self']['composite']))
            if ok: lastclean[b['tracklet']]=f
        tg=r['target']['tracklet']
        if tg in tt:
            h=[x for x in hist[tg] if f-x[0]<=15 and x[0]<f]
            if len(h)>=3:
                base=float(np.median([x[1] for x in h]))
                for x in r['impostors']:
                    if x['composite'] is not None: I.append((base,x['composite']))
  for name,arr in G.items():
    a=np.array([x[:2] for x in arr]); drop=a[:,0]-a[:,1]; br=(drop>=0.06)|(a[:,1]<0.85)
    print(mode,name,len(a),"breaks",int(br.sum()),round(float(br.mean()),4), collections.Counter(x[2] for x,b in zip(arr,br) if b))
  a=np.array(I); drop=a[:,0]-a[:,1]; print(mode,"impostor",len(a),"caught",round(float(np.mean((drop>=0.06)|(a[:,1]<0.85))),4))
