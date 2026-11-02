# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R2 DEV same-frame identity handoff safety (Nathan, 2026-11-02 20:14): evaluate the relative-drop and
#                 absolute-floor step rules on the DEV handoff measurement (tools_rb/r2_handoff_diag.py output).
"""Read-only: rule rates on the DEV measurement. python3 tools_rb/r2_handoff_rel_rule.py (paths below)."""
import json, collections, numpy as np, sys
sys.path.insert(0,'.')
from tools_rb.r2_handoff_diag import _iou, _fixture
O="/home/claude/r2dev/out/handoff_diag_4a3fd9e"
C="R2_MCFC_MIT_C006 R2_MCFC_MIT_C016 R2_MCFC_MIT_C020 R2_MCFC_MIT_C033 R2_MCFC_MIT_C034 R2_SYO_MAR_C004 R2_SYO_MAR_C012 R2_SYO_MAR_C016 R2_SYO_MAR_C021".split()
fps_of={}
G,I=collections.defaultdict(list),collections.defaultdict(list)
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
    hist=collections.defaultdict(list)  # tracklet -> list of (frame, step)
    for r in recs:
        f=r['frame']
        for b in r['bodies']:
            if b['tracklet'] in tt and b['self'] and b['self']['composite'] is not None:
                h=[x for x in hist[b['tracklet']] if f-x[0]<=15]
                if len(h)>=3:
                    base=float(np.median([x[1] for x in h]))
                    trusted = r['target']['trusted'] and r['target']['tracklet']==b['tracklet']
                    G['trusted' if trusted else 'untrusted'].append((base, b['self']['composite']))
                hist[b['tracklet']].append((f,b['self']['composite']))
        tg=r['target']['tracklet']
        if tg in tt:
            h=[x for x in hist[tg] if f-x[0]<=15 and x[0]<f]
            if len(h)>=3:
                base=float(np.median([x[1] for x in h]))
                for x in r['impostors']:
                    if x['composite'] is not None: I['all'].append((base,x['composite']))
for name,arr in list(G.items())+list(I.items()):
    a=np.array(arr); drop=a[:,0]-a[:,1]
    print(name,len(a),"drop>=0.06",round(float(np.mean(drop>=0.06)),4),"drop>=0.04",round(float(np.mean(drop>=0.04)),4),"drop>=0.08",round(float(np.mean(drop>=0.08)),4),
          "step<0.85",round(float(np.mean(a[:,1]<0.85)),4),"(drop>=0.06 or step<0.85)",round(float(np.mean((drop>=0.06)|(a[:,1]<0.85))),4))
