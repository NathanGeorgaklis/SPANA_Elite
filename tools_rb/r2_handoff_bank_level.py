# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R2 DEV suspended-target recovery (Nathan, 2026-11-03 17:36): measure on DEV whether the recovery test
#                 (bank match >= 0.85 and within 0.06 of the target's recent bank level) separates the true target from
#                 nearby bodies; reads the target-profile DEV run (causal production bank score per body).
"""Read-only. python3 tools_rb/r2_handoff_bank_level.py (reads /home/claude/r2dev/out/profile_800a117; run from repo)."""
import json, collections, numpy as np, sys
sys.path.insert(0,'.')
from tools_rb.r2_profile_eval import _iou, _fixture
O="/home/claude/r2dev/out/profile_800a117"
FLOOR,M=0.85,0.06
for cid in ("R2_MCFC_MIT_C006","R2_MCFC_MIT_C020","R2_MCFC_MIT_C033","R2_SYO_MAR_C012"):
    fx=_fixture('.',cid); lab=json.load(open('R2/labels/'+fx['labels_file'])); s=fx['seed_frame']
    tb={int(k):v for k,v in lab['target_boxes'].items()}; keys=sorted(tb)
    def hbox(cf):
        lo=max([q for q in keys if q<=cf],default=None); hi=min([q for q in keys if q>=cf],default=None)
        if lo is None or hi is None or hi-lo>12: return None
        w=0 if hi==lo else (cf-lo)/(hi-lo); return [a+w*(b-a) for a,b in zip(tb[lo],tb[hi])]
    recs=[json.loads(x) for x in open(f"{O}/{cid}.profile.jsonl")]
    fps=30.0; hist=[]  # (frame, true bank)
    G=[];I=[];runs=collections.defaultdict(int); maxrun=collections.defaultdict(int)
    for r in recs:
        f=r['frame']; hb=hbox(f+s)
        if hb is None: continue
        rows=r['profile']['candidates']
        tr=[x for x in rows if _iou(x['bbox'],hb)>=0.5]
        if not tr or tr[0].get('pre') is None: continue
        t=tr[0]; w=[v for (k,v) in hist if f-k<=15]
        if len(w)>=3:
            L=float(np.median(w))
            G.append((L-t['pre'], t['pre']))
            th=t['bbox'][3]-t['bbox'][1]; tc=((t['bbox'][0]+t['bbox'][2])/2,(t['bbox'][1]+t['bbox'][3])/2)
            seen=set()
            for o in rows:
                if o is t or o.get('pre') is None: continue
                c=((o['bbox'][0]+o['bbox'][2])/2,(o['bbox'][1]+o['bbox'][3])/2)
                if np.hypot(c[0]-tc[0],c[1]-tc[1])>3*th: continue
                ok = o['pre']>=FLOOR and L-o['pre']<M
                I.append(ok); key=o['tracklet']; seen.add(key)
                runs[key]=runs[key]+1 if ok else 0; maxrun[key]=max(maxrun[key],runs[key])
            for k in list(runs):
                if k not in seen: runs[k]=0
        hist.append((f,t['pre']))
    g=np.array(G)
    gpass=np.mean((g[:,1]>=FLOOR)&(g[:,0]<M)) if len(g) else None
    print(cid,"genuine frames",len(g),"pass",None if gpass is None else round(float(gpass),3),"| impostor frames",len(I),"pass",round(float(np.mean(I)),4) if I else None,"| impostor max consecutive passing run",max(maxrun.values()) if maxrun else 0, sorted(maxrun.values())[-5:])
