# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R2 DEV remediation instruction (Nathan, 2026-10-31 05:05, F3b): diagnose why the secondary tracklet
#                 is born, what evidence existed at its birth, and why MULTI_CLAIM treated the pair as two people
"""Read-only F3b birth diagnosis on R2 DEV decision logs (DEV only; never opens video).

    python3 tools_rb/r2_f3b_births.py OUT_DIR OUT_JSON

For every tracklet birth: the largest fraction of the new box inside the box of a tracklet observed in the same
frame (and IoU, height ratio, edge / partial flags); for every MULTI_CLAIM merge frame: whether a claimant was born
>= 0.5 inside another claimant (the F3b signature)."""
import json, sys, glob, os, collections
def inter(a,b): return max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
def area(a): return max(0,a[2]-a[0])*max(0,a[3]-a[1])
def iou(a,b):
    i=inter(a,b); u=area(a)+area(b)-i; return i/u if u>0 else 0
OUT=sys.argv[1]
rows=[]; mc=[]
for d in sorted(glob.glob(os.path.join(OUT,"R2_*"))):
    cid=os.path.basename(d); p=os.path.join(d,cid+"_rb_decision_log.jsonl")
    if not os.path.exists(p): continue
    recs=[json.loads(l) for l in open(p)]
    birth={}  # tid -> (frame, box, best partner tid, rel)
    for r in recs:
        pop=r["population"]; obs=pop["observations"]
        trk={v["tracklet_id"]:v for v in pop["tracklets"]}
        observed={tid:obs[v["detection_index"]]["bbox"] for tid,v in trk.items() if v.get("detection_index") is not None and v["observation_source"]=="DETECTION"}
        for tid,v in trk.items():
            if v["first_frame"]==r["frame"] and tid not in birth and v.get("detection_index") is not None:
                b=obs[v["detection_index"]]["bbox"]; fl=obs[v["detection_index"]]["flags"]
                best=None
                for t2,b2 in observed.items():
                    if t2==tid or trk[t2]["first_frame"]==r["frame"]: continue
                    cb=inter(b,b2)/area(b) if area(b) else 0
                    if best is None or cb>best[1]: best=(t2,cb,iou(b,b2),(b[3]-b[1])/(b2[3]-b2[1]),b2)
                birth[tid]=(r["frame"],b,best,fl)
                rows.append((cid,tid,r["frame"],best[1] if best else 0,best[2] if best else 0,best[3] if best else 0,"EDGE" if "TOUCHES_IMAGE_EDGE" in fl else "", "PARTIAL" if "PARTIAL" in fl else ""))
        mcd=[o for o in obs if "MULTI_CLAIM" in (o.get("merge_reasons") or [])]
        if mcd:
            cl=[c["tracklet_id"] for c in pop["closed"] if c["reason"]=="MERGED_CONTACT"]
            # pair classification
            dup=[]
            for a in cl:
                for b in cl:
                    if a<b:
                        for x,y in ((a,b),(b,a)):
                            bb=birth.get(y)
                            if bb and bb[2] and bb[2][0]==x and bb[2][1]>=0.5: dup.append((y,"born %.2f inside %s"%(bb[2][1],x)))
            mc.append((cid,r["frame"],cl,dup))
json.dump({"births":rows,"multi_claims":mc},open(sys.argv[2],"w"),indent=0)
# summary
b=[x for x in rows]
bins=collections.Counter()
for x in b:
    k=x[3]
    bins["cont>=0.75" if k>=0.75 else "0.5-0.75" if k>=0.5 else "0.25-0.5" if k>=0.25 else "<0.25"]+=1
print("births",len(b),dict(bins))
print("MULTI_CLAIM frames",len(mc),"with a duplicate-born claimant (born >=0.5 inside the other)",sum(1 for m in mc if m[3]))
for m in mc[:40]: print("  ",m)
