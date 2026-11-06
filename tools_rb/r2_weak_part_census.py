# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R2 nested-part stage (Nathan 2026-11-07 03:24): count, from decision logs, weak-part boxes that
#                 escaped nesting (same geometry/score rule as population.py) and how many carried a tracklet.
"""python3 tools_rb/r2_weak_part_census.py OUT_DIR [OUT_DIR ...]  (evaluation tooling; reads *_rb_decision_log.jsonl)"""
import json,glob,sys,collections
def wh(b): return b[2]-b[0], b[3]-b[1]
def cont(a,b):
    ix=max(0,min(a[2],b[2])-max(a[0],b[0])); iy=max(0,min(a[3],b[3])-max(a[1],b[1])); A=(a[2]-a[0])*(a[3]-a[1]); return ix*iy/A if A else 0
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from player_tracker_rb.config import TrackerConfig   # noqa: E402
C = TrackerConfig.default()
def weak(j,i):  # j weak part of i (same rule as population.py, thresholds from the registry)
    bj,bi=j['bbox'],i['bbox']; wj,hj=wh(bj); wi,hi=wh(bi)
    return (wi*hi>wj*hj and cont(bj,bi)>=C.get("population_weak_part_min_containment")
            and j['score']<=C.get("population_weak_part_score_ratio")*i['score'] and hj<hi
            and wj<=C.get("population_weak_part_max_width_ratio")*wi
            and abs((bj[0]+bj[2])-(bi[0]+bi[2]))/2<=C.get("population_weak_part_max_centre_offset")*wi
            and hj>=C.get("population_part_box_min_height_ratio")*hi)
for d in sys.argv[1:]:
    for log in sorted(glob.glob(d+'/*/*_rb_decision_log.jsonl')):
        cid=log.split('/')[-2]; frames=0; wp_frames=0; wp_boxes=0; wp_tracklet=0; wp_on_target=0; nested_already=0
        for l in open(log):
            r=json.loads(l); obs=[o for o in r['population']['observations'] if o.get('duplicate_of') is None]
            frames+=1; hit=False; tid=r['target'].get('observation_tracklet_id')
            for j in obs:
                hosts=[i for i in obs if i is not j and weak(j,i)]
                if hosts:
                    if 'NESTED' in j['flags']: nested_already+=1; continue
                    wp_boxes+=1; hit=True
                    if j.get('tracklet_id'): wp_tracklet+=1
                    if any(i.get('tracklet_id')==tid for i in hosts) and tid: wp_on_target+=1
            wp_frames+=hit
        print(f"{cid:22s} frames {frames:4d}  weak-part boxes not nested {wp_boxes:4d} (on {wp_frames} frames; {wp_tracklet} carry a tracklet; {wp_on_target} on the target)  already nested {nested_already}")
