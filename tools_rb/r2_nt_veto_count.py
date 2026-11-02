# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R2 DEV NOT_TARGET memory instruction (Nathan, 2026-11-02 16:12): count exclusions gained and
#                 true-target exclusions on DEV.
"""Count NOT_TARGET vetoes in a DEV decision log (read-only, refuses non-DEV fixtures).

    python3 tools_rb/r2_nt_veto_count.py REPO DECISION_LOG.jsonl CLIP_ID

TRUE-TARGET veto: the vetoed tracklet's box matches the frozen human target box (IoU >= 0.5) on a labelled frame, or
on an unlabelled frame the box linearly interpolated between labelled frames <= 12 frames apart.
Tracklet truth: a vetoed tracklet that matches the human box (IoU >= 0.5) on ANY labelled frame of the clip is
reported as TOUCHES_TARGET (it may have carried the target at some point)."""
import json, sys, os
repo, log, cid = sys.argv[1:4]
fx = json.load(open(os.path.join(repo, "R2", "labels", cid + ".fixture.json")))
assert fx["split"] == "DEV"
lab = json.load(open(os.path.join(repo, "R2", "labels", fx["labels_file"])))
s = fx["seed_frame"]
tb = {int(k): v for k, v in lab["target_boxes"].items()}
keys = sorted(tb)
def box_at(f):
    if f in tb: return tb[f], "labelled"
    lo = max([k for k in keys if k < f], default=None); hi = min([k for k in keys if k > f], default=None)
    if lo is None or hi is None or hi - lo > 12: return None, None
    w = (f - lo) / (hi - lo)
    return [a + w * (b - a) for a, b in zip(tb[lo], tb[hi])], "interpolated"
def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy; u = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i
    return i / u if u > 0 else 0.0
n = {"vetoes": 0, "veto_frames": 0, "withdrawn": 0, "true_target_vetoes": 0, "checked_labelled": 0,
     "checked_interpolated": 0, "unchecked": 0}
vt, touches = set(), set()
for line in open(log):
    d = json.loads(line)
    if "target" not in d: continue
    obs = {o["tracklet_id"]: o["bbox"] for o in d["population"].get("observations", []) if o.get("tracklet_id")}
    f = d["frame"] + s
    if f in tb:
        for tid, b in obs.items():
            if iou(b, tb[f]) >= 0.5: touches.add(tid)
    nts = ((d["target"].get("decision") or {}).get("reacquisition") or {}).get("not_target_memory") or []
    any_v = False
    for e in nts:
        if not e["veto"]:
            n["withdrawn"] += 1; continue
        any_v = True; n["vetoes"] += 1; vt.add(e["tracklet"])
        hb, kind = box_at(f); b = obs.get(e["tracklet"])
        if hb is None or b is None: n["unchecked"] += 1; continue
        n["checked_" + kind] += 1
        if iou(b, hb) >= 0.5: n["true_target_vetoes"] += 1
    n["veto_frames"] += any_v
n["vetoed_tracklets"] = sorted(vt)
n["vetoed_tracklets_touching_target"] = sorted(vt & touches)
print(json.dumps(n))
