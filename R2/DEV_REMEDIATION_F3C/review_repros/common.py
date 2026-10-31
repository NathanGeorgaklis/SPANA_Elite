import sys, os
TREE = os.environ.get("TREE", "/home/claude/spana_push")
sys.path.insert(0, TREE)
from player_tracker_rb import population as P
from tests_rb import _synth_scene as SS
from tests_rb._synth_people import Frame, box_at, cand, run as prun
SAME = (200, 60, 40); OPP = (40, 60, 200)
def people(kit):
    return SS.make_people([("TARGET", SAME, 1.0), ("OTHER", kit, 0.95), ("THIRD", (40, 200, 60), 0.0)])
def wrong_trusted(r):
    return [k for k, (f, ds) in enumerate(zip(r.frames, r.dets))
            if f.identity_trusted and f.detection_index is not None and ds[f.detection_index].owners != ("TARGET",)]
def tracking_on_union(r):
    return [k for k, (f, ds) in enumerate(zip(r.frames, r.dets))
            if f.rb_state == "TRACKING" and f.detection_index is not None and len(ds[f.detection_index].owners) > 1]
def clean_union(r):
    out = []
    for k, (p, ds) in enumerate(zip(r.pops, r.dets)):
        for v in p.tracklets:
            if v.observed and len(ds[v.detection_index].owners) > 1 and P.MERGE_LINEAGE not in v.flags \
                    and P.MERGED_SUSPECTED not in p.observations[v.detection_index].flags:
                out.append(k)
    return out
def mixed_scene(r):
    seen = {}
    for p, ds in zip(r.pops, r.dets):
        for v in p.tracklets:
            if v.observed and len(ds[v.detection_index].owners) == 1:
                seen.setdefault(v.tracklet_id, set()).update(ds[v.detection_index].owners)
    return {t: o for t, o in seen.items() if len(o) > 1}
def mixed_people(rn):
    seen = {}
    for r, fr in zip(rn.results, rn.frames):
        for v in r.tracklets:
            if v.observed and len(fr.owners[v.detection_index]) == 1:
                seen.setdefault(v.tracklet_id, set()).update(fr.owners[v.detection_index])
    return {t: o for t, o in seen.items() if len(o) > 1}
def summary(r):
    st = r.states()
    return dict(tracking=st.count("TRACKING"), n=len(st), trusted=sum(f.identity_trusted for f in r.frames),
                wrong_trusted=len(wrong_trusted(r)), swaps=len(r.swaps()), pure=r.bank_is_pure(),
                episodes=len({f.target_episode for f in r.frames}), mixed=mixed_scene(r))
def tag():
    return "F3C " if "spana_push" in TREE else "BASE"
