# R4 probe: when does the region disappear (resolved vs aged out), and which tracklet sits on the union afterwards.
from common import *
import r4_third_emerges_then_gap as R4
import itertools
fps = 30.0
for gap in (0.6, 0.0):
    # re-run with a hook on population: monkeypatch update to record region count
    orig = P.Population.update
    log = []
    def upd(self, *a, **k):
        res = orig(self, *a, **k)
        log.append((a[1], len(self._regions)))
        return res
    P.Population.update = upd
    r = R4.scene(fps, SAME, gap)
    P.Population.update = orig
    ch = [(round(t, 2), n) for (t, n), (t0, n0) in zip(log[1:], log[:-1]) if n != n0]
    print(tag(), "gap", gap, "region count changes (t, n):", ch[:8])
    ks = tracking_on_union(r)
    if ks:
        k = ks[0]
        v = [v for v in r.pops[k].tracklets if v.observed and v.detection_index == r.frames[k].detection_index][0]
        print("  first TRACKING-on-union frame t=%.2f tracklet=%s flags=%s episode=%s trusted=%s" % (
            k / fps, v.tracklet_id, v.flags, r.frames[k].target_episode, r.frames[k].identity_trusted))
