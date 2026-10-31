import sys; sys.path.insert(0, sys.argv[1])
from tests_rb import test_population_safety as T
from player_tracker_rb import population as P
orig = P.Population.update
log = []
def upd(self, rel, t, cands, cam):
    before = [(tuple(round(v) for v in r.box), r.n_bodies, r.ref_wh and tuple(round(v,1) for v in r.ref_wh)) for r in self._regions]
    res = orig(self, rel, t, cands, cam)
    log.append((before, [tuple(round(v) for v in c.bbox) for c in cands],
                [(o.tracklet_id, o.merge_reasons) for o in res.observations]))
    return res
P.Population.update = upd
r = T.TestPersistentUnion()._union(30.0, T.SAME, 6.0, third=True, gaps=((4.0, 4.6),))
for k in range(128, 150):
    print(k, log[k])
