import sys; sys.path.insert(0, sys.argv[1])
from tests_rb import _synth_scene as SS
from tests_rb import test_population_safety as T
from player_tracker_rb import population as P
orig_update = P.Population.update
log = []
def upd(self, *a, **k):
    res = orig_update(self, *a, **k)
    log.append([(round(r.created_t, 2), r.n_bodies, round(r.last_evidence_t, 2), sorted(r.claimants)) for r in self._regions])
    return res
P.Population.update = upd
tc = T.TestPersistentUnion()
for fps in (30.0,):
    log.clear()
    r = tc._union(fps, T.SAME, 5.0, third=True)
    base = int(2 * fps) + int(fps / 3)
    for k in range(base - 2, len(log), 6):
        ds = r.dets[k]
        print(k, round((k - base) / fps, 2), log[k], [d.owners for d in ds], r.frames[k].rb_state)
