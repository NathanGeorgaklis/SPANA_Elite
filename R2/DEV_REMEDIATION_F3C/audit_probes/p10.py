"""The test file's own TestPersistentUnion._union scenario, third=True, plus a 0.6 s union detection gap after the
third person has passed (combination not in the test matrix)."""
import sys; sys.path.insert(0, sys.argv[1])
from tests_rb import test_population_safety as T
tc = T.TestPersistentUnion()
for fps in T.FPS_SET:
    for kit in (T.SAME, T.OPP):
        for gaps in ((), ((4.0, 4.6),), ((3.5, 3.8),)):
            r = tc._union(fps, kit, 6.0, third=True, gaps=gaps)
            print(fps, "SAME" if kit == T.SAME else "OPP", gaps, "clean_union", len(T.clean_union_tracklet_frames(r)),
                  "tracking_on_union", len(T.tracking_on_union(r)), "wrong_trusted", len(T.wrong_trusted(r)),
                  "bank_pure", r.bank_is_pure())
