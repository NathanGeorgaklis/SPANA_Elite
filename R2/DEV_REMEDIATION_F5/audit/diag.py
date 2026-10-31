import sys, unittest, collections
sys.path.insert(0, "/home/claude/mut_f5")
import tests_rb.test_reacquisition_f5 as M
from player_tracker_rb import target as T
orig = M.run
cur = {"name": None}; out = collections.defaultdict(collections.Counter)
def run(frames, ppl, fps):
    r = orig(frames, ppl, fps)
    if fps == 30.0:
        for f in r.frames:
            rv = f.decision.get("reverify")
            if rv: out[cur["name"]][(rv.get("phase"), rv.get("status"), rv.get("why"))] += 1
        out[cur["name"]][("trusted_frames", sum(f.identity_trusted for f in r.frames))] += 0
    return r
M.run = run
M.SS_run_orig = None
loader = unittest.TestLoader()
for cls in [M.TestPostAnomalyReverification, M.TestOtherReturnsAreNotReverified, M.TestReviewRound1]:
    for n in loader.getTestCaseNames(cls):
        cur["name"] = cls.__name__ + "." + n
        res = unittest.TestResult(); cls(n).run(res)
        print("==", cur["name"], "ok" if res.wasSuccessful() else "FAIL")
        for k, v in sorted(out[cur["name"]].items(), key=str): print("   ", k, v)
