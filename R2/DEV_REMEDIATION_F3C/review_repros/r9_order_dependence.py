# R9: candidate-order dependence. Committed-style failure-2 scene and R1/R7 part-box scenes with the detector output
# order reversed; compares per-frame (tracklet id sequence, continuity_break) records.
from common import *
import itertools
_orig = SS.detect
def scenes(fps):
    tb = SS.box_at(300, 400, 40, 100)
    a = [SS.SceneFrame({"TARGET": tb}) for _ in range(int(2 * fps))]
    a += [SS.SceneFrame({"TARGET": tb, "OTHER": SS.box_at(300, 425, 36, 50)}, order=("TARGET", "OTHER")) for _ in range(int(fps / 2))]
    a += [SS.SceneFrame({"TARGET": tb, "OTHER": SS.box_at(300, 400, 40, 100)}, order=("TARGET", "OTHER"))]
    a += [SS.SceneFrame({"TARGET": tb, "OTHER": SS.box_at(300 + 200 / fps * (k + 1), 400, 40, 100)}, order=("TARGET", "OTHER"), drop=("TARGET",)) for k in range(int(fps))]
    b = []
    for k in range(int(4 * fps)):
        bb = SS.box_at(300 + 40 * k / fps, 400, 40, 100); o = SS.box_at(370 + 40 * k / fps, 400, 40, 100)
        parts = (("TARGET", (bb[0] + 4, bb[1] + 45, bb[2] - 4, bb[3] - 1)), ("TARGET", (bb[0] + 6, bb[1], bb[2] - 6, bb[1] + 60))) if k % 3 == 0 else ()
        b.append(SS.SceneFrame({"TARGET": bb, "OTHER": o}, parts=parts))
    return {"standup": a, "partbox": b}
for fps in (30.0, 15.0, 10.0):
    for name, fr in scenes(fps).items():
        outs = []
        for rev in (False, True):
            SS.detect = (lambda f, *a, **k: list(reversed(_orig(f, *a, **k)))) if rev else _orig
            r = SS.run(fr, people(OPP), fps=fps)
            # order-free signature: for each frame, the set of (owners, break, tracklet-age-class)
            sig = []
            for p, ds in zip(r.pops, r.dets):
                sig.append(tuple(sorted((ds[k].owners, o.continuity_break, o.tracklet_id is None) for k, o in enumerate(p.observations))))
            outs.append((sig, [f.rb_state for f in r.frames], [f.identity_trusted for f in r.frames]))
        SS.detect = _orig
        same = outs[0] == outs[1]
        print(tag(), f"fps={fps} {name}: identical_under_reversal={same}" + ("" if same else f" first_diff_frame={next(k for k in range(len(outs[0][0])) if outs[0][0][k] != outs[1][0][k] or outs[0][1][k] != outs[1][1][k])}"))
