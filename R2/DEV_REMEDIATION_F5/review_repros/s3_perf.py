"""Performance: an unverified post-anomaly episode that never re-verifies (target stands still after a pose anomaly)
in a frame with N far bodies of the other kit. Counts AppearanceExtractor.extract calls per frame (all callers)."""
import time

from common import SS, KIT_A, KIT_B, pose_then
from player_tracker_rb import appearance as A

calls = {"n": 0}
_orig = A.AppearanceExtractor.extract


def counted(self, *a, **k):
    calls["n"] += 1
    return _orig(self, *a, **k)


A.AppearanceExtractor.extract = counted

if __name__ == "__main__":
    fps, n_far, secs = 30.0, 11, 8.0
    boxes, n0 = pose_then(fps)
    boxes += [SS.box_at(300, 400, 40, 100) for _ in range(int(secs * fps))]
    spec = [("TARGET", KIT_A, 1.0)] + [(f"P{i}", KIT_B, 0.2) for i in range(n_far)]
    ppl = SS.make_people(spec)
    far = {f"P{i}": SS.box_at(500 + 70 * (i % 10), 150 + 250 * (i // 10), 40, 100) for i in range(n_far)}
    frames = [SS.SceneFrame({"TARGET": b, **far}) for b in boxes]
    t0 = time.time()
    r = SS.run(frames, ppl, fps=fps)
    dt = time.time() - t0
    tail = len(frames) - int(2 * fps)
    print(f"total extract calls={calls['n']} frames={len(frames)} wall={dt:.1f}s "
          f"last frame trusted={r.frames[-1].identity_trusted} state={r.frames[-1].rb_state} "
          f"reverify={r.frames[-1].decision.get('reverify')}")
    # per-frame calls over the tail
    calls["n"] = 0
