# R6: ordinary crossings at (almost) the same depth: OTHER walks past behind / in front of a still TARGET.
# Measures CONTAINED_BODY_VANISHED breaks, target episodes, TRACKING/trusted coverage (and safety) F3c vs baseline.
from common import *
import itertools
for fps, kit, order, dy in itertools.product((30.0, 15.0, 10.0), (SAME, OPP), (("OTHER", "TARGET"), ("TARGET", "OTHER")), (-3, 0, 3)):
    tb = SS.box_at(300, 400, 40, 100)
    fr = [SS.SceneFrame({"TARGET": tb}) for _ in range(int(2 * fps))]
    fr += [SS.SceneFrame({"TARGET": tb, "OTHER": SS.box_at(200 + 100 * k / fps, 400 + dy, 38, 96 + dy)}, order=order)
           for k in range(int(2 * fps))]
    fr += [SS.SceneFrame({"TARGET": tb}) for _ in range(int(2 * fps))]
    r = SS.run(fr, people(kit), fps=fps)
    brk = sum(1 for p in r.pops for o in p.observations if getattr(o, "continuity_break", None))
    s = summary(r)
    print(tag(), f"fps={fps:4} kit={'SAME' if kit==SAME else 'OPP '} near={order[1]:6} dy={dy:+d} breaks={brk} "
          f"episodes={s['episodes']} TRACKING={s['tracking']} trusted={s['trusted']}/{s['n']} wt={s['wrong_trusted']} pure={s['pure']} last={r.states()[-1]}")
