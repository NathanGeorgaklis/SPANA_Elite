# R7: the accepted cost (flickering feet-aligned lower-body part box) next to a same-kit / opponent neighbour:
# does the repeated CONTAINED_BODY_VANISHED break + re-verification ever trust the neighbour or contaminate the bank?
from common import *
import itertools
for fps, kit, sep, period in itertools.product((30.0, 15.0, 10.0), (SAME, OPP), (50, 70), (2, 3, 5)):
    fr = []
    for k in range(int(6 * fps)):
        t = k / fps
        b = SS.box_at(300 + 40 * t, 400, 40, 100)
        o = SS.box_at(300 + sep + 40 * t, 400, 40, 100)
        parts = (("TARGET", (b[0] + 4, b[1] + 45, b[2] - 4, b[3] - 1)),) if (t > 1.0 and k % period == 0) else ()
        fr.append(SS.SceneFrame({"TARGET": b, "OTHER": o}, parts=parts))
    r = SS.run(fr, people(kit), fps=fps)
    brk = sum(1 for p in r.pops for o in p.observations if getattr(o, "continuity_break", None))
    s = summary(r)
    print(tag(), f"fps={fps:4} kit={'SAME' if kit==SAME else 'OPP '} sep={sep} period={period} breaks={brk} "
          f"episodes={s['episodes']} TRACKING={s['tracking']} trusted={s['trusted']}/{s['n']} wt={s['wrong_trusted']} pure={s['pure']} mixed={s['mixed']}")
