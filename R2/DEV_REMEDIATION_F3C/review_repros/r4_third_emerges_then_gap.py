# R4: static persistent union (TARGET+OTHER, below merged_area_ratio). A THIRD person, hidden behind the pair since
# before the merge formed (never tracked), steps out sideways: it is born overlapping the region -> "emerged" ->
# F3c counts the filling union box as one body (+1) -> region RESOLVES and is forgotten although the union still
# holds two people. Then the union box goes undetected for `gap` s (> population_max_unmatched_s): when it returns
# no region and no live predecessor remains -> is the union reborn as a clean (lineage-free) tracklet?
from common import *
import itertools
TB, OB = SS.box_at(300, 400, 40, 100), SS.box_at(315, 400, 40, 100)
def scene(fps, kit, gap, seconds=7.0):
    hidden3 = SS.box_at(307, 396, 36, 92)          # wholly behind the pair (not detected: visibility < 0.35)
    fr = [SS.SceneFrame({"TARGET": TB, "THIRD": hidden3}, order=("THIRD", "TARGET")) for _ in range(int(2 * fps))]
    fr += [SS.SceneFrame({"TARGET": TB, "OTHER": OB, "THIRD": hidden3}, order=("THIRD", "OTHER", "TARGET"))
           for _ in range(int(fps / 3))]
    for k in range(int(seconds * fps)):
        t = k / fps
        bx = {"TARGET": TB, "OTHER": OB}
        if 1.0 <= t:
            bx["THIRD"] = SS.box_at(307 + min(1.0, (t - 1.0) / 1.0) * 120, 396, 36, 92)   # steps out to the side
        else:
            bx["THIRD"] = hidden3
        drop = ("TARGET", "OTHER") if 3.5 <= t < 3.5 + gap else ()
        fr.append(SS.SceneFrame(bx, merge=(("TARGET", "OTHER"),) if not drop else (), drop=drop,
                                order=("THIRD", "OTHER", "TARGET")))
    return SS.run(fr, people(kit), fps=fps)
for fps, kit, gap in itertools.product((30.0, 15.0, 10.0), (SAME, OPP), (0.0, 0.6, 1.2)):
    r = scene(fps, kit, gap)
    s = summary(r)
    print(tag(), f"fps={fps:4} kit={'SAME' if kit==SAME else 'OPP '} gap={gap} clean_union={len(clean_union(r))} "
          f"tracking_on_union={len(tracking_on_union(r))} wrong_trusted={s['wrong_trusted']} pure={s['pure']}")
