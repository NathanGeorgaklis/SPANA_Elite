# R3: failure-1 class with a MOVING union (two people jogging side by side, union box below merged_area_ratio).
# The region box is only updated by merged detections, so once the hold lapses the union box drifts out of the
# >= population_region_fill_min fill test and is again counted as "one single body" (the 8c798bc behaviour).
# Variants: no third person / third person crossing the union / third person crossing IN FRONT (its tracklet may be
# reborn inside the region, so it is not an outsider).
from common import *
import itertools
def scene(fps, kit, speed_h, third, seconds=5.0):
    v = speed_h * 100 / fps
    tb0, ob0 = (300, 400), (315, 400)
    fr = [SS.SceneFrame({"TARGET": SS.box_at(*tb0, 40, 100)}) for _ in range(int(2 * fps))]
    fr += [SS.SceneFrame({"TARGET": SS.box_at(*tb0, 40, 100), "OTHER": SS.box_at(*ob0, 40, 100)},
                         order=("OTHER", "TARGET")) for _ in range(int(fps / 3))]
    for k in range(int(seconds * fps)):
        t = k / fps
        dx = v * k
        bx = {"TARGET": SS.box_at(tb0[0] + dx, 400, 40, 100), "OTHER": SS.box_at(ob0[0] + dx, 400, 40, 100)}
        order = ("THIRD", "TARGET", "OTHER")
        if third and 1.5 <= t <= 3.0:
            ux = tb0[0] + dx + 7
            bx["THIRD"] = SS.box_at(ux - 110 + (t - 1.5) / 1.5 * 220, 403 if third == "front" else 395,
                                    42 if third == "front" else 40, 104 if third == "front" else 100)
            if third == "front":
                order = ("TARGET", "OTHER", "THIRD")
        fr.append(SS.SceneFrame(bx, merge=(("TARGET", "OTHER"),), order=order))
    return SS.run(fr, people(kit), fps=fps)
for fps, kit, sp, third in itertools.product((30.0, 15.0, 10.0), (SAME, OPP), (0.0, 0.5, 1.5), (None, "behind", "front")):
    r = scene(fps, kit, sp, third)
    s = summary(r)
    print(tag(), f"fps={fps:4} kit={'SAME' if kit==SAME else 'OPP '} speed={sp} third={third!s:6} clean_union={len(clean_union(r))} "
          f"tracking_on_union={len(tracking_on_union(r))} wrong_trusted={s['wrong_trusted']} pure={s['pure']}")
