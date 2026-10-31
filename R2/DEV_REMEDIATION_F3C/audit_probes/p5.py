"""Vanished-hidden-body check masked by a third (farther) person whose box overlaps the crouched person's last box."""
import sys; sys.path.insert(0, sys.argv[1])
from tests_rb._synth_people import Frame, box_at, cand, run
from tests_rb.test_population_safety import mixed_single_tracklets
def scene(fps, third, rise=2):
    TGT = box_at(300, 400, 40, 100)          # 280..320 x 350..450
    crouch = box_at(300, 425, 36, 50)        # feet-aligned, nested
    far = (285.0, 360.0, 315.0, 435.0)       # a farther person behind, bottom 15 px higher (off the feet line)
    def F(ds, os):
        if third:
            ds = ds + [far]; os = os + [("THIRD",)]
        return Frame([cand(b) for b in ds], os)
    fr = [F([TGT], [("TARGET",)]) for _ in range(int(fps))]
    fr += [F([TGT, crouch], [("TARGET",), ("OTHER",)]) for _ in range(int(fps / 2))]
    for k in range(rise):
        f = (k + 1) / rise; h = 50 + 52 * f
        b = box_at(300, 450 - h / 2, 36 + 6 * f, h)
        fr.append(F([b], [("OTHER",)]) if f >= 0.75 else F([TGT, b], [("TARGET",), ("OTHER",)]))
    fr += [F([box_at(300 + 3 * (k + 1), 399, 42, 102)], [("OTHER",)]) for k in range(int(fps))]
    return fr
for fps in (30.0, 15.0, 10.0):
    for third in (False, True):
        for rise in (1, 2):
            rn = run(scene(fps, third, rise), fps=fps)
            br = [(k, o.continuity_break) for k, r in enumerate(rn.results) for o in r.observations if o.continuity_break]
            print(fps, "third" if third else "none", rise, "mixed:", mixed_single_tracklets(rn), "breaks:", br[:3])
