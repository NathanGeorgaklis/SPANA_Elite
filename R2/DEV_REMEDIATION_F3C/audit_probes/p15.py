"""Crouched person seen steadily for 1 s inside the target's box, missed by the detector on only the LAST frame(s)
before standing up in front."""
import sys; sys.path.insert(0, sys.argv[1])
from tests_rb._synth_people import Frame, box_at, cand, run
from tests_rb.test_population_safety import mixed_single_tracklets
TGT = box_at(300, 400, 40, 100); crouch = box_at(300, 425, 36, 50)
for fps in (30.0, 15.0, 10.0):
    for miss_last in (0, 1, 2):
        fr = [Frame([cand(TGT)], [("TARGET",)]) for _ in range(int(fps))]
        n = int(fps)
        for k in range(n):
            fr.append(Frame([cand(TGT)], [("TARGET",)]) if k >= n - miss_last else
                      Frame([cand(TGT), cand(crouch)], [("TARGET",), ("OTHER",)]))
        fr += [Frame([cand(box_at(300 + 3 * k, 399, 42, 102))], [("OTHER",)]) for k in range(int(fps))]
        rn = run(fr, fps=fps)
        print(fps, "missed on last", miss_last, "frame(s): mixed", mixed_single_tracklets(rn))
