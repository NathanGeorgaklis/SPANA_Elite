"""9869c5c population_hidden_min_s: a crouched person whose (partly occluded) detection flickers (missed 1 frame in
every `period`) for a whole second inside the target's box, then stands up in front of it."""
import sys; sys.path.insert(0, sys.argv[1])
from tests_rb._synth_people import Frame, box_at, cand, run
from tests_rb.test_population_safety import mixed_single_tracklets
TGT = box_at(300, 400, 40, 100); crouch = box_at(300, 425, 36, 50)
for fps in (30.0, 15.0, 10.0):
    for period in (0, 2, 3, 5):
        fr = [Frame([cand(TGT)], [("TARGET",)]) for _ in range(int(fps))]
        for k in range(int(fps)):
            miss = period and k % period == period - 1
            fr.append(Frame([cand(TGT)], [("TARGET",)]) if miss else Frame([cand(TGT), cand(crouch)], [("TARGET",), ("OTHER",)]))
        fr += [Frame([cand(box_at(300 + 3 * k, 399, 42, 102))], [("OTHER",)]) for k in range(int(fps))]
        rn = run(fr, fps=fps)
        print(fps, "miss every", period, "mixed", mixed_single_tracklets(rn))
