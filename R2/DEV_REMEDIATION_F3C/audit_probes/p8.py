"""Persistent union + a small far person (height < population_part_box_min_height_ratio of the union) visible inside
the region box, never touching it as a merged body would. The region must not resolve."""
import sys; sys.path.insert(0, sys.argv[1]); sys.path.insert(0, "/home/claude/mut_tools/probe")
from tests_rb._synth_people import Frame, box_at, cand, run
from probe_union import clean_union
for fps in (30.0, 15.0):
    fr = [Frame([cand(box_at(400, 360, 32, 80)), cand(box_at(412, 360, 32, 80))], [("TARGET",), ("OTHER",)]) for _ in range(int(fps))]
    regs = []
    for k in range(int(6 * fps)):
        t = k / fps
        ds, os = [cand((384.0, 320.0, 428.0, 400.0))], [("TARGET", "OTHER")]
        if 1.5 <= t:
            ds.append(cand(box_at(406, 335, 8, 20))); os.append(("FAR",))
        drop = 4.0 <= t < 4.6
        if drop:
            ds, os = ds[1:], os[1:]
        fr.append(Frame(ds, os))
    rn = run(fr, fps=fps, hook=lambda k, pop, res: regs.append(len(pop._regions)))
    gone = next((k for k in range(int(fps) + 2, len(regs)) if regs[k] == 0), None)
    print(fps, "clean_union", len(clean_union(rn)), "region_gone_at", gone)
