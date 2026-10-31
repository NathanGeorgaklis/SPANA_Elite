import sys; sys.path.insert(0, sys.argv[1]); sys.path.insert(0, "/home/claude/mut_tools/probe")
from player_tracker_rb import population as P
from probe_union import scene, clean_union, run
for fps in (30.0, 15.0):
  for ts in (436.0, 440.0, 444.0, 330.0):
    for gap in (False, True):
        regs = []
        fr = scene(fps, 0.0, third_start=ts, third_speed=60.0 if ts > 400 else -60.0, union_s=6.0)
        if gap:   # union box undetected 0.6 s at t=4.0 into the union, then back
            base = int(fps); a, b = base + int(4.0*fps), base + int(4.6*fps)
            for k in range(a, b):
                f = fr[k]; keep = [j for j, o in enumerate(f.owners) if len(o) == 1]
                f.dets[:] = [f.dets[j] for j in keep]; f.owners[:] = [f.owners[j] for j in keep]
        rn = run(fr, fps=fps, hook=lambda k, pop, res: regs.append(len(pop._regions)))
        first_res = next((k for k in range(int(fps)+2, len(regs)) if regs[k] == 0), None)
        print(fps, ts, gap, "clean", len(clean_union(rn)), "region_gone_at_frame", first_res, "third_born_frame", int(fps)+int(1.5*fps))
