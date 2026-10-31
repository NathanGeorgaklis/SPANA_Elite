"""Stale `contained` boxes across a multi-frame host gap under camera pan (contained is warped by only the current
frame's motion when checked). Nothing vanishes in truth: the crouched person stays put in the world."""
import sys; sys.path.insert(0, sys.argv[1])
from tests_rb._synth_people import Frame, box_at, cand, run, pan
def sh(b, dx): return (b[0] + dx, b[1], b[2] + dx, b[3])
def scene(fps, px_per_frame, gap_frames):
    TGT = box_at(300, 400, 40, 100); crouch = box_at(300, 425, 36, 50)
    fr = []; off = 0.0
    n0 = int(fps); n1 = int(fps)
    for k in range(n0 + n1 + int(fps)):
        cam = None
        if k >= n0 and px_per_frame:
            off += px_per_frame; cam = pan(px_per_frame)
        ds, os = [cand(sh(TGT, off))], [("TARGET",)]
        if k >= n0 // 2:
            ds.append(cand(sh(crouch, off))); os.append(("OTHER",))
        if n0 + 5 <= k < n0 + 5 + gap_frames:       # host box undetected; crouched person still detected
            ds, os = ds[1:], os[1:]
        fr.append(Frame(ds, os, camera=cam if k else None))
    return fr
for fps in (30.0,):
    for pxf in (0.0, 4.0, 8.0, 15.0):
        for gap in (1, 2, 4):
            rn = run(scene(fps, pxf, gap), fps=fps)
            br = [k for k, r in enumerate(rn.results) for o in r.observations if o.continuity_break]
            print("pan px/frame", pxf, "host gap frames", gap, "false breaks at", br)
