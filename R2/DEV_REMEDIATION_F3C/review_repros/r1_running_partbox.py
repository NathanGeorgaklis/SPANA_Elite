# R1: target running with a PERSISTENT (non-flickering) feet-aligned lower-body part box.
# Contained boxes are compared at their previous absolute position (camera-warped only), not moved with the host,
# so at low fps / high speed the part box "vanishes" every frame.
from common import *
for fps in (30.0, 15.0, 10.0):
    for speed_h in (1.0, 2.0, 3.5):           # body heights per second
        v = speed_h * 100 / fps
        fr = []
        for k in range(int(4 * fps)):
            cx = 200 + v * k
            b = SS.box_at(cx, 400, 40, 100)
            part = (b[0] + 4, b[1] + 45, b[2] - 4, b[3] - 1)  # lower body, feet-aligned, persistent
            fr.append(SS.SceneFrame({"TARGET": b}, parts=(("TARGET", part),) if k >= int(fps) else ()))
        r = SS.run(fr, people(OPP), fps=fps)
        ntr = len({v_.tracklet_id for p in r.pops for v_ in p.tracklets if v_.observed})
        brk = sum(1 for p in r.pops for o in p.observations if getattr(o, "continuity_break", None))
        s = summary(r)
        print(tag(), f"fps={fps:4} speed={speed_h}h/s tracklets={ntr} breaks={brk} TRACKING={s['tracking']}/{s['n']} "
              f"last5={r.states()[-5:]} wrong_trusted={s['wrong_trusted']} pure={s['pure']}")
