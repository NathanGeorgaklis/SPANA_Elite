# R2b: as R2 "part", but the standing-up person shows a feet-aligned lower-body part box on every rising frame too
# R2: twins of failure 2 (person crouched inside the target's box stands up in front; target then hidden).
#  A  "test"   : the committed test scene (control)
#  B  "part"   : after standing, the detector also emits a feet-aligned lower-body part box of the standing person
#                (the same artefact F3c accepts exists for the target): the old crouch box is "seen" by it -> no break
#  C  "miss"   : the crouched person is missed by the detector on the last frame before the stand-up frame
#  D  "offset" : the rising person is laterally offset so its rising box is not NESTED (containment < 0.75)
from common import *
def scene(fps, kit, variant, rise_s=0.1):
    tb = SS.box_at(300, 400, 40, 100)
    fr = [SS.SceneFrame({"TARGET": tb}) for _ in range(int(2 * fps))]
    cx = 300 + (14 if variant == "offset" else 0)
    crouch = SS.box_at(300, 425, 36, 50)
    fr += [SS.SceneFrame({"TARGET": tb, "OTHER": crouch}, order=("TARGET", "OTHER")) for _ in range(int(fps / 2))]
    if variant == "miss":
        fr[-1] = SS.SceneFrame({"TARGET": tb, "OTHER": crouch}, order=("TARGET", "OTHER"), drop=("OTHER",))
    n = max(1, int(rise_s * fps))
    for k in range(n):
        f = (k + 1) / n
        h = 50 + 50 * f
        ob = SS.box_at(300 + (cx - 300) * f, 450 - h / 2, 36 + 4 * f, h)
        parts = (("OTHER", (ob[0] + 3, ob[1] + 0.5 * h, ob[2] - 3, ob[3] - 1)),) \
            if variant == "part" else ()
        fr.append(SS.SceneFrame({"TARGET": tb, "OTHER": ob}, order=("TARGET", "OTHER"), parts=parts))
    v = 2 * 100 / fps
    for k in range(int(2 * fps)):
        ob = SS.box_at(cx + v * (k + 1), 400, 40, 100)
        parts = (("OTHER", (ob[0] + 3, ob[1] + 50, ob[2] - 3, ob[3] - 1)),) if variant == "part" else ()
        fr.append(SS.SceneFrame({"TARGET": tb, "OTHER": ob}, order=("TARGET", "OTHER"), drop=("TARGET",), parts=parts))
    return SS.run(fr, people(kit), fps=fps)
for variant in ("part",):
    for fps in (30.0, 15.0, 10.0):
        for kit in (SAME, OPP):
            for rise in (0.1, 0.3):
                r = scene(fps, kit, variant, rise)
                s = summary(r)
                ep_on_other = sorted({r.frames[k].target_episode for k in r.swaps()})
                print(tag(), f"{variant:6} fps={fps:4} kit={'SAME' if kit==SAME else 'OPP '} rise={rise} "
                      f"mixed_pop={s['mixed']} swaps={s['swaps']} swap_episodes={ep_on_other} "
                      f"wrong_trusted={s['wrong_trusted']} pure={s['pure']}")
