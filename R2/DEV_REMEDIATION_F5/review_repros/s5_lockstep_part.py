"""(c2) lockstep replacement where the hidden TARGET's legs stick out below/beside OTHER and the detector reports
them as a separate part box (SceneFrame.parts). Is the extra body evidence used to withhold?"""
from common import SS, KIT_A, summarize
from s2_more_twins import base_frames


def scene(fps):
    ppl = SS.make_people([("TARGET", KIT_A, 1.0), ("OTHER", KIT_A, 0.95)])
    fr = base_frames(fps)
    for k in range(int(4 * fps)):
        x = 300 + 100 * (k + 1) / fps
        ob = SS.box_at(x + 2, 401, 41, 101)
        tb = SS.box_at(x + 14, 404, 40, 100)                 # partly visible to the right of OTHER
        part = (x + 23, 420, x + 34, 454)                    # its visible legs, detected as a part box
        fr.append(SS.SceneFrame({"TARGET": tb, "OTHER": ob}, order=("TARGET", "OTHER"), drop=("TARGET",),
                                parts=(("TARGET", part),)))
    return SS.run(fr, ppl, fps=fps)


if __name__ == "__main__":
    for fps in (30.0, 15.0, 10.0):
        summarize(scene(fps), f"[fps {fps}] (c2) lockstep + visible part box of the hidden target")
