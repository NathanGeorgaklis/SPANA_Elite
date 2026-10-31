"""(e) replacement; uncovered TARGET is detected, walks away and is then occluded by a third, different-kit body.
(f) replacement; both people STILL in the scene (target hidden behind OTHER), camera pans: compensated (ESTIMATED)
    vs. uncompensated (UNKNOWN status, not a cut).
(g) replacement; target hidden behind OTHER, still, camera zooms (compensated)."""
from common import SS, KIT_A, KIT_B, summarize
from player_tracker_rb.camera import UNKNOWN, identity_motion, motion_from_matrix


def base_frames(fps, tx=300.0):
    tb = SS.box_at(tx, 400, 40, 100)
    n_hide, n_union, n_front = int(1.5 * fps), max(2, int(fps / 2)), int(fps)
    fr = [SS.SceneFrame({"TARGET": tb, "OTHER": SS.box_at(tx, 395, 38, 95)}, order=("OTHER", "TARGET"),
                        drop=("OTHER",)) for _ in range(n_hide)]
    fr += [SS.SceneFrame({"TARGET": tb, "OTHER": SS.box_at(tx + 40 * (k + 1) / n_union, 395, 38, 95)},
                         order=("OTHER", "TARGET"), merge=(("TARGET", "OTHER"),)) for k in range(n_union)]
    fr += [SS.SceneFrame({"TARGET": tb, "OTHER": SS.box_at(tx + 2, 401, 41, 101)}, order=("TARGET", "OTHER"),
                         drop=("TARGET",)) for _ in range(n_front)]
    return fr


def scene_e(fps):
    ppl = SS.make_people([("TARGET", KIT_A, 1.0), ("OTHER", KIT_A, 0.95), ("THIRD", KIT_B, 0.2)])
    fr = base_frames(fps)
    n = int(4 * fps)
    for k in range(n):
        s = (k + 1) / fps
        ob = SS.box_at(302 + 100 * s, 401, 41, 101)
        tx = 300 - 100 * min(s, 1.5)           # target walks left 1.5 h, then stops
        tbk = SS.box_at(tx, 400, 40, 100)
        boxes = {"TARGET": tbk, "OTHER": ob}
        order = ("TARGET", "OTHER")
        if s >= 1.5:                           # THIRD steps in front of the target (fully covers it)
            boxes["THIRD"] = SS.box_at(tx, 402, 46, 104)
            order = ("TARGET", "OTHER", "THIRD")
        fr.append(SS.SceneFrame(boxes, order=order))
    return SS.run(fr, ppl, fps=fps)


def scene_f(fps, compensated, pan_px_per_s=150.0):
    ppl = SS.make_people([("TARGET", KIT_A, 1.0), ("OTHER", KIT_A, 0.95)])
    fr = base_frames(fps, tx=600.0)
    n = int(4 * fps)
    dx = pan_px_per_s / fps
    for k in range(n):
        off = -dx * (k + 1)                    # scene still; image content shifts left
        ob = SS.box_at(602 + off, 401, 41, 101)
        tb = SS.box_at(600 + off, 400, 40, 100)
        cam = (motion_from_matrix([[1, 0, -dx], [0, 1, 0]], SS.W, SS.H) if compensated
               else identity_motion(UNKNOWN, reasons=("TOO_FEW_TRACKED",)))
        fr.append(SS.SceneFrame({"TARGET": tb, "OTHER": ob}, order=("TARGET", "OTHER"), drop=("TARGET",),
                                camera=cam))
    return SS.run(fr, ppl, fps=fps)


def scene_g(fps, zoom_per_s=1.25):
    ppl = SS.make_people([("TARGET", KIT_A, 1.0), ("OTHER", KIT_A, 0.95)])
    fr = base_frames(fps, tx=400.0)
    n = int(3 * fps)
    s1 = zoom_per_s ** (1.0 / fps)
    cx0, cy0 = SS.W / 2, SS.H / 2
    sc = 1.0
    for k in range(n):
        sc *= s1
        def z(cx, cy, w, h):
            return SS.box_at(cx0 + (cx - cx0) * sc, cy0 + (cy - cy0) * sc, w * sc, h * sc)
        m = [[s1, 0, cx0 * (1 - s1)], [0, s1, cy0 * (1 - s1)]]
        fr.append(SS.SceneFrame({"TARGET": z(400, 400, 40, 100), "OTHER": z(402, 401, 41, 101)},
                                order=("TARGET", "OTHER"), drop=("TARGET",),
                                camera=motion_from_matrix(m, SS.W, SS.H)))
    return SS.run(fr, ppl, fps=fps)


if __name__ == "__main__":
    for fps in (30.0, 15.0, 10.0):
        summarize(scene_e(fps), f"[fps {fps}] (e) target walks away, then occluded by different-kit THIRD")
        summarize(scene_f(fps, True), f"[fps {fps}] (f1) still pair, compensated pan")
        summarize(scene_f(fps, False), f"[fps {fps}] (f2) still pair, UNCOMPENSATED pan (status UNKNOWN)")
        summarize(scene_g(fps), f"[fps {fps}] (g) still pair, compensated zoom-in")
