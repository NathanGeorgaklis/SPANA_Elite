"""(h) stale _anomaly_origin: after the replacement the real TARGET stays detected in frame for `stay_s` seconds (so
    the origin tracklet, now OTHER, is not re-verified), then walks out of the frame. How long after the anomaly does
    the F5 run still verify the origin tracklet?
(i) lock-out after a wrong re-verification: scenario (e) of s2 extended - the real TARGET re-emerges from behind THIRD
    and stays visible; is it ever trusted again, and is its tracklet recorded as a known other?"""
from common import SS, KIT_A, KIT_B, summarize
from s2_more_twins import base_frames


def scene_h(fps, stay_s):
    ppl = SS.make_people([("TARGET", KIT_A, 1.0), ("OTHER", KIT_A, 0.95)])
    fr = base_frames(fps, tx=400.0)
    n0 = len(fr)
    n = int((stay_s + 4.0) * fps)
    for k in range(n):
        s = (k + 1) / fps
        ob = SS.box_at(402 + 60 * s, 401, 41, 101)                # OTHER walks right at 0.6 h/s throughout
        tx = 330 - 20 * s if s < stay_s else 330 - 20 * stay_s - 300 * (s - stay_s)
        boxes = {"OTHER": ob}
        if tx > -30:
            boxes["TARGET"] = SS.box_at(tx, 400, 40, 100)
        fr.append(SS.SceneFrame(boxes))
    return SS.run(fr, ppl, fps=fps), n0


def scene_i(fps):
    ppl = SS.make_people([("TARGET", KIT_A, 1.0), ("OTHER", KIT_A, 0.95), ("THIRD", KIT_B, 0.2)])
    fr = base_frames(fps)
    n = int(8 * fps)
    for k in range(n):
        s = (k + 1) / fps
        ob = SS.box_at(302 + 100 * min(s, 4.0), 401, 41, 101)
        tx = 300 - 100 * min(s, 1.5)
        boxes = {"TARGET": SS.box_at(tx, 400, 40, 100), "OTHER": ob}
        order = ("TARGET", "OTHER")
        if 1.5 <= s < 4.0:
            boxes["THIRD"] = SS.box_at(tx, 402, 46, 104)
            order = ("TARGET", "OTHER", "THIRD")
        elif s >= 4.0:                                            # THIRD walks off; target visible again
            boxes["THIRD"] = SS.box_at(tx - 100 * (s - 4.0) - 60, 402, 46, 104)
            order = ("TARGET", "OTHER", "THIRD")
        fr.append(SS.SceneFrame(boxes, order=order))
    return SS.run(fr, ppl, fps=fps)


if __name__ == "__main__":
    for fps in (30.0, 15.0):
        for stay in (2.0, 5.0):
            r, n0 = scene_h(fps, stay)
            res = summarize(r, f"[fps {fps}] (h) target stays {stay}s then exits")
            if res["rev"]:
                print(f"   re-verified {(res['rev'][0] - n0) / fps:.2f}s after the anomaly ended")
    r = scene_i(30.0)
    summarize(r, "[fps 30] (i) wrong re-verification, then real target visible 4 s")
    last = r.frames[-1]
    tgt_tids = {r.pops[-1].observations[i].tracklet_id for i, d in enumerate(r.dets[-1]) if d.owners == ("TARGET",)}
    print("   last frame: state", last.rb_state, "trusted", last.identity_trusted,
          "on", r.dets[-1][last.detection_index].owners if last.detection_index is not None else None,
          "| real target tracklet", tgt_tids, "in known_others:", bool(tgt_tids & r.tracker._known_others))
