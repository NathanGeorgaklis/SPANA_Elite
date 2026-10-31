"""Same-kit replacement variants (union-like uncorroborated anomaly; the box ends on OTHER).
Variants: the uncovered TARGET is (a) undetected and walks away, (b) detected and walks off the left edge,
(c) hidden behind OTHER in lockstep, (d) undetected and walks away, slower; various fps / OTHER similarity."""
from common import SS, KIT_A, summarize


def scene(fps, *, tgt_detected, tgt_vx_h, lockstep=False, other_cos=0.95, tx=300.0, n_walk_s=4.0,
          other_vx_h=1.0):
    same = SS.make_people([("TARGET", KIT_A, 1.0), ("OTHER", KIT_A, other_cos)])
    tb = SS.box_at(tx, 400, 40, 100)
    n_hide, n_union, n_front, n_walk = int(1.5 * fps), max(2, int(fps / 2)), int(fps), int(n_walk_s * fps)
    fr = [SS.SceneFrame({"TARGET": tb, "OTHER": SS.box_at(tx, 395, 38, 95)}, order=("OTHER", "TARGET"),
                        drop=("OTHER",)) for _ in range(n_hide)]
    fr += [SS.SceneFrame({"TARGET": tb, "OTHER": SS.box_at(tx + 40 * (k + 1) / n_union, 395, 38, 95)},
                         order=("OTHER", "TARGET"), merge=(("TARGET", "OTHER"),)) for k in range(n_union)]
    fr += [SS.SceneFrame({"TARGET": tb, "OTHER": SS.box_at(tx + 2, 401, 41, 101)}, order=("TARGET", "OTHER"),
                         drop=("TARGET",)) for _ in range(n_front)]
    for k in range(n_walk):
        ob = SS.box_at(tx + 2 + other_vx_h * 100 * (k + 1) / fps, 401, 41, 101)
        if lockstep:
            tbk = SS.box_at(tx + other_vx_h * 100 * (k + 1) / fps, 400, 38, 98)
        else:
            tbk = SS.box_at(tx + tgt_vx_h * 100 * (k + 1) / fps, 400, 40, 100)
        fr.append(SS.SceneFrame({"TARGET": tbk, "OTHER": ob}, order=("TARGET", "OTHER"),
                                drop=() if tgt_detected else ("TARGET",)))
    return SS.run(fr, same, fps=fps)


if __name__ == "__main__":
    for fps in (30.0, 15.0, 10.0):
        summarize(scene(fps, tgt_detected=False, tgt_vx_h=0.0), f"[fps {fps}] (ref, test) undetected, stays")
        summarize(scene(fps, tgt_detected=False, tgt_vx_h=-1.0), f"[fps {fps}] (a) undetected, walks away -1 h/s")
        summarize(scene(fps, tgt_detected=False, tgt_vx_h=-0.5), f"[fps {fps}] (a2) undetected, walks away -0.5 h/s")
        summarize(scene(fps, tgt_detected=True, tgt_vx_h=-3.0, tx=150.0),
                  f"[fps {fps}] (b) detected, exits frame left")
        summarize(scene(fps, tgt_detected=False, tgt_vx_h=0.0, lockstep=True), f"[fps {fps}] (c) lockstep hidden")
        summarize(scene(fps, tgt_detected=False, tgt_vx_h=-1.0, other_cos=0.9),
                  f"[fps {fps}] (a3) undetected, walks away, OTHER cos 0.90")
