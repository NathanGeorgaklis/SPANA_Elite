"""Target-level repro: passer-by first DETECTED already overlapping a persistent union (e.g. emerging from behind
it), then the union box is undetected briefly. Uses the test file's own helpers."""
import sys; sys.path.insert(0, sys.argv[1])
from tests_rb import _synth_scene as SS
from tests_rb.test_population_safety import people, SAME, OPP, clean_union_tracklet_frames, tracking_on_union, wrong_trusted
TB = SS.box_at(300, 400, 40, 100); OB = SS.box_at(315, 400, 40, 100)
for fps in (30.0, 15.0, 10.0):
    for kit in (SAME, OPP):
        for x0 in (345.0, 350.0):
            fr = [SS.SceneFrame({"TARGET": TB}) for _ in range(int(2 * fps))]
            fr += [SS.SceneFrame({"TARGET": TB, "OTHER": OB}, order=("OTHER", "TARGET")) for _ in range(int(fps / 3))]
            for k in range(int(7.0 * fps)):
                t = k / fps
                bx = {"TARGET": TB, "OTHER": OB}
                if 1.5 <= t <= 3.0:
                    bx["THIRD"] = SS.box_at(x0 + (t - 1.5) * 80, 395, 40, 100)
                drop = ("TARGET", "OTHER") if 4.0 <= t < 4.6 else ()
                fr.append(SS.SceneFrame(bx, merge=(("TARGET", "OTHER"),) if not drop else (),
                                        order=("THIRD", "TARGET", "OTHER"), drop=drop))
            r = SS.run(fr, people(kit), fps=fps)
            print(fps, kit, x0, "clean_union", len(clean_union_tracklet_frames(r)), "tracking_on_union",
                  len(tracking_on_union(r)), "wrong_trusted", len(wrong_trusted(r)), "bank_pure", r.bank_is_pure())
