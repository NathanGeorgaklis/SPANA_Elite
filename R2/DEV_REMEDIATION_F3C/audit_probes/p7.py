"""Camera cut (continuity broken) in the middle of a persistent two-person union."""
import sys; sys.path.insert(0, sys.argv[1])
from player_tracker_rb.camera import identity_motion, CUT
from tests_rb import _synth_scene as SS
from tests_rb.test_population_safety import people, SAME, OPP, clean_union_tracklet_frames, tracking_on_union, wrong_trusted
TB = SS.box_at(300, 400, 40, 100); OB = SS.box_at(315, 400, 40, 100)
for fps in (30.0, 10.0):
    for kit in (SAME,):
        fr = [SS.SceneFrame({"TARGET": TB}) for _ in range(int(2 * fps))]
        fr += [SS.SceneFrame({"TARGET": TB, "OTHER": OB}, order=("OTHER", "TARGET")) for _ in range(int(fps / 3))]
        for k in range(int(5 * fps)):
            fr.append(SS.SceneFrame({"TARGET": TB, "OTHER": OB}, merge=(("TARGET", "OTHER"),), order=("TARGET", "OTHER"),
                                    camera=identity_motion(CUT, continuity_broken=True) if k == int(2 * fps) else None))
        r = SS.run(fr, people(kit), fps=fps)
        print(fps, "clean_union", len(clean_union_tracklet_frames(r)), "tracking_on_union", len(tracking_on_union(r)),
              "wrong_trusted", len(wrong_trusted(r)), "bank_pure", r.bank_is_pure(), "states_end", r.states()[-1])
