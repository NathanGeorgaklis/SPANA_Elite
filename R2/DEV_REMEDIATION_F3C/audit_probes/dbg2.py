import sys; sys.path.insert(0, sys.argv[1])
from player_tracker_rb import population as P
from tests_rb._synth_people import Frame, box_at, cand, run
for fps in (30.0, 15.0, 10.0):
    fr = [Frame([cand(box_at(400, 360, 32, 80)), cand(box_at(445, 360, 32, 80))],
                [("TARGET",), ("OTHER",)]) for _ in range(int(fps))]
    fr += [Frame([cand((384.0, 320.0, 428.0, 400.0))], [("TARGET", "OTHER")]) for _ in range(int(fps))]
    fr += [Frame([cand(box_at(400 - 40 * (k + 1) / fps, 360, 32, 80)),
                  cand(box_at(412 + 40 * (k + 1) / fps, 360, 32, 80))], [("TARGET",), ("OTHER",)])
           for k in range(int(2 * fps))]
    pop_regions = []
    rn = run(fr, fps=fps, hook=lambda k, pop, res: pop_regions.append(len(pop._regions)))
    ms = sum(1 for r in rn.results for o in r.observations if P.MERGED_SUSPECTED in o.flags)
    lin = sum(1 for r in rn.results for v in r.tracklets if P.MERGE_LINEAGE in v.flags)
    print(fps, "merged_suspected_obs", ms, "max_regions", max(pop_regions), "lineage_views", lin,
          "ids", sorted({v.tracklet_id for r in rn.results for v in r.tracklets}))
