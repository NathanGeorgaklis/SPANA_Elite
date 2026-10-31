import sys; sys.path.insert(0, sys.argv[1]); sys.path.insert(0, "/home/claude/mut_tools/probe")
from player_tracker_rb import population as P
from probe_union import scene, clean_union, run
import collections
for fps in (30.0,):
    for args in [dict(), dict(third_start=436.0, third_speed=60.0)]:
        regs = []
        rn = run(scene(fps, 0.0, **args), fps=fps, hook=lambda k, pop, res: regs.append([(round(r.last_evidence_t,2), r.n_bodies) for r in pop._regions]))
        st = collections.Counter()
        for k,(r,f) in enumerate(zip(rn.results, rn.frames)):
            for di, o in enumerate(r.observations):
                if len(f.owners[di]) > 1:
                    v = [x for x in r.tracklets if x.observed and x.detection_index == di]
                    st[(P.MERGED_SUSPECTED in o.flags, tuple(o.merge_reasons), bool(v and P.MERGE_LINEAGE in v[0].flags), bool(v))] += 1
        print(args, dict(st), "clean", len(clean_union(rn)), "regions@end", regs[-1], "regions@45", regs[45], regs[80])
