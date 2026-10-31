import sys; sys.path.insert(0, sys.argv[1])
sys.path.insert(0, "/home/claude/mut_tools/probe"); from probe_union import scene, run
rn = run(scene(30.0), fps=30.0)
for k in (29,30,31,35,60):
    r = rn.results[k]
    for o in r.observations:
        print(k, o.ref if hasattr(o,'ref') else '', o.tracklet_id, o.flags, o.merge_reasons, o.claimants)
    print('  views', [(v.tracklet_id, v.observed, v.flags) for v in r.tracklets])
