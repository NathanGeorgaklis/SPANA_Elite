# R5: region lifetime / coverage. TARGET (still) is briefly merged with a farther, smaller OTHER standing just behind
# it (union ~= TARGET's own box), OTHER then walks away BEHIND the target, undetected, and leaves. TARGET's own box
# fills the region (>= 0.9) on every frame, so the region is never forgotten and never resolves (no emergence seen).
# Measures: live regions over time, TARGET trusted coverage after the merge, for 30 s.
from common import *
import itertools
for fps, kit in itertools.product((30.0, 10.0), (SAME, OPP)):
    TB = SS.box_at(300, 400, 40, 100)
    fr = [SS.SceneFrame({"TARGET": TB}) for _ in range(int(2 * fps))]
    ob = SS.box_at(305, 396, 34, 88)
    fr += [SS.SceneFrame({"TARGET": TB, "OTHER": ob}, order=("OTHER", "TARGET"), merge=(("TARGET", "OTHER"),))
           for _ in range(int(fps / 2))]
    for k in range(int(30 * fps)):
        t = k / fps
        x = 305 + 40 * t
        bx = {"TARGET": TB}
        if x < 600:
            bx["OTHER"] = SS.box_at(x, 396, 34, 88)
        fr.append(SS.SceneFrame(bx, order=("OTHER", "TARGET"), drop=("OTHER",) if x < 360 else ()))
    r = SS.run(fr, people(kit), fps=fps)
    pop = r.tracker._pop
    n0 = int(2.5 * fps)
    after = r.frames[n0:]
    tr_last10 = sum(f.identity_trusted for f in r.frames[-int(10 * fps):])
    lin = sum(1 for p in r.pops[-int(10 * fps):] for v in p.tracklets if v.observed and P.MERGE_LINEAGE in v.flags)
    print(tag(), f"fps={fps} kit={'SAME' if kit==SAME else 'OPP '} regions_at_end={len(pop._regions)} "
          f"fillers={[len(g.fillers) for g in pop._regions] if hasattr(pop._regions[0] if pop._regions else None,'fillers') else '-'} "
          f"trusted_after_merge={sum(f.identity_trusted for f in after)}/{len(after)} trusted_last10s={tr_last10}/{int(10*fps)} "
          f"lineage_obs_last10s={lin} wrong_trusted={len(wrong_trusted(r))} pure={r.bank_is_pure()} last={r.states()[-1]}")
