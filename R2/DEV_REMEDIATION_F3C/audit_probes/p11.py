"""(a) compensated fast pan with the crouched person staying put (nothing vanishes): no break expected.
   (b) two feet-aligned nested boxes, only one vanishes: break expected."""
import sys; sys.path.insert(0, sys.argv[1])
from tests_rb._synth_people import Frame, box_at, cand, run, pan
def sh(b, dx): return (b[0] + dx, b[1], b[2] + dx, b[3])
TGT = box_at(300, 400, 40, 100); crouch = box_at(300, 425, 36, 50)
for pxf in (10.0, 25.0, 40.0):
    fr = []; off = 0.0
    for k in range(60):
        cam = None
        if k >= 30: off += pxf; cam = pan(pxf)
        fr.append(Frame([cand(sh(TGT, off)), cand(sh(crouch, off))], [("TARGET",), ("OTHER",)], camera=cam if k else None))
    rn = run(fr, fps=30.0)
    print("(a) pan", pxf, "breaks", [k for k, r in enumerate(rn.results) for o in r.observations if o.continuity_break])
legs = (302.0, 415.0, 318.0, 449.0)     # host lower-leg box (right side), feet-aligned, always present
crouch2 = box_at(290, 427, 18, 46)
fr = [Frame([cand(TGT), cand(legs, 0.6)], [("TARGET",), ("TARGET",)]) for _ in range(30)]
fr += [Frame([cand(TGT), cand(legs, 0.6), cand(crouch2, 0.7)], [("TARGET",), ("TARGET",), ("OTHER",)]) for _ in range(15)]
fr += [Frame([cand(box_at(304, 401, 42, 102)), cand(legs, 0.6)], [("OTHER",), ("TARGET",)])] * 1
fr += [Frame([cand(box_at(304 + 3 * k, 401, 42, 102))], [("OTHER",)]) for k in range(15)]
rn = run(fr, fps=30.0)
print("(b) breaks", [k for k, r in enumerate(rn.results) for o in r.observations if o.continuity_break],
      "contained@44", [len(t.contained) for t in rn.population._live][:3])
