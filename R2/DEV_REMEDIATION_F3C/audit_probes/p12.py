"""Crouched person slightly NEARER (feet 0.12 h below the host's feet line, still NESTED: 76% inside) stands up in
front and takes the host box's place."""
import sys; sys.path.insert(0, sys.argv[1])
from tests_rb._synth_people import Frame, box_at, cand, run
from tests_rb.test_population_safety import mixed_single_tracklets
TGT = box_at(300, 400, 40, 100)                 # bottom 450
for drop_px in (0.0, 8.0, 12.0):
    crouch = (282.0, 412.0 + drop_px, 318.0, 462.0 + drop_px - 12.0 + 0.0)
    crouch = (282.0, 400.0 + drop_px, 318.0, 450.0 + drop_px)     # 50 tall, bottom drop_px below host bottom
    fr = [Frame([cand(TGT)], [("TARGET",)]) for _ in range(30)]
    fr += [Frame([cand(TGT), cand(crouch)], [("TARGET",), ("OTHER",)]) for _ in range(15)]
    fr += [Frame([cand(box_at(300, 400 + drop_px / 2, 40, 100 + drop_px))], [("OTHER",)]) for _ in range(5)]
    fr += [Frame([cand(box_at(300 + 3 * (k + 1), 400 + drop_px / 2, 40, 100 + drop_px))], [("OTHER",)]) for k in range(20)]
    rn = run(fr, fps=30.0)
    print("feet below host by", drop_px, "mixed", mixed_single_tracklets(rn),
          "breaks", [k for k, r in enumerate(rn.results) for o in r.observations if o.continuity_break][:2])
