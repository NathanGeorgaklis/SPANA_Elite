"""Task 3Q: how close is GOLD to each path-B gate? Offline, on the frozen Task 3O capture.
Each constant is moved 30% in the STRICTER direction (one at a time); the verdict must not flip
for the result to be robust rather than threshold-fitted. Also reports the value GOLD actually has."""
import json, os, sys, importlib
from types import SimpleNamespace as NS
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from football_analysis import controlled_duel as cd
HERE = os.path.dirname(os.path.abspath(__file__))
cap = json.load(open(os.path.join(HERE, "../task3o/gold_capture/capture.json")))["frames"]
snaps, balls, boxes, pfs = [], [], [], []
for f in range(len(cap)):
    x = cap[str(f)]; det = x["detector"]
    snaps.append(NS(bbox=det["snapshot_bbox"], tracking_state=det["snapshot_state"]))
    bc = det["ball_centre"]
    balls.append(NS(centre_x=bc[0] if bc else None, centre_y=bc[1] if bc else None, state=det["ball_state"], bridge_info=None))
    boxes.append([(t["box"], t["team"]) for t in x["team"]])
    pfs.append(NS(**det["possession"]))
ctx = cd.ControlledDuelContext(snaps, balls, boxes, fps=30.0)

def run():
    a = cd.evaluate(ctx, pfs, 82, 16)
    return a.valid, a.outcome, a.window(), a.failed_gates
base = run()
print("baseline", base)
# (constant, stricter factor)  -- stricter = shrink radii/tolerances, grow minimums
plan = [("MIN_CONTROL_S", 1.3), ("PRE_MIN_POSSESSION_FRACTION", 1.3), ("PRE_MIN_TRACKING_FRACTION", 1.2),
        ("CHAIN_STEP_H", 0.7), ("ENGAGE_RADIUS_H", 0.7), ("REACH_H", 0.7), ("REACH_ENGAGED_H", 0.7),
        ("CONTACT_CENTRE_H", 0.7), ("RELEASE_LINK_S", 0.7), ("CONTEST_GAP_S", 0.7), ("MIN_CONTEST_S", 1.3),
        ("MIN_CLOSING_H", 1.3), ("BALL_LOCAL_H", 0.7), ("MAX_UNOBSERVED_S", 0.7), ("POST_GROUNDING_S", 0.7),
        ("RECOVERY_S", 0.7), ("OPP_CONTROL_H", 1.3), ("OPP_CONTROL_MIN_S", 0.7)]
rows = []
for name, k in plan:
    old = getattr(cd, name)
    setattr(cd, name, old * k)
    r = run()
    setattr(cd, name, old)
    rows.append({"constant": name, "value": old, "stricter_value": round(old * k, 4), "valid": r[0], "outcome": r[1],
                 "window": r[2], "failed_gates": r[3], "verdict_changed": (r[0], r[1]) != (base[0], base[1])})
    print(name, old, "->", round(old * k, 4), r[0], r[1], r[2], r[3])
json.dump({"baseline": {"valid": base[0], "outcome": base[1], "window": base[2]}, "rows": rows},
          open(os.path.join(HERE, "work_gold_sensitivity.json"), "w"), indent=1)
