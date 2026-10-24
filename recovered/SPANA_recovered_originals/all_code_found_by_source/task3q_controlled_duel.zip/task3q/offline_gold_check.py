"""Offline check of controlled_duel.evaluate on the frozen Task 3O GOLD capture (no pipeline run)."""
import json, sys, os
from types import SimpleNamespace as NS
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from football_analysis import controlled_duel as cd
HERE = os.path.dirname(os.path.abspath(__file__))
cap = json.load(open(os.path.join(HERE, "../task3o/gold_capture/capture.json")))["frames"]
n = len(cap)
snaps, balls, boxes, pfs = [], [], [], []
for f in range(n):
    x = cap[str(f)]; det = x["detector"]
    snaps.append(NS(bbox=det["snapshot_bbox"], tracking_state=det["snapshot_state"]))
    bc = det["ball_centre"]
    balls.append(NS(centre_x=bc[0] if bc else None, centre_y=bc[1] if bc else None, state=det["ball_state"], bridge_info=None))
    boxes.append([(t["box"], t["team"]) for t in x["team"]])
    pfs.append(NS(**det["possession"]))
ctx = cd.ControlledDuelContext(snaps, balls, boxes, fps=30.0)
for i_end, seg in ((82, 16), (163, 116)):
    a = cd.evaluate(ctx, pfs, i_end, seg)
    print(i_end, a.valid, a.failed_gates, a.outcome, a.window(), a.outcome_frame, round(a.score, 3), round(a.candidate_confidence, 3), round(a.duel_confidence, 3))
    print(json.dumps(a.evidence.get("measurements"), indent=0)[:1500]); print(a.evidence.get("components"))
