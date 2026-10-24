"""Task 3G design check: apply ball_interaction_credibility to the captured
(pre-fix) per-frame ball evidence of a clip and show segment verdicts and the
possession segments before/after gating. Uses the real assess(),
gate_possession_frames(), PossessionFrame and segment_possession()."""
import sys, json
sys.path.insert(0, ".")
from football_analysis.ball_interaction_credibility import FrameBallEvidence, assess, gate_possession_frames
from football_analysis.possession_detector import PossessionFrame
from football_analysis.possession_segmentation import segment_possession
from football_analysis.adapter import TrackingAdapter

clip = sys.argv[1]
d = json.load(open(f"event_engine_v1/task3g/capture/{clip}_prefix_capture.json"))
F = d["frames"]
class _S:  # minimal stand-in when the capture already carries per-frame tracking states
    def __init__(self, ts): self.tracking_state = ts
if d.get("snapshots"):
    snaps = {s["frame"]: _S(s["tracking_state"]) for s in d["snapshots"]}
else:
    snaps = {s.frame_number: s for s in TrackingAdapter.load(sys.argv[2])}
ev = []
for f in F:
    b = f["best"]
    trusted = f["had_confirmed_before"] and f["prev_fsc"] <= 10
    ev.append(FrameBallEvidence(frame=f["frame"], state=f["state"], center=tuple(f["center"]) if f["center"] else None,
                                box=b["box"] if b else None, stage=f["stage"] if b else None,
                                det_conf=b["conf"] if b else None, prediction_trusted=trusted,
                                player_bbox=f["player_bbox"]))
labels, verdicts = assess(ev)
for v in verdicts:
    print(f"seg {v.segment_id:2d} frames {v.first_frame}-{v.last_frame} root {v.root_frame} in_zone={v.root_in_zone} -> {v.label}: {v.reasons}")
pfs = []
for f in F:
    s = snaps.get(f["frame"])
    ts = s.tracking_state if s else "LOST"
    st = f["state"]
    dist = f["ball_distance"]
    pfs.append(PossessionFrame(frame=f["frame"], ball_distance=dist, ball_state=st,
                               in_possession=bool(f["in_possession_raw"]), tracking_state=ts,
                               paused=(ts != "TRACKING")))
def segs(pf):
    sf = segment_possession(pf); out = []; i = 0
    while i < len(sf):
        if sf[i].in_possession:
            j = i
            while j < len(sf) and sf[j].in_possession: j += 1
            out.append((F[i]["frame"], F[j - 1]["frame"])); i = j
        else: i += 1
    return out
print("possession segments BEFORE gate:", segs(pfs))
print("possession segments AFTER  gate:", segs(gate_possession_frames(pfs, labels)))
