"""Task 3F audit: applies actor_attribution.assess_actor to every raw frozen-engine
event of a clip and prints the evidence. Read-only."""
import sys, json
sys.path.insert(0, ".")
from event_engine_v1.actor_attribution import (assess_actor, build_frame_index, selected_ids,
                                               conflicts_involve, band_for)
mo, raw, status = sys.argv[1], sys.argv[2], sys.argv[3]
d = json.load(open(mo)); fl = d["frame_log"]
idx = build_frame_index(fl)
conf = conflicts_involve(selected_ids(fl), d["summary"].get("identity_conflict_counts"))
evs = json.load(open(raw))["events"]
print("selected ids:", selected_ids(fl), "clip conflicts involve selected:", conf)
out = []
for e in evs:
    a = assess_actor(e, evs, idx, status, conf)
    row = dict(type=e["type"], frames=[e["start_frame"], e["end_frame"]], event_confidence=e["confidence"],
               event_band=e["confidence_band"], review_required=e["review_required"], success=e.get("success"), **a)
    out.append(row)
    print(json.dumps(row, default=str))
json.dump(out, open(sys.argv[4], "w"), indent=2, default=str)
