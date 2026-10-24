"""Task 3G: produce event_log.json / player_analysis.json from a post-fix engine
run's raw events, using exactly analyze_target_player's own integration code
(actor assessments, normalization, possession events, stats). Avoids a second
full Pass-2 run: the raw events come from the post-fix capture run, which
executed the real engine end to end.

usage: finalize_from_raw.py <clip_id> <mo_json> <video> <raw_events_json> <out_dir> <player_label>
"""
import sys, json, os
sys.path.insert(0, ".")
from event_engine_v1.analyze_target_player import (build_actor_assessments, assign_ids_and_merge,
                                                   assemble_player_analysis)
from player_tracker_mo.tracklet_builder import build_tracklets_from_frame_log
from player_tracker_mo.selected_player_resolver import resolve_selected_player
from player_tracker_mo.offline_identity_enrichment import enrich_frame_log_with_offline_identity
from player_tracker.appearance import AppearanceEncoder

clip, MO, VIDEO, RAW, OUT, LABEL = sys.argv[1:7]
os.makedirs(OUT, exist_ok=True)
d = json.load(open(MO)); fl = d["frame_log"]; S = d["summary"]["start_frame"]; fps = d["summary"]["fps"]
res = resolve_selected_player(fl, build_tracklets_from_frame_log(fl), video_path=VIDEO, encoder=AppearanceEncoder(use_ocr=False))
enriched = enrich_frame_log_with_offline_identity(fl, res)
bres = res["boundary_resolutions"]
status = ("UNRESOLVED" if any(b.identity_state == "UNRESOLVED_IDENTITY" for b in bres) else
          "REVIEW_REQUIRED" if any(b.identity_state == "REVIEW_REQUIRED_IDENTITY" for b in bres) else "RESOLVED")
raw = json.load(open(RAW))
raw_events = raw.get("events", [])
log = assign_ids_and_merge(raw_events, clip, S, fps,
                           build_actor_assessments(raw_events, enriched, status, d["summary"].get("identity_conflict_counts")))
n = d["summary"]["processed_frames"]
pa = assemble_player_analysis(clip, VIDEO, LABEL, status, log, n / fps)
json.dump(log, open(f"{OUT}/{clip}_event_log.json", "w"), indent=2, default=str)
json.dump(pa, open(f"{OUT}/{clip}_player_analysis.json", "w"), indent=2, default=str)
json.dump(raw, open(f"{OUT}/{clip}_events_raw.json", "w"), indent=2, default=str)
print(clip, status, len(log), "events", pa["stats"])
