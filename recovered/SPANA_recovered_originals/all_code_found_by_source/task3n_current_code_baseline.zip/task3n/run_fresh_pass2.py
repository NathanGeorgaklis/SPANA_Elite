"""Task 3N: unchanged current Pass 2 (resolver -> enrichment -> adapter -> frozen engine -> canonical
events -> stats) on a FROZEN fresh Pass-1 JSON, then event-clip export. Measurement only.
usage: run_fresh_pass2.py CLIP_ID MO_JSON VIDEO OUT_DIR PLAYER_LABEL"""
import json, sys
import football_analysis.possession_detector as pd
import football_analysis.opponent_detection as od
from event_engine_v1.analyze_target_player import analyze_target_player
from event_engine_v1.event_clip_exporter import export_event_clips

assert pd.TASK3L_OTHER_PLAYER_CONTEXT is True and od.TASK3L_FRAGMENT_EXCLUSION is True
clip_id, mo_json, video, out_dir, label = sys.argv[1:6]
res = analyze_target_player(clip_id, mo_json, video, out_dir, player_id="P1", player_label=label)
ev = json.load(open(res["event_log_path"]))
clips = export_event_clips(video, ev, out_dir + "/clips")
summary = {k: res[k] for k in ("clip_id", "target_status", "n_boundaries", "n_review_required", "n_unresolved",
                               "event_log_path", "player_analysis_path", "raw_events_path", "snapshots_cache_path")}
summary["enrichment_summary"] = {str(k): v for k, v in res["enrichment_summary"].items()} if isinstance(res["enrichment_summary"], dict) else res["enrichment_summary"]
summary["clips"] = clips
json.dump(summary, open(out_dir + "/pass2_summary.json", "w"), indent=1, default=str)
print(json.dumps(summary, default=str)[:2000])
