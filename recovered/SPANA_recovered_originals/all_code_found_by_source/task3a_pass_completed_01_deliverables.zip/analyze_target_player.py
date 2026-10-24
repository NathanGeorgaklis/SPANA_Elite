"""
event_engine_v1/analyze_target_player.py -- Task 3A, Steps 3 + 5 + 7.

Orchestration-only. Wires together pieces that already exist and are
frozen; no event-detection, possession, or scoring logic lives in this
file. Per the task's explicit instructions: no threshold, weight, or
hard-coded player number/event class/frame range is introduced here, and
no frozen file is edited.

Pipeline (see TARGET_PLAYER_ADAPTER_COMPATIBILITY_REPORT.md for the full
field-by-field trace that justifies every call below):

  1. Load the Pass-1 player_tracker_mo tracking JSON (produced separately,
     ahead of this module, by `python3 -m player_tracker_mo.run_mo ...` --
     that call takes tens of minutes per clip and is a precomputed input
     to this orchestrator, not something it re-triggers itself).
  2. player_tracker_mo.tracklet_builder.build_tracklets_from_frame_log
  3. player_tracker_mo.selected_player_resolver.resolve_selected_player
  4. player_tracker_mo.offline_identity_enrichment.
     enrich_frame_log_with_offline_identity
  5. player_tracker_mo.event_adapter.build_integrated_event_snapshots
  6. football_analysis.adapter.TrackingAdapter.save (cache to disk, in
     exactly the format validation/lib/run_engine_pass.py already reads)
  7. validation.lib.run_engine_pass.run_engine_pass (Pass 2: the frozen
     FootballAnalysisEngine, unmodified)
  8. normalize_events() -- Step 5's thin event-normalization layer
  9. event_engine_v1.possession_events.derive_possession_events, merged in
 10. event_engine_v1.player_stats.compute_player_stats
 11. assemble_player_analysis() -- the approved player_analysis.json shape
     (MILESTONE3_TASK3A_TARGET_PLAYER_EVENT_ENGINE_SPEC.md Section 6)

Step 5 normalization note: the frozen engine's raw event dict
(football_analysis/clip_tagger.py's ClipTagger.tag()) has no `outcome`
string enum and no `actor` field -- it has `success` (bool), `qualifiers`,
`confidence`, `confidence_band`, `review_required`, all for a single
implicit player_id (the one FootballAnalysisEngine was run against, i.e.
the target player). normalize_event() below maps this onto the task's
exact requested fields (event_id, type, start_frame, end_frame,
start_time, end_time, actor, outcome, success, confidence,
review_required, qualifiers, source_event) WITHOUT inventing a new
outcome taxonomy: outcome is a plain rename of (type, success) into the
vocabulary confidence.py itself already documents (PASS/CROSS:
COMPLETE/INCOMPLETE; DUEL: WON/LOST; SHOT: ATTEMPT; GOAL: GOAL). actor is
TARGET_PLAYER unless the frozen engine's own review_required flag is set,
in which case actor=UNKNOWN per the task document's own rule -- this
already carries the identity-trust signal, since event_adapter.py's
OCCLUDED-downgrade for untrusted-identity frames (see
TARGET_PLAYER_ADAPTER_COMPATIBILITY_REPORT.md Section 3) feeds directly
into the frozen ConfidenceEngine's own review_required judgment; this
module does not re-derive identity trust a second time.
"""
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
for p in (REPO_ROOT, REPO_ROOT / "validation" / "lib"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from player_tracker.appearance import AppearanceEncoder  # noqa: E402
from player_tracker_mo.tracklet_builder import build_tracklets_from_frame_log  # noqa: E402
from player_tracker_mo.selected_player_resolver import resolve_selected_player  # noqa: E402
from player_tracker_mo.offline_identity_enrichment import (  # noqa: E402
    enrich_frame_log_with_offline_identity, summarize_enrichment,
)
from player_tracker_mo.event_adapter import build_integrated_event_snapshots  # noqa: E402
from football_analysis.adapter import TrackingAdapter  # noqa: E402
from football_analysis.engine import FootballAnalysisEngine  # noqa: E402
from football_analysis.colour import shirt_color_hist  # noqa: E402
import cv2  # noqa: E402

from event_engine_v1.possession_events import derive_possession_events  # noqa: E402
from event_engine_v1.player_stats import compute_player_stats  # noqa: E402


def run_engine_pass_with_offset(video_path, snapshots, start_frame_offset, n_frames, out_json_path):
    """Mirrors validation/lib/run_engine_pass.py's own logic exactly (same
    tracked_color_hist computation, same FootballAnalysisEngine.run_analysis_pass
    call, same conf=0.15) with ONE change: seeks to the TRUE absolute video frame
    (start_frame_offset) rather than snapshots[0].frame_number. See the long
    comment in analyze_target_player() above for why that helper's hard-coded
    seek target is wrong whenever Pass 1 started tracking at a non-zero frame.
    Still calls the frozen FootballAnalysisEngine.run_analysis_pass unmodified --
    only the seek target this wrapper passes to it differs from run_engine_pass.py."""
    filtered = [s for s in snapshots if s.frame_number < n_frames]
    fps = cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FPS) or 25.0
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_offset)
    ok, frame0 = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"could not read frame {start_frame_offset} of {video_path}")
    tracked_color_hist = shirt_color_hist(frame0, filtered[0].bbox)

    engine = FootballAnalysisEngine(fps=fps)
    return engine.run_analysis_pass(
        video_path, filtered, tracked_color_hist,
        start_frame=start_frame_offset, conf=0.15,
        out_json_path=out_json_path,
    )

OUTCOME_MAP = {
    ("PASS", True): "COMPLETE", ("PASS", False): "INCOMPLETE",
    ("CROSS", True): "COMPLETE", ("CROSS", False): "INCOMPLETE",
    ("DUEL", True): "WON", ("DUEL", False): "LOST",
}


def normalize_event(raw: dict, event_id: str, start_frame_offset: int = 0, fps: float = 25.0) -> dict:
    """raw: one entry from the frozen engine's result["events"] (clip_tagger.py
    shape). Returns the event_engine_v1 schema dict. Pure field mapping --
    no new classification, no threshold -- except for adding back
    start_frame_offset, which converts the engine's own seek-relative
    start_frame/end_frame/start_time/end_time into clip-absolute values (see
    the frame-registration note in analyze_target_player()). start_frame_offset
    is 0 for any clip whose Pass-1 run started at frame 0 (every clip this
    project validated before pass_completed_01), so this is a no-op there."""
    t = raw["type"]
    success = raw.get("success")
    outcome = OUTCOME_MAP.get((t, success))
    if t == "GOAL":
        outcome = "GOAL"
    elif t == "SHOT":
        outcome = "ATTEMPT"

    review_required = bool(raw.get("review_required"))
    # clip_tagger.py's own "qualifiers" field is a LIST of tag strings (e.g.
    # ["CARRY"]), not a dict -- confirmed against this run's actual raw output
    # (an earlier version of this function wrongly assumed dict without
    # checking real data; fixed here). Wrapped into a dict alongside
    # confidence_band so this schema field stays a dict as the task document
    # specifies, without discarding the engine's own tags.
    qualifiers = {"tags": list(raw.get("qualifiers") or []), "confidence_band": raw.get("confidence_band")}

    start_frame = raw.get("start_frame")
    end_frame = raw.get("end_frame")
    abs_start_frame = start_frame + start_frame_offset if start_frame is not None else None
    abs_end_frame = end_frame + start_frame_offset if end_frame is not None else None

    return {
        "event_id": event_id,
        "type": t,
        "start_frame": abs_start_frame,
        "end_frame": abs_end_frame,
        "start_time": round(abs_start_frame / fps, 2) if abs_start_frame is not None else raw["start_time"],
        "end_time": round(abs_end_frame / fps, 2) if abs_end_frame is not None else raw["end_time"],
        "actor": "UNKNOWN" if review_required else "TARGET_PLAYER",
        "outcome": outcome,
        "success": success,
        "confidence": raw.get("confidence"),
        "review_required": review_required,
        "qualifiers": qualifiers,
        "source_event": "football_analysis.engine.FootballAnalysisEngine.run_analysis_pass (frozen)",
    }


def assign_ids_and_merge(raw_events: list, clip_id: str, start_frame_offset: int = 0, fps: float = 25.0) -> list:
    """Normalizes every raw engine event, derives POSSESSION_WON/LOST over
    the normalized TARGET_PLAYER subset, assigns final sequential ids to
    the possession events, and returns everything merged, sorted by
    start_time. Never removes or reorders the engine's own events --
    possession events are additive."""
    normalized = [normalize_event(e, f"{clip_id}_e{i:03d}", start_frame_offset, fps)
                  for i, e in enumerate(raw_events)]
    target_events = [e for e in normalized if e["actor"] == "TARGET_PLAYER"]
    possession_raw = derive_possession_events(target_events)
    next_idx = len(normalized)
    possession_events = []
    for pe in possession_raw:
        pe["event_id"] = f"{clip_id}_p{next_idx:03d}"
        next_idx += 1
        possession_events.append(pe)
    merged = normalized + possession_events
    merged.sort(key=lambda e: e["start_time"])
    return merged


def assemble_player_analysis(clip_id: str, video_path: str, player_label: str,
                              target_status: str, event_log: list, duration_analyzed_s: float) -> dict:
    """The approved schema, MILESTONE3_TASK3A_TARGET_PLAYER_EVENT_ENGINE_SPEC.md
    Section 6, plus the user's own Step 7 addition of player.target_status."""
    stats = compute_player_stats(event_log)
    events_review_required = stats.pop("events_review_required")

    confidence_summary = {"high": 0, "medium": 0, "low": 0}
    review_required_events = []
    for e in event_log:
        band = (e.get("qualifiers") or {}).get("confidence_band")
        band_key = band.lower() if isinstance(band, str) else band
        if band_key in confidence_summary:
            confidence_summary[band_key] += 1
        if e.get("review_required"):
            review_required_events.append(e["event_id"])

    event_log_sha256 = hashlib.sha256(
        json.dumps(event_log, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()

    return {
        "player": {"label": player_label, "target_status": target_status},
        "match": {"source_video": str(video_path), "duration_analyzed_s": round(duration_analyzed_s, 2)},
        "minutes_analyzed": round(duration_analyzed_s / 60.0, 3),
        "stats": stats,
        "events": event_log,
        "confidence_summary": confidence_summary,
        "review_required_events": review_required_events,
        "events_review_required_count": events_review_required,
        "source_event_log_sha256": event_log_sha256,
    }


def analyze_target_player(clip_id: str, mo_tracking_json_path: str, video_path: str,
                           out_dir: str, player_id: str = "P1", player_label: str = "target player") -> dict:
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    d = json.load(open(mo_tracking_json_path))
    fps = d["summary"]["fps"]
    frame_log = d["frame_log"]

    tracklets = build_tracklets_from_frame_log(frame_log)
    encoder = AppearanceEncoder(use_ocr=False)
    resolver_result = resolve_selected_player(frame_log, tracklets, video_path=video_path, encoder=encoder)

    enriched = enrich_frame_log_with_offline_identity(frame_log, resolver_result)
    enrichment_summary = summarize_enrichment(enriched)

    snapshots = build_integrated_event_snapshots(enriched, fps, player_id=player_id)
    snapshots_cache_path = out_root / f"{clip_id}_integrated_snapshots.json"
    TrackingAdapter.save(snapshots, str(snapshots_cache_path))

    # BUG FOUND during Step 9's first run (pass_completed_01), fixed here.
    # frame_log's "frame" values (and therefore these snapshots' frame_number) are
    # 0-based RELATIVE to wherever run_mo.py's start_frame argument seeded the online
    # tracker -- here, absolute video frame 28, chosen to reuse an already-validated
    # seed bbox from an old cached tracking run rather than frame 0. That is exactly
    # what football_analysis/engine.py's run_analysis_pass() expects for its MATCHING
    # loop (by_frame.get(frame_idx), frame_idx counting 0,1,2,... frames read SINCE
    # the cap.set seek) -- so snapshot.frame_number must stay untouched, 0-based.
    # What must NOT stay as validation/lib/run_engine_pass.py has it is the SEEK
    # TARGET itself: that helper always seeks to snapshots[0].frame_number (=0),
    # i.e. the video's own true start -- correct only when Pass 1 also started
    # tracking at frame 0, true for every PRIOR validated clip in this project, but
    # not here. Left as run_engine_pass leaves it, Pass 2 would read 28 frames
    # (~0.5s) of the WRONG footage against every tracked position: a frame-
    # registration integration bug, not a detection-logic or threshold issue, and
    # entirely a consequence of this orchestrator's own choice of a non-zero
    # start_frame -- not a frozen-file defect. Fixed here by calling
    # football_analysis.engine.FootballAnalysisEngine.run_analysis_pass directly
    # (still the frozen, unmodified engine -- just not going through
    # run_engine_pass's helper, which hard-codes the wrong seek target for this
    # case) with the TRUE absolute start_frame, while leaving snapshot.frame_number
    # alone. Reported event start_frame/end_frame come back relative to that same
    # seek, so clip-absolute times are start_frame_offset/fps + (raw/fps) --
    # applied in normalize_event() below, not here.
    start_frame_offset = d["summary"].get("start_frame", 0)

    n_frames = (snapshots[-1].frame_number + 1) if snapshots else 0
    events_out_path = out_root / f"{clip_id}_events_raw.json"
    engine_result = run_engine_pass_with_offset(
        video_path, snapshots, start_frame_offset, n_frames, str(events_out_path))

    n_boundaries = len(resolver_result["boundary_resolutions"])
    n_review_required = sum(1 for b in resolver_result["boundary_resolutions"]
                             if b.identity_state == "REVIEW_REQUIRED_IDENTITY")
    n_unresolved = sum(1 for b in resolver_result["boundary_resolutions"]
                        if b.identity_state == "UNRESOLVED_IDENTITY")
    target_status = ("UNRESOLVED" if n_unresolved > 0 else
                      "REVIEW_REQUIRED" if n_review_required > 0 else "RESOLVED")

    event_log = assign_ids_and_merge(engine_result.get("events", []), clip_id, start_frame_offset, fps)
    # snapshots[*].timestamp is seek-relative (see the frame-registration note
    # above) -- duration_analyzed_s only needs the SPAN, which that offset cancels
    # out of, so this is correct as-is without adding start_frame_offset back in.
    duration_analyzed_s = (snapshots[-1].timestamp - snapshots[0].timestamp) if len(snapshots) > 1 else 0.0

    player_analysis = assemble_player_analysis(
        clip_id, video_path, player_label, target_status, event_log, duration_analyzed_s)

    event_log_path = out_root / f"{clip_id}_event_log.json"
    json.dump(event_log, open(event_log_path, "w"), indent=2, default=str)
    player_analysis_path = out_root / f"{clip_id}_player_analysis.json"
    json.dump(player_analysis, open(player_analysis_path, "w"), indent=2, default=str)

    return dict(
        clip_id=clip_id, fps=fps, player_id=player_id, target_status=target_status,
        n_boundaries=n_boundaries, n_review_required=n_review_required, n_unresolved=n_unresolved,
        enrichment_summary=enrichment_summary,
        event_log_path=str(event_log_path), player_analysis_path=str(player_analysis_path),
        snapshots_cache_path=str(snapshots_cache_path), raw_events_path=str(events_out_path),
        player_analysis=player_analysis,
    )


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clip-id", required=True)
    ap.add_argument("--mo-tracking-json", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--player-id", default="P1")
    ap.add_argument("--player-label", default="target player")
    args = ap.parse_args()
    result = analyze_target_player(args.clip_id, args.mo_tracking_json, args.video,
                                    args.out_dir, player_id=args.player_id, player_label=args.player_label)
    print(f"target_status={result['target_status']} "
          f"n_events={len(result['player_analysis']['events'])} "
          f"boundaries={result['n_boundaries']} "
          f"(review_required={result['n_review_required']}, unresolved={result['n_unresolved']})")
    print(f"wrote {result['event_log_path']}")
    print(f"wrote {result['player_analysis_path']}")


if __name__ == "__main__":
    main()
