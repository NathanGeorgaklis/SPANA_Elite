# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: player_tracker_rb/output_schema.py builders; R1/R1_DESIGN_DECISIONS.md D05-D07, D19
"""Test helper: builds small, hand-specified R1 tracking documents with the output_schema builders.

These are contract fixtures only. They contain no tracking decisions: every state below is written by hand.
"""
from player_tracker_rb import output_schema as S
from player_tracker_rb.config import TrackerConfig

FPS = 30.0
START = 180
SEED_BOX = [635.3, 304.1, 757.8, 438.3]
BOX = [659.5, 309.3, 758.2, 444.9]


def confidences(identity, overall=None):
    return {"team_confidence": 0.0, "motion_confidence": 1.0, "appearance_confidence": identity,
            "reid_confidence": identity, "overall_track_confidence": identity if overall is None else overall,
            "visibility_confidence": 1.0, "position_confidence": 1.0, "track_continuity_confidence": 1.0,
            "identity_confidence": identity}


def target_record(cfg, frame, rb_state, *, inside=True, episode=1, last_confirmed=None, reasons=None,
                  merged=False, reidentified=False, occluders=()):
    hist, occ, pos = S.TARGET_TRIPLE_BY_RB_STATE[(rb_state, inside if rb_state == "LOST" else True)]
    observed = rb_state == "TRACKING"
    trusted = observed and not merged and not reasons
    ceiling = cfg.get("untrusted_identity_confidence_ceiling")
    idc_rb = 0.86 if trusted else 0.30
    idc_hist = idc_rb if trusted else min(idc_rb, ceiling)
    if observed:
        src, held_age, fso = "DETECTION", 0.0, 0
        lc = frame
    else:
        src = "HELD" if rb_state in ("OCCLUDED", "AMBIGUOUS") else "PREDICTED"
        lc = last_confirmed if last_confirmed is not None else frame - 1
        fso = frame - lc
        held_age = fso / FPS
    v2 = {
        "record_kind": "TARGET", "rb_state": rb_state, "observation_source": src,
        "bbox_is_observed": src == "DETECTION", "bbox_is_held": src == "HELD", "bbox_is_predicted": src == "PREDICTED",
        "held_age_s": held_age, "last_confirmed_frame": lc, "time_since_confirmed_s": (frame - lc) / FPS,
        "observed_or_contested_box": BOX if rb_state in ("AMBIGUOUS", "REACQUIRING") else None,
        "observation_tracklet_id": "P0001" if observed else None, "target_episode": episode,
        "identity_trusted": trusted, "identity_confidence_rb": idc_rb,
        "identity_uncertainty_reasons": [] if trusted else list(reasons or ["OCCLUDER_OVERLAP"]),
        "occluder_track_ids": list(occluders), "merged_suspected": merged,
        "raw_detection_conf": 0.81 if observed else None, "reidentified_this_frame": reidentified,
        "target_similarity": 0.8 if observed else None, "best_competitor_similarity": 0.4 if observed else None,
    }
    return S.build_track_record(
        cfg, track_id=cfg.get("target_track_id"), state=hist, age=frame, last_seen_frame=lc, bbox=BOX,
        observed_position_box=BOX, occlusion_status=occ, position_source=pos, selected_player=True,
        confidences=confidences(idc_hist), frames_since_observed_position=fso,
        frames_since_confirmed_identity=fso, v2=v2)


def population_record(cfg, frame, tid, detected=True, first_frame=1):
    hist, occ, pos = S.POPULATION_TRIPLE_DETECTED if detected else S.POPULATION_TRIPLE_COASTING
    v2 = {"record_kind": "POPULATION", "detected_this_frame": detected, "raw_detection_conf": 0.6 if detected else None,
          "is_competitor": True, "first_frame": first_frame}
    return S.build_track_record(
        cfg, track_id=tid, state=hist, age=frame - first_frame + 1, last_seen_frame=frame, bbox=BOX,
        observed_position_box=BOX, occlusion_status=occ, position_source=pos, selected_player=False,
        confidences=confidences(0.0, overall=0.5), frames_since_observed_position=0 if detected else 1,
        frames_since_confirmed_identity=0, v2=v2)


def build_doc(cfg=None, states=None, episodes=None, n_events=None):
    """A complete R1 document. `states` is a list of (rb_state, inside) per frame."""
    cfg = cfg or TrackerConfig.default()
    states = states or [("TRACKING", True)] * 3 + [("OCCLUDED", True)] * 2 + [("TRACKING", True)]
    episodes = episodes or [1] * len(states)
    frame_log = []
    last_conf = None
    for i, ((rb, inside), ep) in enumerate(zip(states, episodes), start=1):
        if rb == "TRACKING":
            last_conf = i
        t = target_record(cfg, i, rb, inside=inside, episode=ep,
                          last_confirmed=last_conf if last_conf is not None else 0,
                          occluders=["T002"] if rb == "OCCLUDED" else ())
        pop = population_record(cfg, i, "T002")
        contests = [["T001", "T002"]] if rb == "OCCLUDED" else []
        frame_log.append(S.build_frame_record(
            cfg, frame=i, is_cut=False, camera_shift_px=[0.0, 0.0], n_detections=2, overlap_contests=contests,
            tracks=[t, pop],
            v2={"video_frame_index": START + i, "timestamp_s": (START + i) / FPS, "frame_thumb_sha1": "0" * 40,
                "cut_detected": False, "camera_shift_h": [0.0, 0.0], "target_height_px": 135.6}))
    hist_states = [fr["tracks"][0]["state"] for fr in frame_log]
    rb_states = [fr["tracks"][0]["v2"]["rb_state"] for fr in frame_log]
    increments = sum(1 for a, b in zip(episodes, episodes[1:]) if b != a)
    events = [{"episode": k + 2, "lost_from_frame": 1, "uncertain_from_frame": 1, "reacq_started_frame": 2,
               "reacq_accepted_frame": 3, "uncertain_duration_s": 0.1, "evidence": {}}
              for k in range(increments if n_events is None else n_events)]
    finals = []
    for t in frame_log[-1]["tracks"]:
        f = {k: v for k, v in t.items() if k != "v2"}
        finals.append(f)
    summary = {
        "video": "synthetic.mp4", "start_frame": START, "fps": FPS, "processed_frames": len(frame_log),
        "processing_time_s": 1.0, "selected_state_pct": S.selected_state_pct(hist_states, cfg.get("state_pct_round_dp")),
        "n_tracks_ever_created": 2, "n_tracks_final_active": 2, "clip_output": None, "final_confirmed_count": 2,
        "final_provisional_count": 0, "spawn_guard_stats": {}, "identity_conflicts_logged": 0,
        "identity_conflict_counts": {},
        "health_metrics": {k: 0 for k in S.HEALTH_METRIC_KEYS},
        "v2": {"schema": S.SCHEMA_NAME, "schema_version": S.V2_SCHEMA_VERSION, "tracker": S.TRACKER_NAME,
               "tracker_version": "test", "config_hash": cfg.config_hash(),
               "provenance": {"video_path": "synthetic.mp4", "video_sha256": "0" * 64, "fps": FPS, "frame_count": None,
                              "decode_mode": S.DECODE_MODE, "seed_frame": START, "seed_bbox": SEED_BOX, "models": [],
                              "config_hash": cfg.config_hash(), "git_commit": None, "git_dirty": None,
                              "schema_version": S.V2_SCHEMA_VERSION, "tracker_version": "test",
                              "run_started_utc": "2026-10-24T00:00:00Z"},
               "reacquisition_events": events,
               "rb_state_pct": S.selected_state_pct(rb_states, cfg.get("state_pct_round_dp")),
               "notes": []},
    }
    return {"summary": summary, "frame_log": frame_log, "final_tracks": finals}
