"""
event_engine_v1/actor_attribution.py -- Task 3F.

Decouples ACTOR confidence ("is the player who did this the resolved target
player?") from EVENT confidence ("did this event happen, and is its type /
outcome right?"). Before Task 3F, analyze_target_player.normalize_event()
set actor = UNKNOWN whenever the frozen engine's review_required flag was
set. That flag is computed by football_analysis/confidence.py as

    raw = 0.35 * tracking_quality + 0.45 * ball_quality + 0.20 * team_quality
    review_required (outcome events) = band != HIGH

i.e. it is dominated by BALL evidence quality and an always-unknown team
term (team_sim is hard-coded None -> 0.4), not by target identity. Using
it as an identity proxy is the conflation this module removes.

Evidence used (all pre-existing, nothing new is modelled or trained):

  * Clip level: the selected-player resolver's target_status
    (RESOLVED / REVIEW_REQUIRED / UNRESOLVED), and the online tracker's
    logged identity conflicts (summary["identity_conflict_counts"]).
  * Frame level, from the enriched Pass-1 frame_log record of the selected
    track (the exact record event_adapter.py turned into the TrackingSnapshot
    the frozen engine consumed for that frame -- engine frame f IS frame_log
    frame f): online state, identity_state, position_source,
    occlusion_status, identity_confidence, online_track_id, and whether the
    selected track appears in that frame's overlap_contests.

Actor window -- which frames' identity matters for an event. The frozen
detector (football_analysis/event_detector.py finalize()) emits events only
from the TARGET's own possession segments: a control event (TOUCH /
RECEIVE / CARRY) spans a segment [seg_start, seg_end], and a release event
(PASS / CROSS / SHOT / DUEL / UNKNOWN_RELEASE) starts exactly at that
segment's seg_end. So:

  * control event          -> its own [start_frame, end_frame]
  * PASS/CROSS/SHOT/UNKNOWN_RELEASE
                           -> the parent possession segment (the control
                              event whose end_frame == release.start_frame);
                              who released the ball is decided by who had
                              it, not by where the ball travels afterwards
  * DUEL                   -> parent segment PLUS the release window itself
                              (the target must still be the tracked box
                              through the contest it is credited with)
  * GOAL                   -> same window as the SHOT it follows
  * no parent found        -> the single release frame (narrowest window)

Per-frame identity trust (all must hold; thresholds are re-used from
football_analysis/confidence.py, none are new):
  selected record present, state == ACTIVE, identity_state not in
  event_adapter.UNTRUSTED_IDENTITY_STATES, position_source ==
  CONFIRMED_DETECTION, occlusion_status == NONE, selected track not in an
  overlap contest, identity_confidence >= MEDIUM_THRESH (0.5).

actor_confidence = resolver_factor x trusted_fraction_of_window
  resolver_factor: RESOLVED 1.0, REVIEW_REQUIRED 0.5, UNRESOLVED 0.0
  further capped below MEDIUM if >1 distinct online_track_id occurs inside
  the window, and halved if the clip's logged identity conflicts involve the
  selected track.
Band: HIGH >= 0.75, MEDIUM >= 0.5, LOW below (confidence.py's own bands).

Attribution rule: actor = TARGET_PLAYER iff actor band == HIGH; otherwise
UNKNOWN. Event confidence / review_required are never read here.

What actor_confidence does NOT claim: it measures whether the tracked box
was continuously, uncontestedly the resolved seed track during the window.
It does not re-verify that the seed itself was the right person -- that is
the job of the benchmark provenance/seed verification (Task 3D).
"""
from typing import Dict, List, Optional

from football_analysis.confidence import HIGH_THRESH, MEDIUM_THRESH

UNTRUSTED_IDENTITY_STATES = ("REVIEW_REQUIRED_IDENTITY", "UNRESOLVED_IDENTITY")  # == event_adapter.py
CONTROL_TYPES = {"TOUCH", "RECEIVE", "CARRY"}
RELEASE_TYPES = {"PASS", "SHOT", "CROSS", "DUEL", "UNKNOWN_RELEASE"}
RESOLVER_FACTOR = {"RESOLVED": 1.0, "REVIEW_REQUIRED": 0.5, "UNRESOLVED": 0.0}


def band_for(score: Optional[float]) -> Optional[str]:
    if score is None:
        return None
    return "HIGH" if score >= HIGH_THRESH else ("MEDIUM" if score >= MEDIUM_THRESH else "LOW")


def _selected(rec: dict) -> Optional[dict]:
    for tr in rec.get("tracks", []):
        if tr.get("selected_player"):
            return tr
    return None


def frame_identity(rec: Optional[dict]) -> Dict:
    """Per-frame identity verdict + the evidence behind it."""
    if rec is None:
        return {"trusted": False, "reasons": ["no frame_log record"], "online_track_id": None}
    sel = _selected(rec)
    if sel is None:
        return {"trusted": False, "reasons": ["no selected-player record (LOST)"], "online_track_id": None}
    reasons = []
    if sel.get("state") != "ACTIVE":
        reasons.append(f"state={sel.get('state')}")
    if sel.get("identity_state") in UNTRUSTED_IDENTITY_STATES:
        reasons.append(f"identity_state={sel.get('identity_state')}")
    if sel.get("position_source") != "CONFIRMED_DETECTION":
        reasons.append(f"position_source={sel.get('position_source')}")
    if sel.get("occlusion_status", "NONE") != "NONE":
        reasons.append(f"occlusion_status={sel.get('occlusion_status')}")
    ids = {sel.get("online_track_id"), sel.get("track_id")}
    if any(ids & set(pair) for pair in rec.get("overlap_contests", []) or []):
        reasons.append("selected track in overlap contest")
    idc = sel.get("identity_confidence")
    if idc is None or idc < MEDIUM_THRESH:
        reasons.append(f"identity_confidence={idc}")
    return {
        "trusted": not reasons, "reasons": reasons,
        "online_track_id": sel.get("online_track_id"),
        "identity_confidence": idc, "reid_confidence": sel.get("reid_confidence"),
        "overall_track_confidence": sel.get("overall_track_confidence"),
    }


def actor_window(raw_event: dict, raw_events: List[dict]) -> List[int]:
    t = raw_event["type"]
    s, e = raw_event["start_frame"], raw_event["end_frame"]
    if t in CONTROL_TYPES:
        return list(range(s, e + 1))
    if t == "GOAL":
        shots = [x for x in raw_events if x["type"] == "SHOT" and x["end_frame"] <= s]
        if shots:
            return actor_window(max(shots, key=lambda x: x["end_frame"]), raw_events)
        return [s]
    parents = [x for x in raw_events if x["type"] in CONTROL_TYPES and x["end_frame"] == s]
    frames = list(range(parents[0]["start_frame"], s + 1)) if parents else [s]
    if t == "DUEL":
        frames = sorted(set(frames) | set(range(s, e + 1)))
    return frames


def assess_actor(raw_event: dict, raw_events: List[dict], frame_index: Dict[int, dict],
                 target_status: str, selected_ids_in_conflicts: bool = False) -> Dict:
    """Returns the actor assessment for one raw frozen-engine event."""
    window = actor_window(raw_event, raw_events)
    verdicts = {f: frame_identity(frame_index.get(f)) for f in window}
    trusted = [f for f, v in verdicts.items() if v["trusted"]]
    online_ids = sorted({v["online_track_id"] for v in verdicts.values() if v["online_track_id"]})
    trusted_fraction = len(trusted) / len(window) if window else 0.0

    score = RESOLVER_FACTOR.get(target_status, 0.0) * trusted_fraction
    notes = []
    if len(online_ids) > 1:
        score = min(score, MEDIUM_THRESH - 0.01)
        notes.append(f"multiple online track ids in window: {online_ids}")
    if selected_ids_in_conflicts:
        score *= 0.5
        notes.append("clip-level identity conflict involves the selected track")
    score = round(score, 3)
    band = band_for(score)

    idcs = [v["identity_confidence"] for v in verdicts.values() if v.get("identity_confidence") is not None]
    untrusted = {f: v["reasons"] for f, v in verdicts.items() if not v["trusted"]}
    return {
        "actor": "TARGET_PLAYER" if band == "HIGH" else "UNKNOWN",
        "actor_confidence": score,
        "actor_confidence_band": band,
        "actor_evidence": {
            "target_status": target_status,
            "actor_window_frames": [window[0], window[-1]] if window else None,
            "window_len": len(window),
            "trusted_frames": len(trusted),
            "online_track_ids": online_ids,
            "mean_identity_confidence": round(sum(idcs) / len(idcs), 3) if idcs else None,
            "untrusted_frames": {str(k): v for k, v in list(untrusted.items())[:10]},
            "notes": notes,
        },
    }


def build_frame_index(enriched_frame_log: List[dict]) -> Dict[int, dict]:
    return {rec["frame"]: rec for rec in enriched_frame_log}


def conflicts_involve(selected_ids, identity_conflict_counts: Optional[dict]) -> bool:
    for pair in (identity_conflict_counts or {}):
        if set(pair.split("|")) & set(selected_ids):
            return True
    return False


def selected_ids(enriched_frame_log: List[dict]) -> List[str]:
    ids = set()
    for rec in enriched_frame_log:
        sel = _selected(rec)
        if sel:
            ids.add(sel.get("online_track_id"))
            ids.add(sel.get("track_id"))
    return sorted(i for i in ids if i)
