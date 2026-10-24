"""
touch_episode.py -- MILESTONE 2 TASK 1B: contested touch episode clustering.

A read-only INTERPRETATION layer on top of the frozen touch_detector.py.
Imports its pure functions unchanged -- nothing in touch_detector.py (or
anything it depends on: player tracking, BallTracker, observed_position_box
contract, interaction radius 0.45, feet geometry, player position quality,
ball quality, touch score weights/thresholds, release gate, identity logic)
is modified here. The only new logic is temporal/physical EPISODE
interpretation of the already-frozen raw per-frame candidate scores and
raw touch events.

See MILESTONE_2_TASK1B_TOUCH_EPISODE_SPEC.md for the full design rationale.
"""
import json
import sys
from pathlib import Path

MILESTONE2_DIR = Path(__file__).resolve().parent
MILESTONE1_DIR = MILESTONE2_DIR.parent / "milestone1"
sys.path.insert(0, str(MILESTONE2_DIR))
sys.path.insert(0, str(MILESTONE1_DIR))

from touch_detector import (  # noqa: E402  (frozen Task 1 module, reused not modified)
    generate_frame_candidates, score_candidates, classify_frame, cluster_events,
    TEMPORAL_WINDOW_S, MIN_SUSTAINED_FRAMES, COMPETITION_MARGIN,
)
from attribute_actors import run_ball_pass  # noqa: E402  (frozen, reused)

# ---------------------------------------------------------------------------
# Episode-level constants. All of these REUSE Task 1's already-justified,
# fps-normalized constants rather than introducing new tuned numbers
# (Section 11/30: do not hard-code a new window, do not tune solely to one
# segment). They are simply applied at the episode-interpretation layer.
#
#   EPISODE_BRIDGE_S          -- how close in time two raw touch events must
#                                be to even be considered part of the same
#                                physical interaction. Reuses Section 9's own
#                                TEMPORAL_WINDOW_S (the same horizon Task 1
#                                already uses for "is this a sustained
#                                approach, not just one frame").
#   dominance margin           -- reuses COMPETITION_MARGIN (Section 13),
#                                 the same margin Task 1 already uses to
#                                 decide two frame-level candidates are
#                                 "too close to call".
#   sustained-frames count      -- reuses MIN_SUSTAINED_FRAMES (Section 9).
EPISODE_BRIDGE_S = TEMPORAL_WINDOW_S


def _bridge_frames(fps):
    return max(1, round(EPISODE_BRIDGE_S * fps))


# ---------------------------------------------------------------------------
# Section 12/13: candidate-set overlap and leader-change evidence between
# two temporally-adjacent raw touch events.

def _event_track_set(event, scored_candidates=None):
    """The *reported* candidate set for this raw touch event -- its own
    leader (single-leader events) or its own multi-candidate list
    (AMBIGUOUS events), exactly as Task 1 already surfaced it. This, not the
    full per-frame physical-eligibility set (which includes many weak,
    merely-nearby bystanders within the 0.45 radius), is what Section 12's
    'compare candidate sets' example means by a cluster's candidates --
    using the raw-eligibility set instead over-merges genuinely unrelated
    touches that happen to share a weakly-eligible nearby player."""
    return {c["track_id"] for c in event["candidate_tracks"]}


def _gap(event_a, event_b):
    return event_b["start_frame"] - event_a["end_frame"] - 1


# ---------------------------------------------------------------------------
# Section 14/15: per-track dominance within a contested run, and the rolling
# frame-level scan that decides whether/where a contested run resolves into
# one clearly dominant actor (Section 19 -- determined from evidence, not
# assumed; this is the one mechanism that produces different answers for
# SEG_A (never resolves) and SEG_F (resolves partway through) from the same
# rule, rather than hand-picking per segment).

def _resolve_contested_run(run_events, scored_candidates, fps):
    """run_events: raw touch events (Task 1 output, unmodified) linked end to
    end by CONTESTED links. Returns a list of sub-episode dicts (1 or 2+),
    each with keys: state, start_frame, end_frame, source_touch_ids,
    primary_actor, leader_change_count, candidate_set.

    Section 15/19: find the point, if any, from which ONE track holds sole
    frame-level leadership (using Task 1's own already-computed
    competition_penalty -- a frame is a 'clean' single-leader frame when
    Task 1's own frame classifier found no close rival that frame) all the
    way through to the END of the contested run, tolerating brief ambiguous
    dips in between but never a genuinely different track's clean
    leadership. That is scanned for by walking BACKWARD from the run's last
    frame: the terminal run of frames belonging to one track is the
    resolved SINGLE_ACTOR_TOUCH phase; everything before it (if the
    terminal track never achieves at least MIN_SUSTAINED_FRAMES clean
    leader frames, or a different track still holds clean leadership
    earlier) stays AMBIGUOUS_CONTESTED_TOUCH. This is what gives SEG_A
    (leadership never settles -- a different clean leader always appears
    further back) and SEG_F (leadership settles on one track for the rest
    of the run) two different, evidence-derived answers from one rule."""
    lo, hi = run_events[0]["start_frame"], run_events[-1]["end_frame"]
    frames = sorted(f for f in scored_candidates if lo <= f <= hi and scored_candidates.get(f))
    clean_leader = {}
    for f in frames:
        top = scored_candidates[f][0]
        clean_leader[f] = None if top["competition_penalty"] > 0 else top["track_id"]

    # Repeatedly peel a terminal same-leader suffix off the remaining frame
    # range (Section 19: a contest can resolve, and later become a fresh
    # contest again -- e.g. SEG_F's T015 touch followed by T021 regaining
    # and holding the ball -- so this is not a single two-way split; it is
    # scanned backward once per remaining prefix until nothing more peels).
    phases = []  # list of (state, start_frame_idx, end_frame_idx, leader)
    remaining = frames
    while remaining:
        terminal_track, clean_count, suffix_start_i = None, 0, None
        broke = False
        for i in range(len(remaining) - 1, -1, -1):
            f = remaining[i]
            leader = clean_leader[f]
            if leader is None:
                if terminal_track is not None:
                    suffix_start_i = i
                continue
            if terminal_track is None:
                terminal_track, clean_count, suffix_start_i = leader, 1, i
            elif leader == terminal_track:
                clean_count += 1
                suffix_start_i = i
            else:
                broke = True
                break
        if terminal_track is not None and clean_count >= MIN_SUSTAINED_FRAMES:
            seg = remaining[suffix_start_i:]
            phases.append(("SINGLE_ACTOR_TOUCH", seg[0], seg[-1], terminal_track))
            remaining = remaining[:suffix_start_i]
        else:
            # Nothing in the remaining prefix resolves -- it is all one
            # honest ambiguous phase (Section 18: SEG_A's expected shape
            # when this is the only phase found at all).
            phases.append(("AMBIGUOUS_CONTESTED_TOUCH", remaining[0], remaining[-1], None))
            remaining = []
    phases.reverse()

    def touch_ids_in(start_f, end_f):
        return [e["touch_id"] for e in run_events if start_f <= e["touch_frame"] <= end_f]

    out = []
    assigned = set()
    for state, seg_start, seg_end, leader in phases:
        ids = [t for t in touch_ids_in(seg_start, seg_end) if t not in assigned]
        assigned.update(ids)
        phase_events = [e for e in run_events if e["touch_id"] in ids]
        phase_frames = [f for f in frames if seg_start <= f <= seg_end]
        leaders_seq = [clean_leader[f] for f in phase_frames if clean_leader[f] is not None]
        changes = sum(1 for i in range(1, len(leaders_seq)) if leaders_seq[i] != leaders_seq[i - 1])
        cand_set = sorted(set().union(*[_event_track_set(e) for e in phase_events])) if phase_events else []
        out.append({
            "state": state,
            "start_frame": phase_events[0]["start_frame"] if phase_events else seg_start,
            "end_frame": phase_events[-1]["end_frame"] if phase_events else seg_end,
            "source_touch_ids": ids, "primary_actor": leader if state == "SINGLE_ACTOR_TOUCH" else None,
            "candidate_set": cand_set, "leader_change_count": 0 if state == "SINGLE_ACTOR_TOUCH" else changes,
        })
    # any raw event whose touch_frame fell in a gap between phases (rare) --
    # attach it to the nearest phase by frame distance rather than drop it
    leftover = [e for e in run_events if e["touch_id"] not in assigned]
    for e in leftover:
        nearest = min(range(len(out)), key=lambda i: min(abs(e["touch_frame"] - out[i]["start_frame"]),
                                                            abs(e["touch_frame"] - out[i]["end_frame"])))
        out[nearest]["source_touch_ids"].append(e["touch_id"])
        out[nearest]["start_frame"] = min(out[nearest]["start_frame"], e["start_frame"])
        out[nearest]["end_frame"] = max(out[nearest]["end_frame"], e["end_frame"])
    return [o for o in out if o["source_touch_ids"]]


# ---------------------------------------------------------------------------
# Standalone (unbridged) raw events, and the UNRESOLVED_TOUCH downgrade for
# isolated weak single-frame evidence (Section 6).

def _classify_standalone(event):
    if event["touch_state"] == "AMBIGUOUS_TOUCH":
        return {"state": "AMBIGUOUS_CONTESTED_TOUCH", "start_frame": event["start_frame"],
                "end_frame": event["end_frame"], "source_touch_ids": [event["touch_id"]],
                "primary_actor": None,
                "candidate_set": sorted({c["track_id"] for c in event["candidate_tracks"]}),
                "leader_change_count": 0}
    tp = (event.get("provenance") or {}).get("component_scores", {}).get("temporal_proximity")
    isolated_weak = (tp == 0.3 and event["touch_state"] != "CONFIRMED_TOUCH")
    state = "UNRESOLVED_TOUCH" if isolated_weak else "SINGLE_ACTOR_TOUCH"
    return {"state": state, "start_frame": event["start_frame"], "end_frame": event["end_frame"],
            "source_touch_ids": [event["touch_id"]], "primary_actor": event["actor_track_id"],
            "candidate_set": [event["actor_track_id"]] if event["actor_track_id"] else [],
            "leader_change_count": 0}


# ---------------------------------------------------------------------------
# Top-level episode builder.

def build_episodes(raw_events, scored_candidates, fps, play_id, source_note):
    if not raw_events:
        return []
    events = sorted(raw_events, key=lambda e: e["start_frame"])

    # 1. link every temporally-adjacent pair against a RUNNING contest set
    #    (Section 12/13): once a run of events shares a candidate, later
    #    events joining that same run are tested against the whole running
    #    set, not just the immediately previous event -- this is what lets
    #    SEG_F's T007-alone / T021-alone / T015-alone single-leader frames
    #    still be recognized as one ongoing contest (each individually
    #    shares a track with the broader scrum that opened it) without
    #    accidentally gluing two genuinely unrelated touches together just
    #    because a bystander was weakly eligible in both (bounded by the
    #    temporal gap test, which still applies pairwise).
    links = []
    running_set = _event_track_set(events[0])
    for a, b in zip(events, events[1:]):
        if _gap(a, b) > _bridge_frames(fps):
            links.append(None)
            running_set = _event_track_set(b)
            continue
        b_set = _event_track_set(b)
        if running_set & b_set:
            links.append("CONTESTED")
            running_set = running_set | b_set
        else:
            links.append("SEQUENTIAL")
            running_set = b_set

    # 2. cut into super-chains at hard breaks (link is None)
    super_chains = []
    cur = [events[0]]
    cur_links = []
    for ev, link in zip(events[1:], links):
        if link is None:
            super_chains.append((cur, cur_links))
            cur = [ev]
            cur_links = []
        else:
            cur.append(ev)
            cur_links.append(link)
    super_chains.append((cur, cur_links))

    prelim = []  # list of (episode_dict_without_id, link_before_as_sequential: bool)
    for chain_events, chain_links in super_chains:
        if len(chain_events) == 1:
            prelim.append((_classify_standalone(chain_events[0]), False))
            continue
        # split chain into maximal CONTESTED-only runs at SEQUENTIAL links
        run, run_was_sequential_start = [chain_events[0]], False
        pending_sequential_before_next = False
        for ev, link in zip(chain_events[1:], chain_links):
            if link == "CONTESTED":
                run.append(ev)
            else:  # SEQUENTIAL -- hard sub-boundary, flush current run
                sub_episodes = _resolve_contested_run(run, scored_candidates, fps) if len(run) > 1 \
                    else [_classify_standalone(run[0])]
                for i, se in enumerate(sub_episodes):
                    prelim.append((se, pending_sequential_before_next and i == 0))
                pending_sequential_before_next = True
                run = [ev]
        sub_episodes = _resolve_contested_run(run, scored_candidates, fps) if len(run) > 1 \
            else [_classify_standalone(run[0])]
        for i, se in enumerate(sub_episodes):
            prelim.append((se, pending_sequential_before_next and i == 0))

    # 3. merge adjacent SEQUENTIAL-linked pairs into SEQUENTIAL_TOUCHES only
    #    when BOTH sides independently resolved to a clean single actor
    #    (Section 9: preserve genuine separate touches; do not force a label
    #    implying more certainty than the evidence supports).
    merged = []
    i = 0
    eps_only = [p[0] for p in prelim]
    seq_flags = [p[1] for p in prelim]
    while i < len(eps_only):
        if (seq_flags[i] and merged and merged[-1]["state"] == "SINGLE_ACTOR_TOUCH"
                and eps_only[i]["state"] == "SINGLE_ACTOR_TOUCH"):
            prev = merged.pop()
            cur_ep = eps_only[i]
            merged.append({
                "state": "SEQUENTIAL_TOUCHES", "start_frame": prev["start_frame"],
                "end_frame": cur_ep["end_frame"],
                "source_touch_ids": prev["source_touch_ids"] + cur_ep["source_touch_ids"],
                "primary_actor": None,
                "candidate_set": sorted(set(prev["candidate_set"]) | set(cur_ep["candidate_set"])),
                "leader_change_count": prev["leader_change_count"] + cur_ep["leader_change_count"],
                "sequence": [prev.get("primary_actor"), cur_ep.get("primary_actor")],
            })
        else:
            merged.append(eps_only[i])
        i += 1

    # 4. finalize schema (Section 25)
    out = []
    for idx, ep in enumerate(merged, 1):
        all_rows = []
        for f in range(ep["start_frame"], ep["end_frame"] + 1):
            all_rows.extend(scored_candidates.get(f, []))
        bq_vals = [r["ball_quality"] for r in all_rows] or [0.0]
        pq_vals = [r["player_quality"] for r in all_rows] or [0.0]
        dom = {}
        for r in all_rows:
            d = dom.setdefault(r["track_id"], {"frames_as_candidate": 0, "frames_as_leader": 0,
                                                "sum_score": 0.0, "best_score": 0.0})
            d["frames_as_candidate"] += 1
            d["sum_score"] += r["touch_confidence"]
            d["best_score"] = max(d["best_score"], r["touch_confidence"])
        for f in range(ep["start_frame"], ep["end_frame"] + 1):
            rows = scored_candidates.get(f, [])
            if rows:
                dom.setdefault(rows[0]["track_id"], {"frames_as_candidate": 0, "frames_as_leader": 0,
                                                      "sum_score": 0.0, "best_score": 0.0})
                dom[rows[0]["track_id"]]["frames_as_leader"] += 1
        for t, d in dom.items():
            d["mean_score"] = round(d["sum_score"] / max(1, d["frames_as_candidate"]), 3)
            d["best_score"] = round(d["best_score"], 3)
            del d["sum_score"]
        out.append({
            "episode_id": f"EPISODE_{idx:03d}", "play_id": play_id,
            "start_frame": ep["start_frame"], "end_frame": ep["end_frame"],
            "start_timestamp": round(ep["start_frame"] / fps, 3), "end_timestamp": round(ep["end_frame"] / fps, 3),
            "episode_state": ep["state"], "primary_actor_track_id": ep.get("primary_actor"),
            "candidate_tracks": ep["candidate_set"],
            "actor_dominance_scores": dom,
            "leader_change_count": ep["leader_change_count"],
            "candidate_set_overlap": round(
                (len(set(ep["candidate_set"])) / max(1, len({r["track_id"] for r in all_rows}))), 3
            ) if all_rows else None,
            "ball_quality_summary": {"mean": round(sum(bq_vals) / len(bq_vals), 3), "min": round(min(bq_vals), 3)},
            "position_quality_summary": {"mean": round(sum(pq_vals) / len(pq_vals), 3), "min": round(min(pq_vals), 3)},
            "review_required": ep["state"] != "SINGLE_ACTOR_TOUCH",
            "source_touch_ids": ep["source_touch_ids"],
            "provenance": {"fps": fps, "source": source_note,
                            "sequence": ep.get("sequence")} ,
        })
    return out


# ---------------------------------------------------------------------------
# Convenience top-level entry point mirroring touch_detector.detect_touches,
# but returning BOTH the unmodified raw touch events and the new episode
# interpretation (Section 26/27: raw events are never deleted).

def detect_touches_and_episodes(clip_path, tracking_result_dir, play_id, max_frames=None, ball_history=None):
    with open(f"{tracking_result_dir}/clip_mo_tracking_results.json") as f:
        tracking = json.load(f)
    frame_log_by_frame = {rec["frame"]: rec for rec in tracking["frame_log"]}
    fps = tracking.get("fps", 29.97)
    n_frames = max(frame_log_by_frame.keys()) + 1 if frame_log_by_frame else 0
    if max_frames:
        n_frames = min(n_frames, max_frames)

    if ball_history is None:
        tracker, ball_states = run_ball_pass(clip_path, n_frames, fps)
        ball_by_frame = {}
        for f, bs in ball_states.items():
            ball_by_frame[f] = {"centre_x": bs.centre_x, "centre_y": bs.centre_y, "state": bs.state,
                                 "confidence": bs.confidence, "velocity_x": bs.velocity_x, "velocity_y": bs.velocity_y}
        source_note = f"{clip_path} + {tracking_result_dir}/clip_mo_tracking_results.json (fresh ball pass)"
    else:
        ball_by_frame = ball_history
        source_note = f"{tracking_result_dir}/clip_mo_tracking_results.json (ball history supplied)"

    frame_candidates = generate_frame_candidates(frame_log_by_frame, ball_by_frame, fps)
    scored = score_candidates(frame_candidates, ball_by_frame, fps)
    raw_events = cluster_events(scored, fps, play_id, source_note)
    episodes = build_episodes(raw_events, scored, fps, play_id, source_note)
    meta = {"fps": fps, "n_frames": n_frames, "n_candidate_frames": len(frame_candidates)}
    return raw_events, episodes, meta


if __name__ == "__main__":
    clip_path, result_dir, play_id, out_path = sys.argv[1:5]
    raw_events, episodes, meta = detect_touches_and_episodes(clip_path, result_dir, play_id)
    with open(out_path, "w") as f:
        json.dump({"play_id": play_id, "meta": meta, "raw_touch_events": raw_events,
                   "touch_episodes": episodes}, f, indent=2, default=str)
    print(json.dumps({"play_id": play_id, "meta": meta, "n_raw_events": len(raw_events),
                       "n_episodes": len(episodes),
                       "episodes_summary": [{"episode_id": e["episode_id"], "start_frame": e["start_frame"],
                                              "end_frame": e["end_frame"], "episode_state": e["episode_state"],
                                              "primary_actor_track_id": e["primary_actor_track_id"],
                                              "candidate_tracks": e["candidate_tracks"]} for e in episodes]},
                      indent=2, default=str))
