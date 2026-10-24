"""
event_adapter.py -- Pipeline Integration deliverable 4/5 (event-engine
compatibility adapter + frame/timestamp sync check).

Extends the EXISTING, already-working new_tracker_snapshots.py bridge
(player_tracker_mo frame_log -> football_analysis.adapter.TrackingSnapshot)
to additionally honour OFFLINE identity trust, without changing:
  - football_analysis/*.py (the frozen event engine) at all,
  - the TrackingSnapshot schema (still exactly player_id/frame_number/
    timestamp/bbox/centre_x/centre_y/tracking_confidence/tracking_state),
  - WHICH per-frame box a frame uses. This is a deliberate, structural
    constraint inherited from selected_player_resolver.py itself: per its
    own Phase 3 safety rule, an offline override of the online tracker's
    own choice can NEVER reach AUTO_RESOLVED_IDENTITY (it is always forced
    to REVIEW_REQUIRED_IDENTITY -- see selected_player_resolver.py's tail).
    So the resolver never actually substitutes a different online_track_id's
    position into the trajectory; it only ever tags the online tracker's
    OWN chosen box as trustworthy or not. This adapter mirrors that
    constraint exactly: it changes the frame's TRUST label, never its
    position, team, or identity.

What's new relative to new_tracker_snapshots.py: the legacy_state mapping
is additionally gated on offline_resolved_selected_track / identity_state
(from offline_identity_enrichment.enrich_frame_log_with_offline_identity's
output). A span the online tracker reported ACTIVE but the offline resolver
marked REVIEW_REQUIRED_IDENTITY or UNRESOLVED_IDENTITY is downgraded to
"OCCLUDED" for event-engine purposes -- i.e. "a candidate position exists
but identity is contested," using new_tracker_snapshots.py's own existing
vocabulary and its own stated discipline: this can only ever make the
engine trust a frame LESS than the live state claimed, never more. A span
with identity_state AUTO_RESOLVED_IDENTITY (or None -- no boundary ever
touched this run, i.e. a single continuous online_track_id for the whole
clip) passes through NEW_TO_LEGACY_STATE unchanged, byte-identical to
new_tracker_snapshots.py's existing behaviour.

Frame/timestamp sync (deliverable 5): this adapter assigns timestamp =
frame_number / fps using the SAME fps the online pass itself recorded
(summary["fps"], read from the clip's own video capture) and the SAME
frame-numbering convention frame_log already uses (1-indexed step() calls;
frame 0 is the synthetic bootstrap record, exactly as new_tracker_snapshots.
py already documents) -- no independent re-derivation of fps or frame
numbering is introduced here, which is what "validate synchronisation
without redesigning ball tracking" means in practice: reuse the one
fps/frame-index source of truth that already exists, never compute a
second one that could silently drift from it.
"""
import sys
from pathlib import Path
from typing import List, Optional

from football_analysis.adapter import TrackingAdapter, TrackingSnapshot

# new_tracker_snapshots.py lives at the repo root (a pre-existing script, not part of
# this package) -- same sys.path convention new_clip_test_runner.py already uses to
# reach repo-root-level modules from inside player_tracker_mo/. Reusing its
# NEW_TO_LEGACY_STATE table directly (rather than copying the dict a second time)
# keeps the online-state-name vocabulary mapping in exactly one place.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from new_tracker_snapshots import NEW_TO_LEGACY_STATE  # noqa: E402

# Offline identity states that must never be reported to the event engine as
# a trusted, confirmed position -- mirrors offline_identity_enrichment.py's
# own IdentityState vocabulary (imported lazily inside the function below to
# avoid a hard import-order dependency for callers who only ever use the
# LEGACY, non-enriched path through this same module).
UNTRUSTED_IDENTITY_STATES = ("REVIEW_REQUIRED_IDENTITY", "UNRESOLVED_IDENTITY")


def build_integrated_event_snapshots(enriched_frame_log: List[dict], fps: float,
                                      player_id: str = "P1") -> List[TrackingSnapshot]:
    """enriched_frame_log: exactly offline_identity_enrichment.
    enrich_frame_log_with_offline_identity()'s return value (every track
    record additionally carries identity_state / offline_resolved_selected_
    track). Returns a TrackingSnapshot list, same shape/length/frame-0
    bootstrap convention as new_tracker_snapshots.load_new_tracker_snapshots,
    so it is a drop-in replacement wherever that function's `snapshots` were
    consumed -- the only change is the additional offline-trust gate.

    Never raises on an empty frame_log (the empty-clip / zero-tracklets
    failure-injection case): returns a one-element list containing only the
    synthetic frame-0 bootstrap record in that case, exactly as
    new_tracker_snapshots.py's own frame-0 convention already does before
    its loop runs."""

    def sel_record(rec):
        for tr in rec["tracks"]:
            if tr.get("selected_player"):
                return tr
        return None

    first_sel = None
    for rec in enriched_frame_log:
        s = sel_record(rec)
        if s is not None:
            first_sel = s
            break

    if first_sel is None:
        # No selected-player record anywhere -- the online pass never had a
        # live selected track this clip (e.g. init box never matched a
        # detection). Fail gracefully: a single LOST bootstrap snapshot,
        # never a crash and never a fabricated TRACKING claim.
        return [TrackingSnapshot(
            player_id=player_id, frame_number=0, timestamp=0.0,
            bbox=[0.0, 0.0, 0.0, 0.0], centre_x=0.0, centre_y=0.0,
            tracking_confidence=0.0, tracking_state="LOST",
        )]

    init_box = first_sel["bbox"]
    snapshots = [TrackingSnapshot(
        player_id=player_id, frame_number=0, timestamp=0.0,
        bbox=list(init_box), centre_x=round((init_box[0] + init_box[2]) / 2.0, 2),
        centre_y=round((init_box[1] + init_box[3]) / 2.0, 2),
        tracking_confidence=1.0, tracking_state="TRACKING",
    )]

    missing_frames = []
    downgraded_frames = []
    for rec in enriched_frame_log:
        frame_number = rec["frame"]
        sel = sel_record(rec)
        if sel is None:
            missing_frames.append(frame_number)
            last = snapshots[-1]
            snapshots.append(TrackingSnapshot(
                player_id=player_id, frame_number=frame_number,
                timestamp=round(frame_number / fps, 4), bbox=list(last.bbox),
                centre_x=last.centre_x, centre_y=last.centre_y,
                tracking_confidence=0.0, tracking_state="LOST",
            ))
            continue

        legacy_state = NEW_TO_LEGACY_STATE.get(sel["state"], "LOST")
        identity_state = sel.get("identity_state")
        if legacy_state == "TRACKING" and identity_state in UNTRUSTED_IDENTITY_STATES:
            legacy_state = "OCCLUDED"
            downgraded_frames.append(frame_number)

        x1, y1, x2, y2 = sel["bbox"]
        snapshots.append(TrackingSnapshot(
            player_id=player_id, frame_number=frame_number,
            timestamp=round(frame_number / fps, 4), bbox=[x1, y1, x2, y2],
            centre_x=round((x1 + x2) / 2.0, 2), centre_y=round((y1 + y2) / 2.0, 2),
            tracking_confidence=sel.get("overall_track_confidence", 0.0),
            tracking_state=legacy_state,
        ))

    return snapshots


def snapshot_trust_summary(snapshots: List[TrackingSnapshot]) -> dict:
    """Diagnostic-only (feeds tracking_health.py): never consumed by the
    event engine itself."""
    total = len(snapshots) or 1
    counts = {}
    for s in snapshots:
        counts[s.tracking_state] = counts.get(s.tracking_state, 0) + 1
    return {k: round(100 * v / total, 1) for k, v in counts.items()}
