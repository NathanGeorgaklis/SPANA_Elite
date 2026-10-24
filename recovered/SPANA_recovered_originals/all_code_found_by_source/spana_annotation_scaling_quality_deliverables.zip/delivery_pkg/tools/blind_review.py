#!/usr/bin/env python3
"""blind_review.py -- ANNOTATION-SCALING-QUALITY task items #24-27: a
genuinely independent BLINDED SECOND ANNOTATOR workflow.

Why this exists: this dataset's existing "double annotation" support
(status=DOUBLE_PASS_AGREED/DISAGREEMENT) has so far always been produced by
the SAME analyst doing a second look at their own first-pass clip -- which
is not blinding and does not count as a genuine second opinion (task's own
explicit framing). This tool makes that actually impossible to fudge: it
never shows Annotator B anything from Annotator A's record before B submits.

Workflow (deliberately three separate, simple steps -- no UI, no server):

  1. `prepare`  -- given a clip_id that ALREADY has a first-pass annotation
     record, copies clip.mp4 to a neutral staging path and writes a
     blind_task.json containing ONLY {clip_id, clip_path}. It does NOT read
     event_type, outcome, selected_player, notes, or any pipeline
     prediction from the existing annotation -- it does not even open that
     file. Nathan (or anyone) watches the clip and works out their own
     answer from scratch.

  2. `submit`   -- records Annotator B's own from-scratch judgement as a
     blind_review record (schema/blind_review.schema.json). Still never
     reads the Annotator-A record.

  3. `compare`  -- ONLY NOW, after B has submitted, loads Annotator A's
     locked record and B's blind review and diffs them field by field
     (event_type, outcome, team, identity confidence exactly; interaction
     timing within a tolerance, not exact equality -- task item #27). This
     is a report, not a schema mutation -- it never edits either record.

Nothing here trains anything, and nothing here is consulted before or
during Annotator B's pass -- the ordering is enforced by the CLI itself
(compare refuses to run if no blind_review record exists yet for the clip).
"""
import argparse
import json
import shutil
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
DATASET_ROOT = TOOLS_DIR.parent
CLIPS_DIR = DATASET_ROOT / "dataset" / "clips"
ANNOTATIONS_DIR = DATASET_ROOT / "dataset" / "annotations"
BLIND_TASKS_DIR = DATASET_ROOT / "dataset" / "blind_tasks"
BLIND_REVIEWS_DIR = DATASET_ROOT / "dataset" / "blind_reviews"

TOLERANCE_S = 2.0  # timing agreement tolerance (task item #27: "defensible tolerances, not exact")


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def find_locked_annotation(clip_id: str) -> dict | None:
    """Used ONLY by compare() -- never by prepare() or submit(). Returns the
    first matching annotation record for this clip_id, or None."""
    for p in sorted(ANNOTATIONS_DIR.glob("*.json")):
        try:
            d = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if d.get("clip_id") == clip_id:
            return d
    return None


def prepare_blind_task(clip_id: str, out_dir: Path | None = None) -> dict:
    """Step 1. Copies the clip to a neutral path and writes a blind_task.json
    holding NOTHING but {clip_id, clip_path, prepared_at}. Deliberately does
    not open dataset/annotations/*.json at all."""
    src_clip = CLIPS_DIR / clip_id / "clip.mp4"
    if not src_clip.exists():
        raise FileNotFoundError(f"no extracted clip for {clip_id}: {src_clip}")

    out_dir = Path(out_dir) if out_dir else (BLIND_TASKS_DIR / clip_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    dst_clip = out_dir / "clip_for_review.mp4"
    shutil.copyfile(src_clip, dst_clip)

    task = {"clip_id": clip_id, "clip_path": str(dst_clip), "prepared_at": _now_iso(),
            "instructions": "Watch the clip. Record your OWN judgement of event_type, outcome, "
                             "the acting player's team, and how confidently you can identify that "
                             "player -- before looking at anything else about this clip."}
    (out_dir / "blind_task.json").write_text(json.dumps(task, indent=2))
    return task


def submit_blind_review(clip_id: str, annotator_b_id: str, event_type: str, outcome: str,
                         selected_player_team: str, manual_identity_confidence: str,
                         approx_interaction_time_s: float | None = None, notes: str | None = None) -> dict:
    """Step 2. Writes Annotator B's from-scratch judgement. Never reads the
    Annotator-A record."""
    review_id = str(uuid.uuid4())
    record = {
        "schema_version": "1.0.0",
        "blind_review_id": review_id,
        "clip_id": clip_id,
        "locked_annotation_id": "UNKNOWN_AT_SUBMIT_TIME",  # filled in by compare(), never by submit()
        "annotator_b_id": annotator_b_id,
        "annotator_b_label": {
            "event_type": event_type, "outcome": outcome,
            "selected_player_team": selected_player_team,
            "manual_identity_confidence": manual_identity_confidence,
            "approx_interaction_time_s": approx_interaction_time_s,
            "notes": notes,
        },
        "submitted_at": _now_iso(),
    }
    BLIND_REVIEWS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = BLIND_REVIEWS_DIR / f"{clip_id}__{review_id}.json"
    out_path.write_text(json.dumps(record, indent=2))
    return record


def compare_annotations(clip_id: str) -> dict:
    """Step 3. ONLY callable after a blind_review record exists. Diffs
    Annotator A's locked record against Annotator B's blind submission."""
    reviews = sorted(BLIND_REVIEWS_DIR.glob(f"{clip_id}__*.json"))
    if not reviews:
        raise RuntimeError(f"no blind_review submitted yet for {clip_id} -- run `submit` first")
    review = json.loads(reviews[-1].read_text())

    locked = find_locked_annotation(clip_id)
    if locked is None:
        raise RuntimeError(f"no locked (Annotator-A) annotation found for {clip_id}")

    a_hl = locked["human_label"]
    b = review["annotator_b_label"]

    def agree(a_val, b_val):
        return a_val == b_val

    a_interaction = a_hl["ground_truth_event_window"].get("interaction_time_s")
    b_interaction = b.get("approx_interaction_time_s")
    timing_agree = None
    if a_interaction is not None and b_interaction is not None:
        timing_agree = abs(a_interaction - b_interaction) <= TOLERANCE_S

    comparison = {
        "clip_id": clip_id,
        "locked_annotation_id": locked["annotation_id"],
        "blind_review_id": review["blind_review_id"],
        "annotator_a_id": locked["provenance"]["annotator_id"],
        "annotator_b_id": review["annotator_b_id"],
        "fields": {
            "event_type": {"a": a_hl["event_type"], "b": b["event_type"], "agree": agree(a_hl["event_type"], b["event_type"])},
            "outcome": {"a": a_hl["outcome"], "b": b["outcome"], "agree": agree(a_hl["outcome"], b["outcome"])},
            "selected_player_team": {"a": a_hl["selected_player"]["team"], "b": b["selected_player_team"],
                                      "agree": agree(a_hl["selected_player"]["team"], b["selected_player_team"])},
            "manual_identity_confidence": {"a": a_hl["selected_player"]["manual_identity_confidence"],
                                            "b": b["manual_identity_confidence"],
                                            "agree": agree(a_hl["selected_player"]["manual_identity_confidence"], b["manual_identity_confidence"])},
            "interaction_timing": {"a": a_interaction, "b": b_interaction, "tolerance_s": TOLERANCE_S, "agree": timing_agree},
        },
        "compared_at": _now_iso(),
    }
    n_comparable = sum(1 for k, v in comparison["fields"].items() if v["agree"] is not None)
    n_agree = sum(1 for k, v in comparison["fields"].items() if v["agree"] is True)
    comparison["agreement_rate"] = (n_agree / n_comparable) if n_comparable else None
    return comparison


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("prepare", help="Step 1: stage a neutral clip for Annotator B")
    p1.add_argument("--clip-id", required=True)
    p1.add_argument("--out-dir", default=None)

    p2 = sub.add_parser("submit", help="Step 2: record Annotator B's from-scratch judgement")
    p2.add_argument("--clip-id", required=True)
    p2.add_argument("--annotator-b-id", required=True)
    p2.add_argument("--event-type", required=True)
    p2.add_argument("--outcome", required=True)
    p2.add_argument("--team", required=True)
    p2.add_argument("--identity-confidence", required=True)
    p2.add_argument("--interaction-time-s", type=float, default=None)
    p2.add_argument("--notes", default=None)

    p3 = sub.add_parser("compare", help="Step 3: diff Annotator A vs B (only after submit)")
    p3.add_argument("--clip-id", required=True)

    args = ap.parse_args()
    if args.cmd == "prepare":
        print(json.dumps(prepare_blind_task(args.clip_id, args.out_dir), indent=2))
    elif args.cmd == "submit":
        print(json.dumps(submit_blind_review(
            args.clip_id, args.annotator_b_id, args.event_type, args.outcome,
            args.team, args.identity_confidence, args.interaction_time_s, args.notes), indent=2))
    elif args.cmd == "compare":
        print(json.dumps(compare_annotations(args.clip_id), indent=2))


if __name__ == "__main__":
    main()
