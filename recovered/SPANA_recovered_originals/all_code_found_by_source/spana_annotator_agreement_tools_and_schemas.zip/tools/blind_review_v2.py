#!/usr/bin/env python3
"""blind_review_v2.py -- ANNOTATOR-AGREEMENT task (gap #1): a genuinely
blinded second-annotator workflow with NEUTRAL clip identifiers.

Why v2 exists: v1 (tools/blind_review.py) was self-tested and worked
mechanically, but it leaked the real clip_id to Annotator B via
blind_task.json and the staged file path -- e.g. "src2_clip_011". The new
task's item #3 explicitly requires neutral identifiers ("blind_001", not
"pass_completed_01") specifically so Annotator B cannot infer anything from
the filename. v1's self-test artifacts were already deleted and it was never
used for real data, so this is a genuine infrastructure correction, not a
new system -- same three-step shape (prepare / submit / compare), same
never-look-before-you-submit discipline, same "never mutate the original
annotation" guarantee. It adds one step (adjudicate) that the new task's
item #12 requires, and a batch-prepare / CSV-ingest path so a human
(Nathan, acting as Annotator B) doesn't need to run the CLI once per clip.

Directory layout:
  dataset/blind_mapping/mapping.json   -- HIDDEN. blind_id -> {clip_id,
      annotation_id, team_hint}. Never shipped to Annotator B. Only
      compare_annotations() reads it, and only after B has already
      submitted (enforced the same way v1 enforced ordering: compare()
      raises if no submission exists yet).
  dataset/blind_package/blind_0NN/clip.mp4  -- what ships to Annotator B.
      Contains ONLY the clip video. No clip_id, no event/outcome/notes,
      no pipeline prediction anywhere in this tree.
  dataset/blind_package/INSTRUCTIONS.md, annotator_b_answer_sheet.csv --
      shared instructions + a single fillable answer sheet covering all
      12 clips (so Annotator B fills in one CSV rather than running a CLI
      12 times).
  dataset/blind_reviews_v2/blind_0NN__<review_id>.json -- Annotator B's
      submitted judgement, keyed by blind_id only.
  dataset/blind_comparisons/blind_0NN__comparison.json -- compare() output.
  dataset/blind_adjudications/blind_0NN__adjudication.json -- adjudicate()
      output. Original Annotator-A and Annotator-B records are NEVER
      edited by any step in this file.
"""
import argparse
import csv
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
MAPPING_PATH = DATASET_ROOT / "dataset" / "blind_mapping" / "mapping.json"
PACKAGE_DIR = DATASET_ROOT / "dataset" / "blind_package"
REVIEWS_DIR = DATASET_ROOT / "dataset" / "blind_reviews_v2"
COMPARISONS_DIR = DATASET_ROOT / "dataset" / "blind_comparisons"
ADJUDICATIONS_DIR = DATASET_ROOT / "dataset" / "blind_adjudications"

TIMING_TOLERANCE_S = 2.0  # ~60 frames at 30fps -- see IDENTITY_FEASIBILITY_GATE.md /
                          # ANNOTATOR_AGREEMENT_REPORT.md for the justification; NOT
                          # chosen to maximize agreement (task item #10).

DISAGREEMENT_TAXONOMY = [
    "EVENT_TYPE_DISAGREEMENT", "OUTCOME_DISAGREEMENT", "SUBTYPE_DISAGREEMENT",
    "PLAYER_IDENTITY_DISAGREEMENT", "TIMING_DISAGREEMENT", "VISUAL_AMBIGUITY",
    "ANNOTATION_MISTAKE", "TAXONOMY_AMBIGUITY", "OTHER",
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_mapping() -> dict:
    if not MAPPING_PATH.exists():
        return {}
    return json.loads(MAPPING_PATH.read_text())


def _save_mapping(m: dict) -> None:
    MAPPING_PATH.parent.mkdir(parents=True, exist_ok=True)
    MAPPING_PATH.write_text(json.dumps(m, indent=2, sort_keys=True))


def _find_annotation(clip_id: str) -> dict | None:
    for p in sorted(ANNOTATIONS_DIR.glob("*.json")):
        try:
            d = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if d.get("clip_id") == clip_id:
            return d
    return None


def register_blind_clip(blind_id: str, clip_id: str, team_hint: str) -> dict:
    """Adds one entry to the HIDDEN mapping and stages a neutral clip.mp4
    under dataset/blind_package/<blind_id>/clip.mp4. Does not read
    event_type/outcome/notes -- only clip_id (to locate the video file) and
    the pre-written team_hint string (kit colour only, written by the human
    preparing the batch, never derived from anything B shouldn't see)."""
    src_clip = CLIPS_DIR / clip_id / "clip.mp4"
    if not src_clip.exists():
        raise FileNotFoundError(f"no extracted clip for {clip_id}: {src_clip}")
    ann = _find_annotation(clip_id)
    if ann is None:
        raise RuntimeError(f"no Annotator-A record found for {clip_id} -- cannot register")

    out_dir = PACKAGE_DIR / blind_id
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src_clip, out_dir / "clip.mp4")
    (out_dir / "player_hint.txt").write_text(
        f"Selected player to watch: {team_hint}\n\n"
        "If several players are wearing this colour, focus on whichever one "
        "is most clearly the central, ball-involved player in this clip.\n"
    )

    mapping = _load_mapping()
    mapping[blind_id] = {
        "clip_id": clip_id,
        "annotation_id": ann["annotation_id"],
        "team_hint": team_hint,
        "registered_at": _now_iso(),
    }
    _save_mapping(mapping)
    return mapping[blind_id]


def submit_blind_review(blind_id: str, annotator_b_id: str, selected_player_confirmed: str,
                         event_type: str, annotation_confidence: str, event_subtype: str | None = None,
                         outcome: str | None = None, event_start_s: float | None = None,
                         interaction_release_s: float | None = None, event_end_s: float | None = None,
                         notes: str | None = None) -> dict:
    """Records Annotator B's own from-scratch judgement, keyed by blind_id
    only. Never opens dataset/blind_mapping/mapping.json or any
    dataset/annotations/*.json file -- cannot see Annotator A's answer even
    if it wanted to."""
    review_id = str(uuid.uuid4())
    record = {
        "schema_version": "2.0.0",
        "blind_review_id": review_id,
        "blind_id": blind_id,
        "annotator_b_id": annotator_b_id,
        "annotator_b_label": {
            "selected_player_confirmed": selected_player_confirmed,
            "event_type": event_type,
            "event_subtype": event_subtype,
            "outcome": outcome,
            "event_start_s": event_start_s,
            "interaction_release_s": interaction_release_s,
            "event_end_s": event_end_s,
            "annotation_confidence": annotation_confidence,
            "notes": notes,
        },
        "submitted_at": _now_iso(),
    }
    REVIEWS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REVIEWS_DIR / f"{blind_id}__{review_id}.json"
    out_path.write_text(json.dumps(record, indent=2))
    return record


def ingest_csv(csv_path: str, annotator_b_id: str = "nathan") -> list[dict]:
    """Batch version of submit_blind_review() for a human filling in one
    CSV covering all 12 clips instead of running the CLI 12 times. Expected
    columns (see dataset/blind_package/annotator_b_answer_sheet.csv):
    blind_id, selected_player_confirmed, event_type, event_subtype, outcome,
    event_start_s, interaction_release_s, event_end_s, annotation_confidence,
    notes. Blank cells become null. Still never reads clip_id or any
    Annotator-A record."""
    results = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            bid = (row.get("blind_id") or "").strip()
            if not bid:
                continue

            def g(k):
                v = (row.get(k) or "").strip()
                return v if v else None

            def gf(k):
                v = g(k)
                return float(v) if v is not None else None

            results.append(submit_blind_review(
                blind_id=bid,
                annotator_b_id=annotator_b_id,
                selected_player_confirmed=g("selected_player_confirmed") or "UNSURE",
                event_type=g("event_type") or "UNKNOWN",
                annotation_confidence=g("annotation_confidence") or "LOW",
                event_subtype=g("event_subtype"),
                outcome=g("outcome"),
                event_start_s=gf("event_start_s"),
                interaction_release_s=gf("interaction_release_s"),
                event_end_s=gf("event_end_s"),
                notes=g("notes"),
            ))
    return results


def _classify_disagreement(field: str, a_val, b_val, extra: dict | None = None) -> str:
    """Fixed taxonomy classification (task item #11). Deliberately simple
    and auditable rather than clever -- a human can re-derive every one of
    these labels by eye from the same two values."""
    if field == "event_type":
        return "EVENT_TYPE_DISAGREEMENT"
    if field == "outcome":
        return "OUTCOME_DISAGREEMENT"
    if field == "event_subtype":
        return "SUBTYPE_DISAGREEMENT"
    if field == "selected_player_confirmed":
        return "PLAYER_IDENTITY_DISAGREEMENT"
    if field in ("event_start_s", "interaction_release_s", "event_end_s"):
        return "TIMING_DISAGREEMENT"
    return "OTHER"


def compare_annotations(blind_id: str) -> dict:
    """Only callable after a v2 blind_review record exists for this
    blind_id. Resolves blind_id -> clip_id via the hidden mapping (never
    exposed to B), loads Annotator A's locked record, and diffs field by
    field. Writes nothing to either original record."""
    reviews = sorted(REVIEWS_DIR.glob(f"{blind_id}__*.json"))
    if not reviews:
        raise RuntimeError(f"no blind_review_v2 submitted yet for {blind_id} -- run `submit`/`ingest-csv` first")
    review = json.loads(reviews[-1].read_text())

    mapping = _load_mapping()
    entry = mapping.get(blind_id)
    if entry is None:
        raise RuntimeError(f"no mapping entry for {blind_id} -- was it registered?")
    clip_id = entry["clip_id"]

    locked = _find_annotation(clip_id)
    if locked is None:
        raise RuntimeError(f"no locked (Annotator-A) annotation found for {clip_id}")

    ctm_path = CLIPS_DIR / clip_id / "clip_time_map.json"
    ctm = json.loads(ctm_path.read_text()) if ctm_path.exists() else {}
    clip_offset_s = ctm.get("source_start_timestamp")
    fps = ctm.get("clip_fps_probed") or ctm.get("source_fps") or 30.0

    a_hl = locked["human_label"]
    a_sp = a_hl["selected_player"]
    b = review["annotator_b_label"]

    fields = {}

    fields["event_type"] = {"a": a_hl.get("event_type"), "b": b.get("event_type")}
    fields["outcome"] = {"a": a_hl.get("outcome"), "b": b.get("outcome")}
    fields["event_subtype"] = {"a": a_hl.get("event_subtype"), "b": b.get("event_subtype")}
    fields["selected_player_confirmed"] = {
        "a_team": a_sp.get("team"), "b_confirmed": b.get("selected_player_confirmed"),
    }

    a_window = a_hl.get("ground_truth_event_window", {})
    a_start = a_window.get("start_time_s")
    a_end = a_window.get("end_time_s")
    a_interaction = a_window.get("interaction_time_s") or a_window.get("release_time_s")

    def to_absolute(clip_relative_s):
        if clip_relative_s is None or clip_offset_s is None:
            return None
        return clip_offset_s + clip_relative_s

    b_start_abs = to_absolute(b.get("event_start_s"))
    b_end_abs = to_absolute(b.get("event_end_s"))
    b_interaction_abs = to_absolute(b.get("interaction_release_s"))

    def timing_pair(name, a_val, b_val_abs, b_val_raw):
        if a_val is None or b_val_abs is None:
            return {"a_absolute_s": a_val, "b_clip_relative_s": b_val_raw, "b_absolute_s": b_val_abs,
                    "diff_s": None, "diff_frames": None, "within_tolerance": None}
        diff_s = abs(a_val - b_val_abs)
        return {
            "a_absolute_s": a_val, "b_clip_relative_s": b_val_raw, "b_absolute_s": b_val_abs,
            "diff_s": round(diff_s, 3), "diff_frames": round(diff_s * fps, 1),
            "within_tolerance": diff_s <= TIMING_TOLERANCE_S,
        }

    fields["event_start"] = timing_pair("event_start", a_start, b_start_abs, b.get("event_start_s"))
    fields["interaction_release"] = timing_pair("interaction_release", a_interaction, b_interaction_abs, b.get("interaction_release_s"))
    fields["event_end"] = timing_pair("event_end", a_end, b_end_abs, b.get("event_end_s"))

    fields["annotation_confidence"] = {"a": a_hl.get("annotation_confidence"), "b": b.get("annotation_confidence")}
    fields["manual_identity_confidence_a_only"] = {"a": a_sp.get("manual_identity_confidence"), "b": None,
        "note": "B does not independently state an identity-confidence LEVEL in the same taxonomy as A by design (B confirms YES/NO/UNSURE against a hint instead) -- see selected_player_confirmed above for the comparable field."}

    disagreements = []
    if fields["event_type"]["a"] != fields["event_type"]["b"]:
        disagreements.append({"field": "event_type", "a": fields["event_type"]["a"], "b": fields["event_type"]["b"],
                               "category": _classify_disagreement("event_type", fields["event_type"]["a"], fields["event_type"]["b"])})
    if fields["outcome"]["a"] != fields["outcome"]["b"]:
        disagreements.append({"field": "outcome", "a": fields["outcome"]["a"], "b": fields["outcome"]["b"],
                               "category": _classify_disagreement("outcome", fields["outcome"]["a"], fields["outcome"]["b"])})
    if fields["event_subtype"]["a"] != fields["event_subtype"]["b"]:
        disagreements.append({"field": "event_subtype", "a": fields["event_subtype"]["a"], "b": fields["event_subtype"]["b"],
                               "category": _classify_disagreement("event_subtype", fields["event_subtype"]["a"], fields["event_subtype"]["b"])})
    if b.get("selected_player_confirmed") not in ("YES",):
        disagreements.append({"field": "selected_player_confirmed", "a": fields["selected_player_confirmed"]["a_team"],
                               "b": b.get("selected_player_confirmed"),
                               "category": _classify_disagreement("selected_player_confirmed", None, None)})
    for tname in ("event_start", "interaction_release", "event_end"):
        t = fields[tname]
        if t["within_tolerance"] is False:
            disagreements.append({"field": tname, "a": t["a_absolute_s"], "b": t["b_absolute_s"],
                                   "diff_s": t["diff_s"], "diff_frames": t["diff_frames"],
                                   "category": _classify_disagreement(tname, t["a_absolute_s"], t["b_absolute_s"])})

    n_core_fields = 4  # event_type, outcome, event_subtype, selected_player_confirmed
    n_core_agree = n_core_fields - sum(1 for d in disagreements if d["field"] in
                                       ("event_type", "outcome", "event_subtype", "selected_player_confirmed"))

    comparison = {
        "blind_id": blind_id,
        "clip_id_resolved_at_compare_time": clip_id,
        "locked_annotation_id": locked["annotation_id"],
        "blind_review_id": review["blind_review_id"],
        "annotator_a_id": locked["provenance"]["annotator_id"],
        "annotator_b_id": review["annotator_b_id"],
        "fields": fields,
        "disagreements": disagreements,
        "n_disagreements": len(disagreements),
        "core_field_agreement_rate": round(n_core_agree / n_core_fields, 3),
        "compared_at": _now_iso(),
    }
    COMPARISONS_DIR.mkdir(parents=True, exist_ok=True)
    (COMPARISONS_DIR / f"{blind_id}__comparison.json").write_text(json.dumps(comparison, indent=2))
    return comparison


def adjudicate(blind_id: str, verdict: str, gold_label: dict | None = None, rationale: str | None = None) -> dict:
    """verdict in {ANNOTATOR_A, ANNOTATOR_B, ADJUDICATION_REQUIRED}. Does
    NOT automatically favour either annotator -- the caller (a human
    reviewing the comparison) supplies the verdict. gold_label is optional
    and, if given, is a small dict of the final agreed fields. Writes a
    separate adjudication record; NEVER edits the original Annotator-A
    annotation or the Annotator-B blind_review record."""
    if verdict not in ("ANNOTATOR_A", "ANNOTATOR_B", "ADJUDICATION_REQUIRED"):
        raise ValueError("verdict must be ANNOTATOR_A, ANNOTATOR_B, or ADJUDICATION_REQUIRED")
    comp_path = COMPARISONS_DIR / f"{blind_id}__comparison.json"
    if not comp_path.exists():
        raise RuntimeError(f"no comparison found for {blind_id} -- run `compare` first")
    record = {
        "blind_id": blind_id,
        "verdict": verdict,
        "gold_label": gold_label,
        "rationale": rationale,
        "adjudicated_at": _now_iso(),
    }
    ADJUDICATIONS_DIR.mkdir(parents=True, exist_ok=True)
    (ADJUDICATIONS_DIR / f"{blind_id}__adjudication.json").write_text(json.dumps(record, indent=2))
    return record


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p0 = sub.add_parser("register", help="add one clip to the hidden blind_id map + stage its neutral clip.mp4")
    p0.add_argument("--blind-id", required=True)
    p0.add_argument("--clip-id", required=True)
    p0.add_argument("--team-hint", required=True)

    p2 = sub.add_parser("submit", help="record Annotator B's from-scratch judgement for one blind_id")
    p2.add_argument("--blind-id", required=True)
    p2.add_argument("--annotator-b-id", required=True)
    p2.add_argument("--selected-player-confirmed", required=True, choices=["YES", "NO", "UNSURE"])
    p2.add_argument("--event-type", required=True)
    p2.add_argument("--annotation-confidence", required=True, choices=["HIGH", "MEDIUM", "LOW"])
    p2.add_argument("--event-subtype", default=None)
    p2.add_argument("--outcome", default=None)
    p2.add_argument("--event-start-s", type=float, default=None)
    p2.add_argument("--interaction-release-s", type=float, default=None)
    p2.add_argument("--event-end-s", type=float, default=None)
    p2.add_argument("--notes", default=None)

    p2b = sub.add_parser("ingest-csv", help="batch-submit from a filled answer sheet CSV")
    p2b.add_argument("--csv", required=True)
    p2b.add_argument("--annotator-b-id", default="nathan")

    p3 = sub.add_parser("compare", help="diff Annotator A vs B for one blind_id (only after submit)")
    p3.add_argument("--blind-id", required=True)

    p4 = sub.add_parser("adjudicate", help="record a non-automatic adjudication verdict")
    p4.add_argument("--blind-id", required=True)
    p4.add_argument("--verdict", required=True, choices=["ANNOTATOR_A", "ANNOTATOR_B", "ADJUDICATION_REQUIRED"])
    p4.add_argument("--rationale", default=None)

    args = ap.parse_args()
    if args.cmd == "register":
        print(json.dumps(register_blind_clip(args.blind_id, args.clip_id, args.team_hint), indent=2))
    elif args.cmd == "submit":
        print(json.dumps(submit_blind_review(
            args.blind_id, args.annotator_b_id, args.selected_player_confirmed, args.event_type,
            args.annotation_confidence, args.event_subtype, args.outcome, args.event_start_s,
            args.interaction_release_s, args.event_end_s, args.notes), indent=2))
    elif args.cmd == "ingest-csv":
        results = ingest_csv(args.csv, args.annotator_b_id)
        print(json.dumps({"ingested": len(results)}, indent=2))
    elif args.cmd == "compare":
        print(json.dumps(compare_annotations(args.blind_id), indent=2))
    elif args.cmd == "adjudicate":
        print(json.dumps(adjudicate(args.blind_id, args.verdict, rationale=args.rationale), indent=2))


if __name__ == "__main__":
    main()
