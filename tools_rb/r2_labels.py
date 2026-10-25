# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R2 instruction (Nathan, 2026-10-26 07:53: target box where identifiable, visibility, identity
#                 resolvable, other-person boxes where needed to verify swaps, occlusion / merge state, first clean
#                 reappearance, HUMAN_UNCERTAIN; freeze, hash and commit before inference); Nathan's answer
#                 2026-10-26 (boxes at 5 Hz + every frame within 0.5 s of events); R2/R2_PREREGISTRATION.md §4
"""R2 human-truth label schema ``r2_labels_v1``: validation and freezing. Evaluation tooling only.

One JSON file per clip, written by the R2 annotation page (tools_rb/r2_annotator.html) from Nathan's annotations:

{
 "schema": "r2_labels_v1",
 "clip_id": str, "clip_sha256": str (lossless clip), "fps": float, "frames": int, "width": int, "height": int,
 "annotator": str, "tool": str, "saved_utc": str,
 "clip_rejected": null | str          # annotator rejects the clip (reason); a rejected clip is never scored
 "seed": {"frame": int, "box": [x1, y1, x2, y2]},                  # clip frame index (0-based), image pixels
 "per_frame": [ {                                                   # exactly `frames` entries, index = clip frame
     "visibility": "VISIBLE" | "PARTIAL" | "HIDDEN" | "OUT_OF_FRAME",
     "resolvable": "YES" | "NO",                                    # could a careful human tell which body is the target
     "occlusion": "NONE" | "PARTIAL_BY_PERSON" | "FULL_BY_PERSON" | "PARTIAL_BY_OTHER" | "FULL_BY_OTHER",
     "merged": bool,                                                # target and another person form one blob
     "contact_with": [person_id, ...],                              # people overlapping / touching the target
     "camera_cut": bool,                                            # this frame is the first frame after a hard cut
     "human_uncertain": bool,                                       # HUMAN_UNCERTAIN: annotator not sure of this frame
     "note": str } ],
 "target_boxes": {"<frame>": [x1, y1, x2, y2]},
 "people": {"<person_id>": {"description": str, "relation": "SAME_TEAM" | "OPPONENT" | "OTHER",
                            "boxes": {"<frame>": [x1, y1, x2, y2]}}},
 "first_clean_reappearance": [frame, ...]
}

Required boxes (``box_frames``): every ``round(fps / BOX_HZ)``-th frame from the seed, plus every frame within
EVENT_PAD_S of an event frame (contact, occlusion != NONE, merged, a visibility change, OUT_OF_FRAME boundaries,
camera cut). On a required frame:
  * the target box is required when the target is VISIBLE / PARTIAL, resolvable, and not human_uncertain;
  * the target box is forbidden when the target is HIDDEN / OUT_OF_FRAME (nothing to box);
  * every person in contact_with needs a box (to verify swaps).
Boxes are allowed on other frames too (more truth is fine).
"""
import datetime
import hashlib
import json
import os
import sys
from typing import Any, Dict, List, Set

SCHEMA = "r2_labels_v1"
BOX_HZ = 5.0
EVENT_PAD_S = 0.5
VISIBILITY = ("VISIBLE", "PARTIAL", "HIDDEN", "OUT_OF_FRAME")
RESOLVABLE = ("YES", "NO")
OCCLUSION = ("NONE", "PARTIAL_BY_PERSON", "FULL_BY_PERSON", "PARTIAL_BY_OTHER", "FULL_BY_OTHER")
RELATION = ("SAME_TEAM", "OPPONENT", "OTHER")
FRAME_KEYS = ("visibility", "resolvable", "occlusion", "merged", "contact_with", "camera_cut", "human_uncertain",
              "note")


class LabelError(Exception):
    pass


def _box_ok(b, w, h) -> bool:
    return (isinstance(b, list) and len(b) == 4 and all(isinstance(v, (int, float)) for v in b)
            and 0 <= b[0] < b[2] <= w and 0 <= b[1] < b[3] <= h)


def event_frames(lab: Dict[str, Any]) -> List[int]:
    pf = lab["per_frame"]
    out = []
    for i, f in enumerate(pf):
        prev = pf[i - 1] if i else None
        if (f["contact_with"] or f["occlusion"] != "NONE" or f["merged"] or f["camera_cut"]
                or f["visibility"] == "OUT_OF_FRAME"
                or (prev is not None and (prev["visibility"] != f["visibility"]
                                          or prev["resolvable"] != f["resolvable"]))):
            out.append(i)
    return out


def box_frames(lab: Dict[str, Any]) -> List[int]:
    n, fps = lab["frames"], float(lab["fps"])
    s = lab["seed"]["frame"]
    step = max(1, round(fps / BOX_HZ))
    pad = int(round(EVENT_PAD_S * fps))
    req: Set[int] = set(range(s, n, step))
    for e in event_frames(lab):
        req.update(range(max(s, e - pad), min(n, e + pad + 1)))
    return sorted(req)


def validate(lab: Dict[str, Any], clip: Dict[str, Any] = None) -> List[str]:
    """All problems found (empty = valid). `clip` is the clip's extract record, when available."""
    err: List[str] = []
    try:
        if lab.get("schema") != SCHEMA:
            return [f"schema must be {SCHEMA}"]
        for k in ("clip_id", "clip_sha256", "fps", "frames", "width", "height", "annotator", "seed", "per_frame",
                  "target_boxes", "people", "first_clean_reappearance"):
            if k not in lab:
                err.append(f"missing {k}")
        if err:
            return err
        if clip is not None:
            for k_lab, k_clip in (("clip_id", "clip_id"), ("clip_sha256", "clip_sha256"), ("frames", "frames"),
                                  ("fps", "fps")):
                if lab[k_lab] != clip[k_clip]:
                    err.append(f"{k_lab} {lab[k_lab]!r} != clip record {clip[k_clip]!r}")
        if not str(lab["annotator"]).strip():
            err.append("annotator is empty")
        if lab.get("clip_rejected"):
            return err                                              # a rejected clip needs no further labels
        n, w, h = lab["frames"], lab["width"], lab["height"]
        pf = lab["per_frame"]
        if len(pf) != n:
            return err + [f"per_frame has {len(pf)} entries, expected {n}"]
        people = lab["people"]
        for pid, p in people.items():
            if p.get("relation") not in RELATION:
                err.append(f"person {pid}: relation {p.get('relation')!r}")
            for fk, b in p.get("boxes", {}).items():
                if not (fk.isdigit() and int(fk) < n and _box_ok(b, w, h)):
                    err.append(f"person {pid}: bad box at {fk}")
        for i, f in enumerate(pf):
            if set(f) != set(FRAME_KEYS):
                err.append(f"frame {i}: keys {sorted(set(f) ^ set(FRAME_KEYS))}")
                continue
            if f["visibility"] not in VISIBILITY:
                err.append(f"frame {i}: visibility {f['visibility']!r}")
            if f["resolvable"] not in RESOLVABLE:
                err.append(f"frame {i}: resolvable {f['resolvable']!r}")
            if f["occlusion"] not in OCCLUSION:
                err.append(f"frame {i}: occlusion {f['occlusion']!r}")
            for k in ("merged", "camera_cut", "human_uncertain"):
                if not isinstance(f[k], bool):
                    err.append(f"frame {i}: {k} must be a bool")
            if not isinstance(f["contact_with"], list) or any(p not in people for p in f["contact_with"]):
                err.append(f"frame {i}: contact_with must list known people")
            if f["visibility"] == "OUT_OF_FRAME" and (f["occlusion"] != "NONE" or f["merged"] or f["contact_with"]):
                err.append(f"frame {i}: OUT_OF_FRAME cannot be occluded / merged / in contact")
            if f["visibility"] == "HIDDEN" and f["occlusion"] in ("NONE", "PARTIAL_BY_PERSON", "PARTIAL_BY_OTHER"):
                err.append(f"frame {i}: HIDDEN needs a FULL_* occlusion")
        for fk, b in lab["target_boxes"].items():
            if not (fk.isdigit() and int(fk) < n and _box_ok(b, w, h)):
                err.append(f"target box at {fk} invalid")
                continue
            if pf[int(fk)]["visibility"] in ("HIDDEN", "OUT_OF_FRAME"):
                err.append(f"frame {fk}: target box given while target is {pf[int(fk)]['visibility']}")
        s = lab["seed"]
        if not (isinstance(s.get("frame"), int) and 0 <= s["frame"] < n - 1 and _box_ok(s.get("box"), w, h)):
            err.append("seed must have a frame before the last frame and a valid box")
        else:
            sf = pf[s["frame"]]
            if not (sf["visibility"] == "VISIBLE" and sf["resolvable"] == "YES" and not sf["merged"]
                    and not sf["human_uncertain"] and not sf["contact_with"]):
                err.append("seed frame must be VISIBLE, resolvable, not merged, not uncertain, no contact")
            if lab["target_boxes"].get(str(s["frame"])) != s["box"]:
                err.append("seed box must equal the target box on the seed frame")
        for i in box_frames(lab):
            f = pf[i]
            needs = (f["visibility"] in ("VISIBLE", "PARTIAL") and f["resolvable"] == "YES"
                     and not f["human_uncertain"])
            if needs and str(i) not in lab["target_boxes"]:
                err.append(f"frame {i}: target box required")
            for pid in f["contact_with"]:
                if str(i) not in people[pid].get("boxes", {}):
                    err.append(f"frame {i}: box required for contact person {pid}")
        for c in lab["first_clean_reappearance"]:
            if not (isinstance(c, int) and 0 < c < n):
                err.append(f"first_clean_reappearance {c!r} out of range")
                continue
            f, p = pf[c], pf[c - 1]
            if not (f["visibility"] in ("VISIBLE", "PARTIAL") and f["resolvable"] == "YES" and not f["merged"]):
                err.append(f"first_clean_reappearance {c}: target not cleanly identifiable there")
            if p["visibility"] in ("VISIBLE", "PARTIAL") and p["resolvable"] == "YES" and not p["merged"]:
                err.append(f"first_clean_reappearance {c}: previous frame already clean")
    except (KeyError, TypeError, AttributeError) as e:
        err.append(f"malformed: {e!r}")
    return err


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def freeze(labels_json: str, clip_record_json: str, out_dir: str, now=None) -> Dict[str, Any]:
    """Validate and freeze one clip's labels: copy to OUT_DIR/<clip_id>.labels_v1.json, write
    OUT_DIR/<clip_id>.labels_v1.FROZEN.json and (unless the clip is rejected) the run fixture
    OUT_DIR/<clip_id>.fixture.json with the human seed. Refuses to overwrite (a frozen label is never replaced)."""
    with open(labels_json, "rb") as f:
        data = f.read()
    lab = json.loads(data)
    with open(clip_record_json) as f:
        clip = json.load(f)
    errs = validate(lab, clip)
    if errs:
        raise LabelError(f"{len(errs)} problems, first: {errs[:5]}")
    name = f"{lab['clip_id']}.labels_v1.json"
    dst = os.path.join(out_dir, name)
    rec_path = os.path.join(out_dir, f"{lab['clip_id']}.labels_v1.FROZEN.json")
    if os.path.exists(dst) or os.path.exists(rec_path):
        raise LabelError(f"{lab['clip_id']} already frozen; corrections need labels_v2")
    os.makedirs(out_dir, exist_ok=True)
    with open(dst, "xb") as f:
        f.write(data)
    sha = hashlib.sha256(data).hexdigest()
    rec = {"status": "FROZEN", "valid": True, "errors": [], "labels_file": name, "sha256": sha, "bytes": len(data),
           "schema": SCHEMA, "clip_id": lab["clip_id"], "clip_sha256": clip["clip_sha256"], "split": clip["split"],
           "fps": clip["fps"], "frames": clip["frames"], "annotator": lab["annotator"],
           "clip_rejected": lab.get("clip_rejected"),
           "seed_frame": lab["seed"]["frame"], "seed_bbox": lab["seed"]["box"],
           "box_frames_required": len(box_frames(lab)) if not lab.get("clip_rejected") else 0,
           "frozen_at_utc": (now or datetime.datetime.now(datetime.timezone.utc)).replace(microsecond=0)
           .isoformat().replace("+00:00", "Z"),
           "rule": "Do not modify this label file. Corrections require labels_v2 with the reason recorded."}
    with open(rec_path, "x") as f:
        json.dump(rec, f, indent=1, sort_keys=True)
        f.write("\n")
    if not lab.get("clip_rejected"):
        # run fixture for the scorer (tracker_eval main): the human seed, never a model output
        fx = {"clip_id": lab["clip_id"], "split": clip["split"], "video_sha256": clip["clip_sha256"],
              "video_file": clip["clip_id"] + ".mp4", "fps": clip["fps"], "seed_frame": lab["seed"]["frame"],
              "seed_bbox": lab["seed"]["box"], "seed_source": "HUMAN (frozen r2_labels_v1)",
              "labels_file": name, "labels_sha256": sha}
        with open(os.path.join(out_dir, f"{lab['clip_id']}.fixture.json"), "x") as f:
            json.dump(fx, f, indent=1, sort_keys=True)
            f.write("\n")
    if _sha(dst) != sha:
        raise LabelError("frozen copy hash mismatch")
    return rec


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "validate":
        lab = json.load(open(sys.argv[2]))
        clip = json.load(open(sys.argv[3])) if len(sys.argv) > 3 else None
        e = validate(lab, clip)
        print("\n".join(e) if e else "VALID")
        sys.exit(1 if e else 0)
    elif len(sys.argv) == 5 and sys.argv[1] == "freeze":
        print(json.dumps(freeze(sys.argv[2], sys.argv[3], sys.argv[4]), indent=1))
    else:
        raise SystemExit("usage: r2_labels.py validate LABELS [CLIP_RECORD] | freeze LABELS CLIP_RECORD OUT_DIR")
