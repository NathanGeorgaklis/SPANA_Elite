# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R2 DEV jersey-number validation dataset instruction (Nathan, 2026-11-04 08:24): per clip label the
#                 selected player, expected jersey number, frames where the number is genuinely readable, frames where it
#                 is unreadable/occluded, same-kit competitor numbers nearby; truth frozen before any reader is tested;
#                 no change to the frozen R2 benchmark labels.
"""R2 jersey-number human-truth schema ``r2_jersey_labels_v1``: validation, freezing and the dataset record.
Evaluation tooling only (no tracker, detector or OCR model is involved).

One JSON file per clip, written by the jersey annotation page (tools_rb/r2_jersey_annotator.html):

{
 "schema": "r2_jersey_labels_v1",
 "clip_id": str, "clip_sha256": str, "fps": float, "frames": int, "width": int, "height": int,
 "annotator": str, "tool": str, "saved_utc": str,
 "clip_rejected": null | str,                         # rejected clips are never scored (reason required)
 "target": {"description": str,                       # who the selected player is, in words
            "team": "A" | "B" | "OTHER",              # kit as seen in this clip (A/B are the annotator's own names)
            "expected_number": int 1-99 | null,       # null only when the number is never readable in this clip
            "number_visibility": "READABLE_IN_SOME_FRAMES" | "NEVER_VISIBLE"},
 "keyframes": {"<frame>": {                            # EXACTLY the keyframe grid (keyframe_grid), no other keys
     "visibility": "VISIBLE" | "PARTIAL" | "HIDDEN" | "OUT_OF_FRAME",
     "box": [x1, y1, x2, y2] | null,                  # target box, image pixels; required when VISIBLE / PARTIAL
     "number": "READABLE" | "UNREADABLE" | "NOT_APPLICABLE",   # NOT_APPLICABLE iff HIDDEN / OUT_OF_FRAME
     "reason": null | one of UNREADABLE_REASONS,      # required iff UNREADABLE
     "human_uncertain": bool, "note": str}},
 "competitors": {"<id>": {"description": str, "relation": "SAME_TEAM" | "OPPONENT" | "OTHER",
                          "number": int 1-99 | null,   # null: number never readable for this person
                          "readable_keyframes": [frame, ...],   # keyframes where THEIR number is readable
                          "boxes": {"<frame>": [x1, y1, x2, y2]}}},   # optional, keyframes only
 "distractors": [str, ...]                             # e.g. "substitution number board shows 9" (free text)
}

KEYFRAMES: every round(fps * KEYFRAME_S)-th frame from frame 0 (0.5 s grid: 16 keyframes for an 8 s clip). Truth is
sampled on this grid only; any reader evaluation must sample the same grid and crop the HUMAN target box.

Rules (validate): READABLE keyframes need an expected_number, a VISIBLE / PARTIAL target with a box, and
human_uncertain false; NEVER_VISIBLE <=> expected_number null <=> no READABLE keyframe; a SAME_TEAM competitor with a
number must name keyframes where it is readable; a competitor sharing the target's number is a DUPLICATE (derived
in the dataset record, never typed). The freeze copies the file byte for byte to a read-only, versioned FROZEN file
with a freeze record (sha256); nothing is overwritten and a correction is a new version with a reason.
"""
import datetime
import hashlib
import json
import os
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools_rb import r2_protocol as P  # noqa: E402

SCHEMA = "r2_jersey_labels_v1"
KEYFRAME_S = 0.5
VIS = ("VISIBLE", "PARTIAL", "HIDDEN", "OUT_OF_FRAME")
NUMBER = ("READABLE", "UNREADABLE", "NOT_APPLICABLE")
UNREADABLE_REASONS = ("FRONT_ON", "SIDE_ON", "OCCLUDED_BY_PERSON", "OCCLUDED_OTHER", "MOTION_BLUR", "TOO_SMALL",
                      "PARTIAL_CROP", "LIGHTING", "FOLDED_SHIRT", "OTHER")
NUMBER_VISIBILITY = ("READABLE_IN_SOME_FRAMES", "NEVER_VISIBLE")
RELATIONS = ("SAME_TEAM", "OPPONENT", "OTHER")
TEAMS = ("A", "B", "OTHER")


def _round_half_even(x: float) -> int:
    return int(round(x))


def keyframe_grid(fps: float, frames: int) -> List[int]:
    step = max(1, _round_half_even(fps * KEYFRAME_S))
    return list(range(0, frames, step))


def _shirt(n) -> bool:
    return isinstance(n, int) and not isinstance(n, bool) and 1 <= n and len(str(n)) <= 2


def _box_ok(b, w, h) -> bool:
    return (isinstance(b, list) and len(b) == 4 and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                                                        for v in b)
            and 0 <= b[0] < b[2] <= w and 0 <= b[1] < b[3] <= h)


def validate(lab: Dict[str, Any], clip: Dict[str, Any] = None) -> List[str]:
    """All problems found (empty list = valid). clip: the registry record, checked when given."""
    p: List[str] = []
    if lab.get("schema") != SCHEMA:
        return [f"schema must be {SCHEMA}"]
    for k in ("clip_id", "clip_sha256", "fps", "frames", "width", "height", "annotator"):
        if not lab.get(k):
            p.append(f"{k} missing")
    if p:
        return p
    if clip is not None:
        for k in ("clip_id", "clip_sha256", "fps", "frames"):
            if lab[k] != clip[k]:
                p.append(f"{k} {lab[k]!r} differs from the registry {clip[k]!r}")
        if clip.get("source") in P.SOURCES:
            src = P.SOURCES[clip["source"]]
            if (lab["width"], lab["height"]) != (src["width"], src["height"]):
                p.append(f"image size {lab['width']}x{lab['height']} differs from the source "
                         f"{src['width']}x{src['height']}")
        else:
            p.append(f"registry source {clip.get('source')!r} is not an R2 source")
    if lab.get("clip_rejected"):
        if not isinstance(lab["clip_rejected"], str) or not lab["clip_rejected"].strip():
            p.append("clip_rejected needs a reason")
        return p
    w, h, grid = lab["width"], lab["height"], keyframe_grid(lab["fps"], lab["frames"])
    t = lab.get("target") or {}
    exp, nv = t.get("expected_number"), t.get("number_visibility")
    if not str(t.get("description", "")).strip():
        p.append("target.description missing")
    if t.get("team") not in TEAMS:
        p.append("target.team must be one of " + "/".join(TEAMS))
    if nv not in NUMBER_VISIBILITY:
        p.append("target.number_visibility must be one of " + "/".join(NUMBER_VISIBILITY))
    if exp is not None and not _shirt(exp):
        p.append("target.expected_number must be an integer 1-99 or null")
    if (nv == "NEVER_VISIBLE") != (exp is None):
        p.append("expected_number must be null exactly when number_visibility is NEVER_VISIBLE")
    kf = lab.get("keyframes") or {}
    if sorted(kf, key=lambda s: int(s) if str(s).isdigit() else -1) != [str(g) for g in grid]:
        p.append(f"keyframes must be exactly the 0.5 s grid {grid[:3]}...{grid[-1]} ({len(grid)} keys)")
    readable = 0
    for g in grid:
        x = kf.get(str(g))
        if not isinstance(x, dict):
            p.append(f"keyframe {g}: missing or not an object")
            continue
        tag = f"keyframe {g}"
        vis, num, box = x.get("visibility"), x.get("number"), x.get("box")
        if vis not in VIS:
            p.append(f"{tag}: visibility unlabelled")
            continue
        if num not in NUMBER:
            p.append(f"{tag}: number state unlabelled")
            continue
        if not isinstance(x.get("human_uncertain"), bool):
            p.append(f"{tag}: human_uncertain must be true/false")
        seen = vis in ("VISIBLE", "PARTIAL")
        if seen and not x.get("human_uncertain") and box is None:
            p.append(f"{tag}: target {vis} needs a box (or mark HUMAN_UNCERTAIN)")
        if not seen and box is not None:
            p.append(f"{tag}: target {vis} must have no box")
        if box is not None and not _box_ok(box, w, h):
            p.append(f"{tag}: box {box} invalid for {w}x{h}")
        if seen != (num != "NOT_APPLICABLE"):
            p.append(f"{tag}: number must be NOT_APPLICABLE exactly when the target is HIDDEN / OUT_OF_FRAME")
        if num == "UNREADABLE" and x.get("reason") not in UNREADABLE_REASONS:
            p.append(f"{tag}: UNREADABLE needs a reason")
        if num != "UNREADABLE" and x.get("reason") is not None:
            p.append(f"{tag}: reason only allowed when UNREADABLE")
        if num == "READABLE":
            readable += 1
            if exp is None:
                p.append(f"{tag}: READABLE but no expected_number")
            if x.get("human_uncertain"):
                p.append(f"{tag}: READABLE cannot be HUMAN_UNCERTAIN (use UNREADABLE / OTHER)")
            if box is None:
                p.append(f"{tag}: READABLE needs the target box")
    if nv == "READABLE_IN_SOME_FRAMES" and readable == 0:
        p.append("number_visibility READABLE_IN_SOME_FRAMES but no READABLE keyframe")
    for cid, c in (lab.get("competitors") or {}).items():
        tag = f"competitor {cid}"
        if not str(c.get("description", "")).strip():
            p.append(f"{tag}: description missing")
        if c.get("relation") not in RELATIONS:
            p.append(f"{tag}: relation must be one of " + "/".join(RELATIONS))
        if c.get("number") is not None and not _shirt(c.get("number")):
            p.append(f"{tag}: number must be 1-99 or null")
        rk = c.get("readable_keyframes") or []
        if any(r not in grid for r in rk):
            p.append(f"{tag}: readable_keyframes must be keyframes")
        if c.get("number") is None and rk:
            p.append(f"{tag}: readable_keyframes given but number is null")
        if c.get("number") is not None and c.get("relation") == "SAME_TEAM" and not rk:
            p.append(f"{tag}: SAME_TEAM number given: name the keyframes where it is readable")
        for fk, b in (c.get("boxes") or {}).items():
            if not str(fk).isdigit() or int(fk) not in grid:
                p.append(f"{tag}: box key {fk} is not a keyframe")
            elif not _box_ok(b, w, h):
                p.append(f"{tag}: box {b} invalid")
    if not isinstance(lab.get("distractors", []), list):
        p.append("distractors must be a list")
    return p


def summary(lab: Dict[str, Any]) -> Dict[str, Any]:
    """Coverage of one VALID label file (derived, never typed)."""
    if lab.get("clip_rejected"):
        return {"clip_id": lab["clip_id"], "rejected": lab["clip_rejected"]}
    kf = lab["keyframes"].values()
    reasons: Dict[str, int] = {}
    for x in kf:
        if x["number"] == "UNREADABLE":
            reasons[x["reason"]] = reasons.get(x["reason"], 0) + 1
    exp = lab["target"]["expected_number"]
    comps = lab.get("competitors") or {}
    return {"clip_id": lab["clip_id"], "expected_number": exp,
            "number_visibility": lab["target"]["number_visibility"], "keyframes": len(lab["keyframes"]),
            "readable": sum(x["number"] == "READABLE" for x in kf),
            "unreadable": sum(x["number"] == "UNREADABLE" for x in kf),
            "not_visible": sum(x["number"] == "NOT_APPLICABLE" for x in kf),
            "human_uncertain": sum(bool(x["human_uncertain"]) for x in kf),
            "unreadable_reasons": dict(sorted(reasons.items())),
            "same_team_competitor_numbers": sorted({c["number"] for c in comps.values()
                                                    if c["relation"] == "SAME_TEAM" and c["number"] is not None}),
            "duplicate_number": exp is not None and any(c.get("number") == exp for c in comps.values()),
            "same_team_number_clash": len([c for c in comps.values() if c["relation"] == "SAME_TEAM"
                                           and c["number"] is not None])
                                      != len({c["number"] for c in comps.values() if c["relation"] == "SAME_TEAM"
                                              and c["number"] is not None}),
            "distractors": len(lab.get("distractors") or [])}


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _load_registry(registry_json: str) -> Dict[str, Dict[str, Any]]:
    with open(registry_json) as f:
        return {c["clip_id"]: c for c in json.load(f)["clips"]}


def _check_dev(c: Dict[str, Any]) -> None:
    """Fail closed unless the clip is DEV both in the registry and by the protocol's own split of its window."""
    if c.get("split") != "DEV" or P.split_of(c["source"], c["start_s"], c["end_s"]) != "DEV":
        raise SystemExit(f"{c['clip_id']}: not a DEV clip (jersey truth is DEV only; fail closed)")


def _versions(out_dir: str, clip_id: str) -> List[int]:
    if not os.path.isdir(out_dir):
        return []
    pre, suf, out = clip_id + ".jersey.v", ".FROZEN.json", []
    for n in os.listdir(out_dir):
        if n.startswith(pre) and n.endswith(suf) and n[len(pre):-len(suf)].isdigit():
            out.append(int(n[len(pre):-len(suf)]))
    return sorted(out)


def frozen_path(out_dir: str, clip_id: str, v: int) -> str:
    return os.path.join(out_dir, f"{clip_id}.jersey.v{v}.FROZEN.json")


def freeze(labels_json: str, registry_json: str, out_dir: str, reason: Optional[str] = None,
           now=None) -> Dict[str, Any]:
    """Validate against the registry, then copy the label file BYTE FOR BYTE to OUT_DIR/<clip_id>.jersey.v<N>.FROZEN.json
    (read-only) with a sidecar <...>.FROZEN.record.json holding its sha256. Never overwrites: a correction is version
    N+1, which requires a reason and names the version it supersedes (all versions are kept)."""
    with open(labels_json) as f:
        lab = json.load(f)
    reg = _load_registry(registry_json)
    if lab.get("clip_id") not in reg:
        raise SystemExit(f"{labels_json}: clip {lab.get('clip_id')!r} is not in the jersey registry (fail closed)")
    c = reg[lab["clip_id"]]
    _check_dev(c)
    probs = validate(lab, c)
    if probs:
        raise SystemExit(f"{labels_json}: NOT frozen, {len(probs)} problems:\n  " + "\n  ".join(probs))
    os.makedirs(out_dir, exist_ok=True)
    prev = _versions(out_dir, lab["clip_id"])
    v = (prev[-1] + 1) if prev else 1
    if prev and not (reason and reason.strip()):
        raise SystemExit(f"{lab['clip_id']}: v{prev[-1]} is already frozen; a new version needs a reason")
    out = frozen_path(out_dir, lab["clip_id"], v)
    rec_path = out[:-len(".json")] + ".record.json"
    if os.path.exists(out) or os.path.exists(rec_path):
        raise SystemExit(f"{out} exists; a frozen label is never overwritten")
    src_sha = _sha(labels_json)
    with open(labels_json, "rb") as fi, open(out, "xb") as fo:
        fo.write(fi.read())
    os.chmod(out, 0o444)
    if _sha(out) != src_sha:
        raise SystemExit(f"{out}: copy hash differs from the source labels (fail closed)")
    stamp = (now or datetime.datetime.now(datetime.timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ")
    rec = {"schema": "r2_jersey_freeze_record_v1", "label_schema": SCHEMA, "clip_id": lab["clip_id"], "version": v,
           "frozen_file": os.path.basename(out), "sha256": src_sha, "source_file": os.path.basename(labels_json),
           "frozen_utc": stamp, "registry_sha256": _sha(registry_json), "clip_sha256": c["clip_sha256"],
           "supersedes": os.path.basename(frozen_path(out_dir, lab["clip_id"], prev[-1])) if prev else None,
           "reason": reason.strip() if prev else None, "summary": summary(lab)}
    with open(rec_path, "x") as f:
        json.dump(rec, f, indent=1, sort_keys=True)
        f.write("\n")
    os.chmod(rec_path, 0o444)
    return rec


def dataset_record(frozen_dir: str, registry_json: str) -> Dict[str, Any]:
    """Record over the LATEST frozen version of every registry clip: per-file sha256 (checked against its freeze record),
    a dataset digest over the sorted (file, sha256) list, superseded versions, and coverage. Fails closed when a clip
    has no frozen label, a frozen file differs from its freeze record, or a file no longer validates."""
    reg = _load_registry(registry_json)
    rows, per, superseded = [], [], []
    for cid, c in reg.items():
        _check_dev(c)
        vs = _versions(frozen_dir, cid)
        if not vs:
            raise SystemExit(f"{cid}: no frozen label (fail closed: the dataset is incomplete)")
        for v in vs:
            path = frozen_path(frozen_dir, cid, v)
            rp = path[:-len(".json")] + ".record.json"
            if not os.path.exists(rp):
                raise SystemExit(f"{path}: no freeze record (fail closed)")
            with open(rp) as f:
                rec = json.load(f)
            if rec.get("sha256") != _sha(path) or rec.get("version") != v or rec.get("clip_id") != cid:
                raise SystemExit(f"{path}: differs from its freeze record (fail closed)")
            if v != vs[-1]:
                superseded.append({"file": os.path.basename(path), "sha256": rec["sha256"]})
        path = frozen_path(frozen_dir, cid, vs[-1])
        with open(path) as f:
            lab = json.load(f)
        probs = validate(lab, c)
        if probs:
            raise SystemExit(f"{path}: frozen file no longer validates: {probs[:3]}")
        rows.append((os.path.basename(path), _sha(path)))
        per.append(summary(lab))
    digest = hashlib.sha256("\n".join(f"{n}  {s}" for n, s in sorted(rows)).encode()).hexdigest()
    ok = [s for s in per if "rejected" not in s]
    return {"schema": "r2_jersey_dataset_record_v1", "label_schema": SCHEMA, "registry_sha256": _sha(registry_json),
            "n_clips": len(per), "n_rejected": len(per) - len(ok), "keyframe_s": KEYFRAME_S,
            "files": [{"file": n, "sha256": s} for n, s in sorted(rows)], "superseded": superseded,
            "dataset_sha256": digest,
            "coverage": {"keyframes": sum(s["keyframes"] for s in ok), "readable": sum(s["readable"] for s in ok),
                         "unreadable": sum(s["unreadable"] for s in ok),
                         "not_visible": sum(s["not_visible"] for s in ok),
                         "human_uncertain": sum(s["human_uncertain"] for s in ok),
                         "clips_never_visible": sum(s["number_visibility"] == "NEVER_VISIBLE" for s in ok),
                         "clips_with_same_team_competitor_numbers": sum(bool(s["same_team_competitor_numbers"])
                                                                        for s in ok),
                         "clips_with_duplicate_number": sum(s["duplicate_number"] for s in ok)},
            "per_clip": per}


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) in (2, 3) and a[0] == "validate":
        lab = json.load(open(a[1]))
        clip = None
        if len(a) == 3:
            clip = _load_registry(a[2]).get(lab.get("clip_id"))
            if clip is None:
                raise SystemExit(f"clip {lab.get('clip_id')!r} is not in the registry (fail closed)")
        pr = validate(lab, clip)
        print("VALID" if not pr else "INVALID\n  " + "\n  ".join(pr))
        sys.exit(1 if pr else 0)
    elif len(a) in (4, 5) and a[0] == "freeze":
        print(json.dumps(freeze(a[1], a[2], a[3], a[4] if len(a) == 5 else None), indent=1))
    elif len(a) == 3 and a[0] == "record":
        print(json.dumps(dataset_record(a[1], a[2]), indent=1, sort_keys=True))
    else:
        raise SystemExit("usage: r2_jersey_labels.py validate LABELS [REGISTRY] | freeze LABELS REGISTRY OUT_DIR [REASON]"
                         " | record FROZEN_DIR REGISTRY")
