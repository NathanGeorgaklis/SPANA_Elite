# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R2 label-freeze instruction (Nathan, 2026-10-27 18:01: validate all files without modifying human
#                 labels; preserve HUMAN_UNCERTAIN; verify clip id and DEV/EVAL membership; freeze each valid file
#                 unchanged; stop for any invalid file and report the exact issue, no automatic repair; list uncertain
#                 frames and clips excluded from identity scoring); tools_rb/r2_labels.py; R2/R2_PREREGISTRATION.md
"""R2 label intake: check every incoming label file against the clip registry, freeze only the valid ones.

Evaluation tooling only. No tracker, detector or model is imported or run.

For each registered clip it records, in one ledger:
  * whether ``<clip_id>.labels_v1.json`` arrived, its sha256 and size (as received);
  * filename / clip_id agreement, clip sha256 agreement (labels = extract record = registry [= clip bytes]);
  * DEV / EVAL membership recomputed from r2_protocol and compared with the extract record and the registry;
  * the full list of r2_labels.validate problems (the file is never repaired);
  * HUMAN_UNCERTAIN frames (as ranges), unresolvable / clean frame counts after the seed;
  * identity-scoring status (see ``identity_scoring``);
  * the outcome: MISSING, INVALID, VALID (check only), FROZEN, or ALREADY_FROZEN.

Freezing goes through ``r2_labels.freeze`` (atomic, never overwrites). After freezing, the frozen copy must be
byte-identical to the incoming file. Incoming files are opened read-only and never changed or moved.

Usage:
    python3 tools_rb/r2_label_intake.py INCOMING_DIR CLIPS_DIR REGISTRY_JSON OUT_DIR LEDGER_JSON [--freeze] [--hash-clips]
"""
import argparse
import datetime
import hashlib
import json
import os
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools_rb import r2_labels as L          # noqa: E402
from tools_rb import r2_protocol as P        # noqa: E402

INTAKE_VERSION = "r2-intake-1"
SUFFIX = ".labels_v1.json"


def _sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _sha_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def ranges(xs: List[int]) -> List[List[int]]:
    """Sorted integer list -> [[first, last], ...] runs."""
    out: List[List[int]] = []
    for x in sorted(xs):
        if out and x == out[-1][1] + 1:
            out[-1][1] = x
        else:
            out.append([x, x])
    return out


def identity_scoring(lab: Dict[str, Any], valid: bool) -> Dict[str, Any]:
    """Whether the clip's human truth can support identity scoring.

    EXCLUDED when the annotator rejected the clip, or when no frame after the seed is clean (VISIBLE / PARTIAL,
    resolvable, not merged, not HUMAN_UNCERTAIN): then there is nothing on which identity could be checked.
    NOT_ASSESSED when the file is invalid. Otherwise INCLUDED, with the counts that let a reader judge ambiguity.
    No other threshold is applied (any further exclusion is Nathan's decision, not a tuned rule)."""
    if lab.get("clip_rejected"):
        return {"status": "EXCLUDED", "reason": f"clip rejected by annotator: {lab['clip_rejected']}"}
    if not valid:
        return {"status": "NOT_ASSESSED", "reason": "labels invalid"}
    pf, s = lab["per_frame"], lab["seed"]["frame"]
    after = pf[s + 1:]
    clean = sum(1 for f in after if L._clean(f))
    if clean == 0:
        return {"status": "EXCLUDED", "reason": "no clean, identifiable target frame after the seed"}
    return {"status": "INCLUDED", "reason": None}


def _frame_summary(lab: Dict[str, Any]) -> Dict[str, Any]:
    pf = lab.get("per_frame") or []
    hu = [i for i, f in enumerate(pf) if isinstance(f, dict) and f.get("human_uncertain") is True]
    s = lab.get("seed") if isinstance(lab.get("seed"), dict) else {}
    sf = s.get("frame") if isinstance(s.get("frame"), int) else None
    out = {"human_uncertain_frames": len(hu), "human_uncertain_ranges": ranges(hu)}
    if sf is not None and not lab.get("clip_rejected"):
        after = [f for f in pf[sf + 1:] if isinstance(f, dict)]
        out.update({"frames_after_seed": len(after),
                    "unresolvable_after_seed": sum(1 for f in after if f.get("resolvable") == "NO"),
                    "unlabelled_after_seed": sum(1 for f in after if L.UNLABELLED in
                                                 (f.get("visibility"), f.get("resolvable"), f.get("occlusion"))),
                    "clean_after_seed": sum(1 for f in after if all(k in f for k in L.FRAME_KEYS) and L._clean(f))})
    return out


def check_one(clip_id: str, reg_clip: Dict[str, Any], incoming_dir: str, clips_dir: str, out_dir: str,
              clip_bytes_sha: Optional[str]) -> Dict[str, Any]:
    e: Dict[str, Any] = {"clip_id": clip_id, "registry_split": reg_clip["split"]}
    path = os.path.join(incoming_dir, clip_id + SUFFIX)
    if not os.path.isfile(path):
        e.update({"outcome": "MISSING", "problems": ["no label file"]})
        return e
    with open(path, "rb") as f:
        data = f.read()
    e.update({"incoming_file": os.path.basename(path), "incoming_sha256": _sha_bytes(data), "incoming_bytes": len(data)})
    problems: List[str] = []
    try:
        lab = json.loads(data)
    except ValueError as x:
        e.update({"outcome": "INVALID", "problems": [f"not JSON: {x}"]})
        return e
    if not isinstance(lab, dict):
        e.update({"outcome": "INVALID", "problems": ["not a JSON object"]})
        return e
    if lab.get("clip_id") != clip_id:
        problems.append(f"file name says {clip_id} but clip_id is {lab.get('clip_id')!r}")
    ex_path = os.path.join(clips_dir, clip_id + ".extract.json")
    if not os.path.isfile(ex_path):
        e.update({"outcome": "INVALID", "problems": problems + ["clip extract record missing (fail closed)"]})
        return e
    with open(ex_path) as f:
        ex = json.load(f)
    try:
        split = P.split_of(ex["source"], ex["start_s"], ex["end_s"])
    except (KeyError, ValueError) as x:
        e.update({"outcome": "INVALID", "problems": problems + [f"split not computable: {x}"]})
        return e
    e["split"] = split
    if not (split == ex.get("split") == reg_clip["split"]):
        problems.append(f"split mismatch: protocol {split}, extract {ex.get('split')}, registry {reg_clip['split']}")
    if not (ex.get("clip_sha256") == reg_clip.get("clip_sha256")):
        problems.append("clip sha256 differs between extract record and registry")
    if clip_bytes_sha is not None and clip_bytes_sha != reg_clip.get("clip_sha256"):
        problems.append("clip file bytes do not match the registered clip sha256")
    e["annotator"] = lab.get("annotator")
    e["tool"] = lab.get("tool")
    e["saved_utc"] = lab.get("saved_utc")
    e["clip_rejected"] = lab.get("clip_rejected")
    src = P.SOURCES.get(ex["source"], {})
    vproblems = L.validate(lab, dict(ex, width=src.get("width"), height=src.get("height")))
    problems += vproblems
    e.update(_frame_summary(lab))
    e["seed_frame"] = (lab.get("seed") or {}).get("frame") if isinstance(lab.get("seed"), dict) else None
    e["problems"] = problems
    e["identity_scoring"] = identity_scoring(lab, not problems)
    frozen = os.path.join(out_dir, clip_id + SUFFIX)
    if os.path.exists(frozen) or os.path.exists(os.path.join(out_dir, clip_id + ".labels_v1.FROZEN.json")):
        same = os.path.isfile(frozen) and _sha_file(frozen) == e["incoming_sha256"]
        e["outcome"] = "ALREADY_FROZEN"
        e["frozen_matches_incoming"] = same
        return e
    e["outcome"] = "INVALID" if problems else "VALID"
    return e


def intake(incoming_dir: str, clips_dir: str, registry_json: str, out_dir: str, do_freeze: bool = False,
           hash_clips: bool = False, now=None) -> Dict[str, Any]:
    with open(registry_json) as f:
        reg = json.load(f)
    clips = {c["clip_id"]: c for c in reg["clips"]}
    entries = []
    for cid in sorted(clips):
        mp4 = os.path.join(clips_dir, cid + ".mp4")
        cb = _sha_file(mp4) if hash_clips and os.path.isfile(mp4) else None
        e = check_one(cid, clips[cid], incoming_dir, clips_dir, out_dir, cb)
        if hash_clips:
            e["clip_bytes_sha256_checked"] = cb is not None
        if do_freeze and e["outcome"] == "VALID":
            rec = L.freeze(os.path.join(incoming_dir, e["incoming_file"]),
                           os.path.join(clips_dir, cid + ".extract.json"), out_dir, now=now)
            if rec["sha256"] != e["incoming_sha256"] or _sha_file(os.path.join(out_dir, rec["labels_file"])) != \
                    e["incoming_sha256"]:
                raise L.LabelError(f"{cid}: frozen copy differs from the incoming file")
            e["outcome"] = "FROZEN"
            e["frozen_sha256"] = rec["sha256"]
            e["frozen_at_utc"] = rec["frozen_at_utc"]
        entries.append(e)
    unknown = sorted(n for n in os.listdir(incoming_dir) if n.endswith(".json")
                     and n[:-len(SUFFIX)] not in clips) if os.path.isdir(incoming_dir) else []
    count = lambda pred: sum(1 for e in entries if pred(e))                          # noqa: E731
    done = ("FROZEN", "ALREADY_FROZEN")
    return {"intake_version": INTAKE_VERSION, "schema": L.SCHEMA,
            "run_utc": (now or datetime.datetime.now(datetime.timezone.utc)).replace(microsecond=0)
            .isoformat().replace("+00:00", "Z"),
            "freeze_requested": do_freeze, "clip_bytes_hashed": hash_clips,
            "registered": len(clips), "found": count(lambda e: e["outcome"] != "MISSING"),
            "valid_or_frozen": count(lambda e: e["outcome"] in ("VALID",) + done),
            "frozen": count(lambda e: e["outcome"] in done),
            "invalid": count(lambda e: e["outcome"] == "INVALID"),
            "frozen_dev": count(lambda e: e["outcome"] in done and e.get("split") == "DEV"),
            "frozen_eval": count(lambda e: e["outcome"] in done and e.get("split") == "EVAL"),
            "registered_dev": sum(1 for c in clips.values() if c["split"] == "DEV"),
            "registered_eval": sum(1 for c in clips.values() if c["split"] == "EVAL"),
            "unknown_incoming_files": unknown, "clips": entries}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("incoming_dir")
    ap.add_argument("clips_dir")
    ap.add_argument("registry_json")
    ap.add_argument("out_dir")
    ap.add_argument("ledger_json")
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--hash-clips", action="store_true")
    a = ap.parse_args(argv)
    if os.path.exists(a.ledger_json):
        raise SystemExit(f"{a.ledger_json} exists (ledgers are never overwritten)")
    led = intake(a.incoming_dir, a.clips_dir, a.registry_json, a.out_dir, a.freeze, a.hash_clips)
    with open(a.ledger_json, "x") as f:
        json.dump(led, f, indent=1, sort_keys=True)
        f.write("\n")
    for e in led["clips"]:
        print(f"{e['clip_id']:18} {e.get('split', e['registry_split']):4} {e['outcome']:14} "
              f"problems={len(e.get('problems', []))} HU={e.get('human_uncertain_frames', '-')} "
              f"id={e.get('identity_scoring', {}).get('status', '-')}")
    print(f"found {led['found']}/{led['registered']}  frozen {led['frozen']}  invalid {led['invalid']}  "
          f"DEV {led['frozen_dev']}/{led['registered_dev']}  EVAL {led['frozen_eval']}/{led['registered_eval']}  "
          f"unknown files {led['unknown_incoming_files']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
