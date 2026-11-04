# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: Nathan, 2026-11-04 18:44 and 18:52 (jersey number primary; shorts number optional supporting evidence
#                 only when genuinely present and readable, its absence never counts against identity; keep frozen v1
#                 unchanged; v2
#                 adds competitor annotations for J10/J04/J14; record competitor jersey and shorts numbers separately;
#                 regions JERSEY_BACK / JERSEY_FRONT / SHORTS_LEFT / SHORTS_RIGHT inspected independently; never assume
#                 shirt and shorts match unless both are readable; non-player numbers such as boards / bibs never count)
#                 and his answer 18:4x (v2 also holds region truth for the selected player on all 15 clips).
"""R2 jersey truth v2: an ADDITIVE region layer ``r2_jersey_regions_v2`` over a frozen v1 label (never modifies it).
Evaluation tooling only (no tracker, detector or OCR model is involved).

One JSON file per clip:
{
 "schema": "r2_jersey_regions_v2", "clip_id", "clip_sha256", "fps", "frames", "width", "height",
 "annotator", "tool", "saved_utc",
 "base_v1": {"file": "<clip>.jersey.v1.FROZEN.json", "sha256": str},     # the frozen v1 this layer extends
 "target": {"shorts_number": int 1-99 | null,                             # null: never readable on the shorts
            "regions": {"<keyframe>": REGIONS}},                         # exactly v1's VISIBLE / PARTIAL keyframes
 "competitors": {"<id>": {"description": str, "relation": "SAME_TEAM" | "OPPONENT",
                          "jersey_number": int | null, "shorts_number": int | null,   # recorded separately
                          "keyframes": {"<keyframe>": {"visibility": VIS, "box": [x1,y1,x2,y2] | null,
                                                       "human_uncertain": bool, "regions": REGIONS | null}}}},
 "competitors_note": str,          # required (non-empty) when a REQUIRED_COMPETITOR clip has no competitor
 "distractors": [{"keyframe": int, "box": [x1,y1,x2,y2], "kind": DISTRACTOR_KINDS, "text": str}]
}
REGIONS = {"JERSEY_BACK": R, "JERSEY_FRONT": R, "SHORTS_LEFT": R, "SHORTS_RIGHT": R}, each
R = {"state": "READABLE" | "UNREADABLE" | "NOT_VISIBLE" | "NO_NUMBER", "number": int | null,
     "reason": UNREADABLE_REASONS | null, "note": str}
  * READABLE: a human reads the number in THIS region on THIS keyframe image (number required);
  * UNREADABLE: the region is in view but its number cannot be read (reason required);
  * NOT_VISIBLE: the region is not in view (facing away, occluded entirely, out of frame);
  * NO_NUMBER: the region is in view and has no number printed.

Consistency with frozen v1 (v1 is truth for the SHIRT; this layer refines it, never contradicts it silently):
  * target regions exist on exactly the keyframes where v1's target is VISIBLE / PARTIAL and not HUMAN_UNCERTAIN;
  * v1 READABLE <=> JERSEY_BACK or JERSEY_FRONT READABLE; a READABLE jersey region's number == v1 expected_number.
    A disagreement is allowed only with a non-empty region note starting "V1_CONFLICT:" (reported, never hidden);
  * shorts numbers are independent observations: a READABLE shorts region must equal target.shorts_number (which
    must then be set, and may be set only if some shorts region is READABLE); never assumed equal to the shirt number.
    The V1_CONFLICT override exists only for the target's jersey regions (v1-bound); competitors have none.
Competitors: SAME_TEAM / OPPONENT people near the target; per keyframe visibility, box (when seen) and regions (when
seen and not human_uncertain); READABLE jersey regions must equal jersey_number, shorts regions shorts_number.
Distractors: non-player numbers (board, bib, scoreboard, advertising, other) with a box; readings inside them never count.
"""
import datetime
import hashlib
import json
import os
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools_rb import r2_jersey_labels as L1  # noqa: E402
from tools_rb import r2_protocol as P  # noqa: E402

SCHEMA = "r2_jersey_regions_v2"
REGIONS = ("JERSEY_BACK", "JERSEY_FRONT", "SHORTS_LEFT", "SHORTS_RIGHT")
JERSEY = ("JERSEY_BACK", "JERSEY_FRONT")
SHORTS = ("SHORTS_LEFT", "SHORTS_RIGHT")
STATES = ("READABLE", "UNREADABLE", "NOT_VISIBLE", "NO_NUMBER")
REASONS = ("MOTION_BLUR", "TOO_SMALL", "OCCLUDED_BY_PERSON", "OCCLUDED_OTHER", "FOLDED", "ANGLE", "LIGHTING",
           "PARTIAL_CROP", "OTHER")
RELATIONS = ("SAME_TEAM", "OPPONENT")
DISTRACTOR_KINDS = ("BOARD", "BIB", "SCOREBOARD", "ADVERTISING", "OTHER")
REQUIRED_COMPETITOR_CLIPS = ("R2_JERSEY_MCFC_MIT_J04", "R2_JERSEY_MCFC_MIT_J10", "R2_JERSEY_SYO_MAR_J14")
CONFLICT_TAG = "V1_CONFLICT:"


def _region_problems(tag: str, regs: Any, jersey_no, shorts_no, who: str, allow_conflict: bool) -> List[str]:
    p: List[str] = []
    if not isinstance(regs, dict) or sorted(regs) != sorted(REGIONS):
        return [f"{tag}: regions must be exactly {'/'.join(REGIONS)}"]
    for r in REGIONS:
        x, t = regs[r], f"{tag} {r}"
        if not isinstance(x, dict) or x.get("state") not in STATES:
            p.append(f"{t}: state unlabelled")
            continue
        st, n, why = x["state"], x.get("number"), x.get("reason")
        if st == "READABLE":
            if not L1._shirt(n):
                p.append(f"{t}: READABLE needs a number 1-99")
            else:
                want = jersey_no if r in JERSEY else shorts_no
                kind = "jersey" if r in JERSEY else "shorts"
                if want is None:
                    p.append(f"{t}: READABLE {n} but {who} {kind} number is not set")
                elif n != want and not (allow_conflict and r in JERSEY
                                        and str(x.get("note", "")).startswith(CONFLICT_TAG)):
                    p.append(f"{t}: reads {n} but {who} {kind} number is {want}"
                             + (f" (note '{CONFLICT_TAG} ...' if real)" if allow_conflict and r in JERSEY else ""))
        elif n is not None:
            p.append(f"{t}: number only allowed when READABLE")
        if st == "UNREADABLE" and why not in REASONS:
            p.append(f"{t}: UNREADABLE needs a reason")
        if st != "UNREADABLE" and why is not None:
            p.append(f"{t}: reason only allowed when UNREADABLE")
    return p


def validate(lab: Dict[str, Any], v1: Dict[str, Any], v1_sha: str, clip: Dict[str, Any] = None) -> List[str]:
    """All problems of a v2 layer against its frozen v1 (and the registry record when given)."""
    if lab.get("schema") != SCHEMA:
        return [f"schema must be {SCHEMA}"]
    p: List[str] = []
    for k in ("clip_id", "clip_sha256", "fps", "frames", "width", "height"):
        if lab.get(k) != v1.get(k):
            p.append(f"{k} {lab.get(k)!r} differs from frozen v1 {v1.get(k)!r}")
    if not lab.get("annotator"):
        p.append("annotator missing")
    b = lab.get("base_v1") or {}
    if b.get("sha256") != v1_sha or b.get("file") != f"{v1.get('clip_id')}.jersey.v1.FROZEN.json":
        p.append("base_v1 must name the frozen v1 file and its sha256")
    if L1.validate(v1, clip):
        p.append("base v1 does not validate (fail closed)")
    if v1.get("clip_rejected"):
        p.append("v1 rejected this clip: no v2 layer")
    if p:
        return p
    w, h = lab["width"], lab["height"]
    grid = L1.keyframe_grid(lab["fps"], lab["frames"])
    t = lab.get("target") or {}
    exp, sn = v1["target"]["expected_number"], t.get("shorts_number")
    if sn is not None and not L1._shirt(sn):
        p.append("target.shorts_number must be 1-99 or null")
    seen = [g for g in grid if v1["keyframes"][str(g)]["visibility"] in ("VISIBLE", "PARTIAL")
            and not v1["keyframes"][str(g)]["human_uncertain"]]
    tr = t.get("regions") or {}
    if sorted(tr, key=lambda s: int(s) if str(s).isdigit() else -1) != [str(g) for g in seen]:
        p.append(f"target.regions must cover exactly v1's visible keyframes {seen}")
    for g in seen:
        regs = tr.get(str(g))
        p += _region_problems(f"target kf {g}", regs, exp, sn, "target", allow_conflict=True)
        if isinstance(regs, dict) and all(isinstance(regs.get(r), dict) for r in REGIONS):
            v1r = v1["keyframes"][str(g)]["number"] == "READABLE"
            v2r = any(regs[r].get("state") == "READABLE" for r in JERSEY)
            noted = any(str(regs[r].get("note", "")).startswith(CONFLICT_TAG) for r in JERSEY)
            if v1r != v2r and not noted:
                p.append(f"target kf {g}: v1 says shirt {'READABLE' if v1r else 'not readable'} but jersey regions "
                         f"say {'READABLE' if v2r else 'not readable'} (note '{CONFLICT_TAG} ...' if real)")
    if sn is not None and not any(isinstance(tr.get(str(g)), dict) and isinstance(tr[str(g)].get(r), dict)
                                  and tr[str(g)][r].get("state") == "READABLE" for g in seen for r in SHORTS):
        p.append("target.shorts_number is set but no keyframe has a READABLE shorts region")
    comps = lab.get("competitors")
    if not isinstance(comps, dict):
        p.append("competitors must be an object")
        comps = {}
    if lab["clip_id"] in REQUIRED_COMPETITOR_CLIPS and not comps and not str(lab.get("competitors_note", "")).strip():
        p.append("this clip needs its nearby competitors, or competitors_note saying why there are none")
    for cid, c in comps.items():
        tag = f"competitor {cid}"
        if not str(c.get("description", "")).strip():
            p.append(f"{tag}: description missing")
        if c.get("relation") not in RELATIONS:
            p.append(f"{tag}: relation must be SAME_TEAM or OPPONENT")
        jn, cs = c.get("jersey_number"), c.get("shorts_number")
        for k, v in (("jersey_number", jn), ("shorts_number", cs)):
            if v is not None and not L1._shirt(v):
                p.append(f"{tag}: {k} must be 1-99 or null")
        kfs = c.get("keyframes") or {}
        if not kfs:
            p.append(f"{tag}: no keyframes labelled")
        n_read = {"jersey": 0, "shorts": 0}
        for k, x in kfs.items():
            if not str(k).isdigit() or int(k) not in grid:
                p.append(f"{tag}: {k} is not a keyframe")
                continue
            kt = f"{tag} kf {k}"
            vis = (x or {}).get("visibility")
            if vis not in L1.VIS:
                p.append(f"{kt}: visibility unlabelled")
                continue
            if not isinstance(x.get("human_uncertain"), bool):
                p.append(f"{kt}: human_uncertain must be true/false")
            on = vis in ("VISIBLE", "PARTIAL")
            if on and x.get("box") is None and not x.get("human_uncertain"):
                p.append(f"{kt}: {vis} needs a box")
            if not on and (x.get("box") is not None or x.get("regions") is not None):
                p.append(f"{kt}: {vis} must have no box and no regions")
            if x.get("box") is not None and not L1._box_ok(x["box"], w, h):
                p.append(f"{kt}: box invalid")
            if on and not x.get("human_uncertain"):
                p += _region_problems(kt, x.get("regions"), jn, cs, f"competitor {cid}", allow_conflict=False)
                for grp, r_ in (("jersey", JERSEY), ("shorts", SHORTS)):
                    n_read[grp] += sum(isinstance((x.get("regions") or {}).get(r), dict)
                                       and x["regions"][r].get("state") == "READABLE" for r in r_)
            elif x.get("human_uncertain") and x.get("regions") is not None:
                p.append(f"{kt}: HUMAN_UNCERTAIN keyframes carry no region truth")
        if jn is not None and n_read["jersey"] == 0:
            p.append(f"{tag}: jersey_number is set but no keyframe has a READABLE jersey region")
        if cs is not None and n_read["shorts"] == 0:
            p.append(f"{tag}: shorts_number is set but no keyframe has a READABLE shorts region")
    ds = lab.get("distractors")
    if not isinstance(ds, list):
        p.append("distractors must be a list")
        ds = []
    for i, d in enumerate(ds):
        kf_ok = isinstance(d, dict) and type(d.get("keyframe")) is int and d.get("keyframe") in grid
        if not kf_ok or d.get("kind") not in DISTRACTOR_KINDS \
                or not L1._box_ok(d.get("box"), w, h):
            p.append(f"distractor {i}: needs a keyframe, a valid box and a kind ({'/'.join(DISTRACTOR_KINDS)})")
    return p


def summary(lab: Dict[str, Any], v1: Dict[str, Any]) -> Dict[str, Any]:
    """Coverage of a VALID layer (derived): per-region states over person-keyframes, and how much the shorts add."""
    cnt = {r: {s: 0 for s in STATES} for r in REGIONS}
    tk = {"person_keyframes": 0, "jersey_readable": 0, "shorts_readable": 0, "shorts_only": 0, "either": 0}

    def add(regs, acc):
        for r in REGIONS:
            cnt[r][regs[r]["state"]] += 1
        j = any(regs[r]["state"] == "READABLE" for r in JERSEY)
        s = any(regs[r]["state"] == "READABLE" for r in SHORTS)
        acc["person_keyframes"] += 1
        acc["jersey_readable"] += j
        acc["shorts_readable"] += s
        acc["shorts_only"] += s and not j
        acc["either"] += j or s
    for regs in lab["target"]["regions"].values():
        add(regs, tk)
    ck = {k: 0 for k in tk}
    for c in lab["competitors"].values():
        for x in c["keyframes"].values():
            if x.get("regions"):
                add(x["regions"], ck)
    conflicts = sum(str(regs[r].get("note", "")).startswith(CONFLICT_TAG)
                    for regs in lab["target"]["regions"].values() for r in REGIONS)
    tn, sn = v1["target"]["expected_number"], lab["target"]["shorts_number"]
    return {"clip_id": lab["clip_id"], "target_jersey_number": tn, "target_shorts_number": sn,
            "target_shorts_differs_from_jersey": sn is not None and tn is not None and sn != tn,
            "target": tk, "competitors": ck, "n_competitors": len(lab["competitors"]),
            "same_team_competitors": sorted([(c.get("jersey_number"), c.get("shorts_number"))
                                             for c in lab["competitors"].values() if c["relation"] == "SAME_TEAM"],
                                            key=str),
            "competitor_shares_target_number": any(tn is not None and tn in (c.get("jersey_number"),
                                                                             c.get("shorts_number"))
                                                   for c in lab["competitors"].values()),
            "region_states": cnt, "v1_conflict_notes": conflicts, "distractors": len(lab["distractors"])}


def _sha(path):
    return L1._sha(path)


def _versions(out_dir, clip_id):
    pre, suf, out = clip_id + ".regions.v", ".FROZEN.json", []
    if os.path.isdir(out_dir):
        for n in os.listdir(out_dir):
            if n.startswith(pre) and n.endswith(suf) and n[len(pre):-len(suf)].isdigit():
                out.append(int(n[len(pre):-len(suf)]))
    return sorted(out)


def frozen_path(out_dir, clip_id, v):
    return os.path.join(out_dir, f"{clip_id}.regions.v{v}.FROZEN.json")


def _v1(v1_dir, registry, clip_id):
    """The single frozen v1 of a clip, checked against its freeze record (fail closed)."""
    vs = L1._versions(v1_dir, clip_id)
    if vs != [1]:
        raise SystemExit(f"{clip_id}: expected exactly frozen v1, found versions {vs} (fail closed)")
    path = L1.frozen_path(v1_dir, clip_id, 1)
    with open(path[:-len(".json")] + ".record.json") as f:
        rec = json.load(f)
    sha = _sha(path)
    if rec.get("sha256") != sha:
        raise SystemExit(f"{path}: differs from its freeze record (fail closed)")
    with open(path) as f:
        v1 = json.load(f)
    c = registry[clip_id]
    L1._check_dev(c)
    if L1.validate(v1, c):
        raise SystemExit(f"{path}: frozen v1 no longer validates (fail closed)")
    return v1, sha, c


def freeze(labels_json, registry_json, v1_dir, out_dir, reason=None, now=None):
    """Validate against frozen v1 + registry; byte-copy to <clip>.regions.v<N>.FROZEN.json (read-only) + record."""
    with open(labels_json) as f:
        lab = json.load(f)
    reg = L1._load_registry(registry_json)
    if lab.get("clip_id") not in reg:
        raise SystemExit(f"{labels_json}: clip not in the registry (fail closed)")
    v1, v1_sha, c = _v1(v1_dir, reg, lab["clip_id"])
    probs = validate(lab, v1, v1_sha, c)
    if probs:
        raise SystemExit(f"{labels_json}: NOT frozen, {len(probs)} problems:\n  " + "\n  ".join(probs))
    os.makedirs(out_dir, exist_ok=True)
    prev = _versions(out_dir, lab["clip_id"])
    v = (prev[-1] + 1) if prev else 2          # the region layer starts the dataset's v2
    if prev and not (reason and reason.strip()):
        raise SystemExit(f"{lab['clip_id']}: v{prev[-1]} already frozen; a new version needs a reason")
    out = frozen_path(out_dir, lab["clip_id"], v)
    rp = out[:-len(".json")] + ".record.json"
    if os.path.exists(out) or os.path.exists(rp):
        raise SystemExit(f"{out} exists; never overwritten")
    src = _sha(labels_json)
    with open(labels_json, "rb") as fi, open(out, "xb") as fo:
        fo.write(fi.read())
    os.chmod(out, 0o444)
    if _sha(out) != src:
        raise SystemExit(f"{out}: copy differs from the source (fail closed)")
    stamp = (now or datetime.datetime.now(datetime.timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ")
    rec = {"schema": "r2_jersey_regions_freeze_record_v1", "label_schema": SCHEMA, "clip_id": lab["clip_id"],
           "version": v, "frozen_file": os.path.basename(out), "sha256": src, "base_v1_sha256": v1_sha,
           "source_file": os.path.basename(labels_json), "frozen_utc": stamp, "registry_sha256": _sha(registry_json),
           "supersedes": os.path.basename(frozen_path(out_dir, lab["clip_id"], prev[-1])) if prev else None,
           "reason": reason.strip() if prev else None, "summary": summary(lab, v1)}
    with open(rp, "x") as f:
        json.dump(rec, f, indent=1, sort_keys=True)
        f.write("\n")
    os.chmod(rp, 0o444)
    return rec


def dataset_record(v1_dir, v2_dir, registry_json):
    """Dataset v2 = frozen v1 files + the latest region layer per clip. Digest over all (file, sha256) pairs."""
    reg = L1._load_registry(registry_json)
    rows, per, superseded = [], [], []
    for cid in reg:
        v1, v1_sha, c = _v1(v1_dir, reg, cid)
        rows.append((os.path.basename(L1.frozen_path(v1_dir, cid, 1)), v1_sha))
        vs = _versions(v2_dir, cid)
        if not vs:
            raise SystemExit(f"{cid}: no frozen region layer (fail closed: dataset v2 incomplete)")
        for v in vs:
            path = frozen_path(v2_dir, cid, v)
            rp = path[:-len(".json")] + ".record.json"
            if not os.path.exists(rp):
                raise SystemExit(f"{path}: no freeze record (fail closed)")
            with open(rp) as f:
                rec = json.load(f)
            if rec.get("sha256") != _sha(path) or rec.get("version") != v or rec.get("base_v1_sha256") != v1_sha:
                raise SystemExit(f"{path}: differs from its freeze record or its v1 (fail closed)")
            if v != vs[-1]:
                superseded.append({"file": os.path.basename(path), "sha256": rec["sha256"]})
        path = frozen_path(v2_dir, cid, vs[-1])
        with open(path) as f:
            lab = json.load(f)
        probs = validate(lab, v1, v1_sha, c)
        if probs:
            raise SystemExit(f"{path}: no longer validates: {probs[:3]}")
        rows.append((os.path.basename(path), _sha(path)))
        per.append(summary(lab, v1))
    digest = hashlib.sha256("\n".join(f"{n}  {s}" for n, s in sorted(rows)).encode()).hexdigest()

    def tot(who, k):
        return sum(s[who][k] for s in per)
    keys = ("person_keyframes", "jersey_readable", "shorts_readable", "shorts_only", "either")
    return {"schema": "r2_jersey_dataset_record_v2", "label_schemas": [L1.SCHEMA, SCHEMA],
            "registry_sha256": _sha(registry_json), "n_clips": len(per),
            "files": [{"file": n, "sha256": s} for n, s in sorted(rows)], "superseded": superseded,
            "dataset_sha256": digest,
            "coverage": {"target": {k: tot("target", k) for k in keys},
                         "competitors": {k: tot("competitors", k) for k in keys},
                         "n_competitors": sum(s["n_competitors"] for s in per),
                         "clips_target_shorts_differs": sum(s["target_shorts_differs_from_jersey"] for s in per),
                         "clips_competitor_shares_target_number": sum(s["competitor_shares_target_number"]
                                                                      for s in per),
                         "distractors": sum(s["distractors"] for s in per),
                         "v1_conflict_notes": sum(s["v1_conflict_notes"] for s in per)},
            "per_clip": per}


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) == 4 and a[0] == "validate":
        lab = json.load(open(a[1]))
        reg = L1._load_registry(a[2])
        if lab.get("clip_id") not in reg:
            raise SystemExit("clip not in the registry (fail closed)")
        v1, sha, c = _v1(a[3], reg, lab["clip_id"])
        pr = validate(lab, v1, sha, c)
        print("VALID" if not pr else "INVALID\n  " + "\n  ".join(pr))
        sys.exit(1 if pr else 0)
    elif len(a) in (5, 6) and a[0] == "freeze":
        print(json.dumps(freeze(a[1], a[2], a[3], a[4], a[5] if len(a) == 6 else None), indent=1))
    elif len(a) == 4 and a[0] == "record":
        print(json.dumps(dataset_record(a[1], a[2], a[3]), indent=1, sort_keys=True))
    else:
        raise SystemExit("usage: r2_jersey_regions.py validate LAYER REGISTRY V1_DIR | "
                         "freeze LAYER REGISTRY V1_DIR OUT_DIR [REASON] | record V1_DIR V2_DIR REGISTRY")
