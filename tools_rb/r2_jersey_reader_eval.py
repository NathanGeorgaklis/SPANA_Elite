# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R2/JERSEY_DATASET/READER_EVAL_PREREGISTRATION.md (committed c207de2, fixed before any reader output);
#                 Nathan 2026-11-05 17:18 (run the committed plan exactly; jersey primary; shorts optional support only,
#                 never veto; no threshold changes after results; no tracker; no EVAL).
"""Jersey / shorts number reader evaluation on the frozen DEV jersey dataset v2. Evaluation tooling only: no tracker.

Three decoupled steps (the reader is replaceable and runs in its own environment):
    python3 tools_rb/r2_jersey_reader_eval.py units REGISTRY V1_DIR V2_DIR CLIPS_DIR RUN_DIR
        verifies every clip sha256 against the registry and the dataset v2 freeze records (fail closed), decodes the
        0.5 s keyframes, writes RUN_DIR/units.json and one crop per person-keyframe (human boxes only)
    <reader>  RUN_DIR  ->  RUN_DIR/raw_reads.json   {unit_id: [[text, conf, [x1, y1, x2, y2]], ...]}  (crop pixels)
        e.g. tools_rb/r2_jersey_ocr_easyocr.py (pinned EasyOCR, separate environment)
    python3 tools_rb/r2_jersey_reader_eval.py score RUN_DIR
        applies the pre-registered rules and writes RUN_DIR/results.json and RUN_DIR/RESULTS.md

Every rule below is the pre-registration's; section numbers refer to it.
"""
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from player_tracker_rb.config import TrackerConfig  # noqa: E402
from tools_rb import r2_jersey_labels as L1  # noqa: E402
from tools_rb import r2_jersey_regions as R  # noqa: E402

PREREG = "R2/JERSEY_DATASET/READER_EVAL_PREREGISTRATION.md"
# §3.3 region bands (fractions of the person crop; fixed before any reader output)
JERSEY_BAND = (0.10, 0.55)
SHORTS_BAND = (0.55, 0.80)
LR_SPLIT = 0.5
# §7 decision constants
Q1_PP = 10.0                 # percentage points
Q1_CLIPS = 2
Q2_MIN_TARGET_CLIPS = 10
Q2_MIN_COMP_CLIPS = 3
Q2_MIN_READINGS = 100
Q2_PRECISION = 0.95
Q2_WILSON_LB = 0.90
WILSON_Z = 1.959963984540054

CORRECT, WRONG, ABSTAIN, FALSE, OK_ABSTAIN = "CORRECT", "WRONG", "ABSTAIN", "FALSE_READING", "CORRECT_ABSTAIN"
CONFIRMED, CONFLICT, UNKNOWN = "CONFIRMED", "CONFLICT", "UNKNOWN"
SUPPORT, VETO = "SUPPORT", "VETO"


def cfg_values() -> Dict[str, Any]:
    c = TrackerConfig.default()
    return {"jersey_min_box_h_px": c.get("jersey_min_box_h_px"), "jersey_min_read_conf": c.get("jersey_min_read_conf"),
            "identity_number_min_reads": c.get("identity_number_min_reads"),
            "jersey_confirm_span_s": c.get("jersey_confirm_span_s"), "config_hash": c.config_hash()}


def _sha(p):
    return L1._sha(p)


# ----------------------------------------------------------------------------------------------- step 1: units
def build_units(registry_json, v1_dir, v2_dir, clips_dir, run_dir) -> Dict[str, Any]:
    import cv2                                                            # only needed for this step
    reg = L1._load_registry(registry_json)
    rec = R.dataset_record(v1_dir, v2_dir, registry_json)                 # fails closed on any tampering
    os.makedirs(os.path.join(run_dir, "crops"), exist_ok=False)
    cfg = cfg_values()
    units = []
    for cid, c in reg.items():
        clip = os.path.join(clips_dir, cid + ".mp4")
        if _sha(clip) != c["clip_sha256"]:
            raise SystemExit(f"{clip}: sha256 differs from the registry (fail closed)")
        v1 = json.load(open(L1.frozen_path(v1_dir, cid, 1)))
        vs = R._versions(v2_dir, cid)
        v2 = json.load(open(R.frozen_path(v2_dir, cid, vs[-1])))
        grid = L1.keyframe_grid(c["fps"], c["frames"])
        with tempfile.TemporaryDirectory() as td:
            sel = "+".join(f"eq(n\\,{g})" for g in grid)
            subprocess.run(["ffmpeg", "-v", "error", "-i", clip, "-vf", f"select='{sel}'", "-vsync", "vfr",
                            os.path.join(td, "k_%03d.png")], check=True)
            frames = {g: cv2.imread(os.path.join(td, f"k_{i + 1:03d}.png")) for i, g in enumerate(grid)}
        if any(im is None for im in frames.values()):
            raise SystemExit(f"{cid}: keyframe decode incomplete (fail closed)")
        persons = [("TARGET", "TARGET", v1["target"]["expected_number"], v2["target"]["shorts_number"],
                    {str(g): {"visibility": v1["keyframes"][str(g)]["visibility"], "box": v1["keyframes"][str(g)]["box"],
                              "regions": v2["target"]["regions"].get(str(g))} for g in grid
                     if str(g) in v2["target"]["regions"]})]
        for pid, p in v2["competitors"].items():
            persons.append((pid, p["relation"], p["jersey_number"], p["shorts_number"],
                            {k: x for k, x in p["keyframes"].items() if x.get("regions") and x.get("box")}))
        for pid, rel, jn, sn, kfs in persons:
            for k, x in sorted(kfs.items(), key=lambda kv: int(kv[0])):
                g = int(k)
                x1, y1, x2, y2 = [int(round(v)) for v in x["box"]]
                im = frames[g]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(im.shape[1], x2), min(im.shape[0], y2)
                uid = f"{cid}|{pid}|{g}"
                crop = os.path.join("crops", f"{cid}_{pid}_{g:06d}.png")
                cv2.imwrite(os.path.join(run_dir, crop), im[y1:y2, x1:x2])
                dis = [d["box"] for d in v2["distractors"] if d["keyframe"] == g]
                units.append({"unit_id": uid, "clip_id": cid, "person": pid, "relation": rel, "keyframe": g,
                              "t_s": g / c["fps"], "visibility": x["visibility"], "box": [x1, y1, x2, y2],
                              "crop": crop, "policy_abstain": (y2 - y1) < cfg["jersey_min_box_h_px"],
                              "jersey_number": jn, "shorts_number": sn, "expected_number": v1["target"]["expected_number"],
                              "truth": x["regions"], "distractor_boxes": dis, "source": c["source"],
                              "fps": c["fps"], "classes": c.get("classes", [])})
    out = {"schema": "r2_jersey_reader_units_v1", "prereg": PREREG, "dataset_v2_sha256": rec["dataset_sha256"],
           "registry_sha256": _sha(registry_json), "config": cfg, "units": units}
    with open(os.path.join(run_dir, "units.json"), "x") as f:
        json.dump(out, f, indent=1)
    return out


# ----------------------------------------------------------------------------------------------- step 3: scoring
def valid_number(text: str, conf: float, min_conf: float) -> Optional[int]:
    """§3.2: 1-2 digits, no leading zero, 1-99, confidence >= min_conf; anything else is discarded."""
    t = "".join(ch for ch in str(text) if not ch.isspace())
    if not t or not t.isdigit() or len(t) > 2 or t[0] == "0" or conf is None or float(conf) < min_conf:
        return None
    return int(t)


def assign_region(cx_n: float, cy_n: float) -> Optional[str]:
    """§3.3: JERSEY band / SHORTS_LEFT / SHORTS_RIGHT by the text-box centre normalised by the crop; else None."""
    if JERSEY_BAND[0] <= cy_n < JERSEY_BAND[1]:
        return "JERSEY"
    if SHORTS_BAND[0] <= cy_n < SHORTS_BAND[1]:
        return "SHORTS_LEFT" if cx_n < LR_SPLIT else "SHORTS_RIGHT"
    return None


def _digit_token(text: str, conf: float, min_conf: float) -> Optional[str]:
    """A confident all-digit OCR token (before the 1-2 digit / leading-zero validity rule), else None."""
    t = "".join(ch for ch in str(text) if not ch.isspace())
    if not t or not t.isdigit() or conf is None or float(conf) < min_conf:
        return None
    return t


def region_readings(unit: Dict[str, Any], raw: Sequence, min_conf: float) -> Dict[str, Any]:
    """Readings per region of one person-keyframe (§3.2-3.7).

    Ruling (recorded before the run, from §3.5's own text): separate text boxes are never merged, and "OCR splitting
    '10' into '1' and '0' gives a conflict and an abstention". So every confident all-digit token assigned to a region
    is kept, valid or not; a region holding more than one distinct token is a CONFLICT (abstains). A region holding
    one token reads it only if it passes the §3.2 validity rule; a lone invalid token ('0', '07', '123') is discarded."""
    x1, y1, x2, y2 = unit["box"]
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        raise SystemExit(f"{unit['unit_id']}: empty person box (fail closed)")
    per: Dict[str, set] = {"JERSEY": set(), "SHORTS_LEFT": set(), "SHORTS_RIGHT": set()}
    cnt = {"invalid": 0, "unassigned": 0, "distractor_hits": 0, "distractor_leaks": 0}
    accepted = []
    for text, conf, b in raw:
        tok = _digit_token(text, conf, min_conf)
        if tok is None:
            cnt["invalid"] += 1
            continue
        cx, cy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
        ix, iy = x1 + cx, y1 + cy
        if any(d[0] <= ix <= d[2] and d[1] <= iy <= d[3] for d in unit["distractor_boxes"]):
            cnt["distractor_hits"] += 1
            continue
        r = assign_region(cx / w, cy / h)
        if r is None:
            cnt["unassigned"] += 1
            continue
        per[r].add(tok)
        accepted.append((ix, iy))
    # independent check (§7 Q2): no accepted token centre lies in a distractor box
    cnt["distractor_leaks"] = sum(any(d[0] <= ix <= d[2] and d[1] <= iy <= d[3] for d in unit["distractor_boxes"])
                                  for ix, iy in accepted)
    out = {}
    for r, toks in per.items():
        n = valid_number(next(iter(toks)), 1.0, 0.0) if len(toks) == 1 else None
        if len(toks) == 1 and n is None:
            cnt["invalid"] += 1
        out[r] = {"number": n, "conflict": len(toks) > 1, "tokens": sorted(toks)}
    out["counts"] = cnt
    return out


def _unit_outcome(reading: Optional[int], truth_numbers: List[int]) -> str:
    if truth_numbers:
        if reading is None:
            return ABSTAIN
        return CORRECT if reading in truth_numbers else WRONG
    return FALSE if reading is not None else OK_ABSTAIN


def score_unit(unit: Dict[str, Any], rr: Dict[str, Any]) -> Dict[str, Any]:
    t = unit["truth"]
    jt = [t[r]["number"] for r in R.JERSEY if t[r]["state"] == "READABLE"]
    out = {"JERSEY": _unit_outcome(rr["JERSEY"]["number"], jt),
           "jersey_truth_side": [r for r in R.JERSEY if t[r]["state"] == "READABLE"],
           "jersey_truth_states": {r: t[r]["state"] for r in R.JERSEY}}
    for r in R.SHORTS:
        st = t[r]["state"]
        out[r] = _unit_outcome(rr[r]["number"], [t[r]["number"]] if st == "READABLE" else [])
        out[r + "_truth_state"] = st
    outs = [out["JERSEY"], out["SHORTS_LEFT"], out["SHORTS_RIGHT"]]
    readable = bool(jt) or any(t[r]["state"] == "READABLE" for r in R.SHORTS)
    if any(o in (WRONG, FALSE) for o in outs):
        comb = WRONG
    elif any(o == CORRECT for o in outs):
        comb = CORRECT
    else:
        comb = ABSTAIN if readable else OK_ABSTAIN
    out["COMBINED"] = comb
    out["readable_any"] = readable
    # the S pool's single reading for this keyframe (§6: L + R form one pool; one reading per pool per keyframe)
    st_nums = [t[r]["number"] for r in R.SHORTS if t[r]["state"] == "READABLE"]
    nums = {rr[r]["number"] for r in R.SHORTS if rr[r]["number"] is not None}
    s_read = None if (rr["SHORTS_LEFT"]["conflict"] or rr["SHORTS_RIGHT"]["conflict"] or len(nums) != 1) else next(iter(nums))
    out["SHORTS_POOL"] = _unit_outcome(s_read, st_nums)
    out["v1_conflict_note"] = any(str(t[r].get("note", "")).startswith(R.CONFLICT_TAG) for r in R.REGIONS)
    return out


def consensus(entries: List[Tuple[float, Any]], min_reads: int, span_s: float) -> Dict[str, Any]:
    """§6: entries = (t, number) or (t, 'CONFLICT') per keyframe (at most one per keyframe per pool)."""
    if any(v == CONFLICT for _, v in entries) or len({v for _, v in entries}) > 1:
        return {"status": CONFLICT, "number": None, "reads": len(entries)}
    if len(entries) >= min_reads and L1._shirt(entries[0][1]):
        ts = sorted(t for t, _ in entries)
        if ts[-1] - ts[0] >= span_s - 1e-9:
            return {"status": CONFIRMED, "number": entries[0][1], "reads": len(entries)}
    return {"status": UNKNOWN, "number": None, "reads": len(entries)}


def pool_entries(per_kf: List[Dict[str, Any]], pool: str) -> List[Tuple[float, Any]]:
    out = []
    for u in per_kf:
        rr = u["rr"]
        if pool == "J":
            if rr["JERSEY"]["conflict"]:
                out.append((u["t_s"], CONFLICT))
            elif rr["JERSEY"]["number"] is not None:
                out.append((u["t_s"], rr["JERSEY"]["number"]))
        else:
            if rr["SHORTS_LEFT"]["conflict"] or rr["SHORTS_RIGHT"]["conflict"]:
                out.append((u["t_s"], CONFLICT))
                continue
            s = {rr[r]["number"] for r in R.SHORTS if rr[r]["number"] is not None}
            if len(s) > 1:
                out.append((u["t_s"], CONFLICT))
            elif s:
                out.append((u["t_s"], s.pop()))
    return out


def evidence(j: Dict[str, Any], s: Dict[str, Any], e: Optional[int], mode: str) -> str:
    """§6 table. mode: combined / jersey / shorts. Shorts never VETO; absent shorts never count against identity."""
    if e is None:
        return UNKNOWN
    if mode in ("combined", "jersey"):
        if j["status"] == CONFIRMED:
            return SUPPORT if j["number"] == e else VETO
        if j["status"] == CONFLICT or mode == "jersey":
            return UNKNOWN
    if s["status"] == CONFIRMED and s["number"] == e:
        return SUPPORT
    return UNKNOWN


def wilson(k: int, n: int) -> Tuple[Optional[float], Optional[float]]:
    if n == 0:
        return None, None
    p, z = k / n, WILSON_Z
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - m, c + m


def _tally(outcomes: List[str]) -> Dict[str, Any]:
    t = {k: outcomes.count(k) for k in (CORRECT, WRONG, ABSTAIN, FALSE, OK_ABSTAIN)}
    readable = t[CORRECT] + t[WRONG] + t[ABSTAIN]
    unread = t[FALSE] + t[OK_ABSTAIN]
    emitted = t[CORRECT] + t[WRONG] + t[FALSE]
    t.update(truth_readable=readable, truth_unreadable=unread, emitted=emitted,
             correct_rate=(t[CORRECT] / readable) if readable else None,
             wrong_rate=(t[WRONG] / readable) if readable else None,
             abstain_rate=(t[ABSTAIN] / readable) if readable else None,
             false_reading_rate=(t[FALSE] / unread) if unread else None,
             precision=(t[CORRECT] / emitted) if emitted else None,
             precision_wilson95=wilson(t[CORRECT], emitted), false_rate_wilson95=wilson(t[FALSE], unread))
    return t


def _tables(S: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"JERSEY (shirt-only)": _tally([u["out"]["JERSEY"] for u in S]),
            "JERSEY_BACK readable": _tally([u["out"]["JERSEY"] for u in S if "JERSEY_BACK" in u["out"]["jersey_truth_side"]]),
            "JERSEY_FRONT readable": _tally([u["out"]["JERSEY"] for u in S if "JERSEY_FRONT" in u["out"]["jersey_truth_side"]]),
            "SHORTS_LEFT": _tally([u["out"]["SHORTS_LEFT"] for u in S]),
            "SHORTS_RIGHT": _tally([u["out"]["SHORTS_RIGHT"] for u in S]),
            "SHORTS (shorts-only pool reading)": _tally([u["out"]["SHORTS_POOL"] for u in S]),
            "COMBINED (person-keyframe)": _tally([u["out"]["COMBINED"] for u in S]),
            "n_units": len(S)}


def _condition(u):
    night = "NIGHT_FLOODLIT" in u.get("classes", [])
    return f"{'night' if night else 'day'} {int(u.get('fps', 0))}fps"


def score(run_dir: str) -> Dict[str, Any]:
    U = json.load(open(os.path.join(run_dir, "units.json")))
    raw = json.load(open(os.path.join(run_dir, "raw_reads.json")))
    cfg = U["config"]
    units = U["units"]
    scored, policy = [], []
    for u in units:
        if u["policy_abstain"]:
            policy.append(u["unit_id"])
            continue
        if u["unit_id"] not in raw:
            raise SystemExit(f"{u['unit_id']}: no reader output (fail closed)")
        rr = region_readings(u, raw[u["unit_id"]]["results"], cfg["jersey_min_read_conf"])
        scored.append(dict(u, rr=rr, out=score_unit(u, rr)))
    is_t = lambda u: u["person"] == "TARGET"
    tables = {"all": _tables(scored), "selected": _tables([u for u in scored if is_t(u)]),
              "competitors": _tables([u for u in scored if not is_t(u)]),
              "selected VISIBLE": _tables([u for u in scored if is_t(u) and u["visibility"] == "VISIBLE"]),
              "selected PARTIAL": _tables([u for u in scored if is_t(u) and u["visibility"] == "PARTIAL"]),
              "competitors VISIBLE": _tables([u for u in scored if not is_t(u) and u["visibility"] == "VISIBLE"]),
              "competitors PARTIAL": _tables([u for u in scored if not is_t(u) and u["visibility"] == "PARTIAL"])}
    for cond in sorted({_condition(u) for u in scored}):
        tables["condition " + cond] = _tables([u for u in scored if _condition(u) == cond])
    per_clip = {cid: _tables([u for u in scored if u["clip_id"] == cid]) for cid in sorted({u["clip_id"] for u in units})}
    false_by_state, by_reason = {}, {}
    for u in scored:
        if u["out"]["JERSEY"] == FALSE:
            k = "JERSEY back=" + u["out"]["jersey_truth_states"]["JERSEY_BACK"] + " front=" + \
                u["out"]["jersey_truth_states"]["JERSEY_FRONT"]
            false_by_state[k] = false_by_state.get(k, 0) + 1
        for r in R.SHORTS:
            if u["out"][r] == FALSE:
                k = f"{r}={u['out'][r + '_truth_state']}"
                false_by_state[k] = false_by_state.get(k, 0) + 1
        for r in R.REGIONS:                                 # §5 breakdown by unreadable reason
            x = u["truth"][r]
            if x["state"] == "UNREADABLE":
                o = u["out"]["JERSEY"] if r in R.JERSEY else u["out"][r]
                d = by_reason.setdefault(f"{r}:{x['reason']}", {"units": 0, FALSE: 0})
                d["units"] += 1
                d[FALSE] += o == FALSE
    counts = {k: sum(u["rr"]["counts"][k] for u in scored)
              for k in ("invalid", "unassigned", "distractor_hits", "distractor_leaks")}
    counts["region_conflicts"] = sum(u["rr"][r]["conflict"] for u in scored for r in ("JERSEY",) + R.SHORTS)
    counts["policy_abstain_units"] = len(policy)
    v1_notes = [u["unit_id"] for u in scored if u["out"]["v1_conflict_note"]]
    # combined pool readings: at most one J and one S reading per keyframe (§6 counting; review finding 2)
    reading_t = _tally([u["out"][k] for u in scored for k in ("JERSEY", "SHORTS_POOL")])
    # §6 temporal, per person per clip
    persons: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for u in scored:
        persons.setdefault((u["clip_id"], u["person"]), []).append(u)
    allpersons = {}
    for u in units:
        allpersons.setdefault((u["clip_id"], u["person"]), u)
    P = []
    for key, u0 in sorted(allpersons.items()):
        kfs = sorted(persons.get(key, []), key=lambda u: u["keyframe"])
        j = consensus(pool_entries(kfs, "J"), cfg["identity_number_min_reads"], cfg["jersey_confirm_span_s"])
        sp = consensus(pool_entries(kfs, "S"), cfg["identity_number_min_reads"], cfg["jersey_confirm_span_s"])
        e, jn, sn = u0["expected_number"], u0["jersey_number"], u0["shorts_number"]
        p = {"clip_id": key[0], "person": key[1], "relation": u0["relation"], "jersey_number": jn, "shorts_number": sn,
             "expected_number": e, "J": j, "S": sp,
             "evidence_combined": evidence(j, sp, e, "combined"), "evidence_jersey": evidence(j, sp, e, "jersey"),
             "evidence_shorts": evidence(j, sp, e, "shorts"),
             "wrong_confirmed_J": j["status"] == CONFIRMED and j["number"] != jn,
             "wrong_confirmed_S": sp["status"] == CONFIRMED and sp["number"] != sn,
             "shorts_disagrees": j["status"] == CONFIRMED and sp["status"] == CONFIRMED and j["number"] != sp["number"],
             "readable_truth": any(any(x["truth"][r]["state"] == "READABLE" for r in R.REGIONS)
                                   for x in units if (x["clip_id"], x["person"]) == key),
             "shares_target_shirt": key[1] != "TARGET" and e is not None and jn == e,
             "shares_target_shorts": key[1] != "TARGET" and e is not None and sn == e}
        p["shares_target_number"] = p["shares_target_shirt"] or p["shares_target_shorts"]
        if key[1] == "TARGET":
            p["role"] = "SELECTED"
            p["false_veto"] = p["evidence_combined"] == VETO
            p["established_jersey"] = p["evidence_jersey"] == SUPPORT
            p["established_combined"] = p["evidence_combined"] == SUPPORT
        else:
            p["role"] = "COMPETITOR"
            p["false_support"] = (not p["shares_target_number"]) and SUPPORT in (p["evidence_combined"],
                                                                                p["evidence_shorts"])
            p["established_jersey"] = j["status"] == CONFIRMED and j["number"] == jn
            p["established_combined"] = p["established_jersey"] or (j["status"] == UNKNOWN and sp["status"] == CONFIRMED
                                                                   and sp["number"] == sn)
        P.append(p)
    sel = [p for p in P if p["role"] == "SELECTED"]
    comp = [p for p in P if p["role"] == "COMPETITOR"]
    cnt3 = lambda pp, key: {k: sum(p[key] == k for p in pp) for k in (SUPPORT, VETO, UNKNOWN)}
    sep = {"selected": cnt3(sel, "evidence_combined"), "competitors": cnt3(comp, "evidence_combined"),
           "selected_jersey_only": cnt3(sel, "evidence_jersey"), "selected_shorts_only": cnt3(sel, "evidence_shorts"),
           "competitors_jersey_only": cnt3(comp, "evidence_jersey"),
           "competitors_shorts_only": cnt3(comp, "evidence_shorts"),
           "false_support": [f"{p['clip_id']}|{p['person']}" for p in comp if p["false_support"]],
           "false_veto": [f"{p['clip_id']}|{p['person']}" for p in sel if p["false_veto"]],
           "wrong_confirmed": [f"{p['clip_id']}|{p['person']}|{pool}" for p in P for pool in ("J", "S")
                               if p[f"wrong_confirmed_{pool}"]],
           "shorts_disagrees": [f"{p['clip_id']}|{p['person']}" for p in P if p["shorts_disagrees"]],
           "shares_target_shirt": [f"{p['clip_id']}|{p['person']}" for p in comp if p["shares_target_shirt"]],
           "shares_target_shorts": [f"{p['clip_id']}|{p['person']}" for p in comp if p["shares_target_shorts"]],
           "temporal": {pool: {k: sum(p[pool]["status"] == k for p in P) for k in (CONFIRMED, CONFLICT, UNKNOWN)}
                        for pool in ("J", "S")},
           "temporal_correct_confirmed": {"J": sum(p["J"]["status"] == CONFIRMED and not p["wrong_confirmed_J"] for p in P),
                                          "S": sum(p["S"]["status"] == CONFIRMED and not p["wrong_confirmed_S"] for p in P)}}
    # §7 Q0 (truth only), region-keyframe counts and per clip
    def q0_of(T):
        rs = {r: {st: sum(u["truth"][r]["state"] == st for u in T) for st in R.STATES} for r in R.REGIONS}
        return {"person_keyframes": len(T), "region_states": rs,
                "keyframes_shorts_readable": sum(any(u["truth"][r]["state"] == "READABLE" for r in R.SHORTS) for u in T),
                "keyframes_shorts_only": sum(any(u["truth"][r]["state"] == "READABLE" for r in R.SHORTS)
                                             and not any(u["truth"][r]["state"] == "READABLE" for r in R.JERSEY) for u in T)}
    q0 = q0_of(units)
    q0["persons_with_shorts_number"] = sum(p["shorts_number"] is not None for p in P)
    q0["per_clip"] = {cid: q0_of([u for u in units if u["clip_id"] == cid]) for cid in sorted({u["clip_id"] for u in units})}
    # §7 Q1: coverage = person-keyframes with at least one CORRECT reading (review finding 4)
    def q1_over(pp):
        keys = {(p["clip_id"], p["person"]) for p in pp}
        S = [u for u in scored if (u["clip_id"], u["person"]) in keys]
        n = len(S)
        cj = sum(u["out"]["JERSEY"] == CORRECT for u in S)
        cc = sum(any(u["out"][r] == CORRECT for r in ("JERSEY",) + R.SHORTS) for u in S)
        ej = sum(p["established_jersey"] for p in pp)
        ec = sum(p["established_combined"] for p in pp)
        return {"persons": len(pp), "person_keyframes": n, "jersey_correct_cov": (cj / n) if n else None,
                "combined_correct_cov": (cc / n) if n else None, "gain_pp": (100.0 * (cc - cj) / n) if n else None,
                "persons_established_jersey": ej, "persons_established_combined": ec, "extra_persons": ec - ej}
    with_sn = [p for p in P if p["shorts_number"] is not None]
    q1a, q1_all = q1_over(with_sn), q1_over(P)
    new_bad = any(p["wrong_confirmed_S"] for p in P) or any(           # review finding 3: any wrong S is new
        p["evidence_combined"] != p["evidence_jersey"] and (
            (p["role"] == "COMPETITOR" and p["evidence_combined"] == SUPPORT and not p["shares_target_number"])
            or (p["role"] == "SELECTED" and p["evidence_combined"] == VETO)) for p in P)
    q1_yes = bool(with_sn) and q1a["gain_pp"] is not None and \
        (q1a["gain_pp"] >= Q1_PP or q1a["extra_persons"] >= Q1_CLIPS) and not new_bad
    # §7 Q2 (combined evidence only)
    n_sel_decided = sum(p["evidence_combined"] != UNKNOWN for p in sel)
    n_comp_readable = sum(p["readable_truth"] for p in comp)
    lb = reading_t["precision_wilson95"][0]
    minimum = n_sel_decided >= Q2_MIN_TARGET_CLIPS and n_comp_readable >= Q2_MIN_COMP_CLIPS and \
        reading_t["emitted"] >= Q2_MIN_READINGS
    conds = {"false_support_0": not sep["false_support"],
             "shorts_origin_veto_0": all(p["evidence_shorts"] != VETO for p in P),
             "wrong_confirmed_0": not sep["wrong_confirmed"], "false_veto_0": not sep["false_veto"],
             "distractor_readings_discarded": counts["distractor_leaks"] == 0,
             "precision_ge_0.95": reading_t["precision"] is not None and reading_t["precision"] >= Q2_PRECISION,
             "wilson_lb_ge_0.90": lb is not None and lb >= Q2_WILSON_LB}
    q2 = {"minimum_evidence": {"selected_decided": n_sel_decided, "need_selected": Q2_MIN_TARGET_CLIPS,
                               "competitor_readable": n_comp_readable, "need_competitor": Q2_MIN_COMP_CLIPS,
                               "pool_readings_emitted": reading_t["emitted"], "need_readings": Q2_MIN_READINGS,
                               "met": minimum},
          "conditions": conds,
          "answer": "YES" if minimum and all(conds.values()) else ("NO (insufficient evidence)" if not minimum else "NO")}
    res = {"schema": "r2_jersey_reader_results_v1", "prereg": PREREG, "dataset_v2_sha256": U["dataset_v2_sha256"],
           "registry_sha256": U.get("registry_sha256"), "config": cfg, "reader": raw.get("__reader__"),
           "ruling_3_5": "split digit tokens in one region = conflict (abstain); see region_readings docstring",
           "units_total": len(units), "units_scored": len(scored), "counts": counts, "tables": tables,
           "per_clip": per_clip, "false_readings_by_truth_state": false_by_state,
           "unreadable_reason_breakdown": by_reason, "v1_conflict_note_units": v1_notes,
           "readings_combined_pool": reading_t, "separation": sep, "persons": P, "Q0": q0,
           "Q1_with_shorts_number": q1a, "Q1_all": q1_all, "Q1_new_unsafe": new_bad,
           "Q1_answer": "YES" if q1_yes else "NO", "Q2": q2,
           "errors": [{"unit_id": u["unit_id"], "crop": u["crop"], "region": r, "outcome": u["out"][r],
                       "read": u["rr"][r]["number"], "tokens": u["rr"][r]["tokens"]}
                      for u in scored for r in ("JERSEY",) + R.SHORTS if u["out"][r] in (WRONG, FALSE)],
           "per_unit": [{"unit_id": u["unit_id"], "out": u["out"],
                         "rr": {r: u["rr"][r] for r in ("JERSEY",) + R.SHORTS}} for u in scored],
           "policy_abstain_units": policy}
    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump(res, f, indent=1, default=str)
    with open(os.path.join(run_dir, "RESULTS.md"), "w") as f:
        f.write(results_md(res))
    return res


def _pct(x):
    return "n/a" if x is None else f"{100.0 * x:.1f}%"


def results_md(r: Dict[str, Any]) -> str:
    L = [f"# Jersey reader evaluation results (pre-registration `{r['prereg']}`)", "",
         f"Dataset v2 `{r['dataset_v2_sha256']}`; reader: {(r.get('reader') or {}).get('name')}; "
         f"units scored {r['units_scored']} of {r['units_total']} (policy abstentions "
         f"{r['counts']['policy_abstain_units']}).", "", "## Per-unit outcomes (all persons)", "",
         "| Table | correct | wrong | abstain | false | correct-abstain | precision | Wilson95 | false-rate |",
         "|---|---|---|---|---|---|---|---|---|"]
    for name, t in r["tables"]["all"].items():
        if name == "n_units":
            continue
        w = t["precision_wilson95"]
        L.append(f"| {name} | {t['CORRECT']} | {t['WRONG']} | {t['ABSTAIN']} | {t['FALSE_READING']} | "
                 f"{t['CORRECT_ABSTAIN']} | {_pct(t['precision'])} | "
                 f"{'n/a' if w[0] is None else f'{100*w[0]:.1f}-{100*w[1]:.1f}%'} | {_pct(t['false_reading_rate'])} |")
    rp = r["readings_combined_pool"]
    s = r["separation"]
    L += ["", f"Combined-pool readings emitted {rp['emitted']}: precision {_pct(rp['precision'])}, "
              f"Wilson95 lower {_pct(rp['precision_wilson95'][0])}.", "",
          "## Identity evidence (combined; jersey primary, shorts support only)", "",
          f"- Selected players: {s['selected']}; jersey-only {s['selected_jersey_only']}; shorts-only {s['selected_shorts_only']}",
          f"- Competitors: {s['competitors']}",
          f"- FALSE SUPPORT: {s['false_support'] or 'none'}; false VETO: {s['false_veto'] or 'none'}; "
          f"wrong CONFIRMED: {s['wrong_confirmed'] or 'none'}",
          f"- Temporal: J {s['temporal']['J']}, S {s['temporal']['S']}", "",
          f"## Q1 (shorts useful when present): {r['Q1_answer']}", "", f"{r['Q1_with_shorts_number']}", "",
          f"## Q2 (safe for identity support): {r['Q2']['answer']}", "", f"{r['Q2']}", ""]
    return "\n".join(L) + "\n"


def run_record(run_dir: str, repo: str, argv: List[str]) -> Dict[str, Any]:
    """§8 reproducibility record + SHA256SUMS of every run file."""
    rev = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", repo, "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
    sums = []
    for root, _, files in os.walk(run_dir):
        for n in sorted(files):
            pth = os.path.join(root, n)
            rel = os.path.relpath(pth, run_dir)
            if rel in ("SHA256SUMS.txt", "run_record.json"):
                continue
            sums.append(f"{_sha(pth)}  {rel}")
    with open(os.path.join(run_dir, "SHA256SUMS.txt"), "w") as f:
        f.write("\n".join(sorted(sums, key=lambda x: x.split("  ", 1)[1])) + "\n")
    rec = {"git_commit": rev, "git_dirty": bool(dirty), "command": argv,
           "results_sha256": _sha(os.path.join(run_dir, "results.json")),
           "raw_reads_sha256": _sha(os.path.join(run_dir, "raw_reads.json")),
           "units_sha256": _sha(os.path.join(run_dir, "units.json")),
           "scorer_pip_freeze": subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True,
                                               text=True).stdout.splitlines()}
    with open(os.path.join(run_dir, "run_record.json"), "w") as f:
        json.dump(rec, f, indent=1)
    return rec


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) == 6 and a[0] == "units":
        u = build_units(*a[1:])
        print("UNITS", len(u["units"]), "dataset", u["dataset_v2_sha256"])
    elif len(a) == 2 and a[0] == "score":
        r = score(a[1])
        run_record(a[1], os.path.dirname(os.path.dirname(os.path.abspath(__file__))), sys.argv)
        print(json.dumps({"Q1": r["Q1_answer"], "Q2": r["Q2"]["answer"]}))
    else:
        raise SystemExit("usage: r2_jersey_reader_eval.py units REGISTRY V1_DIR V2_DIR CLIPS_DIR RUN_DIR | score RUN_DIR")
