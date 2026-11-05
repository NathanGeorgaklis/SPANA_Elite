# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R2 DEV real-user reacquisition protocol (Nathan, 2026-11-06 00:39): test tap-based re-seeding on fresh
#                 non-EVAL clips with realistic user taps (centre, shoulder / edge, nearby same-kit, crowd / overlap,
#                 wrong tap, target not visible); measure accepted / rejected taps, wrong accepted selections,
#                 checkpoints per clip, coverage gain, safety, usability problems. DEV only; never EVAL.
"""Tap-protocol runs on the FRESH jersey-dataset clips (15 DEV clips, never used for tracker development, with frozen
human truth on a 0.5 s keyframe grid: R2/JERSEY_DATASET + R2/labels/jersey_dataset{,_v2}). Evaluation tooling only.

    python3 tools_rb/r2_tap_dev.py run REPO CLIPS_DIR OUT_ROOT PROTOCOL [CLIP_ID ...]
    python3 tools_rb/r2_tap_dev.py score REPO OUT_ROOT OUT_JSON

PROTOCOL (the simulated user answers ONLY on keyframes, i.e. the product pauses on the checkpoint and the user taps
the paused frame; here the first keyframe at or after the checkpoint opens, <= 0.5 s later):
  SILENT    never answers: identical to the automatic tracker, but records the checkpoints (the baseline);
  CENTRE    taps the centre of the human target box when the target is VISIBLE / PARTIAL and certain, else NOT_VISIBLE;
  SHOULDER  taps the upper-left shoulder region of the human box (25 % in from the left, 22 % down);
  JITTER    taps the centre with a deterministic Gaussian finger error (sigma 15 % of box width / height);
  WRONG     taps a nearby OTHER player: the labelled same-kit competitor's box centre when one is boxed at that
            keyframe, else the nearest detection not on the target (IoU < 0.3 with the human box); when the target
            is not visible the user (correctly) answers NOT_VISIBLE;
  WRONG_HIDDEN  the most dangerous user error: while the target is NOT visible, taps the labelled same-kit
            competitor (else the detection nearest the target's last human box); while visible, taps correctly
            (centre). No geometry can make such a tap ambiguous, so every acceptance is a wrong identity.
The seed is the first keyframe with a certain, boxed, visible target. The tracker never reads the labels; the only
label information that reaches it is the tap point a user would give.

score: on the keyframe grid, per clip and protocol: trusted-correct / trusted-wrong / abstained keyframes (IoU >= 0.5
with the human box), coverage, checkpoints, taps accepted / refused by reason, wrong accepted selections (accepted
box IoU < 0.5 with the human box, or target not visible), and the automatic (SILENT) coverage alongside.
"""
import glob
import hashlib
import json
import os
import random
import sys
import time

PROTOCOLS = ("SILENT", "CENTRE", "SHOULDER", "JITTER", "WRONG", "WRONG_HIDDEN")
VIS_OK = ("VISIBLE", "PARTIAL")
IOU_OK = 0.5
WRONG_MAX_IOU = 0.3
SHOULDER = (0.25, 0.22)
JITTER_SIGMA = 0.15


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def fixtures(repo, only=None):
    """(clip record, frozen v1, frozen v2 layer, seed keyframe) for every jersey-dataset clip (DEV only, verified)."""
    sys.path.insert(0, repo)
    from tools_rb import r2_jersey_labels as L1
    from tools_rb import r2_jersey_regions as R
    reg_path = os.path.join(repo, "R2", "JERSEY_DATASET", "JERSEY_CLIP_REGISTRY.json")
    v1_dir, v2_dir = os.path.join(repo, "R2", "labels", "jersey_dataset"), os.path.join(repo, "R2", "labels", "jersey_dataset_v2")
    R.dataset_record(v1_dir, v2_dir, reg_path)                       # fails closed on any tampering
    out = []
    for cid, c in L1._load_registry(reg_path).items():
        if only and cid not in only:
            continue
        v1, _, _ = R._v1(v1_dir, L1._load_registry(reg_path), cid)
        v2 = json.load(open(R.frozen_path(v2_dir, cid, R._versions(v2_dir, cid)[-1])))
        grid = L1.keyframe_grid(c["fps"], c["frames"])
        seed = next((g for g in grid if v1["keyframes"][str(g)]["visibility"] in VIS_OK
                     and not v1["keyframes"][str(g)]["human_uncertain"] and v1["keyframes"][str(g)]["box"]), None)
        if seed is None:
            continue
        out.append((c, v1, v2, grid, seed))
    return out


def target_truth(v1, cf):
    x = v1["keyframes"].get(str(cf))
    if x is None:
        return None
    return {"visible": x["visibility"] in VIS_OK and not x["human_uncertain"] and x["box"] is not None,
            "uncertain": bool(x["human_uncertain"]), "box": x["box"], "visibility": x["visibility"]}


def competitor_boxes(v2, cf, relation="SAME_TEAM"):
    out = []
    for pid, c in v2.get("competitors", {}).items():
        if c.get("relation") != relation:
            continue
        x = c["keyframes"].get(str(cf))
        if x and x.get("box") and x["visibility"] in VIS_OK and not x.get("human_uncertain"):
            out.append((pid, x["box"]))
    return out


def make_user(protocol, cid, v1, v2, start):
    def resolve(view, res):
        cf = start + view["frame"]
        tr = target_truth(v1, cf)
        if tr is None or protocol == "SILENT":
            return None                                  # not a keyframe: the user has not answered yet
        if not tr["visible"]:
            if protocol != "WRONG_HIDDEN":
                return {"decision": "NOT_VISIBLE"}
            comps = competitor_boxes(v2, cf)
            if comps:
                cb = comps[0][1]
                return {"decision": "TAP", "point": [(cb[0] + cb[2]) / 2.0, (cb[1] + cb[3]) / 2.0]}
            last = next((v1["keyframes"][str(g)]["box"] for g in range(cf, -1, -1)
                         if str(g) in v1["keyframes"] and v1["keyframes"][str(g)]["box"]), None)
            if last is None or not view["candidates"]:
                return {"decision": "NOT_VISIBLE"}
            cx, cy = (last[0] + last[2]) / 2, (last[1] + last[3]) / 2
            o = min(view["candidates"], key=lambda c: ((c["box"][0] + c["box"][2]) / 2 - cx) ** 2
                    + ((c["box"][1] + c["box"][3]) / 2 - cy) ** 2)["box"]
            return {"decision": "TAP", "point": [(o[0] + o[2]) / 2.0, (o[1] + o[3]) / 2.0]}
        b = tr["box"]
        w, h = b[2] - b[0], b[3] - b[1]
        if protocol in ("CENTRE", "WRONG_HIDDEN"):
            pt = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
        elif protocol == "SHOULDER":
            pt = (b[0] + SHOULDER[0] * w, b[1] + SHOULDER[1] * h)
        elif protocol == "JITTER":
            rng = random.Random(f"{cid}:{cf}")
            pt = ((b[0] + b[2]) / 2 + rng.gauss(0.0, JITTER_SIGMA * w), (b[1] + b[3]) / 2 + rng.gauss(0.0, JITTER_SIGMA * h))
        elif protocol == "WRONG":
            comps = competitor_boxes(v2, cf)
            if comps:
                cb = comps[0][1]
                pt = ((cb[0] + cb[2]) / 2, (cb[1] + cb[3]) / 2)
            else:
                others = [c for c in view["candidates"] if _iou(c["box"], b) < WRONG_MAX_IOU]
                if not others:
                    return {"decision": "NOT_VISIBLE"}
                cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                o = min(others, key=lambda c: ((c["box"][0] + c["box"][2]) / 2 - cx) ** 2
                        + ((c["box"][1] + c["box"][3]) / 2 - cy) ** 2)["box"]
                pt = ((o[0] + o[2]) / 2, (o[1] + o[3]) / 2)
        else:
            raise SystemExit(f"unknown protocol {protocol}")
        return {"decision": "TAP", "point": [float(pt[0]), float(pt[1])]}
    return resolve


def run(repo, clips_dir, out_root, protocol, only=None):
    if protocol not in PROTOCOLS:
        raise SystemExit(f"protocol must be one of {PROTOCOLS}")
    sys.path.insert(0, repo)
    from player_tracker_rb import provenance as PV
    from player_tracker_rb import run_rb as R
    head, dirty = PV.git_state()
    if dirty:
        raise SystemExit("git tree is dirty: the DEV run must use a clean commit (fail closed)")
    root = os.path.join(out_root, protocol)
    os.makedirs(root, exist_ok=True)
    for c, v1, v2, grid, seed in fixtures(repo, only):
        cid = c["clip_id"]
        od = os.path.join(root, cid)
        if os.path.exists(os.path.join(od, "DONE.json")):
            print(protocol, cid, "already done", flush=True)
            continue
        os.makedirs(od, exist_ok=True)
        video = os.path.join(clips_dir, cid + ".mp4")
        t0 = time.time()
        print(protocol, cid, "start", time.strftime("%H:%M:%S"), flush=True)
        try:
            R.run(video, seed, v1["keyframes"][str(seed)]["box"], out_dir=od, debug=False,
                  expected_sha256=c["clip_sha256"], checkpoint_resolver=make_user(protocol, cid, v1, v2, seed))
            done = {"clip_id": cid, "split": "DEV", "outcome": "COMPLETED", "protocol": protocol, "seed_frame": seed,
                    "git_commit": head, "seconds": round(time.time() - t0, 1)}
        except R.RunRefusedError as e:
            done = {"clip_id": cid, "split": "DEV", "outcome": "RUN_REFUSED", "reason": str(e), "protocol": protocol,
                    "seed_frame": seed, "git_commit": head, "seconds": round(time.time() - t0, 1)}
        with open(os.path.join(od, "DONE.json"), "x") as f:
            json.dump(done, f, indent=1)
        print(protocol, cid, done["outcome"], done["seconds"], "s", flush=True)


def _reason_class(reason):
    r = reason or ""
    for k in ("TAP_OFF_PLAYER", "TAP_AMBIGUOUS", "TAP_MERGED"):
        if r.startswith(k):
            return k
    if r.startswith("ambiguous selection"):
        return "PRECHECK_AMBIGUOUS"
    if "deferred anchor" in r:
        return "DEFERRED_ANCHOR"
    if r.startswith("seed validation"):
        return "SEED_VALIDATION"
    return "OTHER"


def score_clip(repo, root, c, v1, grid, seed):
    sys.path.insert(0, repo)
    from player_tracker_rb import run_rb as R
    cid = c["clip_id"]
    od = os.path.join(root, cid)
    done = json.load(open(os.path.join(od, "DONE.json")))
    if done["outcome"] != "COMPLETED":
        return {"outcome": done["outcome"], "reason": done.get("reason")}
    paths = R.output_paths(od, cid + ".mp4")
    doc = json.load(open(paths["results"]))
    sel = {}
    for fr in doc["frame_log"]:
        t = [x for x in fr["tracks"] if x.get("selected_player")][0]
        sel[fr["frame"]] = (bool(t["v2"]["identity_trusted"]), t["bbox"])
    n_vis = correct = wrong = abstain = unc = 0
    wrong_frames = []
    for g in grid:
        if g <= seed:
            continue
        tr = target_truth(v1, g)
        rel = g - seed
        if tr is None or rel not in sel:
            continue
        trusted, box = sel[rel]
        if tr["uncertain"]:
            unc += 1
            continue
        if tr["visible"]:
            n_vis += 1
            if trusted and _iou(box, tr["box"]) >= IOU_OK:
                correct += 1
            elif trusted:
                wrong += 1
                wrong_frames.append(g)
            else:
                abstain += 1
        elif trusted:
            wrong += 1
            wrong_frames.append(g)
    cps = json.load(open(paths["checkpoints"]))["checkpoints"] if os.path.exists(paths["checkpoints"]) else []
    taps = {"answered": 0, "accepted": 0, "refused": 0, "declined": 0, "by_reason": {}}
    wrong_accepted = []
    for cp in cps:
        for a in cp["answers"]:
            if a["decision"] == "NOT_VISIBLE":
                taps["declined"] += 1
                continue
            taps["answered"] += 1
            if a["outcome"] == "ACCEPTED":
                taps["accepted"] += 1
                tr = target_truth(v1, seed + a["frame"])
                ok = tr is not None and tr["visible"] and _iou(a["box"], tr["box"]) >= IOU_OK
                if not ok:
                    wrong_accepted.append({"frame": a["frame"], "clip_frame": seed + a["frame"], "box": a["box"],
                                           "truth": None if tr is None else tr["visibility"]})
            else:
                taps["refused"] += 1
                k = _reason_class(a.get("reason"))
                taps["by_reason"][k] = taps["by_reason"].get(k, 0) + 1
    st = [cp["status"] for cp in cps]
    return {"outcome": "SCORED", "seed_frame": seed, "keyframes_visible": n_vis, "trusted_correct": correct,
            "trusted_wrong": wrong, "wrong_keyframes": wrong_frames, "abstained": abstain, "uncertain_skipped": unc,
            "coverage": round(correct / n_vis, 3) if n_vis else None,
            "checkpoints": {"opened": len(cps), "reseeded": st.count("RESEEDED"), "declined": st.count("DECLINED"),
                            "auto_resolved": st.count("AUTO_RESOLVED"), "open_at_end": st.count("OPEN")},
            "taps": taps, "wrong_accepted_selections": wrong_accepted,
            "events": len(doc["summary"]["v2"]["reacquisition_events"]), "git_commit": done["git_commit"]}


def score(repo, out_root, out_json):
    fx = fixtures(repo)
    rows = {}
    for protocol in PROTOCOLS:
        root = os.path.join(out_root, protocol)
        if not os.path.isdir(root):
            continue
        for c, v1, v2, grid, seed in fx:
            if os.path.exists(os.path.join(root, c["clip_id"], "DONE.json")):
                rows.setdefault(protocol, {})[c["clip_id"]] = score_clip(repo, root, c, v1, grid, seed)
    agg = {}
    for protocol, clips in rows.items():
        sc = [r for r in clips.values() if r["outcome"] == "SCORED"]
        nv = sum(r["keyframes_visible"] for r in sc)
        a = {"clips_scored": len(sc), "keyframes_visible": nv,
             "trusted_correct": sum(r["trusted_correct"] for r in sc),
             "trusted_wrong": sum(r["trusted_wrong"] for r in sc),
             "coverage": round(sum(r["trusted_correct"] for r in sc) / nv, 3) if nv else None,
             "checkpoints_opened": sum(r["checkpoints"]["opened"] for r in sc),
             "reseeded": sum(r["checkpoints"]["reseeded"] for r in sc),
             "taps_answered": sum(r["taps"]["answered"] for r in sc),
             "taps_accepted": sum(r["taps"]["accepted"] for r in sc),
             "taps_refused": sum(r["taps"]["refused"] for r in sc),
             "declined": sum(r["taps"]["declined"] for r in sc),
             "wrong_accepted_selections": sum(len(r["wrong_accepted_selections"]) for r in sc),
             "refused_by_reason": {}}
        for r in sc:
            for k, v in r["taps"]["by_reason"].items():
                a["refused_by_reason"][k] = a["refused_by_reason"].get(k, 0) + v
        a["checkpoints_per_clip"] = round(a["checkpoints_opened"] / len(sc), 2) if sc else None
        agg[protocol] = a
    from tools_rb import r2_jersey_labels as L1
    out = {"schema": "r2_tap_dev_scores_v1", "keyframe_grid_s": L1.KEYFRAME_S, "iou_ok": IOU_OK, "aggregate": agg,
           "per_clip": rows}
    with open(out_json, "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    print(json.dumps(agg, indent=1))


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "run" and len(a) >= 5:
        run(a[1], a[2], a[3], a[4], set(a[5:]) or None)
    elif a and a[0] == "score" and len(a) == 4:
        score(a[1], a[2], a[3])
    else:
        raise SystemExit(__doc__)
