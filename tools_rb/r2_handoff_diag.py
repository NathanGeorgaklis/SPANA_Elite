# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R2 DEV same-frame identity handoff safety instruction (Nathan, 2026-11-02 20:14): find the first failing
#                 layer and a continuity rule that separates genuine continuous target motion from a different same-kit
#                 player replacing the target's box. This tool MEASURES, on DEV footage, the appearance step between
#                 consecutive observations (DEV only; the tracker is run unchanged).
"""Read-only DEV measurement of consecutive-observation appearance steps (refuses non-DEV fixtures).

    python3 tools_rb/r2_handoff_diag.py run   REPO CLIPS_DIR OUT_DIR CLIP_ID
    python3 tools_rb/r2_handoff_diag.py score REPO OUT_DIR CLIP_ID [CLIP_ID ...]   (writes OUT_DIR/HANDOFF_DIAG.json)

Per frame, for every body observed on a tracklet that was also observed on the previous frame (at most
2 frames apart): SELF step = composite(current crop, the same tracklet's previous crop) (production extractor,
other bodies as contamination, production composite). For the tracker's target tracklet, IMPOSTOR steps =
composite(the target's previous crop, the current crop of every OTHER body within 3 target heights): what the step
would be if that body replaced the target's box.

Truth (frozen DEV labels): a tracklet is TARGET if most of its labelled-frame observations match the human box
(IoU >= 0.5) and none is elsewhere; SAME-KIT impostor: its LAB similarity to the target's previous crop is in the upper half
of all impostor LAB values in that clip (the closest kits). Impostor quantiles include the upper tail (p90..max). Reported: SELF steps of TARGET tracklets (trusted
continuation frames, all frames), and IMPOSTOR steps (same-kit, all), as quantiles; plus the separation at
candidate thresholds (fraction of genuine steps below / impostor steps at or above).
"""
import collections
import json
import os
import sys

import numpy as np


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def _fixture(repo, cid):
    with open(os.path.join(repo, "R2", "labels", cid + ".fixture.json")) as f:
        fx = json.load(f)
    if fx["split"] != "DEV":
        raise SystemExit(f"{cid} is not a DEV clip: refused")
    return fx


def run_clip(repo, clips_dir, out_dir, cid):
    sys.path.insert(0, repo)
    from player_tracker_rb import population as P
    from player_tracker_rb import run_rb as R
    from player_tracker_rb import target as T
    from player_tracker_rb.appearance import AppearanceExtractor, compare
    from player_tracker_rb.camera import CameraMotionEstimator
    from player_tracker_rb.config import TrackerConfig
    from player_tracker_rb.video_io import SequentialVideoReader
    fx = _fixture(repo, cid)
    path = os.path.join(out_dir, cid + ".handoff.jsonl")
    if os.path.exists(path):
        return path
    cfg = TrackerConfig.default()
    det, emb = R.build_backends(cfg)
    with SequentialVideoReader(os.path.join(clips_dir, fx["video_file"]), fx["seed_frame"], cfg,
                               expected_sha256=fx["video_sha256"]) as reader:
        fps = reader.info.fps
        est = CameraMotionEstimator(cfg)
        pop = P.Population(cfg, reader.info.frame_width, reader.info.frame_height)
        ext = AppearanceExtractor(cfg, emb)
        tt = T.TargetTracker(cfg, pop, ext)
        prev = {}                                   # tracklet -> (rel_frame, feature)
        with open(path + ".partial", "x") as out:
            def step(rel, image, seed):
                cands = det.detect(image)
                cam = est.update(image, [d.bbox for d in cands], 1.0 / fps)
                res = pop.update(rel, rel / fps, cands, cam)
                tf = tt.seed(res, image, fx["seed_bbox"]) if seed else tt.update(res, image)
                bodies = [o for o in res.observations if o.is_body and P.ALTERNATIVE not in o.flags]
                feats = {}
                for o in bodies:
                    others = [x.bbox for x in bodies if x.index != o.index]
                    feats[o.index] = ext.extract(image, o.bbox, others=others,
                                                 merged_suspected=P.MERGED_SUSPECTED in o.flags)
                rows = []
                tgt_t = tf.observation_tracklet_id if tf.rb_state == T.TRACKING else None
                tbox = None
                for o in bodies:
                    f = feats[o.index]
                    row = {"index": o.index, "tracklet": o.tracklet_id, "bbox": list(o.bbox), "flags": list(o.flags),
                           "usable": f.usable, "qflags": list(f.quality.flags), "self": None, "gap": None}
                    if o.tracklet_id in prev and f.usable:
                        pf_, pfeat = prev[o.tracklet_id]
                        if rel - pf_ <= 2 and pfeat.usable:
                            s = compare(f, pfeat, cfg)
                            row["self"] = {"composite": s.composite, "lab": s.lab, "embedding": s.embedding}
                            row["gap"] = rel - pf_
                    if o.tracklet_id == tgt_t:
                        tbox = o.bbox
                    rows.append(row)
                imp = []
                if tgt_t in prev and tbox is not None:
                    pf_, pfeat = prev[tgt_t]
                    th = tbox[3] - tbox[1]
                    tc = ((tbox[0] + tbox[2]) / 2, (tbox[1] + tbox[3]) / 2)
                    for o in bodies:
                        if o.tracklet_id == tgt_t or not feats[o.index].usable or not pfeat.usable or rel - pf_ > 2:
                            continue
                        c = ((o.bbox[0] + o.bbox[2]) / 2, (o.bbox[1] + o.bbox[3]) / 2)
                        if np.hypot(c[0] - tc[0], c[1] - tc[1]) <= 3 * th:
                            s = compare(feats[o.index], pfeat, cfg)
                            imp.append({"index": o.index, "tracklet": o.tracklet_id, "composite": s.composite,
                                        "lab": s.lab, "embedding": s.embedding})
                for o in bodies:
                    if o.tracklet_id is not None:
                        prev[o.tracklet_id] = (rel, feats[o.index])
                out.write(json.dumps({"frame": rel, "target": {"rb_state": tf.rb_state, "trusted": tf.identity_trusted,
                                                                "tracklet": tf.observation_tracklet_id,
                                                                "bank": (tf.decision.get("bank") or {}).get("status")
                                                                if isinstance(tf.decision, dict) else None},
                                      "bodies": rows, "impostors": imp}) + "\n")
            try:
                step(0, reader.seed.image, True)
            except T.SeedError as e:
                out.write(json.dumps({"seed_refused": str(e)}) + "\n")
            else:
                for fr in reader:
                    step(fr.rel_frame, fr.image, False)
    os.rename(path + ".partial", path)
    return path


def _q(xs):
    if not xs:
        return None
    return {"n": len(xs), **{f"p{p}": float(np.percentile(xs, p)) for p in (0, 1, 5, 10, 50, 90, 95, 99, 100)}}


def score(repo, out_dir, cids):
    rep, pool = {}, collections.defaultdict(list)
    for cid in cids:
        fx = _fixture(repo, cid)
        with open(os.path.join(repo, "R2", "labels", fx["labels_file"])) as f:
            lab = json.load(f)
        s = fx["seed_frame"]
        recs = [json.loads(x) for x in open(os.path.join(out_dir, cid + ".handoff.jsonl"))]
        if recs and "seed_refused" in recs[0]:
            rep[cid] = {"seed_refused": recs[0]["seed_refused"]}
            continue
        votes = collections.defaultdict(lambda: [0, 0])
        for r in recs:
            hb = lab["target_boxes"].get(str(r["frame"] + s))
            if hb is None:
                continue
            for b in r["bodies"]:
                if b["tracklet"]:
                    v = _iou(b["bbox"], hb)
                    votes[b["tracklet"]][0 if v >= 0.5 else 1] += v >= 0.5 or v < 0.1
        target_t = {t for t, (m, e) in votes.items() if m > 0 and e == 0}
        selfs = {"target_all": [], "target_trusted": [], "target_lab": []}
        imps = []
        for r in recs:
            for b in r["bodies"]:
                if b["self"] is None or b["tracklet"] not in target_t or b["self"]["composite"] is None:
                    continue
                selfs["target_all"].append(b["self"]["composite"])
                selfs["target_lab"].append(b["self"]["lab"])
                if r["target"]["trusted"] and r["target"]["tracklet"] == b["tracklet"]:
                    selfs["target_trusted"].append(b["self"]["composite"])
            if r["target"]["tracklet"] in target_t:
                imps += [x for x in r["impostors"] if x["composite"] is not None]
        # SAME-KIT impostor: LAB to the target's previous crop >= the 10th percentile of ALL impostor LAB values' upper
        # half, i.e. the impostors in the top half by colour (kit) similarity (the closest kits in the clip)
        labs = [x["lab"] for x in imps if x["lab"] is not None]
        thr = float(np.percentile(labs, 50)) if labs else None
        sk = [x["composite"] for x in imps if thr is not None and x["lab"] is not None and x["lab"] >= thr]
        out = {"target_tracklets": sorted(target_t), "same_kit_lab_threshold": thr,
               "self_target_all": _q(selfs["target_all"]), "self_target_trusted": _q(selfs["target_trusted"]),
               "impostor_same_kit": _q(sk), "impostor_all": _q([x["composite"] for x in imps])}
        rep[cid] = out
        pool["self_trusted"] += selfs["target_trusted"]
        pool["self_all"] += selfs["target_all"]
        pool["imp_sk"] += sk
    grid = {}
    for t in [round(x, 3) for x in np.arange(0.80, 0.991, 0.01)]:
        grid[str(t)] = {"genuine_trusted_below": float(np.mean([x < t for x in pool["self_trusted"]]))
                        if pool["self_trusted"] else None,
                        "genuine_all_below": float(np.mean([x < t for x in pool["self_all"]])) if pool["self_all"] else None,
                        "impostor_same_kit_at_or_above": float(np.mean([x >= t for x in pool["imp_sk"]]))
                        if pool["imp_sk"] else None}
    rep["POOLED"] = {"self_trusted": _q(pool["self_trusted"]), "self_all": _q(pool["self_all"]),
                     "impostor_same_kit": _q(pool["imp_sk"]), "threshold_grid": grid}
    return rep


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "run":
        repo, clips, out, cid = sys.argv[2:6]
        os.makedirs(out, exist_ok=True)
        run_clip(repo, clips, out, cid)
        print(cid, "done", flush=True)
    elif mode == "score":
        repo, out = sys.argv[2], sys.argv[3]
        rep = score(repo, out, sys.argv[4:])
        with open(os.path.join(out, "HANDOFF_DIAG.json"), "w") as f:
            json.dump(rep, f, indent=1)
        print(json.dumps(rep["POOLED"], indent=1))
    else:
        raise SystemExit("mode: run | score")
