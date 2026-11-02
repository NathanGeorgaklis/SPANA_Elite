# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R2 DEV positive target evidence instruction (Nathan, 2026-11-02 17:46): measure the target reference
#                 profile beside the UNCHANGED tracker on C006, C020, C033, SYO C012: anchors per clip, true target vs
#                 hardest same-kit separation pre/post, reacquisition / coverage, safety. DEV only.
"""Read-only DEV evaluation of the target reference profile beside the unchanged tracker (refuses non-DEV fixtures).

    python3 tools_rb/r2_profile_eval.py run   REPO CLIPS_DIR OUT_DIR CLIP_ID      (one clip; resumable per clip)
    python3 tools_rb/r2_profile_eval.py score REPO OUT_DIR CLIP_ID [CLIP_ID ...]   (writes OUT_DIR/PROFILE_EVAL.json)

Per frame: the production detector, camera, population and target tracker exactly as run_rb runs them, plus
player_tracker_rb.target_profile.TargetProfile fed read-only (TargetFrame, the tracker's bank). For every body:
PRE = the production score (the tracker's trusted bank, single best match: the composite reacquisition uses) and the
profile's best / topk / support / temporal values on the same crop.

Truth (frozen DEV labels; only the target is labelled; fixed before any profile result was seen):
  * frames: target_boxes frame, VISIBLE/PARTIAL, resolvable, not merged, not human_uncertain, and the true box
    detected as a body (IoU >= 0.5). Split TRUSTED / UNTRUSTED by the tracker's identity_trusted.
  * SAME-KIT competitor: its kit (median LAB to the target's anchors) >= the 10th percentile of the true target's own
    kit values in that clip (as in the R2 ReID evaluation). The HARDEST same-kit competitor is chosen per method.
  * separation = true score - hardest same-kit score; rank-1 = separation > 0; margin = separation >=
    ambiguity_margin.
  * counterfactual verdicts: PROFILE = the layer's verdict; PRE = the production analogue (production score >=
    reacq_appearance_floor and >= every other body + ambiguity_margin). TRUE / WRONG = on the true box / another box;
    WRONG_TARGET_ABSENT = SUPPORTED on some box on a frame where the target is labelled not visible / unresolvable or
    its box is not detected (open world).
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
    from player_tracker_rb import target_profile as TP
    from player_tracker_rb.appearance import AppearanceExtractor
    from player_tracker_rb.camera import CameraMotionEstimator
    from player_tracker_rb.config import TrackerConfig
    from player_tracker_rb.video_io import SequentialVideoReader
    fx = _fixture(repo, cid)
    path = os.path.join(out_dir, cid + ".profile.jsonl")
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
        prof = TP.TargetProfile(cfg, ext)
        with open(path + ".partial", "x") as out:
            def step(rel, image, seed):
                cands = det.detect(image)
                cam = est.update(image, [d.bbox for d in cands], 1.0 / fps)
                res = pop.update(rel, rel / fps, cands, cam)
                tf = tt.seed(res, image, fx["seed_bbox"]) if seed else tt.update(res, image)
                pf = prof.update(res, image, tf, tt.bank)
                for row in pf["candidates"]:
                    s, _ = tt.bank.best_match(prof.last_features[row["index"]])
                    row["pre"] = None if s is None else s.composite
                out.write(json.dumps({"frame": rel, "target": {"rb_state": tf.rb_state, "trusted": tf.identity_trusted,
                                                                "tracklet": tf.observation_tracklet_id,
                                                                "index": tf.detection_index},
                                      "bank_samples": len(tt.bank.samples), "profile": pf}) + "\n")
            try:
                step(0, reader.seed.image, True)
            except T.SeedError as e:
                out.write(json.dumps({"seed_refused": str(e)}) + "\n")
            else:
                for fr in reader:
                    step(fr.rel_frame, fr.image, False)
        with open(os.path.join(out_dir, cid + ".anchors.json"), "w") as f:
            json.dump({"anchors": [a.as_record() for a in prof.anchors], "decisions": prof.decisions,
                       "bank_samples_final": len(tt.bank.samples)}, f, indent=1)
    os.rename(path + ".partial", path)
    return path


def _q(xs):
    return None if not xs else {"n": len(xs), "median": float(np.median(xs)), "p10": float(np.percentile(xs, 10)),
                                "p90": float(np.percentile(xs, 90))}


def score(repo, out_dir, cids):
    sys.path.insert(0, repo)
    from player_tracker_rb.config import TrackerConfig
    cfg = TrackerConfig.default()
    margin, floor = cfg.get("ambiguity_margin"), cfg.get("reacq_appearance_floor")
    methods = {"PRE_bank_best": lambda r: r.get("pre"),
               "PROFILE_best": lambda r: (r["match"] or {}).get("best"),
               "PROFILE_topk": lambda r: (r["match"] or {}).get("topk"),
               "PROFILE_joint": lambda r: None if r["temporal"] is None or r["match"] is None
               else min(r["temporal"]["topk"], r["match"]["topk"])}
    report = {}
    for cid in cids:
        fx = _fixture(repo, cid)
        with open(os.path.join(repo, "R2", "labels", fx["labels_file"])) as f:
            lab = json.load(f)
        s = fx["seed_frame"]
        recs = [json.loads(x) for x in open(os.path.join(out_dir, cid + ".profile.jsonl"))]
        if recs and "seed_refused" in recs[0]:
            report[cid] = {"seed_refused": recs[0]["seed_refused"]}
            continue
        with open(os.path.join(out_dir, cid + ".anchors.json")) as f:
            anc = json.load(f)
        frames = []                       # (rec, true_row, others, trusted)
        absent = []
        for r in recs:
            cf = r["frame"] + s
            pf = lab["per_frame"][cf] if cf < len(lab["per_frame"]) else None
            hb = lab["target_boxes"].get(str(cf))
            rows = r["profile"]["candidates"]
            true = None
            if hb is not None and pf is not None and pf["visibility"] in ("VISIBLE", "PARTIAL") and \
                    pf["resolvable"] == "YES" and not pf["merged"] and not pf["human_uncertain"]:
                hits = [x for x in rows if _iou(x["bbox"], hb) >= 0.5]
                true = max(hits, key=lambda x: _iou(x["bbox"], hb)) if hits else None
            not_visible = pf is not None and (pf["visibility"] not in ("VISIBLE", "PARTIAL")
                                              or pf["resolvable"] != "YES")
            if true is not None:
                frames.append((r, true, [x for x in rows if x is not true], r["target"]["trusted"]))
            elif not_visible:
                absent.append(r)
        kits = [t["match"]["kit"] for _, t, _, _ in frames if t["match"] and t["match"]["kit"] is not None]
        kit_thr = float(np.percentile(kits, 10)) if kits else None
        out = {"anchors": len(anc["anchors"]), "anchor_aspects": [round(a["aspect"], 3) for a in anc["anchors"]],
               "anchor_frames": [a["rel_frame"] for a in anc["anchors"]], "anchor_decisions": anc["decisions"],
               "bank_samples_final": anc["bank_samples_final"], "same_kit_threshold": kit_thr,
               "frames_true_detected": len(frames),
               "frames_trusted": sum(1 for f in frames if f[3]), "tracker_trusted_all": sum(
                   1 for r in recs if r["target"]["trusted"]), "methods": {}}
        for name, fn in methods.items():
            res = {}
            for split in ("ALL", "UNTRUSTED", "TRUSTED"):
                seps, n_sk, rank1, mar = [], 0, 0, 0
                for r, t, others, trusted in frames:
                    if (split == "UNTRUSTED" and trusted) or (split == "TRUSTED" and not trusted):
                        continue
                    tv = fn(t)
                    sk = [fn(o) for o in others if o["match"] and o["match"]["kit"] is not None and kit_thr is not None
                          and o["match"]["kit"] >= kit_thr and fn(o) is not None]
                    if tv is None or not sk:
                        continue
                    n_sk += 1
                    d = tv - max(sk)
                    seps.append(d)
                    rank1 += d > 0
                    mar += d >= margin
                res[split] = {"frames_with_same_kit": n_sk, "separation": _q(seps),
                              "rank1": rank1 / n_sk if n_sk else None, "margin_rate": mar / n_sk if n_sk else None}
            out["methods"][name] = res
        v = collections.Counter()
        for r, t, others, trusted in frames:
            pv = r["profile"]["verdict"]
            sp = "UNTRUSTED" if not trusted else "TRUSTED"
            if pv["status"] == "SUPPORTED":
                v[f"PROFILE_TRUE_{sp}" if pv["index"] == t["index"] else f"PROFILE_WRONG_{sp}"] += 1
            v[f"PROFILE_{pv['status']}_{sp}"] += 1
            rows = [t] + others
            pre = [(x.get("pre"), x) for x in rows if x.get("pre") is not None]
            if pre:
                bv, bx = max(pre, key=lambda z: z[0])
                if bv >= floor and all(z[0] <= bv - margin for z in pre if z[1] is not bx):
                    v[f"PRE_TRUE_{sp}" if bx is t else f"PRE_WRONG_{sp}"] += 1
        for r in absent:
            if r["profile"]["verdict"]["status"] == "SUPPORTED":
                v["PROFILE_WRONG_TARGET_ABSENT"] += 1
            pre = [x.get("pre") for x in r["profile"]["candidates"] if x.get("pre") is not None]
            if pre:
                srt = sorted(pre, reverse=True)
                if srt[0] >= floor and (len(srt) == 1 or srt[1] <= srt[0] - margin):
                    v["PRE_WRONG_TARGET_ABSENT"] += 1
        out["frames_target_absent"] = len(absent)
        out["verdicts"] = dict(v)
        report[cid] = out
    return report


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
        with open(os.path.join(out, "PROFILE_EVAL.json"), "w") as f:
            json.dump(rep, f, indent=1)
        print(json.dumps(rep, indent=1))
    else:
        raise SystemExit("mode: run | score")
