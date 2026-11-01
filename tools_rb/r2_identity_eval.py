# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R2 DEV persistent football identity layer instruction (Nathan, 2026-11-01 21:39): measure whether
#                 persistent PERSON_IDs narrow the target's candidates safely; DEV only; no target trust change.
"""Read-only DEV evaluation of the persistent identity layer beside the UNCHANGED tracker (refuses EVAL fixtures).

    python3 tools_rb/r2_identity_eval.py REPO CLIPS_DIR OUT_DIR CLIP_ID [CLIP_ID ...]

Per clip: the production detector, camera, population and target tracker exactly as run_rb runs them, plus
player_tracker_rb.identity.PersonIdentityLayer fed read-only (the tracker's TRACKING tracklet, identity_trusted, bank).
Writes OUT_DIR/<clip>.identity.jsonl (per frame: target state, identity frame) and OUT_DIR/IDENTITY_EVAL.json.

Truth (frozen DEV labels; only the target is labelled):
  * a tracklet is TARGET if most of its boxed-frame observations match the human target box (IoU >= 0.5),
    NOT_TARGET if on every boxed frame it is observed the human box is elsewhere (IoU < 0.1), else UNCLEAR;
  * TARGET-person links: a decision LINKED to a person that already holds a TARGET tracklet is CORRECT for a TARGET
    tracklet and WRONG_INTO_TARGET for a NOT_TARGET one; a TARGET tracklet LINKED to a person holding a NOT_TARGET
    tracklet is WRONG_OUT_OF_TARGET. (Other people have no identity truth: their links are counted, not scored.)
  * NARROWING, on frames where the target is available, its true box is detected (on a tracklet), and the tracker
    does NOT trust the target: other bodies in the frame excluded by the layer (not_target_person) versus by the
    tracker's tracklet-level known others; FALSE EXCLUSIONS = the true box's tracklet excluded (must be 0); POSITIVE
    narrowing = the true box's tracklet is the target person, and FALSE TARGET-PERSON = another tracklet is.
"""
import collections
import json
import os
import sys


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def run_clip(repo, clips_dir, out_dir, cid):
    sys.path.insert(0, repo)
    from player_tracker_rb import identity as I
    from player_tracker_rb import population as P
    from player_tracker_rb import run_rb as R
    from player_tracker_rb import target as T
    from player_tracker_rb.appearance import AppearanceExtractor
    from player_tracker_rb.camera import CameraMotionEstimator
    from player_tracker_rb.config import TrackerConfig
    from player_tracker_rb.video_io import SequentialVideoReader
    with open(os.path.join(repo, "R2", "labels", cid + ".fixture.json")) as f:
        fx = json.load(f)
    if fx["split"] != "DEV":
        raise SystemExit(f"{cid} is not a DEV clip: refused")
    path = os.path.join(out_dir, cid + ".identity.jsonl")
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
        lay = I.PersonIdentityLayer(cfg, pop, ext)
        with open(path + ".partial", "x") as out:
            def step(rel, image, seed):
                cands = det.detect(image)
                cam = est.update(image, [d.bbox for d in cands], 1.0 / fps)
                res = pop.update(rel, rel / fps, cands, cam)
                tf = tt.seed(res, image, fx["seed_bbox"]) if seed else tt.update(res, image)
                on = tf.observation_tracklet_id if tf.rb_state == T.TRACKING else None
                idf = lay.update(res, image, target_tracklet=on, target_trusted=tf.identity_trusted,
                                 target_bank=tt.bank)
                obs = [{"index": o.index, "bbox": list(o.bbox), "tracklet_id": o.tracklet_id} for o in res.observations]
                out.write(json.dumps({"frame": rel, "target": {"rb_state": tf.rb_state, "trusted": tf.identity_trusted,
                                                                "tracklet": tf.observation_tracklet_id},
                                      "known_others": sorted(tt._known_others), "observations": obs,
                                      "identity": idf.as_record()}) + "\n")
            try:
                step(0, reader.seed.image, True)
            except T.SeedError as e:
                out.write(json.dumps({"seed_refused": str(e)}) + "\n")
            else:
                for fr in reader:
                    step(fr.rel_frame, fr.image, False)
    os.rename(path + ".partial", path)
    return path


def score(repo, out_dir, cids):
    report = {}
    for cid in cids:
        with open(os.path.join(repo, "R2", "labels", cid + ".fixture.json")) as f:
            fx = json.load(f)
        with open(os.path.join(repo, "R2", "labels", fx["labels_file"])) as f:
            lab = json.load(f)
        s = lab["seed"]["frame"]
        recs = [json.loads(line) for line in open(os.path.join(out_dir, cid + ".identity.jsonl"))]
        if recs and "seed_refused" in recs[0]:
            report[cid] = {"seed_refused": recs[0]["seed_refused"]}
            continue
        # tracklet truth
        votes = collections.defaultdict(lambda: [0, 0, 0])          # [matches, elsewhere, other]
        avail = {}
        for r in recs:
            cf = r["frame"] + s
            hb = lab["target_boxes"].get(str(cf))
            pf = lab["per_frame"][cf] if cf < len(lab["per_frame"]) else None
            if hb is None or pf is None:
                continue
            ok = pf["visibility"] in ("VISIBLE", "PARTIAL") and pf["resolvable"] == "YES" and not pf["merged"] \
                and not pf["human_uncertain"]
            avail[r["frame"]] = (hb, ok)
            for o in r["observations"]:
                if o["tracklet_id"] is None:
                    continue
                v = _iou(o["bbox"], hb)
                votes[o["tracklet_id"]][0 if v >= 0.5 else 1 if v < 0.1 else 2] += 1
        truth = {tid: "TARGET" if m > (e + x) else "NOT_TARGET" if m == 0 and x == 0 and e > 0 else "UNCLEAR"
                 for tid, (m, e, x) in votes.items()}
        # link decisions against the persons' tracklet truth
        person_tracklets = collections.defaultdict(list)
        counts = collections.Counter()
        for r in recs:
            for d in r["identity"]["decisions"]:
                tid, st, pid = d["tracklet"], d["status"], d["person"]
                tr = truth.get(tid, "UNCLEAR")
                counts[f"{st}"] += 1
                if st == "LINKED":
                    held = {truth.get(x, "UNCLEAR") for x in person_tracklets[pid]}
                    if "TARGET" in held and tr == "TARGET":
                        counts["TARGET_RELINK_CORRECT"] += 1
                    elif "TARGET" in held and tr == "NOT_TARGET":
                        counts["WRONG_INTO_TARGET"] += 1
                    elif "NOT_TARGET" in held and tr == "TARGET":
                        counts["WRONG_OUT_OF_TARGET"] += 1
                    elif tr == "TARGET":
                        counts["TARGET_LINK_UNSCORED"] += 1
                if tr == "TARGET" and st != "LINKED":
                    counts[f"TARGET_TRACKLET_{st}"] += 1
                person_tracklets[pid].append(tid)
        # narrowing on untrusted available frames with the true box detected on a tracklet
        nar = collections.Counter()
        for r in recs:
            a = avail.get(r["frame"])
            if a is None or not a[1] or r["target"]["trusted"]:
                continue
            hb = a[0]
            true_t = [o["tracklet_id"] for o in r["observations"] if o["tracklet_id"] and _iou(o["bbox"], hb) >= 0.5]
            if not true_t:
                continue
            tt_ = true_t[0]
            ids = {x["tracklet_id"]: x for x in r["identity"]["tracklets"]}
            others = [o["tracklet_id"] for o in r["observations"] if o["tracklet_id"] and o["tracklet_id"] != tt_]
            nar["frames"] += 1
            nar["other_bodies"] += len(others)
            nar["excluded_by_layer"] += sum(1 for x in others if ids.get(x, {}).get("not_target_person"))
            nar["excluded_by_tracker"] += sum(1 for x in others if x in r["known_others"])
            nar["excluded_by_layer_only"] += sum(1 for x in others if ids.get(x, {}).get("not_target_person")
                                                 and x not in r["known_others"])
            nar["FALSE_EXCLUSION_layer"] += bool(ids.get(tt_, {}).get("not_target_person"))
            nar["FALSE_EXCLUSION_tracker"] += tt_ in r["known_others"]
            nar["true_is_target_person"] += bool(ids.get(tt_, {}).get("is_target_person"))
            nar["FALSE_TARGET_PERSON"] += sum(1 for x in others if ids.get(x, {}).get("is_target_person"))
        report[cid] = {"tracklet_truth": dict(collections.Counter(truth.values())), "decisions": dict(counts),
                       "narrowing": dict(nar)}
    return report


if __name__ == "__main__":
    repo, clips, out = sys.argv[1], sys.argv[2], sys.argv[3]
    cids = sys.argv[4:]
    os.makedirs(out, exist_ok=True)
    for c in cids:
        run_clip(repo, clips, out, c)
        print(c, "done", flush=True)
    rep = score(repo, out, cids)
    with open(os.path.join(out, "IDENTITY_EVAL.json"), "w") as f:
        json.dump(rep, f, indent=1)
    print(json.dumps(rep, indent=1))
