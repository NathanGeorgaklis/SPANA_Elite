# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R2 DEV F2 instruction (Nathan, 2026-11-01 17:38): allow safe initialisation when the correct seed
#                 player is detected but the crop is partial, near the image edge, or has nested/alternative
#                 detections; diagnose the first failing layer for each refused seed.
"""Read-only F2 seed diagnosis for DEV clips (refuses EVAL fixtures; changes no official metric).

    python3 tools_rb/r2_f2_seed_diag.py REPO CLIPS_DIR SECONDS CLIP_ID [CLIP_ID ...] > report.txt

From the seed frame on, for SECONDS, runs the production detector, camera and population exactly as run_rb does
(no target tracker), and for every frame prints, for the observation matching the frozen human target box (IoU >= 0.5,
else the largest detection >= 0.9 inside it with >= 0.5 of its height):
  * its tracklet id, flags, nested_in, and the ordinary bank-refusal reasons (_BANK_REFUSE_FLAGS, CONTAINS_OTHER of
    another body, NOT_ISOLATED IoU > bank_contact_iou, CAMERA_UNVERIFIED);
  * the other observations touching it: IoU, fraction inside the matched box, flags, nested_in;
  * the frozen truth for the frame (visibility, resolvable, contact, merged).
The seed frame's row also shows the seed box's IoU with the matched observation.
Each row also gives the matched box's area (px^2) and its appearance similarity (composite / embedding / LAB) to the
seed-frame crop of the seed match (the production extractor and embedding model), for the F2 anchor-continuity check.
"""
import json
import os
import sys


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def _inside(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area = (a[2] - a[0]) * (a[3] - a[1])
    return ix * iy / area if area > 0 else 0.0


def match(obs, hb):
    best = max(((o, _iou(o.bbox, hb)) for o in obs if o.is_body), key=lambda x: x[1], default=(None, 0.0))
    if best[1] >= 0.5:
        return best[0], best[1]
    cont = [o for o in obs if o.is_body and _inside(o.bbox, hb) >= 0.9
            and (o.bbox[3] - o.bbox[1]) >= 0.5 * (hb[3] - hb[1])]
    if cont:
        o = max(cont, key=lambda o: o.bbox[3] - o.bbox[1])
        return o, _iou(o.bbox, hb)
    return None, best[1]


def main(repo, clips_dir, seconds, cids):
    sys.path.insert(0, repo)
    from player_tracker_rb import population as P
    from player_tracker_rb import target as T
    from player_tracker_rb import run_rb as R
    from player_tracker_rb.camera import CameraMotionEstimator, ESTIMATED
    from player_tracker_rb.config import TrackerConfig
    from player_tracker_rb.video_io import SequentialVideoReader
    from player_tracker_rb.appearance import AppearanceExtractor, compare
    cfg = TrackerConfig.default()
    det, emb = R.build_backends(cfg)
    ext = AppearanceExtractor(cfg, emb)
    for cid in cids:
        with open(os.path.join(repo, "R2", "labels", cid + ".fixture.json")) as f:
            fx = json.load(f)
        if fx["split"] != "DEV":
            raise SystemExit(f"{cid} is not a DEV clip: refused")
        with open(os.path.join(repo, "R2", "labels", fx["labels_file"])) as f:
            lab = json.load(f)
        s = lab["seed"]["frame"]
        print("=" * 100)
        print(cid, "seed frame (clip)", s, "seed box", fx["seed_bbox"])
        with SequentialVideoReader(os.path.join(clips_dir, fx["video_file"]), fx["seed_frame"], cfg,
                                   expected_sha256=fx["video_sha256"]) as reader:
            fps = reader.info.fps
            print("fps", fps, "size", reader.info.frame_width, reader.info.frame_height)
            est = CameraMotionEstimator(cfg)
            pop = P.Population(cfg, reader.info.frame_width, reader.info.frame_height)
            n_max = int(seconds * fps)

            def step(rel, image):
                cands = det.detect(image)
                cam = est.update(image, [d.bbox for d in cands], 1.0 / fps)
                return pop.update(rel, rel / fps, cands, cam)

            ref = {}

            def report(rel, res, image):
                cf = rel + s
                hb = lab["target_boxes"].get(str(cf)) if rel else fx["seed_bbox"]
                pf = lab["per_frame"][cf] if cf < len(lab["per_frame"]) else {}
                truth = {k: pf.get(k) for k in ("visibility", "resolvable", "contact_with", "merged")}
                if hb is None:
                    print(f"  f{rel:3d} no human box  truth {truth}")
                    return
                o, iou = match(res.observations, hb)
                if o is None:
                    print(f"  f{rel:3d} TARGET NOT DETECTED (best IoU {iou:.2f})  truth {truth}")
                    return
                refuse = [fl for fl in T._BANK_REFUSE_FLAGS if fl in o.flags]
                others = [x for x in res.observations if x.is_body and x.index != o.index]
                for x in others:
                    if _inside(x.bbox, o.bbox) >= cfg.get("population_nested_containment") and x.nested_in != o.index:
                        refuse.append("CONTAINS_OTHER")
                        break
                touch = [(x.index, round(_iou(x.bbox, o.bbox), 2), round(_inside(x.bbox, o.bbox), 2), x.nested_in,
                          x.tracklet_id, list(x.flags)) for x in others
                         if _iou(x.bbox, o.bbox) > 0 or _inside(x.bbox, o.bbox) > 0]
                grp = {o.index} | {x.index for x in others if x.nested_in == o.index}
                if o.nested_in is not None:
                    grp.add(o.nested_in)
                out_ious = [_iou(x.bbox, o.bbox) for x in others if x.index not in grp]
                if out_ious and max(out_ious) > cfg.get("bank_contact_iou"):
                    refuse.append("NOT_ISOLATED")
                if res.camera.status != ESTIMATED and rel:
                    refuse.append("CAMERA_UNVERIFIED")
                feat = ext.extract(image, o.bbox, others=[x.bbox for x in others if x.index not in grp])
                if rel == 0:
                    ref["feat"] = feat
                sim = compare(ref["feat"], feat, cfg) if feat.usable and ref["feat"].usable else None
                simtxt = "n/a" if sim is None else \
                    f"{sim.composite:.3f}/{sim.embedding:.3f}/{sim.lab:.3f}" if sim.composite is not None else "none"
                area = (o.bbox[2] - o.bbox[0]) * (o.bbox[3] - o.bbox[1])
                print(f"  f{rel:3d} match iou {iou:.2f} area {area:.0f} sim_to_seed {simtxt} trk {o.tracklet_id} "
                      f"nested_in {o.nested_in} "
                      f"flags {list(o.flags)} REFUSE {sorted(set(refuse)) or 'NONE (bank-clean)'} "
                      f"touching {touch}  truth {truth}")

            res = step(0, reader.seed.image)
            report(0, res, reader.seed.image)
            for fr in reader:
                if fr.rel_frame > n_max:
                    break
                res = step(fr.rel_frame, fr.image)
                report(fr.rel_frame, res, fr.image)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4:])
