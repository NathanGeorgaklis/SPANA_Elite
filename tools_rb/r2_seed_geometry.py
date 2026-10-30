# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R2 DEV remediation instruction (Nathan, 2026-10-30 17:41, Part 1: before choosing thresholds,
#                 measure seed IoU, containment, centre offset and size ratios across frozen DEV seeds and compare
#                 against competing detections)
"""Read-only seed geometry measurement on frozen DEV fixtures.

    python3 tools_rb/r2_seed_geometry.py REPO CLIPS_DIR OUT_JSON

Rebuilds each DEV seed frame exactly as run_rb.run does (SequentialVideoReader -> detector -> camera -> population
update 0) and reports, for every body observation the target's seed() would consider, its geometry against the
human seed box. Normalised by the seed box height (scale-free, rule 15). DEV only (refuses EVAL). Changes nothing.

Per body: IoU; contain = fraction of the detection inside the seed box; cover = fraction of the seed box covered by
the detection; centre offset dx, dy in seed heights; height and width ratio detection / seed; population flags.
"""
import glob
import json
import os
import sys


def geometry(sb, b):
    """The tracker's own seed geometry (player_tracker_rb.target._seed_geometry, so the measurement cannot drift from
    the rule) plus IoU and the fraction of the seed box covered by the detection."""
    from player_tracker_rb.target import _seed_geometry
    g = dict(_seed_geometry(sb, b))
    ix = max(0.0, min(sb[2], b[2]) - max(sb[0], b[0]))
    iy = max(0.0, min(sb[3], b[3]) - max(sb[1], b[1]))
    inter = ix * iy
    a_s = (sb[2] - sb[0]) * (sb[3] - sb[1])
    a_b = (b[2] - b[0]) * (b[3] - b[1])
    g["iou"] = inter / (a_s + a_b - inter) if a_s + a_b - inter > 0 else 0.0
    g["cover"] = inter / a_s if a_s > 0 else 0.0
    return g


def main(repo, clips_dir, out_json):
    sys.path.insert(0, repo)
    from player_tracker_rb import population as P
    from player_tracker_rb.camera import CameraMotionEstimator
    from player_tracker_rb.config import TrackerConfig
    from player_tracker_rb.run_rb import build_backends
    from player_tracker_rb.video_io import SequentialVideoReader
    cfg = TrackerConfig.default()
    det, _ = build_backends(cfg)
    out = {}
    for p in sorted(glob.glob(os.path.join(repo, "R2", "labels", "*.fixture.json"))):
        with open(p) as f:
            fx = json.load(f)
        if fx.get("split") != "DEV":
            continue
        video = os.path.join(clips_dir, fx["video_file"])
        with SequentialVideoReader(video, fx["seed_frame"], cfg, expected_sha256=fx["video_sha256"]) as rd:
            info, seed = rd.info, rd.seed
            cands = det.detect(seed.image)
            cam = CameraMotionEstimator(cfg).update(seed.image, [d.bbox for d in cands], 1.0 / info.fps)
            res = P.Population(cfg, info.frame_width, info.frame_height).update(0, 0.0, cands, cam)
        sb = [float(v) for v in fx["seed_bbox"]]
        bodies = []
        for o in res.observations:
            if not o.is_body:
                continue
            g = geometry(sb, [float(v) for v in o.bbox])
            if g["iou"] <= 0 and g["contain"] <= 0:
                continue
            g = {k: round(v, 3) for k, v in g.items()}
            g.update({"index": o.index, "bbox": [round(float(v), 1) for v in o.bbox], "flags": list(o.flags),
                      "nested_in": o.nested_in})
            bodies.append(g)
        bodies.sort(key=lambda r: -r["iou"])
        out[fx["clip_id"]] = {"seed_frame": fx["seed_frame"], "seed_bbox": sb, "frame_w": info.frame_width,
                              "frame_h": info.frame_height, "n_bodies_total": sum(o.is_body for o in res.observations),
                              "overlapping_bodies": bodies}
    with open(out_json, "x") as f:
        json.dump(out, f, indent=1)
    for cid, v in out.items():
        print(cid, "seed h", round(v["seed_bbox"][3] - v["seed_bbox"][1], 1))
        for b in v["overlapping_bodies"]:
            print("    iou %.3f contain %.3f cover %.3f dx %+.3f dy %+.3f h %.3f w %.3f feet %.3f %s" % (
                b["iou"], b["contain"], b["cover"], b["dx_h"], b["dy_h"], b["h_ratio"], b["w_ratio"],
                b["bottom_gap_h"], b["flags"]))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
