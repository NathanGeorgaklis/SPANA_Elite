# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R2 DEV remediation instruction (Nathan, 2026-10-30 17:41, Part 1): F1 seed association must accept
#                 clean unambiguous DEV seeds and still fail closed on ambiguous ones
"""Read-only DEV seed check: runs the tracker's own seed() on every frozen DEV fixture's seed frame and prints the
rule used (IOU / CONTAINMENT) or the refusal. DEV only (EVAL fixtures are skipped unopened). Changes nothing.

    python3 tools_rb/r2_seed_check.py REPO CLIPS_DIR
"""
import glob, json, os, sys
repo, clips = sys.argv[1], sys.argv[2]   # noqa: E702
sys.path.insert(0, repo)
from player_tracker_rb import population as P, target as T
from player_tracker_rb.appearance import AppearanceExtractor
from player_tracker_rb.camera import CameraMotionEstimator
from player_tracker_rb.config import TrackerConfig
from player_tracker_rb.run_rb import build_backends
from player_tracker_rb.video_io import SequentialVideoReader
cfg = TrackerConfig.default(); det, emb = build_backends(cfg)
for p in sorted(glob.glob(os.path.join(repo, "R2/labels/*.fixture.json"))):
    fx = json.load(open(p))
    if fx["split"] != "DEV": continue
    with SequentialVideoReader(os.path.join(clips, fx["video_file"]), fx["seed_frame"], cfg, expected_sha256=fx["video_sha256"]) as rd:
        info, seed = rd.info, rd.seed
        cands = det.detect(seed.image)
        cam = CameraMotionEstimator(cfg).update(seed.image, [d.bbox for d in cands], 1.0 / info.fps)
        pop = P.Population(cfg, info.frame_width, info.frame_height); res = pop.update(0, 0.0, cands, cam)
        tr = T.TargetTracker(cfg, pop, AppearanceExtractor(cfg, emb))
        try:
            fr = tr.seed(res, seed.image, fx["seed_bbox"]); d = fr.decision
            print(fx["clip_id"], "SEEDED", d["seed_rule"], "iou %.3f contain %.3f h %.3f" % (d["seed_iou"], d["seed_containment"], d["seed_height_ratio"]))
        except T.SeedError as e:
            print(fx["clip_id"], "REFUSED", e)
