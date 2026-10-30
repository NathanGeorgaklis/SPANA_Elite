# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R2 DEV inference instruction (Nathan, 2026-10-30 15:31: run the frozen clean tracker commit on the
#                 DEV clips only; do not modify tracker logic; do not run or score EVAL; use frozen labels and
#                 fixtures only; score only identity-scorable DEV clips; C019 excluded)
"""R2 DEV runner and scorer. Evaluation tooling only; changes no tracker logic.

    run    REPO_DIR CLIPS_DIR OUT_DIR [CLIP_ID ...]
           For each frozen DEV fixture (R2/labels/*.fixture.json with split DEV), runs
           player_tracker_rb.run_rb.run with the configured production backends and the debug decision log, from the
           human seed in the fixture, pinned to the fixture's video sha256. Reads ONLY the fixture (never the labels)
           before and during the run. Refuses: any EVAL fixture, a dirty git tree, an existing output directory.
    score  REPO_DIR OUT_DIR
           For each completed DEV run, scores with tools_rb/tracker_eval.py main (format r2_v1) against the frozen
           labels and freeze record. Never touches EVAL.
"""
import glob
import json
import os
import sys
import time


def _fixtures(repo, only=None):
    out = []
    for p in sorted(glob.glob(os.path.join(repo, "R2", "labels", "*.fixture.json"))):
        with open(p) as f:
            fx = json.load(f)
        if fx.get("split") != "DEV":
            continue                      # EVAL is never run here
        if only and fx["clip_id"] not in only:
            continue
        out.append((p, fx))
    return out


def run(repo, clips_dir, out_root, only=None):
    sys.path.insert(0, repo)
    from player_tracker_rb import provenance as PV
    from player_tracker_rb import run_rb as R
    head, dirty = PV.git_state()
    if dirty:
        raise SystemExit("git tree is dirty: the DEV run must use a clean commit (fail closed)")
    os.makedirs(out_root, exist_ok=True)
    for p, fx in _fixtures(repo, only):
        if fx["split"] != "DEV":
            raise SystemExit("non-DEV fixture reached the runner (fail closed)")
        cid = fx["clip_id"]
        od = os.path.join(out_root, cid)
        if os.path.exists(os.path.join(od, "DONE.json")):
            print(cid, "already done", flush=True)
            continue
        os.makedirs(od, exist_ok=True)
        video = os.path.join(clips_dir, fx["video_file"])
        t0 = time.time()
        print(cid, "start", time.strftime("%H:%M:%S"), flush=True)
        try:
            summary = R.run(video, fx["seed_frame"], fx["seed_bbox"], out_dir=od, debug=True,
                            expected_sha256=fx["video_sha256"])
        except R.RunRefusedError as e:
            # the tracker failed closed (e.g. seed not backed by a detection): a result, recorded, never retried
            # with different settings
            ref = {"clip_id": cid, "split": "DEV", "outcome": "RUN_REFUSED", "reason": str(e),
                   "fixture": os.path.relpath(p, repo), "git_commit": head, "git_dirty": dirty,
                   "seconds": round(time.time() - t0, 1)}
            with open(os.path.join(od, "DONE.json"), "x") as f:
                json.dump(ref, f, indent=1)
            print(cid, "RUN_REFUSED", e, flush=True)
            continue
        done = {"clip_id": cid, "split": "DEV", "outcome": "COMPLETED", "fixture": os.path.relpath(p, repo),
                "git_commit": head, "git_dirty": dirty, "seconds": round(time.time() - t0, 1),
                "summary_keys": sorted(summary)}
        with open(os.path.join(od, "DONE.json"), "x") as f:
            json.dump(done, f, indent=1)
        print(cid, "done", done["seconds"], "s", flush=True)


def score(repo, out_root):
    sys.path.insert(0, repo)
    from player_tracker_rb import run_rb as R
    from tools_rb import tracker_eval as E
    rows = {}
    for p, fx in _fixtures(repo):
        cid = fx["clip_id"]
        od = os.path.join(out_root, cid)
        if not os.path.exists(os.path.join(od, "DONE.json")):
            rows[cid] = {"status": "NOT_RUN"}
            continue
        with open(os.path.join(od, "DONE.json")) as f:
            done = json.load(f)
        if done.get("outcome") == "RUN_REFUSED":
            rows[cid] = {"status": "RUN_REFUSED", "reason": done["reason"]}
            continue
        res = R.output_paths(od, fx["video_file"])["results"]
        lab = os.path.join(repo, "R2", "labels", fx["labels_file"])
        frz = os.path.join(repo, "R2", "labels", cid + ".labels_v1.FROZEN.json")
        out = os.path.join(od, cid + ".r2_dev_eval.json")
        if not os.path.exists(out):
            E.main(res, p, lab, frz, "r2_v1", out)
        with open(out) as f:
            rows[cid] = json.load(f)
    with open(os.path.join(out_root, "R2_DEV_EVAL_ALL.json"), "w") as f:
        json.dump(rows, f, indent=1, sort_keys=True)
    print("scored", sum(1 for r in rows.values() if r.get("status") != "NOT_RUN"), "of", len(rows))


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2], sys.argv[3], sys.argv[4], set(sys.argv[5:]) or None)
    elif sys.argv[1] == "score":
        score(sys.argv[2], sys.argv[3])
    else:
        raise SystemExit(__doc__)
