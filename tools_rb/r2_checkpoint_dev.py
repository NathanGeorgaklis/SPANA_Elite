# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R2 DEV human-assisted target reacquisition instruction (Nathan, 2026-11-05 20:22): measure DEV
#                 coverage recoverable with correct user confirmation, wrong trusted frames, bank contamination,
#                 checkpoints required, continuity of downstream analysis. DEV only; never EVAL.
"""R2 DEV runs with SIMULATED user confirmations at identity checkpoints. Evaluation tooling only.

    python3 tools_rb/r2_checkpoint_dev.py run REPO CLIPS_DIR OUT_DIR [CLIP_ID ...]
    python3 tools_rb/r2_checkpoint_dev.py summarise REPO OUT_DIR BASE_OUT_DIR OUT_JSON

run: like tools_rb/r2_dev_run.py run (DEV fixtures only, clean git tree, human seed, pinned video sha, debug log),
plus a checkpoint resolver that SIMULATES A CORRECT USER from the frozen DEV human labels (labels_v1.json, sha256
checked against the fixture). The tracker itself never reads labels: the only label information that reaches it is
the box a correct user would draw when asked at a checkpoint. At checkpoint frame f (clip frame = seed_frame + f):
  * target VISIBLE / PARTIAL, resolvable YES, not HUMAN_UNCERTAIN, with a human box at f -> SELECT that box;
  * target VISIBLE / PARTIAL and resolvable but no human box at f (boxes are on a 5 Hz grid) -> no answer yet;
  * otherwise (HIDDEN, OUT_OF_FRAME, resolvable NO, HUMAN_UNCERTAIN) -> NOT_VISIBLE (the user cannot identify).
Scored afterwards by tools_rb/r2_dev_run.py score and compared with tools_rb/r2_dev_compare.py.
summarise: checkpoint statistics per clip (opened / re-seeded / declined / auto-resolved / refused answers, answer
delays, re-seed frames) and the official before/after metrics side by side.
"""
import glob
import hashlib
import json
import os
import sys
import time

VIS_OK = ("VISIBLE", "PARTIAL")


def _sha(p):
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def correct_user(repo, fx):
    lp = os.path.join(repo, "R2", "labels", fx["labels_file"])
    if _sha(lp) != fx["labels_sha256"]:
        raise SystemExit(f"{lp}: sha256 differs from the fixture (fail closed)")
    with open(lp) as f:
        lab = json.load(f)
    s0 = fx["seed_frame"]

    def resolve(view, res):
        cf = s0 + view["frame"]
        if cf >= len(lab["per_frame"]):
            return None
        x = lab["per_frame"][cf]
        ok = x["visibility"] in VIS_OK and x["resolvable"] == "YES" and not x["human_uncertain"]
        if ok:
            b = lab["target_boxes"].get(str(cf))
            return None if b is None else {"decision": "SELECT", "box": list(b)}
        return {"decision": "NOT_VISIBLE"}
    return resolve


def run(repo, clips_dir, out_root, only=None):
    sys.path.insert(0, repo)
    from player_tracker_rb import provenance as PV
    from player_tracker_rb import run_rb as R
    from tools_rb import r2_dev_run as D
    head, dirty = PV.git_state()
    if dirty:
        raise SystemExit("git tree is dirty: the DEV run must use a clean commit (fail closed)")
    os.makedirs(out_root, exist_ok=True)
    for p, fx in D._fixtures(repo, only):
        if fx["split"] != "DEV":
            raise SystemExit("non-DEV fixture reached the runner (fail closed)")
        cid = fx["clip_id"]
        od = os.path.join(out_root, cid)
        if os.path.exists(os.path.join(od, "DONE.json")):
            print(cid, "already done", flush=True)
            continue
        os.makedirs(od, exist_ok=True)
        t0 = time.time()
        print(cid, "start", time.strftime("%H:%M:%S"), flush=True)
        try:
            R.run(os.path.join(clips_dir, fx["video_file"]), fx["seed_frame"], fx["seed_bbox"], out_dir=od,
                  debug=True, expected_sha256=fx["video_sha256"], checkpoint_resolver=correct_user(repo, fx))
            done = {"clip_id": cid, "split": "DEV", "outcome": "COMPLETED", "mode": "SIMULATED_CORRECT_USER",
                    "fixture": os.path.relpath(p, repo), "git_commit": head, "git_dirty": dirty,
                    "seconds": round(time.time() - t0, 1)}
        except R.RunRefusedError as e:
            done = {"clip_id": cid, "split": "DEV", "outcome": "RUN_REFUSED", "reason": str(e),
                    "mode": "SIMULATED_CORRECT_USER", "fixture": os.path.relpath(p, repo), "git_commit": head,
                    "git_dirty": dirty, "seconds": round(time.time() - t0, 1)}
        with open(os.path.join(od, "DONE.json"), "x") as f:
            json.dump(done, f, indent=1)
        print(cid, done["outcome"], done["seconds"], "s", flush=True)


def summarise(repo, out_root, base_root, out_json):
    sys.path.insert(0, repo)
    from player_tracker_rb import run_rb as R
    from tools_rb import r2_dev_run as D
    rows = {}
    for p, fx in D._fixtures(repo):
        cid = fx["clip_id"]
        od = os.path.join(out_root, cid)
        if not os.path.exists(os.path.join(od, "DONE.json")):
            continue
        row = {}
        ck = R.output_paths(od, fx["video_file"])["checkpoints"]
        if os.path.exists(ck):
            with open(ck) as f:
                cps = json.load(f)["checkpoints"]
            st = [c["status"] for c in cps]
            ans = [a for c in cps for a in c["answers"]]
            row["checkpoints"] = {"opened": len(cps), "reseeded": st.count("RESEEDED"),
                                  "declined": st.count("DECLINED"), "auto_resolved": st.count("AUTO_RESOLVED"),
                                  "open_at_end": st.count("OPEN"),
                                  "refused_answers": sum(a.get("outcome") == "REFUSED" for a in ans),
                                  "refusal_reasons": sorted({a.get("reason", "")[:90] for a in ans
                                                             if a.get("outcome") == "REFUSED"}),
                                  "reseed_frames": [c["closed_frame"] for c in cps if c["status"] == "RESEEDED"],
                                  "answer_delays_s": [round(c["closed_time_s"] - c["opened_time_s"], 3)
                                                      for c in cps if c["status"] == "RESEEDED"]}
        from tools_rb import r2_dev_compare as C
        for tag, root in (("after", out_root), ("before", base_root)):
            if os.path.exists(os.path.join(root, cid, "DONE.json")) and (
                    os.path.exists(os.path.join(root, cid, cid + ".r2_dev_eval.json"))
                    or json.load(open(os.path.join(root, cid, "DONE.json"))).get("outcome") == "RUN_REFUSED"):
                row[tag] = C.official(root, cid)       # the official tracker_eval counts, copied unchanged
        rows[cid] = row
    with open(out_json, "w") as f:
        json.dump(rows, f, indent=1, sort_keys=True)
    print(json.dumps({k: v.get("checkpoints") for k, v in rows.items()}, indent=1))


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "run":
        run(a[1], a[2], a[3], set(a[4:]) or None)
    elif a and a[0] == "summarise" and len(a) == 5:
        summarise(*a[1:])
    else:
        raise SystemExit(__doc__)
