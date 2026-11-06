# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R2 user tracker testing instruction (Nathan, 2026-11-06 12:55): test the frozen 1.0.0-r2prod
#                 tracker on completely new footage; the user identifies the target; export a visual MP4 with the
#                 target box, trusted/untrusted state, rb_state, frame/time and track id. No tracker logic, threshold
#                 or config change; production defaults only. Evaluation tooling.
"""R2 user testing of the frozen production tracker on fresh (non-DEV, non-EVAL) clips.

    python3 tools_rb/r2_user_test.py seeds REPO CLIPS_DIR OUT_DIR        # numbered player stills for the user
    python3 tools_rb/r2_user_test.py run REPO CLIPS_DIR CHOICES_JSON OUT_DIR [CLIP_ID ...]
    python3 tools_rb/r2_user_test.py render REPO CLIPS_DIR OUT_DIR [CLIP_ID ...]   # visual MP4 per completed run
    python3 tools_rb/r2_user_test.py summarise REPO OUT_DIR OUT_JSON

seeds: for every clip, runs the production detector on the first frame and writes <clip>_seed_still.png with every
person candidate numbered (2x upscale), plus <clip>_seed_candidates.json. The user replies with the number of the
target per clip (CHOICES_JSON: {"<clip_id>": <number>, ...}); the chosen box is the seed (the box the user picked,
unchanged). Nothing about the tracker changes: it receives (video, seed_frame 0, box) exactly as in production.
run: TrackerConfig.default() only (no overrides accepted here), clean git tree, sha256 of the clip pinned in the
     registry, debug decision log on. Refusals are recorded, never worked around.
render: per frame draws the target box (green = identity trusted, orange = untrusted TRACKING/AMBIGUOUS/REACQUIRING,
     red dashed = LOST with a held/predicted box), the rb_state, trusted flag, frame/time, the target's tracklet id
     and the other live tracks in thin grey, then encodes H.264 with ffmpeg.
"""
import glob
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np

GREEN = (60, 200, 60)
ORANGE = (0, 165, 255)
RED = (40, 40, 230)
GREY = (170, 170, 170)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _clips(clips_dir, only=None):
    out = []
    for p in sorted(glob.glob(os.path.join(clips_dir, "UT_*.mp4"))):
        cid = os.path.splitext(os.path.basename(p))[0]
        if only is None or cid in only:
            out.append((cid, p))
    return out


def _registry(clips_dir):
    """Clip sha256 list written next to the clips (SHA256SUMS.txt from the extraction machine)."""
    reg = {}
    p = os.path.join(clips_dir, "SHA256SUMS.txt")
    if os.path.exists(p):
        for line in open(p):
            parts = line.split()
            if len(parts) == 2:
                reg[os.path.splitext(parts[1])[0]] = parts[0]
    return reg


def seeds(repo, clips_dir, out_root):
    sys.path.insert(0, repo)
    import cv2
    from player_tracker_rb import run_rb as R
    from player_tracker_rb.config import TrackerConfig
    cfg = TrackerConfig.default()
    det, _ = R.build_backends(cfg)
    os.makedirs(out_root, exist_ok=True)
    for cid, path in _clips(clips_dir):
        cap = cv2.VideoCapture(path)
        ok, img = cap.read()
        cap.release()
        if not ok:
            raise SystemExit(f"{path}: cannot read frame 0")
        cands = det.detect(img)
        cands = sorted(cands, key=lambda c: (c.bbox[0], c.bbox[1]))
        rows = [{"n": i + 1, "bbox": [round(float(v), 1) for v in c.bbox], "score": round(float(c.score), 3),
                 "touches_edge": bool(c.touches_image_edge)} for i, c in enumerate(cands)]
        with open(os.path.join(out_root, cid + "_seed_candidates.json"), "w") as f:
            json.dump({"clip_id": cid, "seed_frame": 0, "frame_size": [img.shape[1], img.shape[0]],
                       "candidates": rows}, f, indent=1)
        big = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        for r in rows:
            x1, y1, x2, y2 = [int(round(v * 2)) for v in r["bbox"]]
            cv2.rectangle(big, (x1, y1), (x2, y2), ORANGE, 1)
            label = str(r["n"])
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            ty = y1 - 4 if y1 - th - 6 > 0 else y2 + th + 4
            cv2.rectangle(big, (x1, ty - th - 4), (x1 + tw + 6, ty + 2), BLACK, -1)
            cv2.putText(big, label, (x1 + 3, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)
        cv2.putText(big, f"{cid}  frame 0  ({len(rows)} candidates)", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    WHITE, 2)
        cv2.imwrite(os.path.join(out_root, cid + "_seed_still.png"), big)
        print(cid, len(rows), "candidates", flush=True)


def run(repo, clips_dir, choices_json, out_root, only=None):
    sys.path.insert(0, repo)
    from player_tracker_rb import provenance as PV
    from player_tracker_rb import run_rb as R
    head, dirty = PV.git_state()
    if dirty:
        raise SystemExit("git tree is dirty: the user-test run must use a clean commit (fail closed)")
    with open(choices_json) as f:
        choices = json.load(f)
    reg = _registry(clips_dir)
    os.makedirs(out_root, exist_ok=True)
    for cid, path in _clips(clips_dir, only):
        if cid not in choices:
            print(cid, "no user choice, skipped", flush=True)
            continue
        od = os.path.join(out_root, cid)
        if os.path.exists(os.path.join(od, "DONE.json")):
            print(cid, "already done", flush=True)
            continue
        if cid not in reg:
            raise SystemExit(f"{cid}: no sha256 in SHA256SUMS.txt (fail closed)")
        cj = os.path.join(out_root, "seeds", cid + "_seed_candidates.json")
        with open(cj) as f:
            cands = {r["n"]: r for r in json.load(f)["candidates"]}
        n = choices[cid]
        if n not in cands:
            raise SystemExit(f"{cid}: candidate {n} does not exist")
        box = cands[n]["bbox"]
        os.makedirs(od, exist_ok=True)
        t0 = time.time()
        print(cid, "start", time.strftime("%H:%M:%S"), "seed", n, box, flush=True)
        done = {"clip_id": cid, "split": "USER_TEST", "seed_frame": 0, "seed_candidate": n, "seed_bbox": box,
                "clip_sha256": reg[cid], "git_commit": head, "git_dirty": dirty}
        try:
            R.run(path, 0, box, out_dir=od, debug=True, expected_sha256=reg[cid])
            done["outcome"] = "COMPLETED"
        except R.RunRefusedError as e:
            done["outcome"] = "RUN_REFUSED"
            done["reason"] = str(e)
        done["seconds"] = round(time.time() - t0, 1)
        with open(os.path.join(od, "DONE.json"), "x") as f:
            json.dump(done, f, indent=1)
        print(cid, done["outcome"], done.get("reason", ""), done["seconds"], "s", flush=True)


def _target_rows(results):
    """frame -> (target v2 block or None, other live track boxes)."""
    rows = {}
    for fl in results["frame_log"]:
        tgt, others = None, []
        for t in fl.get("tracks", []):
            v2 = t.get("v2") or {}
            if t.get("selected_player") or v2.get("record_kind") == "TARGET":
                tgt = dict(v2, bbox=t.get("bbox"), track_id=t.get("track_id"))
            elif t.get("bbox") and t.get("state") not in ("DEAD", "LOST_DEAD"):
                others.append((t.get("track_id"), t["bbox"]))
        rows[fl["frame"]] = (tgt, others, fl.get("v2") or {})
    return rows


def _dashed(img, p1, p2, color, gap=6):
    import cv2
    x1, y1 = p1
    x2, y2 = p2
    for x in range(x1, x2, gap * 2):
        cv2.line(img, (x, y1), (min(x + gap, x2), y1), color, 1)
        cv2.line(img, (x, y2), (min(x + gap, x2), y2), color, 1)
    for y in range(y1, y2, gap * 2):
        cv2.line(img, (x1, y), (x1, min(y + gap, y2)), color, 1)
        cv2.line(img, (x2, y), (x2, min(y + gap, y2)), color, 1)


def render(repo, clips_dir, out_root, only=None):
    sys.path.insert(0, repo)
    import cv2
    from player_tracker_rb import run_rb as R
    for cid, path in _clips(clips_dir, only):
        od = os.path.join(out_root, cid)
        dp = os.path.join(od, "DONE.json")
        if not os.path.exists(dp):
            continue
        done = json.load(open(dp))
        rp = R.output_paths(od, path)["results"]
        rows = _target_rows(json.load(open(rp))) if done["outcome"] == "COMPLETED" else {}
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        scale = 2 if w <= 960 else 1
        raw = os.path.join(od, cid + "_visual_raw.avi")
        vw = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"MJPG"), fps, (w * scale, h * scale))
        f = 0
        while True:
            ok, img = cap.read()
            if not ok:
                break
            if scale != 1:
                img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            tgt, others, fv2 = rows.get(f, (None, [], {}))
            for tid, b in others:
                x1, y1, x2, y2 = [int(round(v * scale)) for v in b]
                cv2.rectangle(img, (x1, y1), (x2, y2), GREY, 1)
            state, trusted, tid = "NO TARGET RECORD", False, "-"
            if f == 0:
                state = "SEED"
                x1, y1, x2, y2 = [int(round(v * scale)) for v in done["seed_bbox"]]
                cv2.rectangle(img, (x1, y1), (x2, y2), GREEN, 2)
            elif tgt is not None:
                state = tgt.get("rb_state") or "?"
                trusted = bool(tgt.get("identity_trusted"))
                tid = str(tgt.get("observation_tracklet_id") or tgt.get("track_id") or "-")
                b = tgt.get("observed_or_contested_box") or tgt.get("bbox")
                if b:
                    x1, y1, x2, y2 = [int(round(v * scale)) for v in b]
                    if state == "LOST" or tgt.get("bbox_is_held") or tgt.get("bbox_is_predicted"):
                        _dashed(img, (x1, y1), (x2, y2), RED if state == "LOST" else ORANGE)
                    else:
                        cv2.rectangle(img, (x1, y1), (x2, y2), GREEN if trusted else ORANGE, 2)
            elif done["outcome"] == "RUN_REFUSED":
                state = "SEED REFUSED"
            col = GREEN if trusted else (RED if state in ("LOST", "SEED REFUSED") else ORANGE)
            t_s = f / fps
            lines = [f"{cid}", f"frame {f}  t={t_s:6.2f}s", f"state {state}   {'TRUSTED' if trusted else 'untrusted'}",
                     f"track {tid}" + ("   CUT" if fv2.get("cut_detected") else "")]
            y = 22
            for i, s in enumerate(lines):
                cv2.putText(img, s, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, BLACK, 3)
                cv2.putText(img, s, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col if i == 2 else WHITE, 1)
                y += 22
            vw.write(img)
            f += 1
        vw.release()
        cap.release()
        mp4 = os.path.join(od, cid + "_visual.mp4")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", raw, "-c:v", "libx264", "-crf", "18", "-pix_fmt",
                        "yuv420p", mp4], check=True)
        os.remove(raw)
        print(cid, "rendered", f, "frames ->", mp4, flush=True)


def summarise(repo, out_root, out_json):
    sys.path.insert(0, repo)
    from player_tracker_rb import run_rb as R
    rows = {}
    for dp in sorted(glob.glob(os.path.join(out_root, "UT_*", "DONE.json"))):
        done = json.load(open(dp))
        cid = done["clip_id"]
        row = {"outcome": done["outcome"], "reason": done.get("reason"), "seed_candidate": done["seed_candidate"]}
        if done["outcome"] == "COMPLETED":
            rp = glob.glob(os.path.join(os.path.dirname(dp), "*" + R.RESULTS_SUFFIX))[0]
            res = json.load(open(rp))
            tr = _target_rows(res)
            n = len(tr)
            states = {}
            trusted = 0
            episodes = []
            cur = None
            for f in sorted(tr):
                tgt = tr[f][0]
                st = (tgt or {}).get("rb_state") or "NONE"
                tru = bool((tgt or {}).get("identity_trusted"))
                states[st] = states.get(st, 0) + 1
                trusted += tru
                key = (st, tru, (tgt or {}).get("observation_tracklet_id"))
                if cur is None or cur["key"] != key:
                    cur = {"key": key, "state": st, "trusted": tru, "tracklet": key[2], "from": f, "to": f}
                    episodes.append(cur)
                else:
                    cur["to"] = f
            for e in episodes:
                e.pop("key")
            row.update({"frames": n, "fps": res["summary"].get("fps"), "trusted_frames": trusted,
                        "trusted_fraction": round(trusted / n, 3) if n else None,
                        "state_frames": states, "rb_state_pct": res["summary"]["v2"].get("rb_state_pct"),
                        "reacquisition_events": res["summary"]["v2"].get("reacquisition_events"),
                        "episodes": episodes})
        rows[cid] = row
    with open(out_json, "w") as f:
        json.dump(rows, f, indent=1, sort_keys=True)
    for cid, r in rows.items():
        print(cid, r["outcome"], r.get("trusted_fraction"), r.get("rb_state_pct"), (r.get("reason") or "")[:80])


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "seeds" and len(a) == 4:
        seeds(*a[1:])
    elif a and a[0] == "run" and len(a) >= 5:
        run(a[1], a[2], a[3], a[4], set(a[5:]) or None)
    elif a and a[0] == "render" and len(a) >= 4:
        render(a[1], a[2], a[3], set(a[4:]) or None)
    elif a and a[0] == "summarise" and len(a) == 4:
        summarise(*a[1:])
    else:
        raise SystemExit(__doc__)
