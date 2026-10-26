# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R2/R2_PREREGISTRATION.md §3 (Task 3J extraction method: lossless libx264 -qp 0, framemd5 match
#                 against the source decode of the same range, sha256 recorded)
"""Extract the selected R2 clips losslessly and verify them. No tracker, detector or model is involved.

    python3 tools_rb/r2_extract.py SELECTION_JSON SOURCE_DIR OUT_DIR [CLIP_ID ...]
    python3 tools_rb/r2_extract.py registry SELECTION_JSON OUT_DIR [SOURCE_DIR] > R2/R2_CLIP_REGISTRY.json
        (re-hashes every clip and assembles the extract records; fails closed on any mismatch)

For each clip (all, or the ids given): validates the window with r2_protocol.check_window (and that its split is the
protocol's), verifies the source sha256 (once per source per run), extracts frames
[round(start_s * fps), round(end_s * fps)) to OUT_DIR/<clip_id>.mp4, then verifies:
  * framemd5 of every extracted frame == framemd5 of the source decode of the same range (lossless);
  * alignment: the first 3 frame md5s also match a source decode that seeks 2 s earlier and skips 2 s of frames
    (a different seek path, so a mis-seek would show);
  * frame count == expected.
Writes OUT_DIR/<clip_id>.extract.json (provenance + verification). Refuses to overwrite an existing clip, and never
deletes anything: a clip that fails verification is renamed *.FAILED_VERIFICATION.mp4 and never published.
"""
import hashlib
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools_rb import r2_protocol as P  # noqa: E402

ALIGN_BACK_S = 2.0
ALIGN_FRAMES = 3


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def _framemd5(args):
    out = subprocess.run(["ffmpeg", "-v", "error"] + args + ["-an", "-pix_fmt", "yuv420p", "-f", "framemd5", "-"],
                         check=True, capture_output=True, text=True).stdout
    return [line.rsplit(",", 1)[1].strip() for line in out.splitlines() if line and not line.startswith("#")]


def extract_one(clip, source_dir, out_dir, verified_sources):
    src_meta = P.SOURCES[clip["source"]]
    split = P.check_window(clip["source"], clip["start_s"], clip["end_s"])
    fps = src_meta["fps"]
    f0, f1 = round(clip["start_s"] * fps), round(clip["end_s"] * fps)
    if abs(clip["start_s"] * fps - f0) > 1e-6 or abs(clip["end_s"] * fps - f1) > 1e-6:
        raise SystemExit(f"{clip['clip_id']}: window is not frame-aligned at {fps} fps (fail closed)")
    n = f1 - f0
    src = os.path.join(source_dir, src_meta["file"])
    if clip["source"] not in verified_sources:
        got = _sha(src)
        if got != src_meta["sha256"]:
            raise SystemExit(f"source {src} sha256 {got} != registered {src_meta['sha256']} (fail closed)")
        verified_sources.add(clip["source"])
    out = os.path.join(out_dir, clip["clip_id"] + ".mp4")
    rec_path = os.path.join(out_dir, clip["clip_id"] + ".extract.json")
    if os.path.exists(out) or os.path.exists(rec_path):
        raise SystemExit(f"{out} exists; refusing to overwrite")
    ss = f"{f0 / fps:.6f}"
    tmp = out + ".partial.mp4"
    cmd = ["ffmpeg", "-v", "error", "-ss", ss, "-i", src, "-frames:v", str(n), "-an", "-c:v", "libx264", "-qp", "0",
           "-preset", "veryfast", "-pix_fmt", "yuv420p", "-r", f"{fps:g}", tmp]
    subprocess.run(cmd, check=True)
    src_md5 = _framemd5(["-ss", ss, "-i", src, "-frames:v", str(n)])
    ext_md5 = _framemd5(["-i", tmp])
    back = max(0.0, f0 / fps - ALIGN_BACK_S)
    skip = round(f0 - back * fps)
    alt = _framemd5(["-ss", f"{back:.6f}", "-i", src, "-frames:v", str(skip + ALIGN_FRAMES)])[skip:]
    ok = (len(src_md5) == n and ext_md5 == src_md5 and alt == src_md5[:ALIGN_FRAMES])
    if not ok:
        os.rename(tmp, out + ".FAILED_VERIFICATION.mp4")      # kept for inspection; never published
        raise SystemExit(f"{clip['clip_id']}: verification failed (n={n}, src={len(src_md5)}, ext={len(ext_md5)}, "
                         f"lossless={ext_md5 == src_md5}, aligned={alt == src_md5[:ALIGN_FRAMES]})")
    if os.path.exists(out):
        raise SystemExit(f"{out} appeared during extraction; refusing to overwrite")
    os.rename(tmp, out)
    rec = {"clip_id": clip["clip_id"], "split": split, "source": clip["source"], "source_file": src_meta["file"],
           "source_sha256": src_meta["sha256"], "fps": fps, "start_s": clip["start_s"], "end_s": clip["end_s"],
           "source_start_frame": f0, "source_end_frame_exclusive": f1, "frames": n,
           "ffmpeg_command": cmd[:-1] + [os.path.basename(out)],
           "framemd5_match": True, "alignment_check": {"seek_back_s": ALIGN_BACK_S, "frames": ALIGN_FRAMES, "match": True},
           "framemd5_sha256": hashlib.sha256("\n".join(ext_md5).encode()).hexdigest(),
           "clip_sha256": _sha(out), "clip_bytes": os.path.getsize(out), "protocol_version": P.PROTOCOL_VERSION}
    with open(rec_path, "x") as f:
        json.dump(rec, f, indent=1, sort_keys=True)
        f.write("\n")
    return rec


def main(selection_json, source_dir, out_dir, ids):
    with open(selection_json) as f:
        sel = json.load(f)
    os.makedirs(out_dir, exist_ok=True)
    clips = [c for c in sel["clips"] if not ids or c["clip_id"] in ids]
    if ids and len(clips) != len(ids):
        raise SystemExit("unknown clip id")
    verified = set()
    for c in clips:
        r = extract_one(c, source_dir, out_dir, verified)
        print(r["clip_id"], r["split"], r["frames"], r["clip_sha256"][:16], r["clip_bytes"])


def source_pts(src, f0, fps):
    """Absolute timestamp of the first frame decoded at the clip's seek point (showinfo with -copyts): proves the clip
    starts at source frame f0 = pts * fps, not merely that two seek paths agree (review finding L11)."""
    out = subprocess.run(["ffmpeg", "-v", "info", "-copyts", "-ss", f"{f0 / fps:.6f}", "-i", src, "-frames:v", "1",
                          "-vf", "showinfo", "-f", "null", "-"], capture_output=True, text=True, check=True).stderr
    for line in out.splitlines():
        if "pts_time:" in line:
            return float(line.split("pts_time:")[1].split()[0])
    raise SystemExit("showinfo produced no pts_time")


def registry(selection_json, out_dir, source_dir=None):
    with open(selection_json) as f:
        sel = json.load(f)
    clips = []
    for c in sel["clips"]:
        with open(os.path.join(out_dir, c["clip_id"] + ".extract.json")) as f:
            rec = json.load(f)
        if _sha(os.path.join(out_dir, c["clip_id"] + ".mp4")) != rec["clip_sha256"]:
            raise SystemExit(f"{c['clip_id']}: clip hash differs from its extract record (fail closed)")
        if (rec["start_s"], rec["end_s"], rec["source"]) != (c["start_s"], c["end_s"], c["source"]):
            raise SystemExit(f"{c['clip_id']}: extract record differs from the selection")
        if rec["split"] != P.split_of(c["source"], c["start_s"], c["end_s"]):
            raise SystemExit(f"{c['clip_id']}: split differs from the protocol")
        extra = {}
        if source_dir:
            src = os.path.join(source_dir, P.SOURCES[c["source"]]["file"])
            pts = source_pts(src, rec["source_start_frame"], rec["fps"])
            if abs(pts * rec["fps"] - rec["source_start_frame"]) > 1e-3:
                raise SystemExit(f"{c['clip_id']}: first frame pts {pts} is not source frame "
                                 f"{rec['source_start_frame']} (fail closed)")
            extra["source_first_frame_pts_s"] = pts
        clips.append(dict(rec, classes=c["classes"], proposed_target=c["proposed_target"],
                          proposed_seed_s=c["proposed_seed_s"], **extra))
    return {"schema": "r2_clip_registry_v1", "protocol_version": P.PROTOCOL_VERSION,
            "source_checks": {k: {"pix_fmt": v["pix_fmt"], "start_time_s": 0.0, "constant_fps": v["fps"]}
                              for k, v in P.SOURCES.items()} if source_dir else None,
            "clips_location": "~/Downloads/_spana_r2_work/clips (not in Git: large binaries; archived separately)",
            "n_clips": len(clips), "n_dev": sum(c["split"] == "DEV" for c in clips),
            "n_eval": sum(c["split"] == "EVAL" for c in clips), "clips": clips}


if __name__ == "__main__":
    if sys.argv[1] == "registry":
        json.dump(registry(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else None), sys.stdout,
                  indent=1, sort_keys=True)
        print()
    else:
        main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:])
