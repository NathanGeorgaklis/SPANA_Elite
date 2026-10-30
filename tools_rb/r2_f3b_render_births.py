# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R2 DEV remediation instruction (Nathan, 2026-10-31 05:05, F3b diagnosis): visual check of F3b-
#                 geometry births on DEV footage (DEV only; asserts the fixture split)
"""Render a contact sheet of a random sample of births >= 0.5 inside another observed body (DEV only).

    python3 tools_rb/r2_f3b_render_births.py REPO OUT_DIR CLIPS_DIR BIRTHS_JSON N OUT_PNG
"""
import json
import random
import sys

import cv2


def main(repo, out_dir, clips, births_json, n, out_png):
    d = json.load(open(births_json))
    rows = [r for r in d["births"] if r[3] >= 0.5]
    random.Random(7).shuffle(rows)
    rows = sorted(rows[:int(n)], key=lambda r: (r[0], r[2]))
    caps, logs, tiles, meta = {}, {}, [], []
    for k, (cid, tid, frame, cb, io, hr, e, p) in enumerate(rows):
        fx = json.load(open(f"{repo}/R2/labels/{cid}.fixture.json"))
        assert fx["split"] == "DEV", cid
        if cid not in logs:
            logs[cid] = [json.loads(line) for line in open(f"{out_dir}/{cid}/{cid}_rb_decision_log.jsonl")]
        pop = logs[cid][frame]["population"]
        obs = pop["observations"]
        trk = {v["tracklet_id"]: v for v in pop["tracklets"]}
        b = obs[trk[tid]["detection_index"]]["bbox"]
        best = None
        for t2, v in trk.items():
            if t2 == tid or v.get("detection_index") is None or v["observation_source"] != "DETECTION" \
                    or v["first_frame"] == frame:
                continue
            b2 = obs[v["detection_index"]]["bbox"]
            ix = max(0, min(b[2], b2[2]) - max(b[0], b2[0])) * max(0, min(b[3], b2[3]) - max(b[1], b2[1]))
            c = ix / ((b[2] - b[0]) * (b[3] - b[1]))
            if best is None or c > best[0]:
                best = (c, b2, t2)
        cap = caps.setdefault(cid, cv2.VideoCapture(f"{clips}/{fx['video_file']}"))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame + fx["seed_frame"])
        ok, img = cap.read()
        b2 = best[1]
        cv2.rectangle(img, (int(b2[0]), int(b2[1])), (int(b2[2]), int(b2[3])), (0, 255, 0), 2)
        cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), 1)
        x0, y0 = min(b[0], b2[0]), min(b[1], b2[1])
        x1, y1 = max(b[2], b2[2]), max(b[3], b2[3])
        s = max(x1 - x0, y1 - y0) * 1.3 + 20
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        crop = img[int(max(0, cy - s / 2)):int(cy + s / 2), int(max(0, cx - s / 2)):int(cx + s / 2)]
        crop = cv2.resize(crop, (220, 220))
        cv2.putText(crop, f"#{k} c{cb:.2f}", (3, 14), 0, 0.45, (255, 255, 255), 1)
        tiles.append(crop)
        meta.append((k, cid, tid, frame, round(cb, 2), round(hr, 2)))
    blank = tiles[0] * 0
    grid = [cv2.hconcat(tiles[i:i + 6] + [blank] * (6 - len(tiles[i:i + 6]))) for i in range(0, len(tiles), 6)]
    cv2.imwrite(out_png, cv2.vconcat(grid))
    json.dump(meta, open(out_png + ".json", "w"))
    print(meta)


if __name__ == "__main__":
    main(*sys.argv[1:7])
