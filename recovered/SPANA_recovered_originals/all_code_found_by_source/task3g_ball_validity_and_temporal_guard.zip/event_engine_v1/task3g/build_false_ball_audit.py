"""Task 3G deliverable 5: DUEL_LOST_FALSE_BALL_CANDIDATE_AUDIT.csv + crops.
Every frame around the three fabricated interactions in duel_lost_01 (pre-fix
capture), with the candidate's own features, its context, and the credibility
gate's verdict. Video read sequentially after one seek (the way both passes read
it). Visual verdicts are from the Task 3F/3G pixel inspections."""
import sys, csv, json, os
sys.path.insert(0, ".")
import cv2
from football_analysis.ball_interaction_credibility import (FrameBallEvidence, assess, in_lower_body_zone,
                                                            physically_sized, split_segments)

VIDEO = "/root/.claude/uploads/98d69dca-3149-5a43-9670-3c417f2af6d4/8a710009-dual_lost_01.mov"
d = json.load(open("event_engine_v1/task3g/capture/duel_lost_01_prefix_capture.json"))
F = d["frames"]; S = d["start_frame"]
OUT = "event_engine_v1/task3g/false_ball_crops"; os.makedirs(OUT, exist_ok=True)
WINDOWS = [(50, 70, "RECEIVE->DUEL"), (126, 160, "CARRY->PASS"), (176, 202, "RECEIVE->CROSS")]
VISUAL = {57: "player shadow (pixel zoom)", 58: "player shadow (pixel zoom)", 59: "player shadow (pixel zoom)",
          145: "player boot (pixel zoom)", 147: "player boot (pixel zoom)",
          183: "white blob on centre-circle arc beside boots -- not confirmable as ball",
          184: "white blob on centre-circle arc beside boots -- not confirmable as ball"}

ev = []
for f in F:
    b = f["best"]
    ev.append(FrameBallEvidence(f["frame"], f["state"], tuple(f["center"]) if f["center"] else None,
                                b["box"] if b else None, f["stage"] if b else None, b["conf"] if b else None,
                                bool(f["had_confirmed_before"] and f["prev_fsc"] <= 10), f["player_bbox"]))
labels, verdicts = assess(ev)
seg_of = {}
for k, idxs in enumerate(split_segments(ev)):
    for i in idxs:
        seg_of[i] = (k, idxs[0])

cap = cv2.VideoCapture(VIDEO); cap.set(cv2.CAP_PROP_POS_FRAMES, S); frames = []
while len(frames) < len(F) + 2:
    ok, img = cap.read()
    if not ok: break
    frames.append(img)

rows = []
for lo, hi, name in WINDOWS:
    for i in range(lo, hi + 1):
        f = F[i]; b = f["best"]; pb = f["player_bbox"]
        row = {"window": name, "frame": f["frame"], "video_time_s": round((S + f["frame"]) / d["fps"], 3),
               "ball_state": f["state"], "source_stage": f["stage"] if b else None}
        c = f["center"]
        row.update({"x": round(c[0], 1) if c else None, "y": round(c[1], 1) if c else None})
        if b:
            x1, y1, x2, y2 = b["box"]; w, h = x2 - x1, y2 - y1
            ph = pb[3] - pb[1]
            row.update({"bbox": [round(v, 1) for v in b["box"]], "detector_conf": round(b["conf"], 4),
                        "size_w": round(w, 1), "size_h": round(h, 1), "aspect": round(max(w, h) / max(1e-6, min(w, h)), 2),
                        "size_over_player_height": round(max(w, h) / ph, 3),
                        "physically_sized": physically_sized(b["box"], pb),
                        "staticdecoy_classification": b["classification"], "decoy_score": b["decoy_score"],
                        "static_probability": b["static_prob"], "residual_motion": b["residual"],
                        "pitch_region": b["pitch_region"], "final_ball_score": b["final_ball_score"]})
            crop = frames[f["frame"]][max(0, int(y1) - 30):int(y2) + 30, max(0, int(x1) - 30):int(x2) + 30]
            if crop.size:
                p = f"{OUT}/f{f['frame']:03d}_{(f['stage'] or 'x')}.png"
                cv2.imwrite(p, cv2.resize(crop, (crop.shape[1] * 4, crop.shape[0] * 4), interpolation=cv2.INTER_NEAREST))
                row["patch_file"] = p
        if c and pb:
            fx, fy = (pb[0] + pb[2]) / 2, pb[3]
            row["dist_to_target_feet"] = round(((c[0] - fx) ** 2 + (c[1] - fy) ** 2) ** 0.5, 1)
            dx = max(pb[0] - c[0], 0, c[0] - pb[2]); dy = max(pb[1] - c[1], 0, c[1] - pb[3])
            row["dist_to_target_bbox"] = round((dx * dx + dy * dy) ** 0.5, 1)
            row["rel_x_in_bbox"] = round((c[0] - pb[0]) / (pb[2] - pb[0]), 2)
            row["rel_y_in_bbox"] = round((c[1] - pb[1]) / (pb[3] - pb[1]), 2)
            row["in_lower_body_zone"] = in_lower_body_zone(c[0], c[1], pb)
        if i > 0 and c and F[i - 1]["center"]:
            p = F[i - 1]["center"]
            row["displacement_px"] = round(((c[0] - p[0]) ** 2 + (c[1] - p[1]) ** 2) ** 0.5, 1)
            bg0, bg1 = F[i - 1]["bg_cum"], f["bg_cum"]
            cdx, cdy = (c[0] - bg1[0]) - (p[0] - bg0[0]), (c[1] - bg1[1]) - (p[1] - bg0[1])
            row["camera_compensated_displacement_px"] = round((cdx * cdx + cdy * cdy) ** 0.5, 1)
        k, root_i = seg_of.get(i, (None, None))
        row.update({"segment_id": k, "track_age_frames": (i - root_i) if root_i is not None else None,
                    "frames_since_confirmed_before_step": f["prev_fsc"],
                    "prediction_trusted": ev[i].prediction_trusted, "reacquire_streak": f["reacquire_streak"],
                    "had_confirmed_ball_before": f["had_confirmed_before"],
                    "touch_trigger_in_possession_raw": f.get("in_possession_raw"),
                    "touch_trigger_in_possession_segmented": f.get("in_possession_segmented"),
                    "credibility_label": labels[i],
                    "segment_reasons": " | ".join(verdicts[k].reasons) if k is not None else None,
                    "visual_verdict": VISUAL.get(f["frame"])})
        rows.append(row)

keys = []
for r in rows:
    for k in r:
        if k not in keys: keys.append(k)
with open("event_engine_v1/task3g/DUEL_LOST_FALSE_BALL_CANDIDATE_AUDIT.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
print(len(rows), "rows,", sum(1 for r in rows if r.get("patch_file")), "crops")
