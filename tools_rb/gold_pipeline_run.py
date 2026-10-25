# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R1-G instruction (Nathan, 2026-10-25: GOLD only as development footage, not final benchmark
#                 scoring); R1/gold_label_sheet (frozen human tracking labels v1, identity/visibility only)
"""Integrated Pass-1 pipeline on a development clip (no events, no benchmark scoring).

    run      VIDEO FIXTURE_JSON LABELS_CSV OUT_DIR
             Runs player_tracker_rb.run_rb.run with the configured (production) backends, debug decision log on.
             From the fixture only video_sha256, seed_frame and seed_bbox are read. The frozen human labels are read
             only AFTER the run, to summarise the tracker's states per labelled frame. Writes OUT_DIR/
             gold_pipeline_report.json and OUT_DIR/gold_pipeline_contact_sheet.jpg next to the pipeline outputs.
    compare  OUT_DIR_A OUT_DIR_B VIDEO
             Determinism: compares two runs' documents (everything except the documented wall-clock fields must be
             identical) and their decision logs (byte-identical).
"""
import hashlib
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from player_tracker_rb import output_schema as S  # noqa: E402
from player_tracker_rb import run_rb as R  # noqa: E402
from tools_rb.target_eval import label_class, read_fixture, read_labels  # noqa: E402

STATE_COLOUR = {"TRACKING": (0, 255, 0), "OCCLUDED": (0, 165, 255), "AMBIGUOUS": (0, 255, 255),
                "LOST": (0, 0, 255), "REACQUIRING": (255, 0, 255)}
SHEET_FROM_BEFORE_LABELS = 10
SHEET_STEP = 2
SHEET_COLS = 4
SHEET_TILE_W = 480
SHEET_JPEG_QUALITY = 80


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def run(video, fx_json, labels_csv, out_dir):
    fx = read_fixture(fx_json)
    summary = R.run(video, fx["seed_frame"], fx["seed_bbox"], out_dir=out_dir, debug=True,
                    expected_sha256=fx["video_sha256"])
    paths = R.output_paths(out_dir, video)
    with open(paths["results"], encoding="ascii") as fh:
        doc = json.load(fh)
    errors = S.validate_document(doc, require_v2=True)
    errors_hist = S.validate_document(S.strip_v2(doc))
    # labels: read only now, after the tracker has finished
    labels = read_labels(labels_csv, fx["seed_frame"])
    sel = {fr["frame"]: [t for t in fr["tracks"] if t["selected_player"]][0] for fr in doc["frame_log"]}
    outside = [rel for rel in labels if rel not in sel]
    if outside:
        raise SystemExit(f"label frames outside the run: {sorted(outside)[:5]} (fail closed)")
    by_class = {}
    for rel, row in sorted(labels.items()):
        cl = label_class(row)
        d = by_class.setdefault(cl, {"frames": 0, "trusted": 0, "states": {}, "trusted_frames": []})
        d["frames"] += 1
        st = sel[rel]["v2"]["rb_state"]
        d["states"][st] = d["states"].get(st, 0) + 1
        if sel[rel]["v2"]["identity_trusted"]:
            d["trusted"] += 1
            d["trusted_frames"].append(rel)
    states = {}
    for t in sel.values():
        states[t["v2"]["rb_state"]] = states.get(t["v2"]["rb_state"], 0) + 1
    transitions = []
    prev = None
    for f in sorted(sel):
        v2 = sel[f]["v2"]
        key = (v2["rb_state"], tuple(v2["identity_uncertainty_reasons"]))
        if key != prev:
            transitions.append({"frame": f, "rb_state": v2["rb_state"],
                                "reasons": v2["identity_uncertainty_reasons"], "episode": v2["target_episode"],
                                "identity_confidence_rb": v2["identity_confidence_rb"]})
            prev = key
    lo, hi = min(labels), max(labels)
    trusted_conf = [sel[f]["v2"]["identity_confidence_rb"] for f in sorted(sel) if sel[f]["v2"]["identity_trusted"]]
    report = {
        "tool": "tools_rb/gold_pipeline_run.py", "development_only": True, "events_evaluated": False,
        "labels_used_by_tracker": False, "labels_file_sha256": _sha(labels_csv),
        "results_file": os.path.basename(paths["results"]), "results_sha256": _sha(paths["results"]),
        "contract_errors_r1": errors, "contract_errors_historical_after_strip": errors_hist,
        "processed_frames": summary["processed_frames"], "processing_time_s": summary["processing_time_s"],
        "fps": summary["fps"], "state_counts": states, "selected_state_pct": summary["selected_state_pct"],
        "rb_state_pct": summary["v2"]["rb_state_pct"], "transitions": transitions,
        "reacquisition_events": summary["v2"]["reacquisition_events"],
        "identity_conflict_counts": summary["identity_conflict_counts"],
        "identity_conflicts_logged": summary["identity_conflicts_logged"],
        "n_tracks_ever_created": summary["n_tracks_ever_created"],
        "trusted_identity_confidence_rb": {"min": min(trusted_conf) if trusted_conf else None,
                                           "max": max(trusted_conf) if trusted_conf else None},
        "label_window": [lo, hi], "by_label_class": by_class,
        "provenance": summary["v2"]["provenance"],
    }
    # contact sheet of the label window from the written records (what Pass 2 would read)
    tiles = []
    cap = cv2.VideoCapture(video)
    for i in range(fx["seed_frame"] + 1):
        cap.grab()
    for f in range(1, max(sel) + 1):
        ok, img = cap.read()
        if not ok:
            break
        if lo - SHEET_FROM_BEFORE_LABELS <= f <= hi and (f - lo) % SHEET_STEP == 0:
            fr = doc["frame_log"][f - 1]
            out = img.copy()
            for t in fr["tracks"]:
                if not t["selected_player"]:
                    x1, y1, x2, y2 = (int(round(v)) for v in t["observed_position_box"])
                    cv2.rectangle(out, (x1, y1), (x2, y2), (160, 160, 160), 1)
            t = sel[f]
            v2 = t["v2"]
            col = STATE_COLOUR[v2["rb_state"]]
            x1, y1, x2, y2 = (int(round(v)) for v in t["observed_position_box"])
            cv2.rectangle(out, (x1, y1), (x2, y2), col, 3 if v2["identity_trusted"] else 1)
            if v2["observed_or_contested_box"] is not None and not v2["identity_trusted"]:
                x1, y1, x2, y2 = (int(round(v)) for v in v2["observed_or_contested_box"])
                cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
            cv2.putText(out, f"f {f} {v2['rb_state']} ep{v2['target_episode']} c{v2['identity_confidence_rb']:.2f}",
                        (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(out, ",".join(v2["identity_uncertainty_reasons"])[:60], (8, 54), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2, cv2.LINE_AA)
            tiles.append(cv2.resize(out, (SHEET_TILE_W, int(round(out.shape[0] * SHEET_TILE_W / out.shape[1])))))
    cap.release()
    if tiles:
        while len(tiles) % SHEET_COLS:
            tiles.append(np.zeros_like(tiles[0]))
        sheet = np.vstack([np.hstack(tiles[k:k + SHEET_COLS]) for k in range(0, len(tiles), SHEET_COLS)])
        cv2.imwrite(os.path.join(out_dir, "gold_pipeline_contact_sheet.jpg"), sheet,
                    [cv2.IMWRITE_JPEG_QUALITY, SHEET_JPEG_QUALITY])
    with open(os.path.join(out_dir, "gold_pipeline_report.json"), "x", encoding="ascii") as fh:
        json.dump(report, fh, indent=1, sort_keys=True)
    print(json.dumps({k: report[k] for k in ("contract_errors_r1", "contract_errors_historical_after_strip",
                                             "state_counts", "by_label_class", "reacquisition_events",
                                             "identity_conflict_counts", "processing_time_s")}, indent=1))


def compare(dir_a, dir_b, video):
    pa, pb = R.output_paths(dir_a, video), R.output_paths(dir_b, video)
    with open(pa["results"], encoding="ascii") as fa, open(pb["results"], encoding="ascii") as fb:
        a, b = json.load(fa), json.load(fb)
    cmp = R.compare_documents(a, b)
    logs_identical = _sha(pa["decision_log"]) == _sha(pb["decision_log"])
    out = {"documents_identical_except_wall_clock": cmp["identical_except_wall_clock"],
           "first_difference": cmp["first_difference"], "wall_clock": cmp["wall_clock"],
           "decision_logs_byte_identical": logs_identical}
    print(json.dumps(out, indent=1))
    with open(os.path.join(dir_b, "determinism_vs_" + os.path.basename(dir_a.rstrip("/")) + ".json"), "x",
              encoding="ascii") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)


if __name__ == "__main__":
    if len(sys.argv) == 6 and sys.argv[1] == "run":
        run(*sys.argv[2:6])
    elif len(sys.argv) == 5 and sys.argv[1] == "compare":
        compare(*sys.argv[2:5])
    else:
        raise SystemExit(__doc__)
