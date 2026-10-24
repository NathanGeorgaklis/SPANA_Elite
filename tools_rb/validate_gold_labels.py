# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R0_TRACKER_ACCEPTANCE_TARGETS.md (Part B truth model); R1_APPROVALS.md §8 (validate/hash/freeze);
#                 tools_rb/make_gold_label_sheet.py (LABEL_COLUMNS, value vocabularies)
"""
validate_gold_labels.py — validates and freezes Nathan's GOLD human identity labels.

This is NEW tooling, not part of the tracker. It never edits the label file.

Checks:
  - The header matches the template columns exactly.
  - There is one row per frame, covering FIRST..LAST, with no duplicates and none missing.
  - Each video_time_s equals frame / fps (to 3 dp).
  - Every value comes from the allowed vocabulary. Blank is allowed only where the instructions allow it.
  - first_clean_reappearance is 'Y' on at most one row.

If the file is valid and --freeze is given, it writes GOLD_HUMAN_LABELS_v1.FROZEN.json next to the CSV. That record holds:
  - the CSV's SHA-256, size and row count;
  - a summary (HUMAN_UNCERTAIN frames, hidden and merged frames, first clean reappearance);
  - a FROZEN flag and a UTC timestamp.

It refuses to overwrite an existing FROZEN record whose hash differs. Corrections go in a v2 file.

Exit codes: 0 = valid; 1 = invalid; 2 = refused (frozen-record conflict).

Usage:
  python3 validate_gold_labels.py LABELS_CSV [--first 250] [--last 290] [--fps 30.0] [--freeze]
"""
import argparse
import csv
import datetime
import hashlib
import json
import os
import sys

COLUMNS = ["frame", "video_time_s", "t23_visibility", "t23_identity_resolvable", "t23_location_note",
           "mar4_visibility", "overlap_23_4", "merged_or_hidden", "first_clean_reappearance", "notes"]
VIS = {"VISIBLE", "PARTIAL", "HIDDEN", "OUT_OF_FRAME", "HUMAN_UNCERTAIN"}
ALLOWED = {
    "t23_visibility": VIS,
    "t23_identity_resolvable": {"YES", "NO"},
    "mar4_visibility": VIS,
    "overlap_23_4": {"YES", "NO", "HUMAN_UNCERTAIN"},
    "merged_or_hidden": {"NONE", "MERGED", "T23_HIDDEN", "HUMAN_UNCERTAIN"},
    "first_clean_reappearance": {"Y", ""},
}
FREE_TEXT = {"t23_location_note", "notes"}  # not vocabulary-checked; may be blank


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def validate(path, first, last, fps):
    errors, rows = [], []
    with open(path, newline="", encoding="utf-8-sig") as f:
        rd = csv.reader(f)
        header = next(rd, None)
        if header is None or [h.strip() for h in header] != COLUMNS:
            return [f"header mismatch: got {header}, expected {COLUMNS}"], []
        for n, r in enumerate(rd, start=2):
            if not any(c.strip() for c in r):
                continue
            if len(r) != len(COLUMNS):
                errors.append(f"line {n}: {len(r)} fields, expected {len(COLUMNS)}")
                continue
            rows.append({k: v.strip() for k, v in zip(COLUMNS, r)} | {"_line": n})
    frames = []
    for r in rows:
        try:
            fr = int(r["frame"])
        except ValueError:
            errors.append(f"line {r['_line']}: frame not an integer: {r['frame']!r}")
            continue
        frames.append(fr)
        try:
            if abs(float(r["video_time_s"]) - fr / fps) > 0.0006:
                errors.append(f"frame {fr}: video_time_s {r['video_time_s']} != {fr / fps:.3f}")
        except ValueError:
            errors.append(f"frame {fr}: video_time_s not a number")
        for col, allowed in ALLOWED.items():
            if r[col].upper() not in allowed:
                errors.append(f"frame {fr}: {col}={r[col]!r} not in {sorted(allowed)}")
    expected = list(range(first, last + 1))
    dup = sorted({f for f in frames if frames.count(f) > 1})
    if dup:
        errors.append(f"duplicate frames: {dup}")
    miss = sorted(set(expected) - set(frames))
    extra = sorted(set(frames) - set(expected))
    if miss:
        errors.append(f"missing frames: {miss}")
    if extra:
        errors.append(f"unexpected frames: {extra}")
    fcr = [int(r["frame"]) for r in rows if r["first_clean_reappearance"].upper() == "Y"]
    if len(fcr) > 1:
        errors.append(f"first_clean_reappearance marked on more than one frame: {fcr}")
    return errors, rows


def summarise(rows):
    def frames_where(pred):
        return [int(r["frame"]) for r in rows if pred(r)]
    uncertain_cols = ["t23_visibility", "mar4_visibility", "overlap_23_4", "merged_or_hidden"]
    return {
        "n_frames": len(rows),
        "human_uncertain_frames": frames_where(lambda r: any(r[c].upper() == "HUMAN_UNCERTAIN" for c in uncertain_cols)),
        "human_uncertain_by_field": {c: frames_where(lambda r, c=c: r[c].upper() == "HUMAN_UNCERTAIN") for c in uncertain_cols},
        "t23_identity_not_resolvable": frames_where(lambda r: r["t23_identity_resolvable"].upper() == "NO"),
        "t23_visibility_counts": {v: sum(1 for r in rows if r["t23_visibility"].upper() == v) for v in sorted(VIS)},
        "t23_hidden_frames": frames_where(lambda r: r["t23_visibility"].upper() == "HIDDEN"),
        "t23_partial_frames": frames_where(lambda r: r["t23_visibility"].upper() == "PARTIAL"),
        "overlap_frames": frames_where(lambda r: r["overlap_23_4"].upper() == "YES"),
        "merged_frames": frames_where(lambda r: r["merged_or_hidden"].upper() == "MERGED"),
        "t23_hidden_behind_4_frames": frames_where(lambda r: r["merged_or_hidden"].upper() == "T23_HIDDEN"),
        "first_clean_reappearance": frames_where(lambda r: r["first_clean_reappearance"].upper() == "Y"),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("labels_csv")
    ap.add_argument("--first", type=int, default=250)
    ap.add_argument("--last", type=int, default=290)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--freeze", action="store_true")
    a = ap.parse_args(argv)
    errors, rows = validate(a.labels_csv, a.first, a.last, a.fps)
    digest = sha256_file(a.labels_csv)
    out = {"labels_csv": os.path.basename(a.labels_csv), "sha256": digest,
           "bytes": os.path.getsize(a.labels_csv), "valid": not errors, "errors": errors}
    if errors:
        print(json.dumps(out, indent=2))
        return 1
    out["summary"] = summarise(rows)
    if a.freeze:
        rec_path = os.path.splitext(a.labels_csv)[0] + ".FROZEN.json"
        if os.path.exists(rec_path):
            old = json.load(open(rec_path))
            if old.get("sha256") != digest:
                print(json.dumps({"refused": "existing FROZEN record has a different sha256; create a v2 file",
                                  "existing": old.get("sha256"), "current": digest}, indent=2))
                return 2
        out.update(status="FROZEN", frozen_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   window=[a.first, a.last], fps=a.fps,
                   rule="Do not modify this label file. Corrections require GOLD_HUMAN_LABELS_v2.csv.")
        with open(rec_path, "w") as f:
            json.dump(out, f, indent=2, sort_keys=True)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
