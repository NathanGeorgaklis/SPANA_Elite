#!/usr/bin/env python3
"""blind_id_generator.py -- deliverable #4: a reusable tool that takes a
list of clip_ids plus a NEUTRAL, non-event-revealing player hint for each,
and produces a genuinely blinded package for Annotator B.

This formalizes (as an actual script, not a one-off manual copy) what the
task requires:

  - blind IDs use a neutral naming convention (default "blind_r2_%03d" --
    "r2" distinguishes this round's IDs from the ORIGINAL 12-clip agreement
    round's "blind_001".."blind_012" and the PREVIOUS V2 round's
    "blind_v2_001".."blind_v2_010", so Nathan/future tooling can never
    confuse which round a blind_id belongs to)
  - the delivered folder for each clip contains ONLY clip.mp4 and a
    neutral_hint.txt -- no clip_time_map.json, no real clip_id anywhere in
    the delivered tree, no old filenames, no event/outcome/timing/
    qualifier/confidence data, no model prediction
  - the hidden mapping (blind_id -> real clip_id, neutral hint, taxonomy
    version, registration timestamp) is written to a SEPARATE directory
    that is never copied into the delivered package

Player hints must be checked by the CALLER before passing them in --
this script does not itself judge whether a hint is event-revealing, but
GOOD/BAD examples from the task are repeated here as a reminder:
  GOOD: "Yellow #8", "Blue #3", "Red player closest to bottom touchline
        at clip start"
  BAD:  "the passer", "the shooter", "the player making the tackle",
        "the player who receives the ball" -- these reveal the event.

Usage:
    python3 blind_id_generator.py --spec clip_hint_spec.json \
        --package-dir ../../dataset/blind_package_v2_round2 \
        --mapping-dir ../../dataset/blind_mapping_v2_round2 \
        --prefix blind_r2

clip_hint_spec.json: {"clip_id": "neutral hint text", ...} (ordering is
preserved from the JSON file, which controls blind-id assignment order --
run once, do not re-run over an already-assigned mapping without --force,
since re-numbering would break any in-flight Annotator B work).
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CLIPS_DIR = REPO_ROOT / "dataset" / "clips"
TAXONOMY_VERSION = "SPANA_EVENT_TAXONOMY_V2"


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def generate(spec: dict, package_dir: Path, mapping_dir: Path, prefix: str,
             force: bool = False) -> dict:
    mapping_path = mapping_dir / "mapping.json"
    mapping = {}
    if mapping_path.exists():
        mapping = json.loads(mapping_path.read_text())
        if mapping and not force:
            raise SystemExit(
                f"{mapping_path} already has {len(mapping)} entries. "
                "Refusing to re-run without --force (re-numbering blind IDs "
                "after any Annotator B work has started would corrupt the "
                "A/B matching). Pass --force only if you are certain no "
                "Annotator B submissions reference the existing IDs.")

    package_dir.mkdir(parents=True, exist_ok=True)
    mapping_dir.mkdir(parents=True, exist_ok=True)

    if force:
        for child in package_dir.iterdir():
            if child.is_dir() and child.name.startswith(prefix):
                shutil.rmtree(child)
        mapping = {}

    for i, (clip_id, hint) in enumerate(spec.items(), start=1):
        blind_id = f"{prefix}_{i:03d}"
        src_clip = CLIPS_DIR / clip_id / "clip.mp4"
        if not src_clip.exists():
            raise SystemExit(f"clip.mp4 not found for {clip_id} at {src_clip}")

        dest_dir = package_dir / blind_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src_clip, dest_dir / "clip.mp4")
        (dest_dir / "neutral_hint.txt").write_text(hint.strip() + "\n")

        mapping[blind_id] = {
            "clip_id": clip_id,
            "neutral_hint": hint.strip(),
            "taxonomy_version": TAXONOMY_VERSION,
            "registered_at": _now_iso(),
        }

    mapping_path.write_text(json.dumps(mapping, indent=2, sort_keys=True))
    return mapping


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", required=True, type=Path, help="JSON file: {clip_id: neutral_hint, ...}")
    ap.add_argument("--package-dir", required=True, type=Path)
    ap.add_argument("--mapping-dir", required=True, type=Path)
    ap.add_argument("--prefix", default="blind_r2")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    spec = json.loads(args.spec.read_text())
    mapping = generate(spec, args.package_dir, args.mapping_dir, args.prefix, force=args.force)
    print(json.dumps({bid: {"clip_id": "***HIDDEN, not printed***"} for bid in mapping}, indent=2))
    print(f"\n{len(mapping)} blind clips written to {args.package_dir}")
    print(f"Hidden mapping written to {args.mapping_dir / 'mapping.json'} (never copy this into the delivered package)")


if __name__ == "__main__":
    main()
