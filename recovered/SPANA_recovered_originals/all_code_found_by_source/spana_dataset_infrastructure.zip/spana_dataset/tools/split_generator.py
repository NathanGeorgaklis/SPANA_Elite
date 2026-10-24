#!/usr/bin/env python3
"""
split_generator.py -- assigns each MATCH (never a clip, never an individual
annotation) to one of TRAIN / VALIDATION / TEST / GOLD_TEST, writing
dataset/splits/assignment.json = {match_id: SPLIT_NAME}.

Why match-level, never clip-level (audit item #24): clips from the same
match share camera angle/rig, kit colours, pitch surface and lighting. A
model that has seen other clips from the same match during training would
have an unfair, footage-level advantage evaluating on a held-out clip from
that same match -- the split would look clean by clip_id/annotation_id but
leak by the thing that actually matters. leakage_checker.py is the
independent auditor of this and of the deeper case where two different
match_ids turn out to share the same underlying source_video_id.

RECOMMENDATION (also see --help below): reserve a genuinely untouched
GOLD_TEST set as early in the project as possible, from whichever match(es)
give you the least incentive to peek, and then DO NOT use it for iterative
debugging, threshold tuning, or "just checking" during development (audit
item #25). Every time GOLD_TEST is looked at while still iterating, its
value as a final, honest evaluation set is a little more spent. TEST exists
for iterative use; GOLD_TEST does not.

Two ways to assign matches:

  1. MANUAL (the primary interface, and the only one you should need while
     the dataset has only a handful of known matches -- an automatic ratio
     split of 2-3 matches is not meaningful):

         python split_generator.py --assign match_001:TRAIN \\
                                    --assign match_002:TEST \\
                                    --assign match_003:GOLD_TEST

  2. AUTOMATIC (a fallback for once there are enough known matches that a
     ratio actually means something -- applied by MATCH COUNT, not by clip
     or annotation count):

         python split_generator.py --auto \\
             --ratios TRAIN=0.6,VALIDATION=0.15,TEST=0.15,GOLD_TEST=0.1

     --auto only assigns matches that are not already covered by this run's
     --assign flags and are not already present in an existing
     assignment.json (unless --allow-reassign is also given).

The known match_id universe is discovered from dataset/annotations/*.json's
`match_id` field, unioned with dataset/metadata/source_videos.json's
per-source-video `match_id` field when that registry exists (it is being
built by other tooling in parallel and may be empty or partially populated
-- that's a normal early-dataset state, not an error).

SAFETY / IDEMPOTENCY (this is the whole point of the tool, not an
afterthought): this script REFUSES to silently reassign a match that is
already recorded in assignment.json to a different split unless
--allow-reassign is explicitly passed. Every run prints, per match, exactly
what would change (or not change, or be blocked) before writing anything,
so a re-run against an unchanged set of inputs is a safe no-op and a run
that WOULD change something is never silent about it.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
DATASET_ROOT = TOOLS_DIR.parent

ANNOTATIONS_DIR = DATASET_ROOT / "dataset" / "annotations"
SOURCE_VIDEOS_PATH = DATASET_ROOT / "dataset" / "metadata" / "source_videos.json"
ASSIGNMENT_PATH = DATASET_ROOT / "dataset" / "splits" / "assignment.json"

VALID_SPLITS = ("TRAIN", "VALIDATION", "TEST", "GOLD_TEST")
DEFAULT_RATIOS = {"TRAIN": 0.6, "VALIDATION": 0.15, "TEST": 0.15, "GOLD_TEST": 0.1}


class SplitGeneratorError(Exception):
    pass


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_known_match_ids(annotations_dir: Path = ANNOTATIONS_DIR,
                              source_videos_path: Path = SOURCE_VIDEOS_PATH) -> set[str]:
    """Unions match_ids referenced by dataset/annotations/*.json with
    match_ids recorded in dataset/metadata/source_videos.json. Tolerant of
    either being empty, partially populated, or (for annotations) containing
    a file that isn't valid JSON yet -- this is expected while other tools
    are still landing their part of the pipeline in parallel."""
    known: set[str] = set()

    if annotations_dir.exists():
        for p in sorted(annotations_dir.glob("*.json")):
            try:
                ann = json.loads(p.read_text())
            except json.JSONDecodeError as e:
                print(f"  (skipping unparseable annotation file {p.name}: {e})", file=sys.stderr)
                continue
            match_id = ann.get("match_id") if isinstance(ann, dict) else None
            if match_id:
                known.add(match_id)

    if source_videos_path.exists():
        try:
            registry = json.loads(source_videos_path.read_text())
        except json.JSONDecodeError as e:
            print(f"  (warning: {source_videos_path} is not valid JSON, ignoring: {e})", file=sys.stderr)
            registry = {}
        if isinstance(registry, dict):
            for rec in registry.values():
                if isinstance(rec, dict) and rec.get("match_id"):
                    known.add(rec["match_id"])

    return known


def load_assignment(path: Path = ASSIGNMENT_PATH) -> dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise SplitGeneratorError(f"{path} exists but is not valid JSON: {e}") from e
    if not isinstance(data, dict):
        raise SplitGeneratorError(f"{path} must contain a JSON object of match_id -> SPLIT_NAME")
    return data


# ---------------------------------------------------------------------------
# Ratio parsing / automatic assignment
# ---------------------------------------------------------------------------

def parse_ratios(ratios_str: str) -> dict:
    """Parses 'TRAIN=0.6,VALIDATION=0.15,TEST=0.15,GOLD_TEST=0.1' into a dict
    and normalizes it to sum to 1.0. Raises SplitGeneratorError on an
    unknown split name or a non-positive total."""
    out = {}
    for part in ratios_str.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise SplitGeneratorError(f"malformed ratio entry {part!r} -- expected SPLIT=fraction")
        name, _, value = part.partition("=")
        name = name.strip().upper()
        if name not in VALID_SPLITS:
            raise SplitGeneratorError(f"unknown split {name!r} in --ratios (valid: {VALID_SPLITS})")
        try:
            out[name] = float(value)
        except ValueError:
            raise SplitGeneratorError(f"ratio for {name} is not a number: {value!r}")
    total = sum(out.values())
    if total <= 0:
        raise SplitGeneratorError(f"--ratios must sum to a positive number, got {total}")
    return {k: v / total for k, v in out.items()}


def auto_assign(match_ids: list[str], ratios: dict) -> dict:
    """Assigns each of `match_ids` (already sorted, for determinism) to a
    split by MATCH COUNT using the largest-remainder method, so the result
    is deterministic and every ratio's rounded share is as close to its true
    share as an integer count of matches allows."""
    n = len(match_ids)
    if n == 0:
        return {}

    names = sorted(ratios.keys())  # deterministic iteration order
    raw_counts = {name: ratios[name] * n for name in names}
    base_counts = {name: int(raw_counts[name]) for name in names}
    remainder = n - sum(base_counts.values())
    # Largest fractional remainder gets the leftover seats, tie-broken by
    # split name so the outcome never depends on dict/set ordering.
    fractional = sorted(names, key=lambda name: (-(raw_counts[name] - base_counts[name]), name))
    for name in fractional[:remainder]:
        base_counts[name] += 1

    assignment = {}
    idx = 0
    for name in names:
        for _ in range(base_counts[name]):
            if idx >= n:
                break
            assignment[match_ids[idx]] = name
            idx += 1
    return assignment


# ---------------------------------------------------------------------------
# Core: compute the diff between existing assignment.json and the requested
# changes, honouring --allow-reassign, and report it either way.
# ---------------------------------------------------------------------------

def plan_changes(existing: dict, requested: dict, allow_reassign: bool) -> tuple[dict, list[dict]]:
    """Returns (new_assignment_dict, change_log). change_log entries have
    shape {"match_id", "old_split", "new_split", "action"} where action is
    one of ADDED / UNCHANGED / REASSIGNED / BLOCKED."""
    new_assignment = dict(existing)
    change_log = []
    for match_id in sorted(requested):
        new_split = requested[match_id]
        old_split = existing.get(match_id)
        if old_split is None:
            new_assignment[match_id] = new_split
            change_log.append({"match_id": match_id, "old_split": None,
                                "new_split": new_split, "action": "ADDED"})
        elif old_split == new_split:
            change_log.append({"match_id": match_id, "old_split": old_split,
                                "new_split": new_split, "action": "UNCHANGED"})
        elif allow_reassign:
            new_assignment[match_id] = new_split
            change_log.append({"match_id": match_id, "old_split": old_split,
                                "new_split": new_split, "action": "REASSIGNED"})
        else:
            change_log.append({"match_id": match_id, "old_split": old_split,
                                "new_split": new_split, "action": "BLOCKED"})
    return new_assignment, change_log


def print_change_log(change_log: list[dict]) -> None:
    if not change_log:
        print("  (no changes requested)")
        return
    for c in change_log:
        old = c["old_split"] or "-"
        if c["action"] == "ADDED":
            print(f"  ADDED       {c['match_id']}: (unassigned) -> {c['new_split']}")
        elif c["action"] == "UNCHANGED":
            print(f"  UNCHANGED   {c['match_id']}: {old} (already this split)")
        elif c["action"] == "REASSIGNED":
            print(f"  REASSIGNED  {c['match_id']}: {old} -> {c['new_split']}  (--allow-reassign was given)")
        elif c["action"] == "BLOCKED":
            print(f"  BLOCKED     {c['match_id']}: {old} -> {c['new_split']}  "
                  f"(pass --allow-reassign to actually make this change)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_assign_arg(value: str) -> tuple[str, str]:
    if ":" not in value:
        raise argparse.ArgumentTypeError(f"--assign expects match_id:SPLIT_NAME, got {value!r}")
    match_id, _, split = value.partition(":")
    match_id, split = match_id.strip(), split.strip().upper()
    if not match_id:
        raise argparse.ArgumentTypeError(f"--assign {value!r}: empty match_id")
    if split not in VALID_SPLITS:
        raise argparse.ArgumentTypeError(f"--assign {value!r}: split must be one of {VALID_SPLITS}")
    return match_id, split


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Reminder: reserve a genuinely untouched GOLD_TEST set as early as possible "
               "and never use it for iterative debugging (audit item #25).")
    ap.add_argument("--assign", action="append", default=[], type=_parse_assign_arg, metavar="match_id:SPLIT",
                     help="Manually assign one match to a split. Repeatable. SPLIT is one of "
                          f"{VALID_SPLITS}. This is the primary interface -- prefer it while the "
                          "dataset has only a handful of known matches.")
    ap.add_argument("--auto", action="store_true",
                     help="Also auto-assign, by ratio, any known match not covered by --assign "
                          "and not already present in assignment.json (unless --allow-reassign). "
                          "Only meaningful once there are enough matches for a ratio to mean "
                          "anything -- with 2-3 matches, use --assign instead.")
    ap.add_argument("--ratios", type=str, default=None, metavar="TRAIN=0.6,VALIDATION=0.15,...",
                     help="Ratios for --auto, applied by MATCH COUNT via largest-remainder "
                          f"rounding. Defaults to {DEFAULT_RATIOS} if --auto is given without this.")
    ap.add_argument("--allow-reassign", action="store_true",
                     help="Without this flag, a match already present in assignment.json under a "
                          "DIFFERENT split is left untouched and reported as BLOCKED. This is the "
                          "tool's core safety property -- pass this only when you mean to move a "
                          "match across splits on purpose.")
    ap.add_argument("--show", action="store_true",
                     help="Print the discovered match_id universe and the current assignment.json "
                          "contents, then exit without writing anything.")
    ap.add_argument("--dry-run", action="store_true",
                     help="Compute and print the change log as usual, but do not write assignment.json.")
    ap.add_argument("--annotations-dir", type=Path, default=ANNOTATIONS_DIR)
    ap.add_argument("--source-videos", type=Path, default=SOURCE_VIDEOS_PATH)
    ap.add_argument("--out", type=Path, default=ASSIGNMENT_PATH)
    args = ap.parse_args()

    known_match_ids = discover_known_match_ids(args.annotations_dir, args.source_videos)
    existing = load_assignment(args.out)

    if args.show:
        print(f"known match_ids ({len(known_match_ids)}): {sorted(known_match_ids)}")
        print(f"current assignment.json ({args.out}):")
        if not existing:
            print("  (does not exist yet, or is empty)")
        else:
            for match_id, split in sorted(existing.items()):
                flag = "" if match_id in known_match_ids else "  (not found in current annotations/registry)"
                print(f"  {match_id}: {split}{flag}")
        unassigned = sorted(known_match_ids - existing.keys())
        if unassigned:
            print(f"unassigned known match_ids ({len(unassigned)}): {unassigned}")
        return

    for match_id, _split in args.assign:
        if match_id not in known_match_ids:
            print(f"  (note: {match_id!r} is not among the match_ids currently discoverable from "
                  f"annotations/source_videos.json -- fine if it just hasn't been ingested yet, "
                  f"assigning it explicitly anyway since --assign is a deliberate manual action)")

    requested = dict(args.assign)  # last --assign for a given match_id in this invocation wins

    if args.auto:
        ratios = parse_ratios(args.ratios) if args.ratios else dict(DEFAULT_RATIOS)
        already_spoken_for = set(requested) | (set(existing) if not args.allow_reassign else set())
        remaining = sorted(known_match_ids - already_spoken_for)
        auto_result = auto_assign(remaining, ratios)
        # --assign entries in THIS invocation take precedence over --auto for the same match_id.
        for match_id, split in auto_result.items():
            requested.setdefault(match_id, split)
        if not remaining:
            print("  (--auto: no unassigned known match_ids left to auto-split)")

    if not requested:
        print("Nothing to do -- no --assign entries and either --auto was not given or nothing "
              "was left to auto-assign. Pass --show to inspect the current state.")
        return

    new_assignment, change_log = plan_changes(existing, requested, args.allow_reassign)
    print(f"split_generator: {len(known_match_ids)} known match_id(s), "
          f"{len(existing)} already in {args.out.name}, {len(requested)} change(s) requested")
    print_change_log(change_log)

    n_blocked = sum(1 for c in change_log if c["action"] == "BLOCKED")
    n_applied = sum(1 for c in change_log if c["action"] in ("ADDED", "REASSIGNED"))

    if args.dry_run:
        print(f"\n--dry-run: would write {n_applied} change(s) to {args.out} "
              f"({n_blocked} blocked); nothing written.")
    elif n_applied:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(dict(sorted(new_assignment.items())), indent=2) + "\n")
        print(f"\nwrote {n_applied} change(s) to {args.out} ({len(new_assignment)} total match(es) assigned).")
    else:
        print(f"\nno changes to write ({n_blocked} blocked, rest unchanged).")

    if n_blocked:
        print(f"\n{n_blocked} reassignment(s) were BLOCKED -- rerun with --allow-reassign if that "
              f"was intentional.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
