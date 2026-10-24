#!/usr/bin/env python3
"""Phase 4 step 4: run the PRE-REGISTERED, UNMODIFIED ab_event_matcher.py
against real Annotator A vs Annotator B data. This file is a NEW driver --
it imports run_batch()/match_clip()/timing_only_proximity_pairs() and calls
them; it does not edit a single line of ab_event_matcher.py. The module's
own __main__ guard (which refuses to run against real data) is bypassed by
never executing ab_event_matcher.py as __main__ -- only importing it."""
import sys, json
from pathlib import Path

REPO_ROOT = Path("/tmp/claude-0/-home-claude/98d69dca-3149-5a43-9670-3c417f2af6d4/scratchpad/spana_dataset")
sys.path.insert(0, str(REPO_ROOT / "tools" / "v2_calibration_round2"))
import ab_event_matcher as M

B_DIR = REPO_ROOT / "dataset" / "blind_reviews_v2_round2"
results = M.run_batch(B_DIR)

out = {}
for blind_id, r in results.items():
    out[blind_id] = {
        "clip_id": r["clip_id"],
        "matches": r["matches"],
        "unmatched_a": r["unmatched_a"],
        "unmatched_b": r["unmatched_b"],
        "timing_only_diagnostic_pairs": r["timing_only_diagnostic_pairs"],
    }

dest = REPO_ROOT / "dataset" / "reports" / "v2_round2_event_matching_output.json"
dest.write_text(json.dumps(out, indent=2, default=str) + "\n")
print(f"Wrote matcher output to {dest}\n")

total_a = total_b = total_matches = 0
for blind_id, r in results.items():
    n_a = len(r["matches"]) + len(r["unmatched_a"])
    n_b = len(r["matches"]) + len(r["unmatched_b"])
    total_a += n_a; total_b += n_b; total_matches += len(r["matches"])
    print(f"{blind_id} ({r['clip_id']}): A={n_a} B={n_b} matched={len(r['matches'])} "
          f"unmatched_a={len(r['unmatched_a'])} unmatched_b={len(r['unmatched_b'])} "
          f"diagnostic_pairs={len(r['timing_only_diagnostic_pairs'])}")
    for m in r["matches"]:
        print(f"    MATCH: {m}")
    for u in r["unmatched_a"]:
        print(f"    A-ONLY: {u['event_type']} outcome={u.get('outcome')} "
              f"t={u.get('interaction_time') or (u.get('start_time'), u.get('end_time'))} "
              f"conf(type={u.get('event_type_confidence')},id={u.get('player_identity_confidence')})")
    for u in r["unmatched_b"]:
        print(f"    B-ONLY: {u['event_type']} outcome={u.get('outcome')} "
              f"t={u.get('interaction_time') or (u.get('start_time'), u.get('end_time'))}")
    for dp in r["timing_only_diagnostic_pairs"]:
        print(f"    DIAGNOSTIC PAIR: {dp}")

print(f"\nTOTALS: A={total_a} B={total_b} MATCHED={total_matches} "
      f"A-ONLY={total_a-total_matches} B-ONLY={total_b-total_matches}")
print(f"RECALL = {total_matches}/{total_a}")
print(f"PRECISION = {total_matches}/{total_b}")
