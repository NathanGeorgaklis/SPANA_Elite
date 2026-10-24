"""Task 3F regression diff: compares a clip's pre-3F (post-3E) event log / raw events /
player_analysis against the Task 3F rerun. Raw engine events must be byte-identical
(frozen engine untouched); normalized events are compared on every pre-3F field."""
import sys, json
old_dir, new_dir, clip = sys.argv[1], sys.argv[2], sys.argv[3]
L = lambda d, n: json.load(open(f"{d}/{clip}_{n}.json"))
ro, rn = L(old_dir, "events_raw"), L(new_dir, "events_raw")
print("raw engine events identical:", ro["events"] == rn["events"])
eo, en = L(old_dir, "event_log"), L(new_dir, "event_log")
OLD_FIELDS = ["type", "start_frame", "end_frame", "start_time", "end_time", "actor", "outcome", "success",
              "confidence", "review_required", "qualifiers", "source_event"]
print(f"event count: {len(eo)} -> {len(en)}")
key = lambda e: (e["type"], e["start_frame"], e["end_frame"])
mo, mn = {key(e): e for e in eo}, {key(e): e for e in en}
for k in sorted(set(mo) | set(mn), key=lambda k: (k[1], k[0])):
    a, b = mo.get(k), mn.get(k)
    if a is None: print("  NEW    ", k, "actor", b["actor"]); continue
    if b is None: print("  REMOVED", k); continue
    diffs = {f: (a.get(f), b.get(f)) for f in OLD_FIELDS if a.get(f) != b.get(f)}
    print("  ", "SAME   " if not diffs else "CHANGED", k, diffs if diffs else "",
          f"| actor_conf={b.get('actor_confidence')} event_conf={b.get('event_confidence')} stat_eligible={b.get('stat_eligible')}")
po, pn = L(old_dir, "player_analysis"), L(new_dir, "player_analysis")
print("stats identical:", po["stats"] == pn["stats"])
for k in po["stats"]:
    if po["stats"][k] != pn["stats"][k]: print(f"   {k}: {po['stats'][k]} -> {pn['stats'][k]}")
print("confidence_summary:", po["confidence_summary"], "->", pn["confidence_summary"])
print("stat_confidence:", pn.get("stat_confidence"))
