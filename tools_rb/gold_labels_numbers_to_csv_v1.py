# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: Nathan's GOLD labels (GOLD_HUMAN_LABELS_v1_SOURCE.numbers) and his two corrections given
#                 in chat 2026-10-24 ~14:20 Sydney (see GOLD_HUMAN_LABELS_v1_CORRECTIONS.md)
"""One-off conversion of Nathan's Numbers label table to GOLD_HUMAN_LABELS_v1.csv.

Cell text is copied exactly as Numbers formats it. Only two corrections are applied, both
authorised by Nathan: merged_or_hidden 'hidden' -> 'T23_HIDDEN' on frames 266-268, and
first_clean_reappearance = 'Y' on frame 273 only (blank on every other frame).
Run in the cloud workspace with numbers-parser 4.22.0 / protobuf 7.36.2.
Usage: python make_v1.py SOURCE.numbers OUT.csv   (prints every changed cell)
"""
import sys, csv, numbers_parser as n
t = n.Document(sys.argv[1]).sheets[0].tables[0]
rows = [[c.formatted_value for c in r] for r in t.iter_rows()]
hdr = rows[0]; iM = hdr.index("merged_or_hidden"); iF = hdr.index("first_clean_reappearance")
changes = []
for r in rows[1:]:
    fr = int(r[0])
    if fr in (266, 267, 268):
        changes.append((fr, "merged_or_hidden", r[iM], "T23_HIDDEN")); r[iM] = "T23_HIDDEN"
    new = "Y" if fr == 273 else ""
    if r[iF] != new:
        changes.append((fr, "first_clean_reappearance", r[iF], new)); r[iF] = new
with open(sys.argv[2], "w", newline="", encoding="utf-8") as f:
    csv.writer(f, lineterminator="\n").writerows(rows)
for c in changes: print(*c, sep="\t")
print("n_changes", len(changes), file=sys.stderr)
