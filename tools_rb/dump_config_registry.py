# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: player_tracker_rb/config.py (the registry is the source of truth; this only exports it)
"""Export the player_tracker_rb parameter registry to CSV, so every constant's units and provenance can be reviewed
without reading code. Usage: python3 tools_rb/dump_config_registry.py OUT.csv"""
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from player_tracker_rb import config as C  # noqa: E402

COLUMNS = ["name", "group", "role", "source", "evidence_level", "validation_status", "type", "units", "default",
           "min_value", "max_value", "allowed", "historical_value", "meaning", "evidence"]


def main(out_path):
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(COLUMNS)
        for rec in C.registry_records():
            w.writerow([json.dumps(rec[c]) if isinstance(rec[c], (list, dict)) else
                        ("" if rec[c] is None else rec[c]) for c in COLUMNS])
    cfg = C.TrackerConfig.default()
    print(json.dumps({"params": len(C.PARAMS), "registry_version": C.CONFIG_REGISTRY_VERSION,
                      "default_config_hash": cfg.config_hash()}))


if __name__ == "__main__":
    main(sys.argv[1])
