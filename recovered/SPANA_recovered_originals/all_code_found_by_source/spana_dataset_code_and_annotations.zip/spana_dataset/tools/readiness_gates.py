#!/usr/bin/env python3
"""
readiness_gates.py -- task item #29: evaluates the four collection
readiness gates (A: source foundation, B: first data milestone, C:
exploratory ML, D: development ML) against the REAL current dataset state.
Never trains anything; never says a gate is passed unless the real numbers
support it.

Usage:
    python readiness_gates.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
DATASET_ROOT = TOOLS_DIR.parent
ANNOTATIONS_DIR = DATASET_ROOT / "dataset" / "annotations"
REPORTS_DIR = DATASET_ROOT / "dataset" / "reports"
sys.path.insert(0, str(TOOLS_DIR))

import collection_tracker as ctr        # noqa: E402
import source_registry as sr            # noqa: E402
import source_diversity_report as sdr   # noqa: E402
import training_eligibility as te       # noqa: E402


def build_report() -> dict:
    records, _ = ctr.load_annotations(ANNOTATIONS_DIR)
    n_annotations = len(records)

    diversity = sdr.build_report(target_matches=12)
    n_confirmed_matches = diversity["n_confirmed_independent_matches"]

    elig = te.run(ANNOTATIONS_DIR, DATASET_ROOT / "dataset" / "clips")
    n_training_eligible = elig["n_training_eligible"]

    gate_a = {
        "name": "GATE A -- SOURCE FOUNDATION",
        "requirement": ">= ~8 confirmed independent matches",
        "current": n_confirmed_matches,
        "target": 8,
        "met": n_confirmed_matches >= 8,
        "unlocks": "Dataset collection can accelerate (broader per-match caps, less strict source-first gating).",
    }
    gate_b = {
        "name": "GATE B -- FIRST DATA MILESTONE",
        "requirement": "50 quality annotations with reasonable match spread",
        "current_annotations": n_annotations,
        "current_matches": n_confirmed_matches,
        "target_annotations": 50,
        "met": n_annotations >= 50 and n_confirmed_matches >= 4,
        "unlocks": "Audit annotation consistency and feature quality across a wider real sample.",
    }
    gate_c = {
        "name": "GATE C -- EXPLORATORY ML",
        "requirement": "~150 quality annotations across sufficient sources",
        "current_annotations": n_annotations,
        "current_matches": n_confirmed_matches,
        "target_annotations": 150,
        "met": n_annotations >= 150 and n_confirmed_matches >= 8,
        "unlocks": "A very early, disposable baseline model MAY be considered -- never production, never this task.",
    }
    gate_d = {
        "name": "GATE D -- DEVELOPMENT ML",
        "requirement": "300+ annotations, stronger diversity, meaningful match-level split",
        "current_annotations": n_annotations,
        "current_matches": n_confirmed_matches,
        "target_annotations": 300,
        "met": n_annotations >= 300 and n_confirmed_matches >= 8,
        "unlocks": "Serious learned-classifier experimentation.",
    }

    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "current_state": {
            "n_annotations": n_annotations,
            "n_training_eligible": n_training_eligible,
            "n_confirmed_independent_matches": n_confirmed_matches,
            "n_diversity_gaps_open": len(diversity["next_source_priorities"]),
        },
        "gates": [gate_a, gate_b, gate_c, gate_d],
        "no_model_trained_this_task": True,
    }


def main():
    report = build_report()
    out = REPORTS_DIR / "readiness_gates.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    print("readiness_gates:")
    for g in report["gates"]:
        print(f"  [{'MET' if g['met'] else 'NOT MET'}] {g['name']} -- {g['requirement']}")
    print(f"\nfull report written to {out}")


if __name__ == "__main__":
    main()
