# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R1-H instruction (Nathan, 2026-10-26); tools_rb/tracker_eval.py
"""Score every R1-G synthetic end-to-end scenario with tools_rb/tracker_eval.py against the harness's COMPLETE box
truth (target box, every other person's box, merged / hidden status). Synthetic evidence only: it shows the metrics
and the tracker agree on controlled sequences; it says nothing about real-footage performance.

    python3 tools_rb/r1h_synthetic_eval.py OUT.json
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests_rb import test_pipeline_e2e as P  # noqa: E402
from tests_rb.test_tracker_eval import truth_from_synthetic  # noqa: E402
from tools_rb import tracker_eval as E  # noqa: E402


def row(name):
    e = P.result(name)
    r = E.evaluate(e.doc, truth_from_synthetic(e))
    cov = r["coverage_and_abstention"]
    delays = [x.get("delay_s") for x in r["reacquisition"] if "delay_s" in x]
    return {"fci": r["priority_1_false_confident_identity"]["count"],
            "fci_status": r["priority_1_false_confident_identity"]["status"],
            "swaps": r["priority_2_identity_swaps"]["count"],
            "wrong_reacq": r["priority_3_wrong_reacquisitions"]["count"],
            "trusted_on_unavailable": r["priority_4_trusted_on_hidden_merged_unresolvable"]["count"],
            "available_frames": cov["available_frames"],
            "trusted_coverage": cov["trusted_coverage_of_available"],
            "rates_on_available": cov["rates_on_available"],
            "uncertainty_when_unavailable": cov["uncertainty_when_unavailable"],
            "reacquisitions": [{k: x.get(k) for k in ("gap", "outcome", "delay_s", "delay_s_at_least",
                                                      "missed_reacquisition") if k in x}
                               for x in r["reacquisition"]],
            "trusted_reacq_delay_s": delays,
            "mean_iou_confident": r["box_quality"]["mean_iou_confident"]}


def main(out):
    P.setUpModule()
    try:
        rows = {name: row(name) for name in P.SCENARIOS}
    finally:
        P.tearDownModule()
    with open(out, "x") as f:
        json.dump(rows, f, indent=1, sort_keys=True)
        f.write("\n")
    for name, r in rows.items():
        rates = r["rates_on_available"]
        print(f"{name:26s} fci={r['fci']}({r['fci_status'][:4]}) swaps={r['swaps']} wrong={r['wrong_reacq']} "
              f"p4={r['trusted_on_unavailable']} cov={r['trusted_coverage']} "
              f"amb={rates.get('AMBIGUOUS', 0)} occ={rates.get('OCCLUDED', 0)} lost={rates.get('LOST', 0)} "
              f"unv={rates.get('OBSERVED_UNVERIFIED', 0)} delay={r['trusted_reacq_delay_s']} "
              f"iou={r['mean_iou_confident']} reacq={[x['outcome'] for x in r['reacquisitions']]}")


if __name__ == "__main__":
    main(sys.argv[1])
