# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R1-G instruction (Nathan, 2026-10-25: report synthetic identity results)
"""Per-scenario summary of the R1-G synthetic end-to-end scenarios (tests_rb/test_pipeline_e2e.SCENARIOS), for the
stage record. Reports error measures AND coverage (project rule 26). The pass/fail assertions live in the tests; this
tool only tabulates.

    python3 tools_rb/synth_scenario_summary.py OUT.json
"""
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests_rb import test_pipeline_e2e as E  # noqa: E402


def summarise(name):
    e = E.result(name)
    states = Counter()
    visible = trusted = trusted_visible = untrusted_observed = 0
    trusted_without_own_detection = 0
    for f in range(1, e.n + 1):
        v2 = e.target(f)["v2"]
        states[v2["rb_state"]] += 1
        own = e.target_det_box(f) is not None
        visible += own
        if v2["identity_trusted"]:
            trusted += 1
            trusted_visible += own
            trusted_without_own_detection += not own
        elif v2["rb_state"] == "TRACKING":
            untrusted_observed += 1
    events = e.summary["v2"]["reacquisition_events"]
    return {"frames": e.n, "target_visible_frames": visible, "rb_states": dict(sorted(states.items())),
            "trusted_frames": trusted,
            "trusted_coverage_of_visible": round(trusted_visible / visible, 3) if visible else None,
            "observed_but_unverified_frames": untrusted_observed,
            "silent_swaps": len(e.swaps()), "trusted_without_own_detection": trusted_without_own_detection,
            "target_written_twice": len(e.target_duplicated()), "body_written_twice": len(e.body_written_twice()),
            "bank_violations": len(e.bank_violations()), "bank_accepted_frames": len(e.accepted_bank_frames()),
            "reacquisitions": len(events),
            "reacquisitions_contrast_verified": sum(1 for r in events if r["evidence"].get("contrast_verified")),
            "identity_conflicts_logged": e.summary["identity_conflicts_logged"]}


def main(out):
    E.setUpModule()
    try:
        rows = {name: summarise(name) for name in E.SCENARIOS}
    finally:
        E.tearDownModule()
    with open(out, "w") as f:
        json.dump(rows, f, indent=1, sort_keys=True)
        f.write("\n")
    for name, r in rows.items():
        print(f"{name:26s} vis={r['target_visible_frames']:4d} trusted={r['trusted_frames']:4d} "
              f"cov={r['trusted_coverage_of_visible']} unverified={r['observed_but_unverified_frames']:3d} "
              f"swaps={r['silent_swaps']} dup={r['target_written_twice']}/{r['body_written_twice']} "
              f"bank_bad={r['bank_violations']} reacq={r['reacquisitions']}({r['reacquisitions_contrast_verified']}v)")


if __name__ == "__main__":
    main(sys.argv[1])
