# SPANA engine v2-rebuild
# REBUILT 2026-10 — NOT ORIGINAL SOURCE
# Evidence basis: R1-H instruction (Nathan, 2026-10-26: "target actor attribution requires identity_trusted = true");
#                 recovered actor_attribution.py (3Q exact, hash-pinned in tests_rb/test_surviving_consumers.py)
"""Measure whether the UNMODIFIED recovered actor_attribution.assess_actor credits TARGET_PLAYER on actor windows that
contain frames R1 marked identity_trusted = false. Evidence for R1/R1H/R1H_DOWNSTREAM_IDENTITY_RULE.md; it changes
nothing. Uses the synthetic mixed scenario of tests_rb/test_surviving_consumers.py (crossing, merge, cut, rival).

    python3 tools_rb/r1h_legacy_attribution_gap.py OUT.json
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests_rb import test_surviving_consumers as C  # noqa: E402

WINDOW_LENGTHS = (5, 10, 15, 20, 30)          # control-event windows in frames (synthetic clip at the harness fps)


def measure():
    e = C._scenarios()
    aa = C._MODS["actor_attribution"]
    idx = aa.build_frame_index(e.doc["frame_log"])
    trusted = {f: e.target(f)["v2"]["identity_trusted"] for f in range(1, e.n + 1)}
    out = {}
    for clip_conflict in (False, True):
        credited = with_untrusted = 0
        examples = []
        for n in WINDOW_LENGTHS:
            for s in range(1, e.n - n + 1):
                raw = {"type": "CARRY", "start_frame": s, "end_frame": s + n}
                a = aa.assess_actor(raw, [raw], idx, "RESOLVED", clip_conflict)
                if a["actor"] != "TARGET_PLAYER":
                    continue
                credited += 1
                bad = [f for f in range(s, s + n + 1) if not trusted.get(f, False)]
                if bad:
                    with_untrusted += 1
                    if len(examples) < 5:
                        examples.append({"window": [s, s + n], "untrusted_frames": bad,
                                         "trusted_fraction": a["actor_evidence"]["trusted_frames"] /
                                         a["actor_evidence"]["window_len"]})
        out["clip_conflict_" + str(clip_conflict).lower()] = {
            "windows_credited_target_player": credited, "of_which_contain_untrusted_frames": with_untrusted,
            "examples": examples}
    return out


def main(path):
    C.setUpModule()
    try:
        out = measure()
    finally:
        C.tearDownModule()
    with open(path, "x") as f:
        json.dump(out, f, indent=1, sort_keys=True)
        f.write("\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
