"""Shared helpers for the F5 review repros. Usage: python script.py <repo_root>"""
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/home/claude/spana_push"
sys.path.insert(0, ROOT)

from player_tracker_rb import population as P  # noqa: E402
from player_tracker_rb import target as T  # noqa: E402
from player_tracker_rb.config import TrackerConfig  # noqa: E402
from tests_rb import _synth_scene as SS  # noqa: E402

KIT_A = (200, 60, 40)
KIT_B = (40, 40, 210)


def summarize(r, label, start=0):
    tw = [k for k in range(start, len(r.frames)) if r.frames[k].identity_trusted
          and "TARGET" not in r.dets[k][r.frames[k].detection_index].owners]
    to = [k for k in range(start, len(r.frames)) if r.frames[k].identity_trusted
          and r.dets[k][r.frames[k].detection_index].owners != ("TARGET",)]
    rev = [k for k, f in enumerate(r.frames) if (f.decision.get("reverify") or {}).get("status") == "REVERIFIED"]
    whys = {}
    for f in r.frames:
        rv = f.decision.get("reverify")
        if rv:
            key = rv.get("status") + ":" + str(rv.get("why", rv.get("vacated_spot", "")))
            whys[key] = whys.get(key, 0) + 1
    bank_untrusted = [k for k, f in enumerate(r.frames) if (f.decision.get("bank") or {}).get("status") == "ACCEPTED"
                      and not f.identity_trusted]
    n_tr = sum(1 for f in r.frames if f.identity_trusted)
    print(f"{label}: frames={len(r.frames)} trusted={n_tr} trusted_without_target={len(tw)} "
          f"(first={tw[:1]}) trusted_not_target_alone={len(to)} reverified_at={rev} swaps={len(r.swaps())} "
          f"bank_pure={r.bank_is_pure()} bank_untrusted={bank_untrusted} bank_n={len(r.tracker.bank.samples)}")
    if whys:
        print("   reverify log:", dict(sorted(whys.items())))
    return dict(tw=len(tw), to=len(to), rev=rev, pure=r.bank_is_pure())


def pose_then(fps, cx=300.0, cy=400.0, w0=40.0, w1=92.0, h=100.0):
    n_still, n_grow, n_hold = int(fps), max(2, int(fps / 2)), max(1, int(fps / 5))
    out = [SS.box_at(cx, cy, w0, h) for _ in range(n_still)]
    out += [SS.box_at(cx, cy, w0 + (w1 - w0) * (k + 1) / n_grow, h) for k in range(n_grow)]
    out += [SS.box_at(cx, cy, w1, h) for _ in range(n_hold)]
    return out, len(out)
