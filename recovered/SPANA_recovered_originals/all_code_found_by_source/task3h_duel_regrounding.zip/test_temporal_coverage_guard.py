"""Task 3G: self-tests for the temporal-coverage guard (benchmark_fixture.py).
Uses one real file only for the provenance half of evaluation_gate()."""
import sys, copy
sys.path.insert(0, ".")
from event_engine_v1.benchmark_fixture import (BenchmarkFixture, validate_temporal_coverage, evaluation_gate,
                                               require_evaluation_allowed, BenchmarkEvaluationNotAllowed)

VIDEO = "/root/.claude/uploads/98d69dca-3149-5a43-9670-3c417f2af6d4/8a710009-dual_lost_01.mov"
base = BenchmarkFixture.create(clip_id="t", video_path=VIDEO, target_player_label="x", seed_frame=0,
                               seed_bbox=[0, 0, 1, 1], expected_primary_event="DUEL", expected_outcome="LOST",
                               expected_event_window="test", fixture_provenance_notes="self-test")
R = []


def case(name, **kw):
    f = copy.deepcopy(base)
    for k, v in kw.items():
        setattr(f, k, v)
    return f


def check(name, cond, detail=""):
    R.append((name, cond)); print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")


v = validate_temporal_coverage(case("ok", label_window_start_s=2.0, label_window_end_s=3.0, tracking_start_s=1.0, tracking_end_s=4.0))
check("1 covered with context -> VALID", v.status == "TEMPORAL_COVERAGE_VALID")
v = validate_temporal_coverage(case("late", label_window_start_s=2.0, label_window_end_s=3.0, tracking_start_s=2.4, tracking_end_s=5.0))
check("2 tracking starts after label opens -> INVALID", v.status == "BENCHMARK_TEMPORAL_COVERAGE_INVALID", str(v.detail))
v = validate_temporal_coverage(case("duel", label_window_start_s=0.0, label_window_end_s=2.0, tracking_start_s=0.9025, tracking_end_s=5.33))
check("2b duel_lost_01 shape (label 0.0 s, tracking 0.90 s) -> LEFT_BOUNDARY_INSUFFICIENT_CONTEXT", v.status == "LEFT_BOUNDARY_INSUFFICIENT_CONTEXT")
v = validate_temporal_coverage(case("noctx", label_window_start_s=1.0, label_window_end_s=2.0, tracking_start_s=0.8, tracking_end_s=5.0))
check("3 starts before label but inside required context -> INVALID", v.status == "BENCHMARK_TEMPORAL_COVERAGE_INVALID")
v = validate_temporal_coverage(case("end", label_window_start_s=1.0, label_window_end_s=4.0, tracking_start_s=0.0, tracking_end_s=3.0))
check("4 tracking ends before label closes -> INVALID", v.status == "BENCHMARK_TEMPORAL_COVERAGE_INVALID")
v = validate_temporal_coverage(case("missing"))
check("5 pre-3G fixture without temporal fields -> INVALID (fail closed)", v.status == "BENCHMARK_TEMPORAL_COVERAGE_INVALID", str(v.detail))
v = validate_temporal_coverage(case("zero", label_window_start_s=0.0, label_window_end_s=1.2, tracking_start_s=0.0, tracking_end_s=5.0))
check("6 label at clip start, even with tracking from frame 0 -> LEFT_BOUNDARY_INSUFFICIENT_CONTEXT (Task 3H correction)",
      v.status == "LEFT_BOUNDARY_INSUFFICIENT_CONTEXT", str(v.detail))
v = validate_temporal_coverage(case("edge", label_window_start_s=0.5, label_window_end_s=1.2, tracking_start_s=0.0, tracking_end_s=5.0))
check("6b label opening exactly one context-length into the file, tracked from frame 0 -> VALID",
      v.status == "TEMPORAL_COVERAGE_VALID")
v = validate_temporal_coverage(case("short", label_window_start_s=0.3, label_window_end_s=1.2, tracking_start_s=0.0, tracking_end_s=5.0))
check("6c label opening 0.3 s into the file (context would start at -0.2 s) -> LEFT_BOUNDARY_INSUFFICIENT_CONTEXT",
      v.status == "LEFT_BOUNDARY_INSUFFICIENT_CONTEXT")
g = evaluation_gate(case("prov", video_sha256="0" * 64, label_window_start_s=2.0, label_window_end_s=3.0,
                         tracking_start_s=1.0, tracking_end_s=4.0))
check("7 provenance mismatch -> not allowed, temporal not assessed",
      not g.event_evaluation_allowed and g.temporal_coverage_valid is None)
try:
    require_evaluation_allowed(case("raise", label_window_start_s=0.0, label_window_end_s=1.0, tracking_start_s=0.9, tracking_end_s=2.0))
    check("8 raising variant raises on INVALID", False)
except BenchmarkEvaluationNotAllowed:
    check("8 raising variant raises on INVALID", True)
g = evaluation_gate(case("good", label_window_start_s=2.0, label_window_end_s=3.0, tracking_start_s=1.0, tracking_end_s=4.0))
check("9 provenance + coverage valid -> allowed", g.event_evaluation_allowed)
fails = [n for n, ok in R if not ok]
print(f"\n{len(R) - len(fails)}/{len(R)} temporal-guard checks passed")
if fails:
    raise SystemExit(1)
