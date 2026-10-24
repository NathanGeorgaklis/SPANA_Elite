"""Task 3D, self-test of the provenance guard itself, before trusting it for
the real duel_lost_01 validation. Read-only against real files; writes only
to a throwaway tmp JSON for the save()/load() round-trip check."""
import sys, copy
sys.path.insert(0, ".")
from event_engine_v1.benchmark_fixture import BenchmarkFixture, validate_provenance

VIDEO = "/root/.claude/uploads/98d69dca-3149-5a43-9670-3c417f2af6d4/8a710009-dual_lost_01.mov"

fx = BenchmarkFixture.create(
    clip_id="duel_lost_01",
    video_path=VIDEO,
    target_player_label="White #20",
    seed_frame=53,
    seed_bbox=[262.1, 503.4, 335.3, 658.3],
    expected_primary_event="DUEL",
    expected_outcome="LOST",
    expected_event_window="1-2s",
    fixture_provenance_notes="self-test fixture",
)

print("--- TEST 1: fixture validated against the real, unmodified file -> expect MATCH ---")
r1 = validate_provenance(fx, VIDEO)
print(r1.status, r1.mismatches)
assert r1.status == "MATCH", "TEST 1 FAILED: guard did not pass a genuinely matching file"

print("\n--- TEST 2: fixture's own recorded hash corrupted -> expect BENCHMARK_PROVENANCE_MISMATCH ---")
fx_bad_hash = copy.deepcopy(fx)
fx_bad_hash.video_sha256 = "0" * 64
r2 = validate_provenance(fx_bad_hash, VIDEO)
print(r2.status, r2.mismatches)
assert r2.status == "BENCHMARK_PROVENANCE_MISMATCH", "TEST 2 FAILED: guard did not catch a hash mismatch"

print("\n--- TEST 3: fixture's recorded resolution corrupted -> expect BENCHMARK_PROVENANCE_MISMATCH ---")
fx_bad_res = copy.deepcopy(fx)
fx_bad_res.width = 9999
r3 = validate_provenance(fx_bad_res, VIDEO)
print(r3.status, r3.mismatches)
assert r3.status == "BENCHMARK_PROVENANCE_MISMATCH", "TEST 3 FAILED: guard did not catch a resolution mismatch"

print("\n--- TEST 4: fixture's recorded file size corrupted -> expect BENCHMARK_PROVENANCE_MISMATCH ---")
fx_bad_size = copy.deepcopy(fx)
fx_bad_size.file_size_bytes = 1
r4 = validate_provenance(fx_bad_size, VIDEO)
print(r4.status, r4.mismatches)
assert r4.status == "BENCHMARK_PROVENANCE_MISMATCH", "TEST 4 FAILED: guard did not catch a size mismatch"

print("\n--- TEST 5: missing file path -> expect BENCHMARK_PROVENANCE_MISMATCH, not a crash ---")
r5 = validate_provenance(fx, "/does/not/exist.mov")
print(r5.status, r5.mismatches)
assert r5.status == "BENCHMARK_PROVENANCE_MISMATCH", "TEST 5 FAILED: guard did not handle a missing file"

print("\n--- TEST 6: save()/load() round trip preserves every field ---")
h = fx.save("/tmp/claude-0/-home-claude/98d69dca-3149-5a43-9670-3c417f2af6d4/scratchpad/event_engine_v1/duel_lost_01/_selftest_fixture.json")
fx2 = BenchmarkFixture.load("/tmp/claude-0/-home-claude/98d69dca-3149-5a43-9670-3c417f2af6d4/scratchpad/event_engine_v1/duel_lost_01/_selftest_fixture.json")
assert fx2 == fx, "TEST 6 FAILED: round trip altered the fixture"
print("fixture file sha256:", h)

print("\nALL 6 GUARD SELF-TESTS PASSED.")
