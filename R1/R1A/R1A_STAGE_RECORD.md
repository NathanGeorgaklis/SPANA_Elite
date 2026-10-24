# R1-A stage record: contract foundation

**Stage:** R1-A, approved by Nathan on 2026-10-24 (`R1/R1_APPROVALS.md`).

**Scope:** `player_tracker_rb/config.py`, `player_tracker_rb/output_schema.py`, their tests, and a registry export tool.

**Not built:** no tracking, detection, appearance or state-machine logic. No model was downloaded. The event engine, Task 3R and the holdout were not touched.

## Files

| Path | Purpose |
|---|---|
| `player_tracker_rb/__init__.py` | Package marker. It carries the isolation note (D02). |
| `player_tracker_rb/config.py` | The parameter registry: 86 parameters, each with name, type, units, meaning, default, source (HISTORICAL_REFERENCE or NEW_R1), evidence level, validation status, role and evidence. It also does unit conversion: seconds to frames, and target heights *h* to pixels. |
| `player_tracker_rb/output_schema.py` | The Pass-1 JSON contract. The historical part is preserved; the additive `v2` blocks are new. It provides validators, R1→historical state mapping, target safety invariants, `strip_v2`, canonical serialisation and record builders. |
| `tests_rb/test_config.py` | Registry completeness and provenance; override and type checks; hash; unit conversions. |
| `tests_rb/test_output_schema.py` | Both frozen historical exemplars validate; synthetic R1 documents validate; every invariant has a rejection test. |
| `tests_rb/test_package_hygiene.py` | Checks the rebuild header, forbidden imports (D02, D21) and numeric literals outside `config.py`. |
| `tests_rb/_synth_docs.py` | Hand-written contract fixtures. They make no tracking decisions. |
| `tools_rb/dump_config_registry.py` | Exports the registry to `R1/R1A/R1A_CONFIG_REGISTRY.csv` for review without reading code. |
| `fixtures/historical_contract/` | The frozen 3N Pass-1 exemplars and their Pass-2 snapshots, copied from `~/Downloads/task3n_current_code_baseline_core.zip` (sha256 `c8b7113e…0147`). The hashes equal the 3N `PASS1_FROZEN_SHA256.txt` values: GOLD `88246efe…1dc5`, PC `40fa0df1…30fc`. They are listed in `HISTORICAL_CONTRACT_HASHES.sha256`. |

## Registry summary

The registry has 86 parameters (`CONFIG_REGISTRY_VERSION r1a-1`). The default config hash is `7fc5276eead0569bd674b0adf2fcd30ed4bd448df7bf71f332897773fffaeb10`.

| Role | Source | Count |
|---|---|---|
| BEHAVIOURAL | NEW_R1 | 50 |
| RUNTIME | NEW_R1 | 11 |
| CONTRACT | HISTORICAL_REFERENCE | 10 |
| CONTRACT | NEW_R1 | 2 |
| REFERENCE_ONLY | HISTORICAL_REFERENCE | 13 |

- **NEW_R1 behavioural values are UNVALIDATED.** They are design defaults, to be checked by the synthetic suite (R1-G) and by DEV tracking labels (R2). They must never be tuned on event results.
- **REFERENCE_ONLY historical values** (the association weights, floor 0.30, margin 0.08, gates 90/35/350 px, fast path 45 px, and others) cannot be read through `TrackerConfig.get()` and cannot be overridden. Tests enforce this.

## Contract decisions made in R1-A

- **Additive v2 only.** Each new block sits under a key named `v2` in `summary`, in each `frame_log` record, and in each per-frame track record. There are no new top-level keys.
  - `strip_v2(doc)` always leaves a document that validates as historical. This is tested.
- **`final_tracks` records carry no v2 block.** The per-frame v2 data is authoritative.
- **Target record.** The track id stays `T001`.
  - Each R1 state writes exactly one historical (state, occlusion_status, position_source) triple, and every one of those triples was observed in GOLD.json.
  - Only R1 TRACKING maps to legacy TRACKING.
  - Held or predicted boxes are never written as `CONFIRMED_DETECTION` in an ACTIVE record.
  - An untrusted identity forces `identity_confidence` ≤ 0.49, below the 0.5 trust threshold in `actor_attribution`, and must name a reason.
- **`selected_player_confidence`.** R1 writes `identity_confidence` on the target in every state, and 0.0 on every other record. This never exceeds `identity_confidence`, which is the historical bound.

## R0 errata found by the R1-A contract tests (VERIFIED_OUTPUT)

These are corrections to R0 statements, found by validating the frozen exemplars. R0 documents are not edited; the errata are recorded here.

- **E1.** R0 said `detection_confidence` is quantised by `position_source` in "all records". In fact it holds for every frame_log record **except state REIDENTIFYING**. Those records carry `CONFIRMED_DETECTION` but `detection_confidence` 0.0: 11 of 11 such records (GOLD 1, PC 10). This is now a CONTRACT parameter, `detection_confidence_reidentifying = 0.0`.
- **E2.** R0 said `selected_player_confidence` equals `identity_confidence` on the selected track. That holds on ACTIVE selected records only. On OCCLUDED it is about ×0.85, and on AMBIGUOUS about ×0.55; the formula is UNKNOWN (GOLD frames 86–91, 93, 100, 154, 157, 165). It is always ≤ `identity_confidence`, and 0.0 on non-selected records. The validator enforces only that bound.
- **E3.** `final_tracks` also uses state `TERMINATED` (GOLD 20 of 35, PC 45 of 70), and a non-null `merged_into` (for example `T014`). Neither ever appears in `frame_log`. The validator accepts them in `final_tracks` only.

## Test results

- **Cloud workspace** (Python 3.13): 58 tests, OK.
- **Mac** (Python 3.10.12, numpy 2.2.6, cv2 5.0.0, onnxruntime 1.23.2): see `R1/R1A/R1A_TEST_OUTPUT_mac.txt`. That run is authoritative.

## Recovery note

This stage was written partly before a session-limit interruption, at about 14:50 Sydney. On resumption at 16:39, every file was re-read in full and the tests were re-run before anything was written to the Mac.
