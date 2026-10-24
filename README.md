# SPANA engine v2-rebuild

Source of truth for the SPANA football analytics engine **rebuild**.

The original 3A–3Q working codebase was lost in a workspace reset on 2026-10-20. All new code in this tree is **NEW CODE (SPANA engine v2-rebuild)**. It is not the original 3A–3Q engine, and none of it may be described as recovered.

## Layout

| Path | What it is | Rules |
|---|---|---|
| `R0/` | R0 recovery map, tracker interface map, evidence, and the R1 plan | Documentation; verified archive at `~/Documents/SPANA_Engine_Archive/R0_2026-10-24_1237/` |
| `recovered/SPANA_recovered_originals/` | Byte-exact recovered originals, unpacked from `SPANA_recovered_originals.zip` (sha256 1e99c82f…) and verified against its `SHA256SUMS.txt` | **READ-ONLY EVIDENCE.** Never edit, reformat or rename. `task3r/` is deliberately excluded because Task 3R is frozen. |
| `R1/` | R1 design decisions, model and licence record, reports | — |
| `tools_rb/` | NEW support tools (for example, the GOLD human label sheet) | NEW code header required |
| `player_tracker_rb/`, `tests_rb/` | NEW selected-player tracker and its tests | Created only after Nathan approves R1 implementation |

Historical paths (`player_tracker/`, `player_tracker_mo/`, `new_tracker_snapshots.py`) are **never** written here.

## Persistence rules

Nothing counts as built until it is:
- written here;
- hashed;
- committed;
- snapshotted to `~/Documents/SPANA_Engine_Archive/<PHASE>_<ts>/`;
- verified from the archive copy.

Second independent backup: **NOT CONFIGURED** (open risk).
