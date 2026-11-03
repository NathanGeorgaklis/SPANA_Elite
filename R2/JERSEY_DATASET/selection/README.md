Candidate-scan scripts and plans, copied from `~/Downloads/_spana_jersey_work` (read-only on the sources):

- `make_contact.py`, `contact_plan.json`: the FIRST scan. Every 30 s, excluding only the registry clips ±30 s. It did
  not apply the DEV/EVAL split; see the protocol deviation in `../README.md`. No clip was selected from it.
- `make_contact_dev.py`, `dev_plan.json`: the DEV-only rescan used for selection. Times were computed in the repo with
  `tools_rb/r2_protocol` (DEV blocks, excluded windows +60 s, registry clips ±30 s).
