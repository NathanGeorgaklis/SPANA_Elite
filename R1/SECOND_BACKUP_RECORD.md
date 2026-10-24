# SECOND_BACKUP_RECORD.md

This records the second independent backup (R1 implementation approval, 2026-10-24).

## Destination
A private GitHub repository: **https://github.com/NathanGeorgaklis/SPANA_Elite** (owned by Nathan Georgaklis). It is independent of the Mac disk.

## First push: 2026-10-24, about 14:40 Sydney
- Method: a git bundle was made on the Mac (`spana_rebuild_f669e22.bundle`, sha256 `a4dd6e05c8e2641947ae1100ebd46175a8b4ead96034f32b42378a4ad2ed487b`; Git reported "complete history"). It was copied to the Claude session (the hash matched) and pushed with session-scoped GitHub credentials. **No token was stored** in the repository, its config, or the Mac's git config.
- Pushed: `main` at `f669e22b79f5ce7ea4b153f75c94a4d92a2923ff`, with the full history (4 commits back to the R0 baseline `fb90702f3ba2114725933190d83642c8538b5399`).

## Verification from a fresh clone of GitHub
- `git rev-parse HEAD` = `f669e22b79f5ce7ea4b153f75c94a4d92a2923ff`.
- `git fsck --full`: no errors. The R0 baseline commit `fb90702` is present.
- Tracked files: 370. The sorted list of `sha256sum` results for every tracked file has digest `e429fd43a40b3e7520827a57909255fc282e6d4e5c73cb70514a89982fa0bed2`. That is **identical** to the same list computed on the Mac working tree, which was clean at HEAD `f669e22`.
- `GOLD_HUMAN_LABELS_v1.csv` (490a8af1…) and `GOLD_HUMAN_LABELS_v1.FROZEN.json` (419a3cae…) on GitHub match the frozen hashes.
- **SECOND_BACKUP_VERIFICATION = PASS.**

## Limitations
- **The tag `r0-recovery-map` was not pushed.** The session's git proxy refused tag pushes with HTTP 403. The tag exists on the Mac, and the commit it points to (`fb90702`) is in the GitHub history. Nathan can recreate the tag on GitHub, or push it from the Mac, if wanted.
- The `SPANA_Engine_Archive/` snapshot folders and the archive zips are **not** in Git. They remain on the Mac only. The source history they contain is in GitHub.
- Model binaries are excluded from Git (`.gitignore`). They will be backed up separately with hashes when they are downloaded.

## Ongoing rule
After every meaningful R1 commit: push to `SPANA_Elite`, then check that the remote HEAD equals the local HEAD.
