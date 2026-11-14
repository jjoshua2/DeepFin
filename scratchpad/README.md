# scratchpad/

Local, untracked scratch space: probe scripts, run output, logs and notes from a
working session. `.gitignore` ignores everything here except this file and the few
paths listed there. Do not commit new scratch output; publish finished records and
compact evidence under `docs/experiments/` and keep bulk artifacts outside the
checkout (see `docs/workspace_storage.md`).

The content that used to be tracked here (1,458 files, about 111 MB, last modified
August–September 2026) was removed from git on 2026-11-13. It is:

- archived on the project host under `~/chess-artifacts/repo-slim-20261113/scratchpad/`,
  same relative paths, with `MANIFEST.sha256` at the archive root; and
- recoverable from git history at tag `pre-slim-20261113` (also branch `bend`), e.g.
  `git show pre-slim-20261113:scratchpad/<path>`.

Records and the frozen ledger that cite `scratchpad/<path>` refer to that archived
content.

Still tracked, because staying code or current docs read them:

- `seed_shift_control.py` — loaded by `tests/test_seed_shift_control.py`.
- `fpr_measure.py` — reproduction command in `docs/eval_protocol.md`.
- `bt4/inspect_bt4.py` — reproduction command in `docs/bt4.md`.
- `bt4_joint20/sf_close_run02/completed_s100_readout.json` — read by
  `scripts/bt4_confirmation.py` and `scripts/bt4_direct_screen.py`.
- `bt4_joint20/sf_close_run02/optional_sharpened_ties_data_qualification.json` —
  read by `scripts/bt4_one_epoch_screen.py`.
- `bt4_joint20/sf_close_run02/sf-close/manifest.json` and
  `bt4_joint20/confirmation_openings_v1/openings.fen` — inputs named in
  `docs/bt4_confirmation.md`.
