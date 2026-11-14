# scripts/

What belongs here:

- tools that run, build, lint, validate, evaluate or operate the package, such as
  `train.sh`, `lint.sh`, `validate.py`, `build_production_extensions.py`,
  `arena_standard.py`, and the watchdog, ratchet and recovery loops;
- the helpers those tools and the package's tests import;
- drivers for current experiments, meaning those cited by records dated
  2026-10-25 or later.

A one-off research driver that only an older experiment record or the frozen
`docs/experiment_ledger.md` cites does not belong here. When its experiment is
closed, archive it together with any test file that tests only that driver.

## Archived drivers

Scripts removed under this rule, and their dedicated tests, are:

- archived on the project host under
  `~/chess-artifacts/repo-slim-20261113/scripts/` and `.../tests/`, with the same
  relative paths and listed in that directory's `MANIFEST.sha256`;
- kept in git history at tag `pre-slim-20261113` (also branch `bend`).

To bring one back:

```bash
git show pre-slim-20261113:scripts/<name> > scripts/<name>
git show pre-slim-20261113:tests/test_<name>.py > tests/test_<name>.py   # if it had a test
```

Then check that it still runs against the current package before relying on it.
