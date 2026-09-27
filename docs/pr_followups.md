# Pull-request follow-ups

This is the list of gaps that should survive a merge. GitHub review comments
are not the record. When a pull request lands with a remaining hole, add a row
here in the same session. A missing proof or an incomplete check can land when
the core change is useful and the hole is written here. A change whose main
path crashes, or reports success after a failure, stays open until that path
is fixed.

## Landed with a remaining hole

| PR | What landed | What is still open |
| --- | --- | --- |
| #858 | The bounded NumPy huge-page screen and its negative result are banked; no production default changed. | `publication-manifest.json` pins `tests/test_benchmark_numpy_thp_hotload.py` at `7460b94c…`, the scratchpad tests that were executed. The committed tests hash to `a3361c97…` after a typing-only `Popen` seam change, so the sealed manifest does not describe the exact tests in the merged pull request. |

## Open, not merged

| PR | Why it is still open |
| --- | --- |
| #819 | Population and block-size laws are real. They do not read the index `Chess.bend` uses in `slide_mask` / `slide_offset`, so a wrong production lookup does not fail this suite. The same note is in that branch's layout README and experiment record. Later stack PRs, including #873, #875, and #877, take up actual lookup. Merge is a merge commit because #820 is based on it. |
| #852 | The verified-session branch now contains a compatibility shim for both legacy `field.label` and modern `field.is_repeated` protobuf descriptors, plus a fail-closed check that runtime provider fallback was actually disabled. The repaired branch is reconciled with current `main`; it remains open until fresh exact-head CI passes. |
| #868 | The source adapter now rejects the legacy V4/two-game PASS receipts and re-runs the pinned reviewed `verify_bank` before publication, comparing replay facts, terminal facts, hashes, and bank identity. A regression test flips an external verifier oracle after receipt creation and requires publication to reject. It remains open until the repaired exact head passes fresh CI/review. |
| #796 | `factorial_prepare_and_train.py` does start baseline A after the 35 base seals while preparation is still running, and a real failure stays `INCOMPLETE`. `factorial_base_a_runner.py` never gets there. It loads `disk_pause` only as `{operator_runtime}/disk_pause.py`, which is not in the repo. The pinned helper `training/A/disk_pause.py` is hash-checked and on `PYTHONPATH`, but it is not imported. The runner raises `runtime helper missing: disk_pause` and cleanup kills preparation. The tests inject a fake `sys.modules["disk_pause"]`, so CI never loads the file. This stays open until the runner imports that pinned helper. |
