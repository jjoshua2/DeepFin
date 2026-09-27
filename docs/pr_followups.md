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
| #852 | Session open and the fallback guard are fixed on `def127699`. `_field_is_repeated` uses `is_repeated` when that attribute exists and `label == LABEL_REPEATED` otherwise, and `open_verified_bt4_session` refuses the session unless fallback is disabled before the provenance hash is stored. Independent review approves that head. No child is based on it, so it can be squash-merged when the test job passes. After that, one non-blocking hole remains: `getattr(sess, "_enable_fallback", False)` treats a missing private flag as disabled. On ORT 1.17.3 and 1.29.0 the flag is present and is what `run` reads. |
| #868 | The source adapter now rejects the legacy V4/two-game PASS receipts and re-runs the pinned reviewed `verify_bank` before publication, comparing replay facts, terminal facts, hashes, and bank identity. A regression test flips an external verifier oracle after receipt creation and requires publication to reject. It remains open until the repaired exact head passes fresh CI/review. |
| #796 | `disk_pause` is now loaded from the pinned helper path, and the receipt test writes that file instead of injecting `sys.modules["disk_pause"]`. `bootstrap_experiment_operator` is still returned from `sys.modules` when that name is already loaded, and `test_real_base_a_runner_executes_and_binds_honest_receipt` injects a fake module there. The test's `operator_runtime` directory does not contain the file, so CI never opens it. This stays open until that operator helper is loaded from a pinned file the same way. |
| #818 | The pair runner does launch the two storage roots and a crash stays `INCOMPLETE`. The measurement does not reach a real batch. `packed_trainer_probe.observe` requires `ply`, while the sampler's identity column is `ply_index`. The first real batch raises before an optimizer step. The probe tests feed a synthetic batch that already contains `ply`. #823, stacked on this branch, reads `ply_index` instead. This root stays open until the root itself observes the sampler's identity columns. Merge would be a merge commit because #823 is based on it. |
