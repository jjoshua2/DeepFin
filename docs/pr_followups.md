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
| #819 | Population and block-size laws. Merged as `a1979aa5a`. #820 is retargeted to main. | They do not read the index `Chess.bend` uses in `slide_mask` / `slide_offset`, so a wrong production lookup does not fail this suite. The same note is in that branch's layout README and experiment record. Later stack PRs, including #873, #875, and #877, take up actual lookup. |
| #852 | Verified-session open and the fallback guard. Squash-merged as `765f5e35d`. | `getattr(sess, "_enable_fallback", False)` treats a missing private flag as disabled. On ORT 1.17.3, this host's 1.23.2, and 1.29.0 the flag is present and is what `run` reads. |
| #868 | Publication re-runs the pinned verifier inside `inspect_bank`. Squash-merged as `e9c7e8084`. | `write_source` only re-hashes the audit file. A direct caller, and `isolate_source_adapter_from_strict_audit`, skips the replay. The publication test never calls `write_source`. |
| #842 | Host overlap. Squash-merged as `f9fcb4b4e`. The ON arm prepares the next host batch during the optimizer step, and a failed run stays `INCOMPLETE`. | The qualification receipt does not embed `runtime_files` or `cpu_producer_sha256`, so a dirty tree with the same HEAD is tied across CPU and GPU by `runtime_commit` plus each plan's inventory. `make_host_trainer` does not call `Trainer.__init__`. |

## Open, not merged

| PR | Why it is still open |
| --- | --- |
| #796 | `disk_pause` is now loaded from the pinned helper path, and the receipt test writes that file instead of injecting `sys.modules["disk_pause"]`. `bootstrap_experiment_operator` is still returned from `sys.modules` when that name is already loaded, and `test_real_base_a_runner_executes_and_binds_honest_receipt` injects a fake module there. The test's `operator_runtime` directory does not contain the file, so CI never opens it. This stays open until that operator helper is loaded from a pinned file the same way. |
| #818 | The pair runner does launch the two storage roots and a crash stays `INCOMPLETE`. The measurement does not reach a real batch. `packed_trainer_probe.observe` requires `ply`, while the sampler's identity column is `ply_index`. The first real batch raises before an optimizer step. The probe tests feed a synthetic batch that already contains `ply`. #823, stacked on this branch, reads `ply_index` instead. This root stays open until the root itself observes the sampler's identity columns. Merge would be a merge commit because #823 is based on it. |
| #786 | The factorial runners execute, a failed arm stays incomplete, and a changed audited-source manifest fails closed. Independent review approves `cd6e075bf`. #810 and #811 are based on it, so landing is a merge commit after the index conflict is cleared and the test job passes. `training/jobs.frozen.json` still says `FROZEN_REVIEWED_NOT_QUEUED` while `pipeline_adoption.json` says those stages were queued. `--allow-partial-corpus` is on the frozen command, and `run_arm.main` still requires 58,090,688 realized rows. |
| #820 | The eight laws check `Sliders.pext_index` and the imported recurrence. Rewriting that helper body to 0 fails `Correspondence.bend`. Nothing in the diff imports `Chess.bend`, `slide_mask`, or `slide_offset`. `verify.js` only admits `standalone/` and `bitboard_probe/Sliders.bend`, so replacing the production lookup leaves these laws green. The pull request text already says an actual `Chess.slide` read is not proved. #821 is based on it, so a later landing would be a merge commit. This stays open until a proof fails when that production lookup changes. |
