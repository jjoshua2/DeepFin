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
| #759 | Sequential Ceres blocks stop on a nonzero exit or a qualification mismatch. A missing output root raises. `disk_free_gib` takes the item output path. | A `harvest_ceres` item whose status is `failed` is treated like `logged`, so the next queued arena, match, or command still launches. `scripts/supervise_bootstrap_queue.sh` exits 0 on `DEADLINE` and `QUEUE_IDLE`, including after that skip. |

## Open, not merged

| PR | Why it is still open |
| --- | --- |
| #819 | Population and block-size laws are real. They do not read the index `Chess.bend` uses in `slide_mask` / `slide_offset`, so a wrong production lookup does not fail this suite. The same note is in that branch's layout README and experiment record. Later stack PRs, including #873, #875, and #877, take up actual lookup. Merge is a merge commit because #820 is based on it. |
| #852 | `field.is_repeated` crashes session open on the protobuf this repo allows. Use `field.label == field.LABEL_REPEATED`. After admission, ONNX Runtime fallback can replace the session without the verified CUDA device or memory cap, and the code still records the original session hash. The core check does not run, so this stays open until those two paths are fixed. |
| #858 | The huge-page screen is real and its negative result matches the receipts. `publication-manifest.json` still pins `tests/test_benchmark_numpy_thp_hotload.py` at `7460b94c…`, the scratchpad tests that were executed. The committed tests hash to `a3361c97…` after a typing-only `Popen` seam change, so the sealed manifest does not describe the tests in the pull request. The screen can land with that gap. |
| #868 | Publication does not run the strict replay. `verify_audit` accepts `PASS_SAVED_BT4_TWO_GAME_BANK_AUDIT` or `PASS_INDEPENDENT_BT4_V4_READBACK` from a summary hash, a row count, and a status string. It does not call `verify_bank`. The ordinary path allowlists the verifier file and checks that receipt facts are a subset of `terminal.json`, where `status` only has to start with `PASS_`. A hand-written passing terminal publishes rows the verifier would reject. `audit_bank` does call the verifier. `inspect_bank` and `write_source` do not. This stays open until publication calls that verifier. |
