# Opt-in raw BT4 CPU prefetch (2026-09-21)

## Preregistration

Stack from reviewed raw-projection PR814 commit
`74a31a1d3ee18328d12c2e6cf3f519316eb61114`. Add default-off
`label_shard(..., cpu_prefetch=False)` / `--cpu-prefetch`, preparing exactly one
next batch on one CPU thread while the caller runs ONNX. Keep model calls,
policy mapping, output publication and source identity checks on their existing
paths. Batch size remains caller-configured; the GPU comparison will fix 128.

Success for this implementation stage requires exact feeds, output arrays,
identity/provenance and native WDL on CPU fixtures; deterministic evidence that
preparation overlaps main-thread inference without preparing a third batch;
propagated producer/consumer failures and KeyboardInterrupt, closed source
readers, joined owned workers, and no partial publication on these failures.
The producer checks cancellation at row/batch boundaries and cannot drain an
entire corpus during error cleanup. The outer orchestrator still owns STOP and
the process group. This implementation does not create GPU contexts in a worker.

Memory is bounded to one consumed and one future batch. At batch128 a float32
112-plane feed is 3.5MiB (float16 1.75MiB); preparation transiently holds up to
roughly 11MiB of list-plus-stacked 175-plane float32 inputs, plus conversion and
small identity arrays/boards. Complete-shard policy/WDL output arrays already
existed and are unchanged. This is a batch-count bound, not a hard RSS quota;
JSON and chess history sizes depend on the source, and the configured batch size
scales these costs.

CPU tests use cores18/19, nice19, two library threads. No GPU execution or active
runtime edit is authorized in this subtask. The parent owns subsequent original,
optimized-serial and opt-in-prefetch GPU trials (fixed128, paired fresh processes).
Keep default off until those trials establish parity and an end-to-end gain.

## Readout

CPU qualification passed 73 tests, including deterministic preparation during
main-thread fake inference, exactly one future, both float16/float32 feeds,
byte-identical output and identity arrays, native WDL, partial batches,
producer/consumer failures, KeyboardInterrupt while waiting, and a real SIGINT
during shutdown. The independently identified interrupted-join race was fixed by
deferring SIGINT only through bounded worker shutdown and reader close, then
restoring and invoking the original handler. Producer threads never call ONNX.

The full banked 8,236-row shard (SHA256
`3c93d2d127bc6d11db9eb31425e0c795c96b417106d737632ef4b689569ae17d`)
also passed exact serial/prefetch preparation parity: source/input keys, game/ply
IDs, canonical feeds and ordered legal moves all match, with the same 64 batches
of 128 and final batch of 44. A ready float32 batch contains 3,675,648 bytes of
arrays; current plus future ready arrays use at most 7,351,296 bytes, excluding
Python objects and preparation transients. CPU-only parity timing was 5.07s
serial / 6.08s prefetch while lint also ran on those cores. There was no GPU wait
to overlap, and this is not throughput evidence.

Whole-repository lint ran: Ruff/Vulture passed, and all 14 type diagnostics match
the unchanged main baseline. Focused type checks passed. Bulk artifacts and exact
source/test hashes are under
`~/chess-artifacts/operations/bt4-cpu-prefetch-20260921`.

Independent reviewer `factorial_receipt_review` approved the code after its
original real-SIGINT shutdown reproducer passed: reader closed, worker joined,
and KeyboardInterrupt preserved. It independently passed all 73 tests and
verified the full-shard receipt's source hash, ordered hashes and batch schedule.
The review is banked as `bt4-cpu-prefetch-independent-review-20260921.json`.

[Compact CPU qualification receipt](evidence/2026-09-21-bt4-cpu-prefetch.json)
links the exact source, test, full-shard result and review hashes. The flag remains
off by default; no GPU execution, live deployment, or end-to-end throughput claim
in this readout.

