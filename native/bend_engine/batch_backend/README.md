# Native fixed-batch backend

This is a synchronous, Bend-owned **tensor transport boundary**, not a second
search controller. `Batch.bend` retains linear ownership of its input/output arrays
until the bound native model has completed. The probe is an external qualification
entry point; it does not change the UCI engine, create a Python engine coordinator,
start concurrent workers, or claim production Gumbel parity.

## Contract

Use only trusted checkpoint-v3 packages and sidecars. The build checks the exact
package hash, checkpoint identity, corrected root encoding, compact policy width,
CPU/F32 dtype/device and static batch. Backend-only bindings admit 1, 2, 4, 8 and 16;
the normal `build_neural.sh`/`--header` singleton gate still rejects batch > 1.
The C++ singleton open/run API independently rejects a batch-bound package or an
incompatible API mode, so a mislinked probe cannot silently pad UCI searches.

One bound model and one caller own the runtime for its entire lifetime. It is not
thread-safe or asynchronous. Shapes are model-bound, not inferred from capacity.
The caller allocates `2^input_log2`/`2^output_log2` F32 slots from `Capabilities`.
Submit `1 <= real_rows <= batch`; the leading `real_rows * channels * 64` inputs
are valid. The backend clears all physical model input rows, copies only valid
inputs and executes the fixed batch. Input tail bytes are ignored, not consumed.
It returns only `real_rows * 1861` raw F32 values in row-major order:

```
row 0: policy[1858], wdl[3]
row 1: policy[1858], wdl[3]
...
```

Returned arrays have the same storage and untouched output tail. Both tensor shapes,
dtypes and devices are validated before any output bytes are published; neither
capacity nor total element count substitutes for tensor shape. The engine consumer
must still validate finite logits, legal policy and ticket identity before accepting
a row into a search. No accepted-row/EPS metric can be inferred at this boundary.

`Result` exposes real rows, physical rows and logical output count separately.
Each successful call costs `batch` physical rows, including `batch - real_rows`
padding. These counters must reach PR1 accounting when a scheduler adopts the API;
that adoption has NOT happened here. Existing singleton accounting is unchanged.
A `row_independent` declaration is required but is not numerical proof; run the
qualification gate for the package being adopted.

## Build and qualify explicitly

Use a fresh output directory and the verified compiler source directory already
used by the standalone build. The script checks its source fingerprint again.

```sh
bash native/bend_engine/batch_backend/build_probe.sh \
  build/native_batch /path/to/checkpoint_b4.pt2 /path/to/libtorch/share/cmake \
  /path/to/verified-bend-source

python -m native.bend_engine.batch_backend.verify \
  --binary build/native_batch/build/deepfin-bend-batch-probe \
  --package /path/to/checkpoint_b4.pt2 --checkpoint /path/to/checkpoint.pt \
  --report /tmp/native-batch.json
```

Model export stays outside the engine, using the existing
`neural_probe.checkpoint.export_checkpoint(..., batch=4, device='cpu')` on an
immutable checkpoint. Never relabel a singleton package as batched. Build/link
LibTorch must match the package version. The verifier checks package/checkpoint
and complete encoding metadata; export/binding alone is not qualification.

The model test uses deterministic synthetic tensor inputs, independent eager
singleton forwards, and full returned logits (not only checksums). A full batch
is followed by partial batches, then the same first row with different padding;
actual physical input padding must be exact zero and output tail must remain NaN.
It checks persistent addresses, one bridge input-tensor allocation, invalid rows
and undersized buffers. This is not a chess-selected-leaf batching test or an Elo
measurement. It does not establish equivalence for all models, positions or buckets.

`DEEPFIN_BEND_MODEL_TRACE` remains opt-in and requires a new owner-only file. This
API writes version-2 records: six little-endian U32s (magic `0x44464232`, forward
sequence, physical batch, real rows, channels, output width 1861), followed by the
actual physical F32 model input and only real F32 outputs. Never parse these with
the singleton version-1 trace reader. Raw traces can contain model/input data and
are not published by the qualification workflow.

## Scope boundary

The default path documented above remains CPU/F32 only. An explicitly enabled
[CUDA/BF16 backend](CUDA.md) adds pinned staging and a single synchronous CUDA
stream, but has not yet passed actual-device numerical qualification. Its separate
build/verification gates never turn CPU testing into GPU evidence. Trained-network
5090 qualification, batch-bucket scheduling and asynchronous search remain open.
The 4,096-node search arena and fixed-wall comparability limit are unchanged.
