# BT4 verified-session and feed-identity foundation — 2026-09-23

This CPU-tested slice prepares a BT4 session for an opt-in generator without
starting games or writing labels. `open_verified_bt4_session` checks a pinned
ONNX file SHA-256 before and after session creation, rejects external tensor
files including sparse and nested tensors, validates the one 112-plane
float16/float32 input and named floating policy/WDL heads, and rejects a
requested CUDA provider unless it is first, on device 0, with the requested
memory limit realized. The input batch dimension is stamped; only dynamic or
fixed-one inputs are accepted so singleton root evaluation remains valid. It
reuses the raw collector's session, head-resolution and
remap-provenance helpers. The repetition-plane mode must already be configured;
the factory and evaluator only assert it and never flip the process-global flag.

The frozen session provenance records the model path/hash, actual input name,
dtype and shape, realized provider list/options, output names/types/shapes,
explicit W/D/L
side-to-move semantic contract, input encoding/mode, remap blobs and source-file
hashes. Its canonical SHA-256 is attached to each root observation. This stamp
is internal provenance, not independent attestation: a caller can construct a
bare evaluator with an asserted string. A future writer must receive the
`VerifiedBT4Session` object and match its descriptor to the observations, not
trust a nonempty root stamp alone. Source hashes do not pin loaded native
extension or library versions; those remain part of the future writer's full
producer receipt. The WDL
order/POV is an explicit producer contract; graph metadata alone cannot prove
those semantics. A provider list shows the session's registered providers, not
which provider executed every node. The direct evaluator remains usable for CPU
contract tests, but its outputs have no verified-session stamp.

Each root also records `onnx_feed_sha256`, computed over its named, typed,
shaped, contiguous **post-cast** LC0 input row immediately before the shared
session call. This is separate from the original float32 `input_key`. A future
writer must additionally verify the physical source-qualified row, the key of
the actually stored float16 replay input, and the policy/WDL array identities;
this slice does not produce a publishable sidecar receipt. Existing raw
collector receipt `run06_g10/w05-00281.bt4.zarr` pins a real model/head/provider
contract, but its model was not loaded for this change.

Validation used a tiny synthetic ONNX artifact and fake sessions only: 124
focused adapter/conversion/WDL/factory cases pass, including artifact mutation,
external tensor references (also sparse), bad input or head metadata, CUDA
fallback and cap refusal, provider
option changes, batched feed order, and float32 inputs that collapse to the
same float16 feed. Scoped Ruff and basedpyright pass. This is no throughput or
playing-strength result.
