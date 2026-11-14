# Current-P0 SF evidence carrier

`scripts.sf_current_p0_sidecar` produces a separate candidate carrier or validates
one against the original source. It is a CPU-only data tool. It does not query an
engine, train, choose a target/value variant, or admit labels to learning.

```bash
python -m scripts.sf_current_p0_sidecar \
  --protocol /external/protocol.json --protocol-sha256 SHA256 \
  --admission /external/admission.json --admission-sha256 SHA256
```

The protocol selects `phase: "produce"` or `"validate"`. Both require authenticated
engineering selection, the accepted closed census protocol/readout/CPU terminal,
independent census review, original producer authority/proofs and exact source
pins. The root-owned outer CPU owner, affinity `[20,22]`, two threads, hidden CUDA
and finite inner/outer caps are checked by the actual CLI. Root must bind these
inputs and separately admit execution; source preparation is not admission.
The separate `ROOT_ADMITS_CURRENT_P0_SF_SIDECAR_PHASE_V1` record binds that exact
protocol ref, phase, scope, current source/file digest, runtime caps, engineering
selection and census readout. Keeping the admission outside the protocol avoids
a self-hash cycle. Candidate output records it; validation joins it to the exact
closed producer child command as well as its own separately supplied admission.

`scope: "FIXED_ORIGINAL_G1"` fixes all 256 original-G1 TRAIN games / 24,827 rows and
manifest `35279e2e…`. `SYNTHETIC_ONLY` exists for known serialized-document fixtures;
those fictional acceptance documents are never actual root authority. The tool
can also consume the untouched historical four-row direct-CLI fixture using its
actual closed synthetic owner/root acceptance and independent review. That binds
the old executed census separately from the new sidecar's own source/admission;
it does not relabel the fixture's fake producing source as the real learner.
The tool uses the current caller's source directory, not a historical runtime path. The
published census algorithm and encoding/move/value dependencies must match the
historical source pins. `ext_check.py` is pinned to current source separately:
main removed an optional-symbol allowlist entry, not an import/startup function.
This is not a claim that every historical dependency file is byte-identical.

## Source and carrier

The producer freshly replays and checks full history, actual ply, clock, STM/POV,
legal map, input F16 identity, row/game/source roles and inventories through the
unchanged census utilities. Every recomputed metadata row must equal its accepted
census observation. It then reopens only that game's original numeric fields to
copy the retained actual-ply predecessor's SF policy, values and diagnostics.
The census journal contains no policy vector; policy is never made from sparse
PV scores, regret or a new temperature/transform.

Per-game `rows.json` retains all row IDs, original inventory/proof refs, input
context/legal hash, source classes/reasons/storage-presence facts, explicit-P0
evidence, and qualified flags. Invalid and unavailable rows remain in place with
false qualified flags. No-join observed flags are null, not invented negatives.
Numeric padding is non-observation. Invalid means the declared cross-field source
checks failed; it does not prove a wrong position, bad search, or the historical
cause. Bounds/reranking can separate an exact record snapshot from retained PV1;
without original wire the cause remains unknown. No parser change or salvage
transform is selected here.

Plain numeric `.npy` files preserve these distinct observations:

- Original predecessor policy F16 bytes, plus current-P0 legal mask.
- Original stored record-POV F16 WDL and exactly one W/L swap into current/query STM.
- Native query-STM W,D permille and reconstructed L as int16, separately identified
  from stored values. Per-row source relation and producer config distinguish
  native-derived, CP-logistic and clipped/unknown stored-value observations;
  the known four-row fixture is native-derived, not CP-logistic.
- Same-predecessor original int32 record metadata and int16 sparse PV block, with
  observation flags, for coherent later diagnostics.

Qualified flags express only these source-data checks. They are not a recipe,
training eligibility or wire authentication. Full PVs, bounds, request IDs and
per-candidate effort are not recovered. All source fields stay unchanged.

The manifest is `CANDIDATE_CURRENT_P0_SF_SIDECAR_V1`, with complete per-game file
hashes/bytes and counts. Files become read-only and directories sealed only after
all checks; late failures preserve an external `.failed.json` marker. Candidate
completion is not root acceptance or an accepted dataset.

## Separate validation

The validator requires the exact closed producer CPU owner/command/protocol,
read-only candidate manifest and absence of a late failure marker. It refuses
missing, extra, changed or unsafe files and checks NPY headers/byte lengths before
numeric loading (no pickle/object codecs). It freshly reopens the original source,
recomputes every game/context/payload, and compares carrier row metadata, all vector
bytes/dtypes/shapes, flags, counts and full denominator. Before/after inventories
protect input custody. Output is `VALIDATION_CANDIDATE_CURRENT_P0_SF_SIDECAR_V1`;
owner closure and subsequent root acceptance remain separate.

This is an independent process/pass using shared source algorithms, not an
independently authored teacher-quality proof. Synthetic tests use known moves,
POVs, field flags and vector values to expose shared semantic mistakes. The
pytest harness can import Torch through repository conftest; passing those tests
alone does not qualify the direct CLI's no-Torch/native guards. At source
`cf3c71aad826c4fc35ec9031f4450d946748c36b`, all 33 focused cases and the separate
owned four-row direct producer/validator chain closed PASS. The synthetic CLI's
independent actual review accepted the closed chain without findings. This native-derived
fixture does not qualify actual original-G1 CP-logistic payloads or teacher quality;
actual G1 produce/validate remain unrun. See the [dated qualification record](experiments/2026-11-14-sf-current-p0-sidecar-qualification.md).
The CLI refuses project native images; ignored author-checkout images are not
adopted or rebuilt by this tool.

The protocol explicitly bounds metadata per file, original decoded bytes per game
(up to 64 MiB), inventory count, JSONL line/game metadata, carrier decoded bytes
per game and total output bytes (up to 1 GiB). One game's numeric arrays are live
at a time; full-population state contains only small identities/counts/file refs.
Fresh outputs must be outside every input/checkout and cannot reuse a failure
identity. Root chooses actual phase caps and paths before execution.

No finite-consumer, replay, objective, loss, YAML or checkpoint/resume behavior is
changed by these scripts. A later consumer requires its own explicit contract and
qualification; no numeric mixture, head choice, LR, dose, seed or strength test is
selected here.
