# Actual validated-root king selection

Stacked on PR1041 `5c7862dcd87976df8ece4bad6be5073924a7dc04`.
The production code and all earlier proofs and counterexamples are preserved.

`consumer.validated_root` proves a property of the exact `Protocol.validate`
output for an arbitrary input array and `Maybe<Game>`. It takes no caller
king-presence, selected-square, row-partition, canonical-turn or table-geometry
certificate. A `None` result carries no board claim. Every actual `Some<Game>`
result carries the actual `Position.valid` certificate, the existing prepared
king-selection bounds/equality/king-bit/owned-bit result, and an in-range square
for the validator's opponent-side `Chess.in_check` query.

`Kings.bend` proves structurally that U64 popcount one implies nonempty, including
both U32 limbs. It extracts the two singleton-king checks from the actual
`Position.structural` conjunction. Actual `Chess.color` selects a nonempty plane
for every U32 side, including noncanonical values. The prior checked
`prepared_king_selection.use_selection` and `use_required` consumers receive
their presence certificate from this producer rather than from the public caller.

`Validate.bend` follows the actual `valid_board`, `king_checked` and `validate`
branches. A dependent continuation threads one actual affine table/result pair.
`consumer.validator_lookup` is a separate range result for the same actual
computed validity flag and opponent side. It is not a wrapper around the
validator's query dispatch. The accepted-output theorem follows the real
validation result and supplies its query-square bound after acceptance.
`consumer.required` composes the actual accepted output with the existing
selected-king filter requirement and needs no caller presence proof.

The domain is the actual validation boundary. This is distinct from the broader
PR1041 theorem, which still includes kingless and multiking partition-valid boards.
The validator rejects kingless and multiking roots. On arbitrary raw validation
inputs it can still accept a noncanonical turn or overlapping pawn/king row;
this proof does not assert canonical turn, partition validity or king decoder
kind from validation alone. The real position parser is a separate producer.

The two PR1041 counterexamples require different missing premises:

* The stale EP16 board fails the actual `Position.valid_ep` rank/victim check and
  is rejected by root validation. Excluding it throughout a move sequence still
  requires coherent EP preservation, not just the EP range result.
* The pawn32-to48 capture witness has a valid initial root. Its arbitrary leaf
  attack table supplies bit48 where the initialized white-pawn slot must supply
  the actual diagonal mask. The existing checked `attack_geometry.Initialized`
  query producer supplies that mask from the extras builder without expanding
  the giant slider table. Further proof is needed to extract pawn advance/capture
  geometry through actual generated move membership and preserve `valid_ep`.

`Protocol.position` validates before `Protocol.moves`, then applies the command
tail without revalidating each child. This proof stops before that tail. It does
not establish continued king presence, EP coherence, king safety, initialized
reachable play, or full legal-move correctness. The existing PR1041 legal-output
counterexamples are reused as evidence and are not imported or re-normalized.

Qualification uses Bend2.0.21+U64
`aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae`, Bun1.4.2 and the verified84-file
fingerprint `d9550e30dbf17f12013aa5db89b27cf6724957fe68213c9c7e409e99f1405ef4`.
Checks are serial:86400 seconds wall, CPU affinity1,3, two worker threads,
6GiB AS/RSS and16MiB per output file. Typed negative controls must fail at the
intended obligation with distinct expected/observed terms; parse, usage, timeout
and resource failures receive no rejection credit. Raw commands, logs, snapshots
and hashes remain outside the checkout. No runtime/GPU changes, merges or P2
giant concrete normalization are performed.

The frozen fixture/consumer check and eleven controls cover seven contract
couplings and four concrete false claims about rejected roots or accepted raw
inputs. From the repository root, run:

```sh
python3 -B -m native.bend_engine.standalone.proofs.validated_root_kings.qualify_validated_root_kings \
  /path/to/pinned/checker --checker-manifest /path/to/checker-tree.json \
  --report /path/to/fresh-report.json --evidence-dir /path/to/fresh-evidence
```

Durable commands, snapshots, raw logs, hashes and independent review are under
`~/chess-artifacts/deepfin-validated-root-kings-20261005/evidence/2026-10-05`.
