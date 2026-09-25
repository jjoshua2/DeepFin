# Occupied-square decoding and abstract-square correspondence

Opt-in P3 continuation of the board partition suite. No production function,
compiler input, earlier law, ordinary CI budget or perft depth changes.

```sh
bun native/bend_engine/standalone/proofs/decoder/focused.js /path/to/pinned/bend --report /tmp/decoder-proofs.json
bun native/bend_engine/standalone/proofs/decoder/verify_native.js /path/to/pinned/bend --report /tmp/decoder-native.json
# Expensive full chain, not implied by modular retained evidence:
bun native/bend_engine/standalone/proofs/decoder/verify.js /path/to/pinned/bend --report /tmp/decoder-aggregate.json
```

## Four universal contracts

- `square_projection_valid`: the global board partition projects to a valid
  local eight-Boolean row at every bounded square.
- `abstract_square_roundtrip`: classifying that row independently, then encoding
  it, reproduces all six kind and two color observations exactly.
- `guarded_decoder_matches_square`: the explicit occupancy-guarded observation
  using actual `Chess.occupied`, `Chess.piece` and the white bit equals the
  independent classification, including empty squares.
- `occupied_decoder_matches_square`: on an occupied square, the actual raw
  piece tag and observed color identify the independent abstract occupant.

All laws require the existing global partition invariant and `toNat(square)<64`.
The raw decoder law additionally requires an actual occupied-bit observation.
Those are input premises, not desired-result certificates. Metadata is arbitrary.
The consumer combines the last law with roundtrip to reconstruct **all eight
square bits from the actual decoded kind/color**. It also exercises initial-board
validity supplied by the existing proof, fresh real insertions at boundary bits,
empty fallback behavior, and independently invalid row classification.

`Spec.Cell` has Empty, Occupant and Invalid constructors. Its independent
classifier lists the thirteen permitted states, not the priority decoder's
fallback. The guarded observation in this proof module is not a new production
API and does not change `Chess.piece`: on empty squares its raw fallback is still
5. `selected_tag` is only the bridge to the actual implementation, not the
independent specification used on the right side of the contracts.

`Bits` proves actual U64 bit tests correspond to structural word projection for
all supported indices, including both limbs. Word induction projects the global
partition; a finite 256-row Boolean proof establishes the independent local
classification and encoding correspondence. The finite bit-position bridge is
checked in Bend. Neither universal board correctness nor bit projection is
inferred from native samples. No new axiom, unsafe/foreign witness or hole.

## Tests and their evidence layers

The focused command checks the public importing consumer and eighteen controls:
nine ordinary semantic/refinement failures, eight manifest/import policy checks,
and one synthetic warning-output unit test. The last is not a compiler execution.
Seven semantic cases mutate actual functions or the independent classifier;
two use concrete empty/inconsistent-board counterexamples to reject omitted input
premises. A crash, missing file, malformed term, timeout or affine-use error is
not accepted as semantic rejection. Successful checking requires status zero and
exactly `All terms check.`. Each child is bounded; the complete chain is opt-in.

The native verifier uses an independent 64-square optional piece/color model.
The probe executes real Chess reads and actual Position.put/start, not the proof
classifier or a precomputed expected decoder label. Four native modes compare
1,920 square observations and 21,120 output fields per mode, covering every one
of 832 square/local-state pairs and 768 fresh square/kind/color insertions, actual
initial-board queries, and additional random consistent boards. Occupied cases
also reconstruct all eight observed bits. The 169 empty rows explicitly test raw
fallback behavior outside the occupied-law premise. Nine malformed requests are
rejected by the bounded probe, not by the underlying total raw Chess API.

Three disposable native mutations alter pawn decoding, king fallback or occupied
bit combination. Their witnesses are restricted to occupied, consistent boards;
they compile and execute before wrong-value rejection. Modes repeat fixtures and
are not disjoint datasets or exhaustive arbitrary-board coverage. The probe
observes a square, not the complete Board result, whole-board bijection, metadata
validity, parser traversal, move application, legality or reachability.

The expensive wrapper preserves the unchanged board aggregate. Publication may
retain exact-source parent qualification instead of executing the entire chain;
its record must distinguish that modular evidence from a new aggregate run.

## Trust and remaining work

Pinned checker/Base, native lowering/storage, ABI, compiler, OS and hardware
remain separate trust boundaries. These are pointwise abstract-square results,
not a global abstract-board bijection or FIDE movement theorem. The inherited
FIDE-2023 semantic anchor is unchanged. Next steps are parser-derived freshness,
whole-board abstraction and invariant-preserving removal/move operations. No
application responsibility moves from Python into Bend in this proof increment.
