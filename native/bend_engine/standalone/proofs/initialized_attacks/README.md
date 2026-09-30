# Initialized attack composition

Three public, opt-in source laws compose the existing complete-table slider and
non-slider proofs with the actual attack-query pipeline and per-square witness scan.

## Public contracts

- `initialized_piece_attack_matches_geometry`: all six typed piece kinds, including
  a queen's two actual slider reads, return the independent target-centred mask and
  the complete unchanged initialized array.
- `initialized_attacked_matches_geometry`: actual `Chess.attacked` returns the
  independent geometric witness decision and that same complete array.
- `initialized_single_king_check_matches_geometry`: actual `Chess.in_check` returns
  that independent decision at the unique selected-side king, against the other side,
  with the complete initialized array retained.

The initialization is one common actual expression for every query:

```text
Tables.extras(extra, 0,
  Tables.tables(n, 0, 512, Array.new(U64, d, seed)))
```

Caller conditions are `d == 17`, `n == 128`, `extra == 64`, and query square below64.
Seed, occupancy and Board fields are arbitrary. Side/color is Boolean (white or black),
not a claim about arbitrary invalid raw side encodings. The singleton-check law also
requires the actual selected-color king plane to equal exactly the queried bit.
No caller supplies desired masks, correct stored values, five query certificates,
or the desired final Boolean. `Masks` derives read results from the previously proved
actual initialization, and `Composition` supplies them to the internal `Wire` lemma.

## Independent answer and limits

`Spec.mask` uses prior independent bounded-coordinate leaper masks and first-blocker-
inclusive coordinate paths. `Spec.attacked` combines those masks using the independent
per-square piece/color witness scan. The answer does not call production mask builders,
`Chess.attack`, `Chess.attacked`, or `Chess.in_check`. Actual getters and occupancy
select Board bits; overlapping or otherwise inconsistent Boards are not ruled out.

This is target-centred geometric attack semantics. A general theorem identifying it
with an attacker-origin forward-ray specification is not added here. Native tests use
that separate forward-coordinate oracle, but their samples do not prove the universal
forward/reverse correspondence. Singleton preservation through castling, historical
rights, metadata legality, reachable legal positions and full generator soundness/
completeness remain separate. No direct normalization proof of the enormous literal
zero-argument `Tables.build()` term is added; native tests do execute the real builder.
Source array equality is not a native pointer identity, allocation or lifetime proof.

## Reproduction

Compiler is the unchanged standalone pin; `verify_compiler.js` checks its full input
fingerprint. Use Bun1.4.2 and the recorded compiler checkout.

```sh
bun native/bend_engine/standalone/proofs/initialized_attacks/focused.js /path/to/bend --report focused.json
BUN=bun CC=clang python3 native/bend_engine/standalone/proofs/initialized_attacks/verify_native.py /path/to/bend --report native.json
```

The complete focused command executes all three public laws and their importing
consumer, positive checks of the small dispatch/composition/domain modules, and all
19 controls. `--controls-only --report controls.json` is available for separately
paired qualification and explicitly reports that the consumer/full gate was not run.
Safe success requires exit0 and exactly `All terms check.`. Crashes, parser/ownership
errors, missing imports and timeouts are never accepted semantic proof rejections.

Ten semantic/refinement controls exercise new rook/bishop/queen dispatch, reverse-
pawn/diagonal-mask composition, complete-array transport and scalar bounds, plus
explicitly reused attack-witness implementation bridges for color/kind/check-side
routing. Eight checks enforce manifests/imports/regular sources; one synthetic
warning-output check is not a compiler execution. A controls-only result must be
paired with a successful exact-source public consumer, not called a complete gate.

The native driver reuses the accepted, unchanged independent forward-coordinate/set
reference, adds the queen mask, and executes three full-initialization contexts:
actual public zero-seed build and two nonzero-seed full tables/extras builds. Each
mode processes14,090 underlying Board/query requests in each context (42,270 context-
qualified distinct inputs), comparing six masks plus attacked, selected king and
bounded check observations (15 U32 fields per row). Nine malformed requests are
rejected. A missing king skips out-of-domain in_check in the probe and prints sentinel2;
that sentinel is not a production return value. Multiple kings test raw lowest-bit
selection, not a legal-position theorem. Modes repeat fixtures, not disjoint datasets.

Four actual implementation corruptions (pawn direction, queen diagonal, king slot,
blocker observation) must compile and execute before numerical mismatches count as
successful native detection. The candidate never imports the proof model or receives
expected answers. No full-buffer native inspection, GPU, search, training, strength,
benchmark, or perft expansion is included.
