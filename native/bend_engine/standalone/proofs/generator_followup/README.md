# A geometric check after complete move generation

`Runtime.run` calls actual `Chess.legal_moves` and passes its returned array to
actual `Chess.in_check` on a separately supplied query Board. It returns the array,
complete generated move list, and check Boolean. It does not reinitialize between
the operations. This is a proof/test adapter, not a production API change.

Two public contracts:

- `generation_then_check_matches_direct_check`: for arbitrary source arrays, Boards
  and raw side, the entire adapter result equals the original generated list joined
  with a direct check on the original array.
- `initialized_generation_then_check_matches_geometry`: for depth17/128 blocks/64
  extras and arbitrary seed, the following check equals the independent geometric
  witness and the final array equals the initialized array. Only the query Board
  needs the selected singleton at a square below64 and a Boolean selected side.
  The generation Board is independent and unrestricted in the source theorem.

`Transport` obtains the table certificate from the existing public preservation law;
`PROOF` supplies the initialized check certificate from the established geometry law.
Neither correct query answers nor returned-state equality are public premises.

This is not correctness of the generated move set, geometry of internal generator
checks, full castling acceptance, native safety for malformed arrays/Boards, pointer
identity, or native lifetime. The query need not concern a generated child. It is an
explicit subsequent query, not a newly inserted check in production `legal_moves`.

## Verification

```sh
BUN=/path/to/bun python3 native/bend_engine/standalone/proofs/generator_followup/focused.py /path/to/pinned/bend --report /tmp/followup-source.json
BUN=/path/to/bun CC=clang python3 native/bend_engine/standalone/proofs/generator_followup/verify_native.py /path/to/pinned/bend --report /tmp/followup-native.json
```

The source runner checks the full importing consumer and15 controls: six semantic
adapter/certificate failures, eight manifest/import controls, one synthetic output
wrapper unit. Controls-only mode explicitly leaves the consumer NOT_RUN.

Native tests have184 distinct generation/query inputs under two initialization
contexts, repeated in generic,portable,native-target andUBSan modes. Each request
first executes a direct baseline generation, then executes the adapter (a second
real generation and the subsequent query). The following query's Boolean has an
independent forward-coordinate oracle. Complete move-list equality to baseline is
differential, not an independent move-set oracle. Each batch threads returned state
and checks an unused-slot marker. The query Board's raw turn is deliberately opposite
the explicitly selected side. Four adapter mutations compile/run before rejection.
The new probe does not claim full logical-buffer or lifetime verification; the parent
preservation suite's separate full-buffer evidence remains unchanged.

No formal/native pass is implied by this README; completed receipts are recorded in
the continuation readout. Compiler pin and prior sources remain unchanged.
