# Actual parsed/validated singleton full-filter application

This increment stacks on PR1067 exact head
`aea04a41e15194b167936bba259cdee5c0921bff`, tree
`70c67c6cb8384041e91ad9d479676929abc4cf23`. Prior qualified source and sealed
evidence remain unchanged.

`Turn.fen` follows the exact `Position.fen` Maybe binds, including side, rights,
EP and all halfmove/fullmove numeric guards. Every successful output has
`get_turn(board)==Bool.to_u32(U32.is_eq(get_turn(board),1))`. No board, king,
table, accepted-root or caller turn certificate is needed for this parser result.
Rejection carries no board claim.

`Validate.known` follows actual `Protocol.validate`, `valid_board` and
`king_checked`. It threads the actual check pair and the existing actual
table-preservation certificate. Every result retains the exact input table;
every accepted board retains the parser's row and turn certificates plus the
computed `Position.valid` certificate. It preserves both validation rejection
branches and asks for no caller acceptance or king-presence certificate.

`Apply` derives original-side king presence from that computed validity flag,
then invokes PR1067's actual initialized fast/full consumer. Its full path executes
the exact actual prefilter candidate producer, including both castle producers,
and fully filters the resulting pair. Equality of complete ordered result pairs
passes through actual `Protocol.move_found`, `Position.find_move` and
`Position.apply`. The exact text and game are retained, including the full selected
Ply, history, counters and board update; none are recomputed by a proof-only decoder.

`consumer.use_fen_step` relates actual `Protocol.apply_one` on the exact validator
result to that full path. `consumer.use_fen_singleton` connects the same result to
actual `Protocol.moves(Con{text,Nil{}},Protocol.validate(...))`. Both consume one
affine input array through exact structural reification. They require only:

* Exact `a==Init.run(d,seed,n,extra)` with depth17, 128 slider tables and64 extra
  squares. No concrete `Tables.build` conversion is expanded.
* `Board.valid` for a successful actual `Position.fen` output, expressed as a
  conditional result predicate; a failed parse carries `Unit`.

The caller supplies no white Boolean, canonical-turn, king-presence, initial-check,
acceptance, child-validity, lookup, EP-coherence or target-geometry certificate.
This removes two explicit caller certificates by connecting their actual producers;
it does not derive row partition from root validation. The raw validator can
accept noncanonical or overlapping boards, as the unchanged prior counterexamples
show. A structural row-partition producer for the layout parser remains open.

`Fixtures` invokes the actual public singleton consumer with arbitrary FEN fields
and text. Its separate first-match witnesses retain an identical arbitrary move
tail, repeated queen-promotion records and colliding EP/ordinary records with the
same text. `Position.move_text` omits flag; `find_move` still selects the first
complete Ply. These witnesses claim no universal generation of a fixed record.
The full path preserves candidate order and multiplicity; it performs no set or
text-key deduplication.

This is implementation equivalence for one command after the actual validation
boundary. It does not prove arbitrary command-tail replay, child invariant
preservation, reachability/history validity, move-generation completeness, all
king/forward safety, default builder conversion or full chess legal-move correctness.

Qualification uses Bend2.0.21+U64
`aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae`, Bun1.4.2 and the exact84-file
fingerprint `d9550e30dbf17f12013aa5db89b27cf6724957fe68213c9c7e409e99f1405ef4`.
Checks are serial, with86400 seconds per check, CPUs1/3, two worker threads,
6GiB AS/RSS and16MiB per output file. A strict control receives credit only for
one intended typed mismatch with different expected/observed types; parser,
import, inference, ownership, crash, timeout and resource failures receive none.
Frozen source, compiler, snapshots, mutations, raw commands/logs/timing and Git
base blobs are hashed and checked. Completion and independent review are recorded
externally; an exploratory positive or active checker is not a completed gate.

```sh
taskset -c 1,3 python3 -B -m \
  native.bend_engine.standalone.proofs.fen_validated_full.qualify_fen_validated_full \
  "$BEND_CHECKER" --checker-manifest "$CHECKER_MANIFEST" \
  --report "$EVIDENCE_ROOT/qualification.json" \
  --evidence-dir "$EVIDENCE_ROOT/qualification-logs"
```

Evidence belongs outside all checkouts, beneath
`~/chess-artifacts/deepfin-fen-validated-full-20261007/evidence/2026-10-07`.
