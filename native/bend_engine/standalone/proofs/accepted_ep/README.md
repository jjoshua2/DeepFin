# Accepted EP check observation

For an arbitrary affine array `a`, Board `b` and entire Ply `m`, exact
membership in actual `Chess.legal_moves(a,b)` and `flag(m)=1` imply:

```text
Chess.in_check(a, Chess.make_move(b,m), Chess.get_turn(b)) == (a, False)
```

The original moving side is queried, despite the child turn flip. The complete
input array is returned unchanged. No Board validity, EP metadata coherence,
promotion, source/destination range, king cardinality, canonical turn or table
geometry premise is added. The prepared-filter contract applies to arbitrary
candidate lists, including duplicates and arbitrary full-Ply promotion fields.

Accept extracts the check decision from both production filter branches using
the existing generic full/fast acceptance folds. Flag1 forces the full check for
every sensitive mask. Filter.prepare and Generated.prepared_equal connect the
proof to actual state-threading calls. Public reifies the actual affine array
once and transports its complete value identity; consumer invokes these laws.

Both EP directions instantiate the generated law with actual membership. A
discovered-rook fixture proves the predicate returns False while the actual
geometric rook gate contains square39. Arbitrary-array predicate acceptance
therefore does not establish physical attack geometry or full legal-move safety.
Earlier geometric removal and board assumptions retain their original scope.

Filtering starts with an empty accumulator in prepare/legal_moves. TailWitnesses
keeps duplicate candidate multiplicity, exact table and full-Ply distinctions.
It also shows that an unchecked raw accumulator tail is preserved even when its
EP members return True from the check. The inherited fold invariants require a
safe accumulator; no arbitrary-tail acceptance claim is made.

Qualification uses the pinned Bend2.0.21+U64 checker and Bun1.4.2, the complete
loaded closures and exact-base dependency Git blobs, two CPUs, 6GiB memory,
86400-second checks and 16MiB output caps. Strict negatives require one typed
expected/observed mismatch at the named obligation; crashes, inference failures
and resource limits are failures.

```sh
BUN=/path/to/bun python3 -B -m native.bend_engine.standalone.proofs.accepted_ep.qualify_accepted_ep /path/to/pinned/bend --checker-manifest /path/to/checker-tree.json --report /tmp/accepted-ep.json --evidence-dir /tmp/accepted-ep-evidence
```
