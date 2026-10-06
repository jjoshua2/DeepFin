# Child EP coherence from actual initialized occurrences

Stacked on PR1045 `ca44e381a1308802b17d7e1064552781b987185d`.

`Coherence.total` proves the actual child `Position.valid_ep(get_ep(child), child)`
for every full-Ply member of `Chess.legal_moves(Initialized.run(...), board)`.
Initialized depth 17 and extra 64, and canonical parent turn, remain explicit.
There is no caller source-bound, target-hit, double-guard or promotion-shape premise.
PR1045 derives source/target geometry from the exact initialized query.

`Source` strengthens the emitted promotion branch with the implementation's
back-rank guard. `Fields` composes that guard with the actual double-step
occurrence bridge to derive promotion 0 and flag 0 or 1. It excludes castles
using the actual source/destination XOR trigger. Full-Ply identity and every
certified tail node are preserved, including duplicate occurrences.

`Bits` and `Post` prove structural remove/insert facts for arbitrary board planes.
`Move` rewrites the actual flag-0/flag-1 child into those updates with the exact
XOR turn and regenerated EP field. `Files` certifies a start file from the 64
bounded source indices. `Geometry` checks sixteen arithmetic and single-bit
certificates. `Closed` consumes those certificates in one generic board update
proof: the source is vacated, the midpoint stays empty, and the destination
contains a pawn in the previous mover's color. `Coherence` transports
the double-step guard and exact turn into the child validator. EP 64 is immediate.

The public `consumer.legal_member` combines child partition, canonical turn,
EP bounds and EP coherence. Parent partition, canonical turn and parent EP
coherence remain explicit in that contract. The lower EP theorem needs only
canonical turn plus exact initialized full-Ply origin: even malformed parent
EP that emits flag 1 on a double step removes its already-empty midpoint.
This is recorded by a closed update witness; it is not a claim that the witness
was produced by an initialized legal list.

The old stale EP king-loss and arbitrary-table 32-to-48 jump remain checked.
New false witnesses reject promotion, castle flag and noncanonical turn on an
EP-setting double. Actual scan-after fixtures retain full fields and duplicate
counts. No king survival, accepted-root preservation, initialized geometry after
arbitrary array writes, final suffix array equality, or full legal-move correctness
is claimed. The exact input array is the one queried by the theorem.

Qualification uses pinned Bend 2.0.21+U64 `aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae`,
Bun 1.4.2 and the verified 84-file checker fingerprint. One checker runs at a
time, on CPUs 1 and 3, with a 6 GiB address/RSS ceiling, 16 MiB output-file cap
and 86400-second checker deadline. Parse, affine, resource and timeout failures
do not count as successful negative controls.

```sh
BUN="$HOME/.bun/bin/bun" python3 -B -m native.bend_engine.standalone.proofs.child_ep_coherence.qualify_child_ep_coherence /path/to/pinned-checker --checker-manifest /path/to/checker-tree.json --report /fresh/qualification.json --evidence-dir /fresh/evidence
```

Evidence lives outside the source tree and binds hashes, commands, exact HEAD
and base Git blobs, compiler identity, raw output and negative mutations.
Internal independent review precedes draft publication. No merge or runtime
source changes are part of this increment; the PR1043 test fix stays separate.
