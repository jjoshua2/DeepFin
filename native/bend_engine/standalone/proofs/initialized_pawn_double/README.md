# Initialized pawn double-advance producer

`consumer.initialized_double` proves that actual `Chess.pawn_targets` membership
plus `src xor dst == 16` implies the exact guarded double advance:
`empty2 == True` and `dst == second`. `empty2` is the implementation's conjunction
of first-square range/emptiness, source start rank, and second-square emptiness.

The table is exactly `Initialized.run(d,seed,n,key,at,extra)`, with `d=17` and
`extra=64`. Seed and the prior `Tables.tables` loop parameters are arbitrary.
`Initialized.query` supplies the actual queried pawn row and preserves that array.
`Producer.targets` lifts this equality through the real `Chess.pawn_after` pair.
No concrete P2 table is normalized. The only source premise is Nat(`src`) < 64;
XOR cancellation reconstructs `dst=src xor 16`, so no destination bound is assumed.
All other board fields and U32 turn values are arbitrary. Production chooses white
only when turn equals 1 and black otherwise; the fixture with turn 2 checks this.

The structural bit algebra isolates the second selected advance. Two sets of 64
source cases prove that neither initialized pawn diagonals nor the first step can
set the XOR 16 destination; another bounded adapter establishes the U32 roundtrip
and identifies the second destination from the recovered start-rank guard.
The false out-of-range branch is discharged from the explicit source bound.

This is a producer lemma. It assumes membership in the actual target mask, not
membership in `legal_moves` or arbitrary tails. The occurrence-aware bridge through
filtering and scan/destinations remains open. No king safety, child EP coherence,
legal completeness or full reachability theorem is claimed.

Accepted `Protocol.validate` roots and `Board.valid` are incomparable. Root
validation supplies king facts only; it supplies neither partition, canonical
turn nor initialized geometry. The prior PR1042 collision and turn 2 witnesses
remain unchanged. The two counterexamples remain distinct and are preserved here:
stale raw EP 16 tags the actual advance 8 -> 16 with flag 1 and erases the enemy king 24,
while the prior root validator rejects that stale position; an arbitrary leaf
containing bit 48 can propose pawn 32 -> 48 and regenerate EP 40. Initialized pawn
geometry excludes that capture, as the separate target-mask fixture checks.

Qualify the full consumer/fixtures closure and typed negative controls using the
pinned Bend 2.0.21+U64 compiler and its 84-file manifest:

```sh
PYTHONDONTWRITEBYTECODE=1 BUN=/path/to/bun python -m native.bend_engine.standalone.proofs.initialized_pawn_double.qualify_initialized_pawn_double /path/to/pinned-bend --checker-manifest /path/to/checker-tree.json --report /outside/checkout/Q001.json --evidence-dir /outside/checkout/Q001
```

Each checker invocation defaults to 86400 seconds, two allowed CPUs, 6 GiB address
space/RSS ceiling, and 16 MiB per output file. Logs, commands, source snapshots,
hashes and full copied mutant closures are stored outside the checkout.
