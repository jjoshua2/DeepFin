# Initialized actual replay bootstraps and preserves EP coherence

Stacked on exact PR1071 head `25054541f6f3ff37037216574cb3562da0083e26`.
Production source and all prior proof/control files remain exact base blobs.

The missing obligation was the replay consumer of the existing
`child_ep_coherence.Coherence.total` theorem. This suite reuses that checked
producer rather than repeating its double-step arithmetic or board-plane proofs.
`Build.run(d,seed,n,extra)` is definitionally the same actual initialized array as
`Initialized.run(d,seed,n,0,512,extra)`; no closed builder normalization is used.

`Members.self` and `extend` construct full-Ply membership certificates for every
occurrence of the exact list, preserving duplicates. `ep_all` applies the existing
EP theorem to those occurrences. `Members.legal` transports the exact queried
array through its initialization equation. `Find.all` then retains the certificate
for the actual first text match, including its complete promotion and flag fields.
`Apply.moved` connects that exact selected child to actual `Position.apply`.
The metadata and coherence proofs join on the same actual returned pair.

The generic public consumer follows nonempty
`Protocol.moves(Con{text,tail},(table,maybe))` with arbitrary text and arbitrary
ordered, duplicate-containing tail. Its inputs are conditional parent rows and
canonical turn, the exact initialization equation, and literal depth `17`, table
count `128`, extra count `64`. It supplies no parent EP bound or coherence premise.
The first actual application bootstraps child EP coherence; each subsequent step
reuses exact complete-table identity and canonical turn. `None` and failed lookup
remain covered. The output retains all four initialization assumptions, the exact
original array, conditional row partition, canonical turn, EP range, and actual
`Position.valid_ep(get_ep(board),board) == True`.

The FEN consumer derives parent rows and canonical turn internally from the actual
parser/validator providers. The raw EP65 fixture demonstrates the absence of a
parent EP premise. Shared-text, duplicate full-field fixtures exercise actual first
selection and application. Empty replay retains invalid raw EP; the nonempty shape
is load-bearing. The inherited arbitrary pawn-jump witness is a raw update witness,
not a claim of initialized legal-list membership or an actual replay counterexample.

This closes the initialized replay EP-coherence consumer gap. It does not prove
child `Position.valid`, king survival, structural piece counts, castling-right
validity, replay acceptance, full legal-move correctness or fast/full equivalence.
The prior arbitrary-array metadata theorem remains unchanged; it gains no
EP-coherence claim. All earlier controls remain exact base blobs.

Pinned checker: Bend `2.0.21+U64`, commit
`aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae`, Bun 1.4.2, verified 84-file fingerprint
`d9550e30dbf17f12013aa5db89b27cf6724957fe68213c9c7e409e99f1405ef4`.
Checks run serially on CPUs 1 and 3, two threads, 6 GiB address-space/RSS cap,
16 MiB output cap and 86400 seconds per check. Exact source/runtime/Git guards,
fresh raw streams, timing, commands, snapshots and single-declaration mutants are
required. Parser, inference, resource and timeout failures do not count as controls.

From `native/bend_engine` in the isolated checkout:

```sh
PYTHONDONTWRITEBYTECODE=1 BUN="$HOME/.bun/bin/bun" taskset -c 1,3 \
  python3 -m standalone.proofs.protocol_replay_ep_coherence.qualify_protocol_replay_ep \
  /tmp/deepfin-king-away-checker-aaeb9bc \
  --checker-manifest /absolute/new-evidence/checker-tree.json \
  --report /absolute/new-evidence/qualification.json \
  --evidence-dir /absolute/new-evidence/fresh-checks
```

PR1070's CI repair changed only the README reproduction command from a literal
home path to quoted `$HOME`. Theorems, executable Bend/Python, checker and lint/test
tooling were unchanged; its existing path guard and all exact-head CI passed.
Sealed prior evidence is retained unchanged.
