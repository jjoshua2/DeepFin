# Pawn double advance from an actual legal-list occurrence

`consumer.legal_member` starts with membership of the complete `Chess.Ply`
in the actual `Chess.legal_moves(Initialized.run(...), board)` result. It
derives `Nat(src) < 64` and the actual pawn-target mask hit from that same
occurrence. When the exact `make_move` EP trigger is true, PR1044 then proves
the implementation's `empty2` guard and `dst == second`; `Bridge.tag` proves
that the actual child has EP `(src + dst) / 2`. The caller supplies no separate
source bound, destination bound, target membership, or target geometry.

The initialized-array premises remain explicit: depth 17 and all 64 extra
rows, with arbitrary seed and earlier table-loop parameters. The theorem
uses the exact `Initialized.run` array. It does not cover arbitrary writes to
that array. The inherited child metadata contract additionally requires
`Board.valid` (piece/color partition) and canonical turn. Those premises are
separate from root validation, which proves king facts but does not imply
the partition or canonical-turn contract.

The bridge is structural. `Emission` uses the checked exact destination
factorization, the all-set-bit result, and the actual fuel 64/zero-mask flag.
It preserves every certified incoming tail node, including duplicates.
`Pipeline` carries equality to the input array through the ordinary source
queries and scan. It then transports the per-occurrence source/target
predicate through castle generation and the full/fast filters. This theorem
does not return final table equality; that result belongs to the separate
existing table-preservation proof. Its generic legal-member
theorem accepts arbitrary arrays and boards; it conditionally supplies a
pawn-target mask hit and a source bound. Castles cannot trigger XOR 16.
Membership and the certificate copied for the two consumers use all four
Ply fields and the original occurrence path.

The fixtures check actual multi-bit promotion emission (promotions 1 through
4, flag 0), the ordinary raw EP flag, exact array preservation, and duplicate
tail counts. They retain the arbitrary-array jump `32 -> 48`: it is a legal
occurrence and regenerates EP 40 despite a false double-advance guard and
unavailable child EP target. They also retain the stale EP 16 advance that
erases the enemy king. This increment proves neither child EP coherence,
king preservation, root reachability, nor full legal-move correctness.

The portable qualifier pins the exact PR1044 base and unchanged dependency
blobs. It checks every term in the fixture/consumer closure, then reconstructs
typed negative controls for source and target premises, original-array
transport, tail multiplicity, full-Ply consumer linkage, initialized depth,
and the child EP tag. Concrete false witnesses test duplicates, promotions,
raw EP flags, and the arbitrary-array counterexample. Rejection requires a
typed error at the expected obligation; parse, affine-consumption, resource,
and timeout failures do not qualify.

Use the verified Bend 2.0.21+U64 checker at commit
`aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae` with Bun 1.4.2 and the verified
84-file manifest (fingerprint
`d9550e30dbf17f12013aa5db89b27cf6724957fe68213c9c7e409e99f1405ef4`):

```sh
BUN=/path/to/bun python3 -B -m native.bend_engine.standalone.proofs.pawn_ep_occurrence.qualify_pawn_ep_occurrence /path/to/pinned/checker --checker-manifest /path/to/checker-tree.json --report /path/to/fresh/report.json --evidence-dir /path/to/fresh/evidence
```

Positive and negative checks default to 86400 seconds, two CPUs, 6 GiB
address space, and 16 MiB per output file. Durable evidence stays outside the
checkout and records exact commands, snapshots, hashes, resources, and logs.
The proof changes no production runtime and uses no GPU.
