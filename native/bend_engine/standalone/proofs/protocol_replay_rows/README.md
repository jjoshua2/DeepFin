# Actual protocol replay preserves row partition and exact input storage

Stacked on exact PR1069 head `142697dd3fe5e7bda1cd2419879c6fc4802c1c6c`.
This suite follows the next actual-source obligation after parsed FEN rows.
It adds proofs only; the production engine and prior suites are unchanged.

`Find.all` transports a property of every candidate occurrence through actual
`Position.find_move` to its exact first matching full `Ply`. Text omits the flag;
the proof neither reconstructs a move from text nor assumes unique candidates.
`Apply.legal` uses the actual `Generated.legal` continuation and table-preservation
provider for the same `Chess.legal_moves` result. Generated full-Ply preservation
then reaches actual `Position.apply`, including its unmodified board update,
counters and history. Failed lookup follows actual `Protocol.move_apply`.

`Replay.moves` inducts over an arbitrary `List<&2,String>` through actual
`Protocol.moves`. Each current returned array is transported to the complete
`Observe.Cells` image before the next `apply_one`. Equality covers every leaf and
the tree structure. Commands, including repeated commands, retain their original
order and multiplicity. `None` remains covered, including initial `None` and
failure after a successful command. The generic consumer requires only conditional
input `Board.valid` rows and works for arbitrary affine arrays and raw Games.

`use_fen_replay` derives rows internally from PR1069 `Rows.fen`, runs actual
`Protocol.validate`, and proves conditional rows for the final actual replay
result. Its result carries both exact original-table equality and the existing
initialization equation with literal `d == 17n`, `n == 128n`, `extra == 64n`.
These four explicit assumptions remain load-bearing. No caller row, turn, king,
target geometry or chess-board validity assumptions are added. Each successful
prefix also falls under the arbitrary-list theorem.

This proves representation row partition and exact storage preservation. It does
not prove replay acceptance, child `Position.valid`, EP coherence, king presence,
legal-move correctness, or arbitrary per-command fast/full equivalence. The latter
still needs stronger preserved premises. The concrete zero-leaf fixtures exercise
actual lookup/apply/failure and do not certify initialized chess tables. Closed
`Init.run` normalization is deliberately absent.

Qualification uses pinned Bend `2.0.21+U64` commit
`aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae`, its verified 84-file fingerprint
`d9550e30dbf17f12013aa5db89b27cf6724957fe68213c9c7e409e99f1405ef4`, and Bun 1.4.2.
The qualifier freezes complete entry closures and all Python support, checks exact
base blobs, and guards source/runtime/Git identities before and after every check.
Every positive and strict negative runs serially on CPUs 1 and 3, two threads,
6 GiB address-space/RSS cap, 16 MiB output cap, and 86400-second wall limit.
Only exact `All terms check.\n` with zero exit is positive; strict controls require
one unequal expected/observed type error at the intended declaration, without
timeout, parser/import/resource/inference errors. Prior controls remain unchanged.

From `native/bend_engine` in this isolated checkout:

```sh
PYTHONDONTWRITEBYTECODE=1 BUN=/home/josh/.bun/bin/bun taskset -c 1,3 \
  python3 -m standalone.proofs.protocol_replay_rows.qualify_protocol_replay_rows \
  /tmp/deepfin-king-away-checker-aaeb9bc \
  --checker-manifest /absolute/new-evidence/checker-tree.json \
  --report /absolute/new-evidence/qualification.json \
  --evidence-dir /absolute/new-evidence/fresh-checks
```

Reports, raw streams, `/usr/bin/time -v` records, exact source snapshots and
mutation copies live outside all checkouts. A finished checked candidate must be
independently reviewed before publishing a stacked draft; there is no merge gate
or stronger theorem claim in a raw successful checker result alone.
