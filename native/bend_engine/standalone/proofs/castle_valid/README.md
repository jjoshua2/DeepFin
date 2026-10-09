# Generated-child castling validity: white kingside increment

`consumer.preserves_white_kingside` proves the actual child predicate
`Position.valid_right(Chess.make_move(b,m),1,7,4,1) == True` for a complete
`Chess.Ply` member of the actual `Chess.legal_moves(table,b)` output.

The public interface takes parent row partition (`board/Spec.valid`), canonical
turn, coherent parent en passant, and all four parent `valid_right` checks:
`(1,7,4,1)`, `(2,0,4,1)`, `(4,63,60,0)`, `(8,56,60,0)`. There is no zero-right
assumption, initialized-table premise, child row or home-piece premise, or assumed
child conclusion. The proof actually uses only the white kingside parent check;
canonical turn and the other three parent checks can be omitted for this increment.

The other three child checks remain unproved. This is not a theorem of full child
`Position.valid`, preservation of king counts with nonzero rights, legal-move
completeness, replay preservation, or reachability from a conventional root.

`Emission` imports the existing ordinary scan certificate and composes both actual
castle producers and both actual final filter paths. Full promotion and flag
fields, duplicate list entries, arbitrary table shape, ordinary moves, promotion
selectors 1–4, flag1 captures, and flag2 castles remain in scope. `storage.reify`
consumes the affine table once; equality transport permits independent generation
and EP-tag certificates on exact constructor images of that same table.

If the child right is clear, its validity is immediate. If it survives, rights
decrease forces the parent right to have been set, and parent validity supplies
the white rook7 and king4. Actual rights clearing excludes a source or destination
at rook7 and excludes moving the owned king4. The generated destination cannot
capture king4. The actual flag1 tag plus coherent EP puts the XOR-8 victim away
from both home squares. Pointwise bitboard lemmas then preserve all eight home
row observations through actual `make_move`, including the decoder and color.
White castles clear both own rights via the actual king rule; black castles leave
both white home rows unchanged. No axioms are added.

`Fixtures` records nonzero rights15 surviving quiet and promotion moves, complete
producer fields, and both sides' castle effects. A row-valid, canonical-turn board
with all four parent rights valid and malformed EP target15 retags black capture
22→15 as flag1: actual `make_move` removes rook7 while right1 survives and the
child check is false. The existing malformed-EP king-loss suite remains unchanged;
its exact board and conclusions are reproduced locally because importing the
complete old fixture module triggers a qualified `tail` binder parser collision.
`GeneratedNegative` strengthens the missing-EP boundary with
a closed actual legal-list witness: white king4/rook7, black king60/pawn23,
turn0, rights1, EP15, and the actual empty `Array.new` table generate exactly
`Ply{23,15,0,1}`. Parent rows, turn and all four rights are valid; EP coherence
alone fails, and the actual generated child loses rook7 while right1 survives.
The separate producer fixtures do not assume full legal membership for a concrete
table; the public theorem independently quantifies actual legal membership.

The suite is stacked on PR1081 exact head
`0ec611bad5b79682936e7dc5e36dd2498a227b67`, tree
`cf8fdc1681f49a3510153ffa5eb0acccd58e573c`, whose parent is
`91e4eadded397ffe388379e2566e06ceb03145d8`. It changes only this new proof suite.
Engine sources and existing proof suites are reused unchanged.

Qualification requires a clean committed isolated worktree, Bun1.4.2, the
qualified Bend2.0.21+U64 compiler pin
`aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae`, and the independently reviewed v2
unlimited-VA sampled-RSS watchdog with SHA256
`bdafed929dd1d6ed5a4cf72148340423da1ca7bd2513ce78e45ee8beab57b279`.
Obtain physical CPU/RSS admission from the current resource owner before running.

```sh
python3 -B native/bend_engine/standalone/proofs/castle_valid/qualify.py \
  --compiler /path/to/qualified-compiler \
  --bun /path/to/bun \
  --watchdog /path/to/bend_sampled_rss_watchdog_v2_unlimited_va.py \
  --admission /path/to/resource-admission.json \
  --cpus 29,30 --wall-seconds 600 \
  --evidence-dir /external/fresh/qualification-directory
```

The runner validates admission status, exact worktree/base/compiler identity,
authorized CPUs, serial count, nice value, sampled-RSS policy, wall and stream
limits before launching any checker. An unrelated admission or different CPU
selection is rejected. The current admission is specific to this worktree and
CPU29/30; a new environment needs a newly coordinated admission and a reviewed
schema update rather than silently reusing this approval.

The runner serializes three positive checks and five intended-obligation mutants,
uses nice19, two CPUs, 6GiB sampled aggregate RSS, 600s per check, and 16MiB per
stream. The RSS threshold is sampled, not a hard ceiling. It verifies all 84 pinned
compiler files and every reused Git blob; before/after guards enforce unchanged
source, snapshot, compiler, admission, watchdog, and frozen Git identity. Parser,
inference, affine-use, timeout, resource, or unrelated-location failures earn no
negative-control credit. Exact mutant sources, raw streams, rusage, commands,
source/runtime manifests, and process-cleanup receipts remain external.

After a change outside the consumer's Bend import closure, optional
`--reuse-consumer-report /external/prior/qualification.json` can reuse an actual
passing consumer receipt. It verifies the original raw streams, process cleanup,
resource admission, runtime fingerprint, before/after guards, old Git identity,
and every old snapshot/hash/Git blob against the entire current consumer closure.
A failed overall prior gate supplies no general qualification credit; only its
individually passing, identical consumer check can be reused, and the report
labels the reuse explicitly. Fixtures and all mutants are always checked freshly.

Independent nonauthor review of the immutable source and actual receipts is
required before integration. Integration remains separately owned; this suite
does not modify live training or evaluation.
