# All four castling-right validity checks after actual generated moves

`all_consumer.preserves_all_rights` imports the shared proof and returns all four
actual child predicates after `Chess.make_move(b,m)`:
`(1,7,4,1)`, `(2,0,4,1)`, `(4,63,60,0)` and `(8,56,60,0)`.
Its parent premises are exactly the original public interface: parent row
partition, canonical turn, coherent en passant, all four parent valid-right
checks, and complete `Chess.Ply` membership in the actual `Chess.legal_moves`
output for the supplied affine table. No zero-right, initialized-table,
child-freshness or child-conclusion premise is added, and no axiom is introduced.
Canonical turn is retained although this proof does not need it.

The original checked white-kingside suite, consumer, fixtures and qualifier remain
unchanged; the original README points to this extension. The extension adds a four-constructor `Right` descriptor and one shared
ordinary proof. Rights clearing excludes both corner endpoints and moving the
owned home king; generated target clearance excludes capturing the home king.
A structural Word singleton proof observes each requested home square through
existing checked U64 bridges. Coherent EP restricts destinations to ranks2 or5:
only those16 squares need concrete six-home victim calculations, and the other48
close by rank contradiction. XOR8 home preimages have rank1 or rank6. Pointwise
update and row-decoder lemmas are reused.
Shared castling logic proves own king moves clear both own bits (3 or12), and
opponent castles preserve the requested two home rows for both castle wings.

`AllProof` consumes actual membership once and combines generation and EP-tag
certificates on the same reified table and exact actual list. It destructures
equality certificates before sharing those leaves; it never duplicates the affine
Array, full membership evidence or an unexamined affine aggregate. All ordinary
moves, promotion selectors1–4, flag1 EP occurrences and both flag2 castles remain
in scope. This does not prove full child `Position.valid`, nonzero-right king-count
preservation, replay preservation, reachability or legal-generation completeness.

`AsymmetricFixtures` independently checks closed actual singleton legal-list
witnesses for the other three home corners: black pawn16→8 removes white rook0,
white pawn47→55 removes black rook63, and white pawn40→48 removes black rook56.
Each parent has valid rows, canonical turn and all four valid-right checks; its
metadata EP target is incoherent. The actual flag1 child removes the corresponding
home rook while its right bit survives and its child predicate is false. Together
with the unchanged earlier rook7 witness these cover all four asymmetric corners.
Concrete rights15 fixtures also check all child predicates after both colors'
castles and after captures between corresponding home-rook corners.

`qualify_all.py` extends the existing fail-closed qualification harness. It checks
the old importing consumer, new all-four consumer, old fixtures, old closed EP
witness and new asymmetric fixtures. It retains the original five strict controls
and adds black-queenside home mapping, black own-mask, rank6 exclusion, shared
parent rows, and black-queenside actual EP-flag controls. A descriptor mutation
must fail at its checked downstream identity, not merely where it was edited.
Parser, affine-use, kind, timeout, resource and unrelated-location errors receive
no semantic rejection credit. Optional reuse applies only to an independently
passing old consumer with all204 identical Bend inputs and exact raw execution
provenance; it does not reuse an old overall qualification verdict.

Use the existing admitted serial CPU29/30, nice19, 6GiB sampled aggregate RSS,
600s/check and16MiB stream envelope, exact Bun1.4.2 and the qualified compiler pin
`aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae`. Run from a clean, frozen isolated
worktree with external evidence as described in the original README, replacing
`qualify.py` with `qualify_all.py`. A different environment requires newly
coordinated admission. Independent nonauthor source and raw-receipt review is
required before later integration, which remains separately owned.

## CI and later integration

Publication retains `[skip ci]` in the head commit. [GitHub documentation](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/skip-workflow-runs) recognizes that marker
for workflows triggered by push or pull_request; it does not suppress an explicit
workflow_dispatch or pull_request_target trigger. The present proof path matches
`bend-u64-map.yml`, whose complete job invokes benchmark drivers and uploads
artifacts, and other complete jobs upload artifacts too. No workflow is dispatched
under the current restrictions. Skipped required checks can remain pending, so
removing the marker blindly is not an integration plan.

The integration owner can run these existing host-only composition CI commands
separately in an isolated environment with dependencies already available:

```sh
python -B -m unittest native.bend_engine.standalone.proofs.generator_contract.test_qualify_castles
python -B -O -m unittest native.bend_engine.standalone.proofs.generator_contract.test_qualify_castles
python -B -OO -m unittest native.bend_engine.standalone.proofs.generator_contract.test_qualify_castles
bun test native/bend_engine/standalone/verify_compiler.test.js
```

They test host classification and synthetic compiler fingerprints and supply no
theorem-checking credit. The current suite's source-only qualifier is also suitable
for a separately admitted CPU-only environment. Static checks from the workflows
can be selected separately after dependency/resource coordination. Complete native
map workflows, benchmark steps and upload steps need separate authorization;
this patch changes no workflow or security setting. Before integration, verify the
stacked base and published source tree, obtain independent review for any later
changes, and arrange required non-benchmark check statuses without launching the
excluded jobs.
