# Complete generator contract and closed-builder normalization blocker

## Scope fixed before qualification

The [end-to-end target](../bend_legal_generator_contract.md) requires actual
Tables.build and optimized Chess.legal_moves to satisfy soundness, completeness,
canonical uniqueness, complete-array threading and successor representation under
independent orthodox-history/representation premises. It has three compositional
milestones. It does not count documentation, native parity or a shadow generator
as a completed source theorem. Training remains Python.

Draft PR [#979](https://github.com/jjoshua2/DeepFin/pull/979) began from main
ae2d095b66dcbf5cabbae96dc3e3e0629a5192fa. Initial candidate
f9ec0dae6e40681ce271ef6256a2add256a85609, tree 3f066d7bc82bfa9866da16bacd44665d6c6dbc70,
contains a zero-premise reflexivity candidate and importing consumer for:

```text
Tables.build() == Init.run(17n,U64.zero(),128n,64n) : Array<U64>
```

The qualified symbolic initialization result and separate public-wrapper token
guard remain unchanged. This is the missing direct closed equality; it is not
assumed from that guard.

## Hosted outcome: not completed

[Run 36807294025, job 110194374813](https://github.com/jjoshua2/DeepFin/actions/runs/36807294025/job/110194374813)
ran the unchanged pinned compiler aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae,
84 source inputs, fingerprint
d9550e30dbf17f12013aa5db89b27cf6724957fe68213c9c7e409e99f1405ef4.
The full importing consumer exceeded its 600-second wall limit after
600.271891773 seconds, with empty output and no normal exit code. The receipt
remains NOT_COMPLETED. **Zero laws were accepted; neither semantic control ran.**

The entire qualifier had a transient cgroup bound of 200% CPU quota, 6 GiB RAM,
no swap and 1,100 seconds. The two proposed controls each retain their separate
180-second bound. No resource bound became a theorem premise or was raised after
failure. This is not a counterexample or proof that the equality is false.

Compiler installation/fingerprint, all 12 original pin tests, three host result
classification tests in each of normal Python/-O/-OO, and the post-attempt
unchanged-source check passed. The final qualifier source/compiler recheck after
controls was not reached. The host tests validate classifications, not Bend terms.
Separate exact-head whole-repository lint passed. Other ordinary CI results do
not upgrade this failed proof attempt.

The [receipt](evidence/bend-generator-contract/attempt-36807294025.json) is
reconstructed from the original receipt printed in the GitHub job log, without
changing its semantic fields. [Provenance](evidence/bend-generator-contract/attempt-36807294025-provenance.json)
retains the candidate/tree, job, source identities and original artifact ZIP
SHA-256. The original artifact expires according to GitHub retention, so the
portable record is kept here. No compiler peak-memory claim is inferred from
systemd's final summary line.

## Review and successor changes

An independent source review verified that the candidate and consumer refer to
actual public construction, have no additional axioms/premises, and retain the
unchanged compiler pin/checker and strict result classification. It found that
the first workflow path filter omitted transitive input paths. The successor
adds the exact production, recipe, host-helper, pin and installer paths. It also
writes source/compiler identities to the NOT_COMPLETED receipt before starting
the expensive child, so a whole-cgroup interruption retains those identities.

The theorem candidate, consumer, controls, resource limits and production code
remain unchanged in that successor. This record is evidence for the initial
candidate only; later hosted runs must keep their own identities and results.
The draft remains blocked on the direct public-builder proof and cannot be
presented as a completed initialization or generator theorem.

## Interpretation and next research decision

The pinned conversion routine in bend2/bend.ts calls term_wnf on both equality
endpoints before comparing their structure. A fully closed initialized array
therefore exposes the concrete construction to normalization. Earlier symbolic
proofs deliberately retain depth/count parameters to avoid that expansion.
The observed timeout is consistent with this conversion boundary, but the
receipt alone does not isolate a particular reduction or establish impossibility.

Next investigate a proof architecture that composes the existing symbolic
theorems with the actual public call without requiring exhaustive eager closed
normalization. Keep checker, Base, axioms, runtime behavior and existing proof
premises unchanged. Any claimed route needs a small checked demonstrator plus
the full importing consumer; merely renaming the symbolic recipe or allowing a
source-token assertion is insufficient. There is no automatic retry with larger
resources and no compiler modification in this PR.

All repository work was through GitHub and GitHub Actions. No local executor,
training, GPU, model export, live state or deployment was touched. The complete
castling, ordinary/promotion/EP enumeration, fast-filter theorem and final
generator composition remain open.

## Final trigger-fix head: second bounded non-completion

[Run 36808544524, job 110198207115](https://github.com/jjoshua2/DeepFin/actions/runs/36808544524/job/110198207115)
qualifies the unchanged candidate on branch head
1e80548473589020e5004d7ba5b4b17d2569c6df, checked via GitHub's merge revision
8eb4e5b85be6ba5b9a6d709c835b95578b8a4638 into base
9369067851896bf3ddbb5d0ad084809b091b2e8f.

The positive consumer again timed out with no checker output, after
600.174135031 seconds. Its [receipt](evidence/bend-generator-contract/attempt-36808544524.json)
remains NOT_COMPLETED; zero accepted laws and zero executed semantic controls.
The compiler, limits, theorem and controls were unchanged. Source identities were
successfully recorded before the child. Original pin tests, all three host-test
optimization modes and the post-attempt unchanged-source check passed. The final
post-control source/compiler verification was not reached. All six other exact-head
ordinary/Bend regression workflows, including CI 36808544572, completed successfully;
those results do not turn this normalization timeout into a proof.

[Provenance](evidence/bend-generator-contract/attempt-36808544524-provenance.json)
binds the printed receipt and original artifact metadata. The retained artifact
ZIP was 1,420 bytes with SHA-256
af0fb5c5de3262054a5e2e1ef86dd7ed3fdfc4273d7f15c1d138a91dc0d3e44d.
No peak-memory inference is made from systemd's final accounting.

## Compositional continuation: both-wing exact projection

The new [castling candidate](../../native/bend_engine/standalone/proofs/generator_contract/CASTLING.md)
targets exact whole-array and tag-2 list equality for actual optimized legal_moves,
plus complete-Ply Boolean uniqueness. The ordinary scan and both real castling
producers stay in the implementation side; a projection theorem connects full
and optimized filtering after execution. Both wing guards are decided inside
the proof, and independent three-stage geometric checks select the two canonical
moves. No intermediate state or desired attack answer is assumed.

This candidate is not qualified at publication. Its consumer retains the existing
symbolic initialized recipe, valid-Board/Boolean-side/home-king singleton premises,
and inherited proof bodies. It does not close actual Tables.build, independently
derive historical rights or cover general king-away positions. No new accepted-law
count, native result or complete-generator result is claimed.

Before running, the qualifier is bounded at two CPU cores of quota, 6 GiB RAM,
no swap and 1,600 seconds; helper 180, complete consumer 600, five semantic controls
120 seconds each. Initial failure or any invalid control fails closed. These are
qualification bounds, not theorem premises. Existing runtime/compiler/proofs and
the closed-builder gate stay unchanged. No local tests or GPU are used.

## Castling full-consumer attempt and evidence-led budget allocation

[Run 36812198118](https://github.com/jjoshua2/DeepFin/actions/runs/36812198118)
on head 5665387303fdb1a0c024548f6564aa93240e8600 passed its generic actual
full/fast projection helper in 8.329775745999996 seconds. The complete importing
consumer timed out after 600.1790122010001 seconds with empty checker output.
The [receipt](evidence/bend-generator-contract/castling-attempt-36812198118.json)
and [provenance](evidence/bend-generator-contract/castling-attempt-36812198118-provenance.json)
retain the source/compiler identities. No controls ran and accepted candidate
credit remains zero. The earlier two attempts stopped promptly on quantity
errors; they are not semantic controls or theorem successes.

Independent review identified and verified fixes for both a duplicated linear
evidence use and the classifier's original ability to misclassify Data/Type
diagnostics. The host test now contains the exact original kind-error diagnostic
and quantity examples. All host tests in three modes and original pin tests
passed at this head; full qualification still did not complete.

The unchanged inherited castle_sequence consumer already has a hosted
641.93163341-second passing receipt. With explicit approval, the next attempt
therefore reallocates the same 1,600-second total budget: helper 180, consumer
900, five controls 90 each, with unchanged 2-CPU quota, 6-GiB RAM/no swap.
It does not extend the closed public-builder's 600-second bound or modify the
checker. A small exact production-wiring lemma, used by the composition and
included in its positive helper baseline, lets the duplicate-wing corruption
be checked without rerunning the large initialization proof graph.

## Complete castling source consumer accepted, control harness refinement

[Run 36813268646](https://github.com/jjoshua2/DeepFin/actions/runs/36813268646)
at 879a996056cb1e2b00839bd159012945aa43f1ec passed the full importing
exact-pair/uniqueness consumer in 579.028494238 seconds, with exact output
"All terms check." The helper including actual production wiring passed in
6.758946540000011 seconds. The [unchanged receipt](evidence/bend-generator-contract/castling-attempt-36813268646.json)
and [artifact provenance](evidence/bend-generator-contract/castling-attempt-36813268646-provenance.json)
retain those source/checker identities.

One intended semantic control passed. The second erased all projection heads
and was rejected in ordinary_cons before castle_retain, so its registered
location test correctly failed. Later controls were not executed and the full
gate stayed NOT_COMPLETED, with zero qualification credit. Whole-repository
lint and PEXT CI passed separately at this head.

The refined mutant erases only the projection's True constructor, preserving
the ordinary bridge and falsifying castling retention directly. The matcher
is not broadened. Short controls are moved ahead of the expensive complete
consumer so malformed controls fail promptly; full consumer plus all controls,
source/compiler checks and original resource limits remain mandatory. No theorem
source, consumer, premise, compiler or runtime change accompanies this correction.

## Final combined castling source gate: PASS

[Run 36814349505, job 110216055951](https://github.com/jjoshua2/DeepFin/actions/runs/36814349505/job/110216055951)
on a2824a0d586a1ca17bb279538866872edb4e03df passes the entire fail-closed gate.
The helper completed in 8.497660053999994 seconds. All five semantic controls
were rejected at their required locations, then the complete importing consumer
returned exact safe output after 697.018755295 seconds. Final source and compiler
identity checks passed. Four host classification methods passed under normal
Python, -O and -OO; all twelve original compiler-pin tests passed. The unchanged
whole-qualifier 1,600-second, 2-CPU quota and 6-GiB/no-swap limits were respected.

The [full source receipt](evidence/bend-generator-contract/castling-qualified-36814349505.json)
and [provenance](evidence/bend-generator-contract/castling-qualified-36814349505-provenance.json)
retain the source/compiler identity, diagnostics and original artifact metadata.
The first stdout JSON copy interleaved systemd status output; the second complete
copy printed by the following record step was parsed unchanged. No compiler
peak-memory claim is inferred from the service's final accounting line.

Independent source review found no remaining blockers after checking actual
generator retention, complete table transport, both wing order/decisions, complete
Ply uniqueness, explicit local premises, source-wiring mutation and fail-closed
gate corrections. Review and final source/control evidence are distinct.

This qualifies the two public component definitions in CastleComposition:
exact and no_duplicates, as requested by the complete castle_consumer. It is
not an aggregate proof-inventory run and does not qualify actual Tables.build,
historical rights/reachability, forward-attack composition, ordinary/promotion/EP
coverage, general skip-check safety or the integrated legal-generator theorem.
The proof is a local home-king/symbolic-initializer component of that finish line.
Runtime, checker, Base, pin, inherited proofs and previous gates are unchanged.

Exact-head whole-repository lint and PEXT passed; ordinary CI was still running
when this record was prepared. Existing native regression results are separate
and do not upgrade the source theorem's domain or residual trusted components.
