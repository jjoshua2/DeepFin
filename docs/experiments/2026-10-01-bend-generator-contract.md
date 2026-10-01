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
