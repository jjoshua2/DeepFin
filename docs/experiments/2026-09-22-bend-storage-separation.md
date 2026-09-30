# Actual array and fill storage separation

## Scope and baseline

Continue PR #821 at `2c807135a17a8abcf478f9b142ac9a860b6111ed`, complete tree
`0491217a0e20ee5ed1c14b1208e7ffc2b6cf75e2`. The source archive and original
commit object were recovered and checked against that entire tree, preserving
all prior source and CI repairs. Repository instructions and the experiment
index were read before modification. No merge, deployment or live-process work.

Compiler stays `jjoshua2/bend@aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae`,
Bend 2.0.21 + U64. Its 84-source fingerprint remains
`d9550e30dbf17f12013aa5db89b27cf6724957fe68213c9c7e409e99f1405ef4`.

## Acceptance recorded after constructive development, before hosted qualification

This is not a backdated preregistration. Local construction and focused checks
preceded this record. Hosted acceptance requires the exact candidate tree, all
unchanged inherited laws/controls, the new four-law source gate, four native modes,
original compiler source/pin gates and unchanged whole-repository lint. No failed
gate may be ignored. A failure is retained, not reclassified as proof rejection.
Only proof/test/docs files change; all checks are opt-in and bounded. No model,
GPU, perft, production or training compute is authorized by this increment.

## Four source laws

| Law | Actual implementation and explicit condition |
| --- | --- |
| `write_preserves_other_location` | Actual get after set returns the complete updated array and original query value, when normalized paths are separated. |
| `write_preserves_separation` | Any actual set preserves the separation predicate for every pair of query indices. |
| `fill_preserves_unwritten_location` | Actual Tables.fill preserves the query value when every incrementing write address has a separated path. |
| `fill_preserves_separation` | Actual Tables.fill preserves the separation predicate for every pair of indices. |

`Spec.separate` derives size/normalization and routes solely from actual array
shape and indices. It contains no expected read value. `clear` applies this
condition to every actual incrementing U32 address, retaining wraparound. Both
are executable specifications, not axioms or assumed lookup-correctness laws.
The consumer proves satisfiable uses with arbitrary U64 cell values, nonuniform
source arrays and positive-count fill across U32_MAX-to-zero. It retains the
explicit 0/2 two-slot alias and a fill-overlap counterexample.

The proof reuses the prior certified structural image and actual Base API
refinements, transferring results back to arbitrary affine `Array<U64>` values.
No new unsafe dependency, foreign witness, hole, axiom or weakened prior law.
A nonuniform tree is covered by source semantics, not inferred to match native
flat-array behavior. Source equality is not physical pointer identity.

## Local results on final executable sources

The four-law importing consumer and all **17 new rejection controls** pass.
Nine controls require ordinary affected-refinement failure: false leaf/left/right
separation, omitted normalization, replacing paths with numeric inequality,
omitted first fill address, repeated instead of incremented fill addresses,
actual fill writing one address ahead and an actual zero-count fill that writes.
Eight controls enforce law/proof manifests, mandatory imports, no holes/foreign
code/symlinks and rejection of an unsafe warning even with raw status zero.

The final native gate passes **1,279 rows and 57,226 complete-array cell
comparisons per mode**: 638 set/read and 641 actual fill cases, all 128 chess keys.
There are **547 protected-query rows**, **732 overlap counterexamples**, and
186 set/read cases where different numeric indices alias. Seven invalid requests
are rejected in each mode. These repeated fixtures are not disjoint datasets or
exhaustive arbitrary-U64 native testing. All four mode output hashes are
`3e61ca04a8aecda814c5f0c9ca03bab82c35609c74af85987935b923f53f8ee8`.

The native probe is the unchanged prior storage probe. The independent reference
uses flat arrays, unsigned index masks, signed board coordinates and direct
BigInt gather/scatter. It compares full output cells. Proof predicates/models
are not executed in the native candidate. Local tools: Bun 1.4.2 / Clang 17.

Whole-repository lint was attempted locally and returned nonzero because Ruff,
Basedpyright and Vulture are absent. Hosted locked-environment lint is required
to resolve this gap. No local full 68-law aggregate is claimed: development used
the explicitly focused gate, with the unchanged parent reserved for hosted
aggregate qualification. The existing parent is not modified or skipped there.

A constructive draft initially consumed the Nat induction count twice. Marking
the duplicable Nat parameter explicitly resolved the linearity diagnostic before
acceptance, with no theorem-domain or compiler change. Preliminary native-driver
cleanup removed unused copied table/extras oracle paths; the final focused and
native reports were rerun after that edit. Earlier draft reports are not the
final candidate evidence.

## Remaining obligations and trust

These are real other-location frame laws, not proof that the current chess
prefix schedule always satisfies their conditions. Numeric prefix no-overflow,
complete allocation bounds, preservation of arbitrary other slots from numeric
interval separation, initialization of every final slider entry and independent
blocker-ray equality remain unfinished. The next decisive step is deriving
`clear` certificates from the actual builder's prefixes and using the frame lemma
with enumeration to prove final contents and lookup refinement.

Self-review only, not independent review. The pinned checker/Base, native
lowering, array ownership/lifetime, effects/ABI, C toolchain, OS/hardware and
model libraries remain trust boundaries. Historical compiler-fork TypeScript
and diagnostic-depth issues are not addressed or suppressed.

No production code or application responsibility moved from Python to Bend.
Python export, external references, data/control orchestration and training remain;
C++/LibTorch/AOTI is still transitional inference. No complete-engine theorem,
new model/GPU qualification, performance or strength result is implied.


## Hosted qualification

Hosted qualification **35728145361**, temporary workflow commit `82b49583a60ddcd9cd29408abcdd59e208eef409`, passes **68 accepted laws and 107 rejection controls**, all four native modes, original compiler source/pin checks and the unchanged whole-repository lint gate on source `4a0e442cc701b38a5ccd93938f28e421be5b4f9a`.

The complete source tree is 893dfc70e26e2bcba0a1536b1315fe12c306f79a, identical to the locally checked candidate. The full aggregate retains the unmodified 64-law/90-control parent and adds four frame laws and 17 controls. The focused contract/control/source report matches its local canonical SHA-256; the full native report matches locally except the separately recorded C compiler identity. Exact local/hosted comparison identities are committed.

Each native mode passes 1,279 fixture rows and 57,226 full-array cell comparisons: 638 set/read and 641 actual fill cases, all 128 keys, 547 protected queries, 732 overlap counterexamples and 186 distinct-index alias rows. Seven invalid requests are rejected per mode. These repeat local fixtures and modes, not disjoint datasets or exhaustive native U64 testing. Neither proof predicates nor proof representations execute in the native probe.

Tools: Bun 1.4.2, Ubuntu clang version 18.1.3 (1ubuntu1), locked Python 3.13 CPU environment with uv 0.12.10 for external tools. Original compiler source laws and all seven controls, including cyclic-template rejection, pass. The 12 compiler-pin contracts pass. Whole-repository Ruff/Basedpyright/Vulture passes unchanged, resolving the retained local missing-tools failure for this exact candidate.

Only qualification evidence, the experiment index and the migration inventory are appended after checking. Executable source hashes are unchanged. Full logs are in the 30-day artifact bend-storage-separation-qualification; compact reports and the original local lint failure are committed. Source publication uses direct Git tree entries with exact complete-tree verification, not encoded patch fragments.

Self-review only, not independent review. No production runtime, prior accepted law, compiler input, existing test, permanent workflow or perft budget changes. No full-engine/model/GPU/training/benchmark qualification. Source equality does not establish native pointer identity, allocation or physical lifetime.

The four laws assume explicit normalized-path separation or a computed clear-write-path condition, which the consumer shows satisfiable. Deriving these certificates for every actual chess prefix remains unfinished. Prefix no-overflow, initialized final table contents and independent blocker-ray lookup refinement are separate acceptance targets. No new Python application responsibility moved into Bend; export/data/control/training and transitional C++/LibTorch/AOTI remain dependencies.

Branch `feat/bend-storage-separation-20260922` is updated only by a fast-forward evidence/documentation commit after qualification. No merge, force push, deployment or live-process change. The temporary development workflow is excluded from this feature branch. Historical compiler TypeScript and diagnostic-depth limitations are not fixed or suppressed.
