# Publication reconciliation and supplementary terminal-path review

## Exact qualified candidate

Hosted run **35996797725** passed preparation, source, controls, runtime/lint,
geometry and evidence publication on its first attempt. Source commit:
`12134356d6e09f303c256a76f83439204292897a`; qualified evidence head:
`811bb144e4b846f5d0b5bda853462cff999ee013`, tree
`b9ed1976131096c7146adb720bc0bfd1cc830748`.

The new branch `feat/bend-lookup-contracts-qualified-20260924` is based directly
on #872 at `9e7f433e0a6e139751b8a3b5f0871868df25a635`. It preserves all twelve
lookup-suite files and all eleven independent-geometry archive files from saved
local commit `a51dc14037b8dc9e85ea2f3706f5b475a13f27b5` byte-for-byte. Its new
commit identity is intentional: the saved local-only logs/status prose were not
silently republished as current hosted status.

The parallel three-contract branch at
`feb2dd7dbd12acc3e3f1eb4d13b2cb1b63d0cb6a` remains untouched. Its
`lookup_from_headers` maps to this candidate's `certified_header_route`;
`selected_state_is_masked` corresponds directly; `initialized_lookup` maps to
`initialized_masked_lookup`. This saved candidate additionally registers
`initialized_indexed_lookup`, rather than leaving that intermediate result only
as a helper. The parallel recovery run 35994246166 failed; no result from that
run is presented as qualification of this four-contract candidate. Do not apply
both same-directory variants together without explicit contract reconciliation.

## Downloaded evidence verification

Final qualification artifact **10807071413**, `lookup-saved-final`, ZIP SHA-256:
`eb060749a806f91a5483a85a0b37a62dedaa2edfbd78ca9077f1946c44d7c36b`.

The downloaded complete source archive reproduces all **3,518 tracked paths**,
the exact evidence-head tree and original Git commit object. All **341** native
source-manifest entries match the inspected local candidate. The sixteen-control
report is JSON-identical to the saved local report. The complete four-mode native
report matches except its separately recorded C compiler identity. The independent
geometry modes, sources and driver hashes match their saved report. Original
reports are retained, not normalized or rewritten to force agreement.

Four source contracts and sixteen controls executed as separate commands. Parent
106-law/260-control evidence is retained on exact sources and successful receipts,
yielding **modular 110-law/276-control coverage**, not a newly executed full110-law
wrapper. Fresh hosted repository lint passed Ruff, Basedpyright and Vulture;
the historical local missing-tools failure is not relabeled as a pass.

Source job: 107623611722. Controls job: 107623611820. Runtime/lint job:
107623611690. Geometry job: 107623611751. Publication job: 107626102780.
All belong to run35996797725 and completed successfully. Consumer success
requires status0 and exactly `All terms check.`. Four native modes each pass
1,024 rows (1,022 distinct inputs), all128 keys, six malformed requests and the
actual shifted-lookup regression. Geometry checks cover all512 step cases and
128 full masks; they are supplementary source results, not extra public laws.

## New independent path lemma

The [terminal-path archive](terminal-path/README.md) contains a newly checked
structural theorem over every finite path and arbitrary pairs of U64 occupancies:
agreement on all path positions except the last implies equal first-blocker-
inclusive attack values. Its premise observes only input occupancy bits, not
expected attacks. The consumer demonstrates terminal-only variation, arbitrary
singleton occupancies, inclusion of a concrete first blocker, and rejection of
an interior blocker difference.

This is a theorem about the **independent path specification**, not an established
connection to actual Tables.ray. Deriving interior agreement after relevant-mask
application and connecting actual production traversal to the path specification
remain open. It therefore is not counted as a fifth public lookup law or added
to hosted110/276 evidence.

The new consumer passed locally in1.182 seconds. Three disposable mutations were
rejected with ordinary diagnostics: omitting the first interior observation,
ignoring all interior observations, and omitting the blocker itself. The latter
fails the singleton proof; no false claim is made that it is a production-code
mutation. Exact source hashes, diagnostic hashes and reproduction steps are
archived. Two initial explicit-duplication-marker errors were corrected before
acceptance without changing the intended theorem or compiler. No native execution
of these three path modules is claimed.

## Scope and trust

This final archive/reconciliation commit changes only documentation/evidence.
The qualified primary lookup, native verifier, geometry archive and all prior
proof sources are unchanged. No full aggregate, native suite or lint was rerun
for this documentation-only addition. The supplementary path consumer and its
controls were newly executed and are labeled separately.

Self-review only, not independent review. Checker/Base, native lowering,
allocation/lifetime, ABI, toolchain and hardware remain trust boundaries.
Existing compiler TypeScript and diagnostic/raw-helper limitations are unchanged.
No production runtime, previous accepted contract, routine perft budget, model,
GPU, training or benchmark work changed. No merge, deployment or live-process
operation. No additional Python application responsibility moved into Bend;
export/references/data/control/training and transitional C++/LibTorch/AOTI remain.

The next decisive P2 acceptance is actual arbitrary-occupancy blocker traversal
and terminal-edge irrelevance, followed by composition into the public initialized
lookup equality. Independent masks and a conditional path-specification lemma
are foundations for that result, not a completed source-to-independent-rays proof.
