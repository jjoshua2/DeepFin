# Relative table addresses and cross-block preservation

## Baseline and bounded acceptance

Continue #829, branch feat/bend-complete-routes-20260922 at
`fc1fdd7cbe4d7e84543a14df9b7336c39ab098f2`, complete tree
`1a182723143434f457daf296b8582a7a7a415f21`. The exact archive, commit object and
whole Git tree were recovered and verified. Current repository guidance,
development/branch lifecycle documents and experiment index were read. The
compiler remains `aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae`, Bend 2.0.21 + U64,
84 inputs with fingerprint
`d9550e30dbf17f12013aa5db89b27cf6724957fe68213c9c7e409e99f1405ef4`.

This acceptance record follows constructive local development; it is not a
backdated preregistration. Required qualification is the unchanged parent source
gate plus six public contracts and 20 new controls, four bounded native modes,
original compiler checks, and unchanged whole-repository lint. No GPU, training,
perft increase, deployment or live-process work is included. Temporary workflow
and payload objects must not enter the feature tree.

## Six public source contracts

| Law | Guarantee and premises |
| --- | --- |
| relative_address_bounds | For valid key and any relative U32 below actual block size: start <= actual sum < next prefix, and actual sum < 131072. |
| relative_addition_exact | Under the same domain, widening the U32 sum equals adding widened prefix and relative index. |
| lookup_address_bounds | The actual arbitrary-occupancy Sliders.pext_index supplies the relative bound internally. |
| ordered_block_addresses | Addresses in ordered valid block regions remain strictly numerically ordered. |
| ordered_block_address_paths | Those addresses have separated routes in a real complete depth-17 affine array. |
| lookup_other_block_write | Writing the actual PEXT-derived address of an earlier block preserves the later block query and returns the complete updated array/value pair. |

Prefix endpoint/no-overflow certificates and actual mask population/lookup bounds
are imported from discharged parent laws. The public caller does not assume those
certificates. Addition carry balance and unsigned-order transitivity are proved
by Word induction, with finite local Boolean/Cmp cases rather than enumeration of
U32 operands. The Nat-to-machine comparison bridge connects the previous actual
PEXT bound to this source arithmetic. Complete shape uses the existing proved
initialization/pipeline invariant; it is not interchangeable with reported size.

## Local checks before hosted qualification

The public consumer checked successfully with exact `All terms check.` and exit
zero. It exercises all six obligations, arbitrary occupancy, strict endpoint
exclusion and derived allocation-mask identity. The 20 rejection controls
passed in the explicitly controls-only development mode. These are separate
commands, not a local full-aggregate or full-focused-run claim. Source identities
and raw successful consumer output are retained with local evidence.

Ten semantic mutations reject shifted/missing relative addresses, incorrect
incoming carries, inclusive order, corrupted comparison coherence, wrong endpoint
certificate extraction, routing to the write address, returning the original
buffer, and corruption of actual Base U64 carry/low-half addition behavior. Ten policy/output controls
retain every obligation and required certificate import, reject holes/foreign/
symlinked inputs, and reject unsafe output even with raw zero status. Missing
imports, crashes and timeouts do not count as semantic rejection.

The unchanged final native driver passes four modes (generic, forced-portable,
native-target and UBSan): **288 rows per mode**, all 128 keys, 256 cross-block rows
(254 in the public theorem's earlier-to-later orientation), 32 same-block controls,
272 protected queries and 16 same-address writes. Six malformed requests are
rejected per mode. Output SHA-256 in every mode:
`7471c94b03a652f0a599384beae228aca99400ff8cbcbf5f9861f3812b39089d`.
Local tools are Bun 1.4.2 and Clang 17.

The candidate builds actual Tables.build, reads actual mask/offset headers,
computes actual PEXT/addition, and executes supported public Array APIs. Independent
signed-coordinate rays, direct BigInt gathering and ordinary integer sums predict
expected values externally. It receives no expected tables/indices. Observations
are headers, indices, addresses, selected values and capacity, not every cell.
Modes repeat fixtures, not disjoint data sets or exhaustive arbitrary-U64 tests.

A standalone certificate-module attempt exceeded its 300-second development
limit; the complete public consumer subsequently passed in about 351.5 seconds.
The timeout remains a failed development attempt, not a rejected theorem or
passing aggregate. Early constructive drafts had computed-tuple match/linearity
errors, and an intentionally missing parent PROOF import was refused as an
unfilled law. Those issues were corrected before acceptance without changing
compiler semantics or weakening any accepted statement. The private arithmetic/order/frame modules also checked independently.
A trailing blank line was removed after the first consumer pass; all controls
were rerun on the resulting exact source, and the full consumer recheck is
recorded separately. That whitespace-only edit is not silently attributed to
the earlier execution.

The exact post-whitespace public consumer passed again in about 320.9 seconds.
Self-review found two initial controls rejecting at the wrong layer: an untyped
zero failed inference, and U32 off-by-one first broke the existing Base
commutativity proof. The original reported controls-only PASS is retained but
is not accepted as semantic qualification of those two cases. Corrected typed
U64 carry/low-half mutations now fail specifically at safe_from_wide and widening.
The harness rejects malformed/inference diagnostics and records exact failure
locations and normalized diagnostic hashes. All 20 corrected controls passed.
Obsolete hosted run 35776424965 was cancelled before native/lint/publication;
no completed aggregate or published feature is attributed to that run. Neither
accepted source theorem nor compiler input changed during this correction.

Whole-repository lint and the combined parent/new source aggregate are not claimed
executed locally. Hosted qualification is recorded separately when completed.

## Scope and next decisive acceptance

These source laws calculate with the certified prefix expression. They do not yet
prove that every stored metadata header equals that expression, or that every
final lookup reads computed rather than seed data. The native probe reads those
headers directly, but its passing comparisons are not the missing source theorem.
The remaining connection is deriving the full fill-address schedule's clear-write
certificates, initialization of final contents, metadata refinement and equality
to independent blocker-ray attacks. No whole-P2 or whole-engine proof is claimed.

No production source, previous accepted law/gate, compiler input, existing test or
permanent workflow changes. No new application responsibility moves from Python
into Bend; Python export/external references/data/control/training and transitional
C++/LibTorch/AOTI remain dependencies. Self-review only, not independent review.
Pinned checker/Base, native lowering, physical storage/lifetime, ABI, toolchain,
libraries, OS/hardware remain trust boundaries. Prior TypeScript, diagnostic-depth
and raw-internal-call lowering limitations are unchanged and unsuppressed.
