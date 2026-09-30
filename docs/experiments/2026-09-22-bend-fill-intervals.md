# Numeric intervals for actual full table fills

## Scope and source

Continue #833 at `d8de6e46fd04c4b34385d754224df37722a6a118`, complete parent tree
`99f19d5ef816a238e89da12ec19529088943012d`. The refreshed relative-address
implementation already proves safe interior addresses and cross-block single
writes; it is reused, not redone. No merge, deployment or live checkout operation.

Compiler remains `aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae`, Bend 2.0.21 + U64,
84 inputs/fingerprint `d9550e30dbf17f12013aa5db89b27cf6724957fe68213c9c7e409e99f1405ef4`.
All code additions are under `standalone/proofs/fill_interval/`. No production,
prior accepted law/gate, existing test, permanent workflow or compiler input changes.

## Acceptance recorded after local construction, before hosted qualification

This is not a backdated preregistration. Hosted acceptance requires exact source
identity, the unchanged 89-law parent, all new laws/controls, four native modes,
original compiler source/pin checks and unchanged whole-repository lint. A failing
stage must be preserved, not ignored. Native Clang selection is scoped to native
checks; locked Python CPU development setup uses its normal compiler. Tests are
bounded and opt-in, with no extra perft/model/GPU/training budget.

## Six new contracts

| Law | Actual result and preconditions |
| --- | --- |
| `bounded_clear_before` | Complete shape, query < start, end < allocation, and Nat count + start <= end derive every normalized fill path avoiding the query. |
| `bounded_clear_after` | Complete shape, query < allocation, and Nat count + start <= query derive every fill path avoiding the query. |
| `bounded_fill_preserves_before` | Apply that derived before certificate to the actual fill, preserving the queried value. |
| `bounded_fill_preserves_after` | Apply the after certificate to the actual fill, preserving the queried value. |
| `block_fill_preserves_before` | For every valid chess key, the full production block fill preserves every query before its prefix, on a complete depth-17 array. |
| `block_fill_preserves_after` | The full block fill preserves every bounded query at or beyond its exclusive endpoint. |

The count and prefix conditions for full blocks are produced from earlier
accepted prefix laws, not added as assumed equalities. Structural Word induction
connects actual increment and addition to Nat arithmetic; the complete-tree route
proof and existing fill frame law then compose. Neither expected values nor a
per-write clear certificate are assumed by the new public caller. This is a
queried-value theorem, not a new equality describing every final array cell.
The consumer includes positive-count before/after use and overlap/overflow exclusions.

## Recorded local evidence

The full importing consumer returned exactly `All terms check.` in 293.831 seconds.
The separately run final controls-only gate passed **19 controls**. These are two
commands, not a claimed execution of the combined focused or inherited aggregate.
The source files were unchanged between them; the final controls report also
identifies the verifier source. No local full 95-law aggregate is claimed.

All four native modes passed **469 rows** each: 384 full-block rows covering all
128 keys, 20 zero-count rows and 25 overlong rows, plus partial fills. There are
331 protected-query and 138 overwritten-query cases. Seven invalid requests per
mode are rejected. Repeated modes are not disjoint datasets or exhaustive U64
coverage. The probe observes actual metadata, query/first/last cell values and
capacity, not every returned cell. The independent reference receives candidate
outputs only after producing its expected masks/prefixes/rays.

Whole-repository lint was attempted unchanged but its local tools are missing:
Ruff, Basedpyright and Vulture. Exit was nonzero; the raw log is retained. Hosted
locked-environment lint must resolve this gap before qualification is called complete.

An initial U32.inc mutation was correctly rejected by the mutation harness for
failing first in inherited `layout/Domain.exact`, not the intended new lemma.
It was never counted as a valid control. The replacement tests the new increment
observation and requires failure at `inc_value`. Two separate production fill
mutations deliberately retain their first failure at imported `storage/Build.fill`;
those are implementation-dependency checks, not claimed new-layer arithmetic failures.
Early local syntax, duplicability and unavailable U32.dec spellings were corrected
before acceptance. No checker or previously accepted statement was modified.

## Remaining scope and trust

This closes numeric-to-clear certificates for bounded fills and outside-location
preservation for each complete slider block. Stored mask/offset header correctness,
inside-block computed contents after later writes, complete builder composition,
and independent blocker-ray lookup equality are still required. The source uses
the certified prefix expression; native tests read actual headers but do not turn
them into a quantified header theorem.

Self-review only, not independent review. Source laws trust the pinned checker/Base;
native lowering, array allocation/ownership/lifetime, ABI, toolchain and hardware
remain separate. Historical compiler TypeScript/diagnostic and direct internal-call
limitations remain unresolved. No application responsibility moved into Bend:
Python export/data/control/training and transitional C++/LibTorch/AOTI remain.
No full-engine/model/GPU/perft increase/training/benchmark/strength result is added.

## Supplementary bounded self-review

A disposable actual Tables.fill mutation performs an extra write in the zero-count
base case. The unchanged native reference rejects row 1 in generic mode: the
exclusive endpoint query is overwritten with zero. This is a wrong-value result,
not a malformed-input rejection, and is not included in the source-control count.
One surplus empty line at LAWS EOF was removed after local source checking; its
before/after hashes and exact formatting-only scope are retained. Hosted source
checking is required on the final bytes; the earlier local check is not relabeled.


## Hosted qualification and clean publication

Hosted run **35790621404**, development workflow `e91f23707a541243b073aca5ecd1e58158039c42`, passes all **95 accepted laws and 195 rejection controls**, four native modes, original compiler source/pin checks and unchanged whole-repository lint on the exact inspected candidate.

The aggregate runs the unchanged 89-law/176-control parent, then the new six-law/19-control gate on final source bytes. The new control diagnostics and complete source identities match the final local controls report. This is a newly executed complete aggregate, not the two separate local development commands relabeled. The final EOF formatting is checked here.

All four native modes match the saved complete local report except the separately recorded C compiler identity: 469 rows, all 128 keys, 384 full-block, 20 zero-count and 25 overlong rows, 331 protected and 138 overwritten queries, seven malformed rejections per mode. The candidate reads actual table headers and executes the real fill; proof predicates and host expected tables are not candidate inputs. Only selected query/first/last cells and capacity are observed, not the complete returned array.

Hosted tools: Bun 1.4.2, Ubuntu clang version 18.1.3 (1ubuntu1) scoped to the native step; locked Python 3.13 CPU dependencies, uv 0.12.10 and the normal Python extension compiler. Original compiler laws/seven controls and 12 pin tests pass. The unchanged Ruff/Basedpyright/Vulture gate resolves the local missing-tools lint failure; the original nonzero log remains retained.

Every one of the 269 native-source manifest entries matches before and after qualification. Patch and complete candidate tree identities are preserved. Publication adds hosted evidence and documentation only. No production code, prior proof/gate, compiler input, existing test, routine workflow or perft budget changed.

The local wrong-layer increment mutation was not counted as a pass; its replacement targets inc_value. Eight new arithmetic/interval refinements and two deliberately inherited actual-fill refinement failures make up the ten semantic controls; nine policy/output guards complete the new 19. The separate actual extra-write native mutation fails at the exclusive-endpoint query and is not added to gate counts.

Fresh branch `feat/bend-fill-intervals-20260922` is created only after all checks. No force push, merge, deployment or live process change. Temporary development workflows and transport payloads are absent from the feature diff. Compact reports are committed; full logs are retained in the 30-day artifact bend-fill-interval-qualification. Self-review only, not independent review.

The new laws prove derived clear-write certificates and outside-value preservation for bounded fills and every full slider block. Stored header correctness, preservation of computed inside-block entries, composition through all metadata/extras writes and independent blocker-ray lookup equality remain P2 obligations. Source prefix expressions are not yet a theorem about every loaded stored header.

Source checker/Base, native lowering, allocation/ownership/lifetime, ABI, toolchain, libraries and hardware remain trust boundaries. Prior compiler TypeScript/diagnostic and raw-internal-call limitations remain unresolved. No Python application responsibility moved: export, external references, data/control and training remain, with C++/LibTorch/AOTI transitional inference. No engine/model/GPU/perft increase/training/strength/benchmark result is added.
