# DeepFin unused-code audit — 2026-10-15

## Revision, scope and outcome

Audit base: `32370b099f6d8d4ec965de135e70a752fc95a7a9`, the remote `main` observed on 2026-10-15. Repository: <https://github.com/jjoshua2/DeepFin>. Isolated worktree: `/home/josh/projects/chess-worktrees/unused-code-audit-20261015`; branch: `cleanup/unused-code-audit-20261015`. The running checkout was not switched or edited.

Three narrowly justified cleanup changes were selected: two unreferenced functions and one redundant shell conditional. No feature, experiment, configuration key, supported driver or historical record is declared retired. The source evidence supports preserving checked-in behavior; deployment and external consumers remain separate questions.

A case-insensitive `git grep -n -i statsig BASE` over all tracked content returned exit 1 with no matches. Dependency/build manifests and configuration/entrypoint surfaces were included. **No checked-in Statsig integration was found.** This is not a statement about external services or untracked deployment assets. There are no verified Statsig production values to report.

The entire tracked inventory was risk-triaged; this is not a claim that every definition or external workload was exhaustively proven reachable or unreachable. No production telemetry, current process sampling, queue inspection, data scan, benchmark or profiling was performed. Unknown usage is not evidence of non-use.

## Coverage and preservation boundaries

All 6,103 tracked paths at the base were inventoried. Detailed source tracing prioritized executing code and dispatch boundaries; evidence/proof assets received preservation and reference triage rather than mechanical deletion screening.

| Subsystem / proposed review owner | Paths | Audit workflow and boundary |
| --- | ---: | --- |
| Runtime maintainers — `chess_anti_engine/` | 224 | Imports, exports, private helpers, optional backends, schema/reload consumers, C API tables/capsules, encoding and recovery contracts. |
| Toolchain maintainers — `scripts/` | 374 | Direct script/module execution, argparse dispatch, adapters, shell sourcing, wrapper calls, dynamic wrapping, sidecar naming and recovery chains. |
| Validation maintainers — `tests/`, `.github/` | 733 + 20 | Existing behavioral regressions, validator selection, CPU/capped/PEXT distinction, lint and advisory canary. Tests are consumers, not proof of production use. |
| Configuration / packaging maintainers | 20 configs; root manifests | Main template versus reference/research snapshots, installed `deepfin` entrypoint, setuptools package/build scope, environment/native flags and optional dependencies. |
| Bend / proof owners — `native/` | 1,484 | Native/proof/qualification assets preserved. 1,259 paths belong to `native/bend_engine/standalone/`; CI and other native probes establish entrypoint surfaces. This lane does not retire active proofs or reproduce their qualification. |
| Research/evidence owners — `docs/`, `scratchpad/` | 1,746 + 1,467 | Historical evidence preserved. `docs/experiments/evidence/` contains 1,382 paths; `docs/experiments/artifacts/` 30. 1,237 scratchpad paths are under `bt4_joint20`. Age, a negative result or lack of package imports is insufficient for deletion. |
| Ceres toolchain owner — `tools/` | 8 | Operator stage dispatch and documented reproducibility entrypoints; historical toolchain retained. |
| Other root/data/guidance assets | 27 | Inventory, dependency/build/CLI instructions and references; no data or original-record removal. |

Counts describe path inventory, not ownership identities or test counts. No CODEOWNERS file was present in the tracked `.github` inventory; owner labels here are subsystem roles for routing review.

### Coordination exclusions

- [Research migration PR #1097](https://github.com/jjoshua2/DeepFin/pull/1097), owner `01a14006-3bc5-7113-a07f-8a4de16370d5` owns branch `docs/living-research-submodule-20261015`, based on this same main revision. Its 121 dated experiment-prose removals, `CLAUDE.md`, experiment-index policy, doc links, `docs/research_repository.md`, `.gitmodules` and `research/` gitlink are excluded. No move is duplicated here. Public research repo observed at `9f4e747540881a91c3aacb09ffdeae6cdd5ab497`.
- [PR #1095](https://github.com/jjoshua2/DeepFin/pull/1095) owns cache digest/replay sampling reuse (`inference_cache.py`, `replay/buffer.py`, `disk_buffer.py`, `sampling.py` and related tests). This audit does not edit those paths.
- [PR #1096](https://github.com/jjoshua2/DeepFin/pull/1096) owns UCI topology/restricted-search work; this audit does not edit UCI.
- V10 repair, Bend compiler/proof work, all qualification inputs/receipts, negative/rejected evidence, original runtime sources and active queue remain with their owners. In particular private tree `6614ae4f678725c4371c8c5fb3f6a6bbbb1fcc94`, V3 baseline/core/runner, Bend V5 and parity/science inputs are not modified or borrowed as validation authority.
- Checkpoints, RNG/sampler cursors, control files, partial BT4 unit1489 and live configuration/services are untouched. Git-tracked removals remain recoverable from the base and isolated commits; no history rewrite or permanent data deletion occurred.

## Independently actionable cleanup changes

Locations below refer to the immutable audit base, before line shifts. Keep each code change in its own commit; a reviewer can accept or drop any one without the others.

| ID / owner | Purpose and source evidence | Callers / configuration evidence | Classification, risk and required validation |
| --- | --- | --- | --- |
| R1 — Replay maintainers; `chess_anti_engine/replay/shard.py:766–767` | `_f16` wraps `x.astype(np.float16, copy=False)`. Whole tracked-content `git grep -n -w _f16 BASE` finds only this definition. No decorator, registration or re-export. `replay/__init__.py:1–17` exports named replay APIs, excluding it. Dynamic access at `shard.py:1151–1230,1702` targets sample fields/array properties, not module helpers. | No in-repository callers. Actual conversions remain inline at base `1164,1166,1222`; dtype `_F16` at `140` feeds schema specs. Neighbor `_u8` remains used at `1170`. No configuration selects `_f16`. External private imports cannot be ruled out by repository search. | **Safe in-repository removal**, low risk. Delete only its four-line definition/separation. Existing NPZ/Zarr roundtrip, metadata, compact-policy, optional-field and invalid-input tests exercise the surviving contract. No source-spelling regression is added. |
| T1 — Corpus toolchain maintainers; `scripts/bt4_raw_corpus_sidecar.py:194–197` | `source_name_from_sidecar` is an unused inverse naming function. It reconstructs only `.jsonl.zst`, whereas active forward naming accepts `.jsonl.zst` and `.jsonl.gz` at `187–191`. Whole tracked-content name search finds only its definition. | Closed-inventory mapping and `sidecar_name` drive discovery at `512–536` and verification at `1149–1167`; adapters call forward naming at `scripts/adapt_raw_bt4_sidecars.py:133,150`. `bt4_batch_benchmark.py:104–106` dynamically wraps four explicitly named functions, excluding this one. CLI parser/dispatch does not resolve this name. No configured activation exists. | **Safe in-repository removal**, low risk; remove only this function, retaining module, naming validation, receipts and restart logic. Existing `tests/test_bt4_raw_corpus_sidecar.py` covers both suffixes, closed-inventory and recovery behavior. External imports remain an explicit limit. |
| V1 — Validation maintainers; `scripts/lint.sh:241–243` | `if [[ ${#JOB_PIDS[@]} -gt 0 ]]` is always true when reached. Unconditional `start_job` at `198–199` appends PIDs at `117`; no reset occurs before this guard. Reset is inside `run_group` at `190`. Startup errors exit under `set -e` before reaching it. | All normal lint modes queue ruff and basedpyright; `--fast` only suppresses vulture. `--help` exits earlier. No configuration yields an empty final job group. | **Safe removal of redundant condition**, low risk. Call `run_group` unconditionally. Keep waits, failure aggregation, advisory handling and `set -e` behavior. Existing hosted whole-repo lint exercises the script; no custom validation harness is introduced. |

## Retained and unresolved candidates, grouped by owner

| Owner / candidate | Classification and reachability evidence | What would justify a later change |
| --- | --- | --- |
| Encoding — repetition-fix native setters | **Retain.** `encoding/rep_fix.py:81–96` builds module names, imports extensions and uses `getattr(..., 'set_history_rep_fix')`. `_lc0_ext.c:1367` and `_mcts_tree.c:6449` register setters. A zero direct-call count would be misleading. | Any retirement must preserve checkpoint encoding identity and native construction-time history semantics, with actual artifact migration evidence. |
| Encoding/native — feature callbacks and providers | **Retain.** `_features_ext.c:132–145` registers callbacks, `features.py:274–276` aliases names as `_c_compute`/`_c_compute_relations`. `_nnue_ext.c:1191–1222` publishes capsules consumed by `_mcts_tree.c:5003,5029–5039`. | Trace runtime registries, aliases and capsule ABI before changing native surfaces. Existing compiler source and qualification remain separate owned work. |
| Encoding — perft extension and ray-walker references | **Retain diagnostics/oracles.** `setup.py:169–181` builds the perft extension, `encoding/perft.py:17` exposes it, `tests/test_perft_snapshot_ep.py:8` uses it and freshness tooling lists it at `scripts/check_c_extensions_fresh.py:55`. Setup macros intentionally preserve reference walkers for differential tests. | A reference implementation is not dead because optimized production code takes another branch. Replace its verification role before retirement. |
| Runtime — FP8, PUCT and legacy history modes | **Retain optional backends/compatibility.** `worker_inference.py:31` dynamically imports `torchao.quantization`; `maybe_compile_inference_model` selects it through `use_fp8`. `mcts/puct_history.py` intentionally omits an argument for older native binaries. Encoding aliases and legacy NPZ reads preserve existing artifacts. | Need a supported-version/deployment inventory and explicit migration policy. Quantized, default-off or old does not mean unused. |
| Training/configuration — `_TRAINER_WEIGHT_KEYS`, dead-key rejection | **Retain guards.** `trainable_init.py:58,322` imports/uses key tables; `trainable_config_ops.py:1388` iterates them. Rejection executes at `trainable_init.py:678`; reload/sync at `trainable.py:1063,1169`. | Removing rejection can turn an invalid key into an accepted-but-ignored one. Schema-to-consumer and reload behavior need independent coverage. |
| Training compatibility — `pairwise_lr`, `pairwise_momentum` | **Needs migration**, not blanket deletion. `tune/gpbt.py:48–49` explicitly accepts/ignores legacy kwargs and vulture exemptions document this boundary in `pyproject.toml:238–245`. | Identify all caller/version/checkpoint/tuning-restore contracts before removing accepted arguments. No live/queued consumer absence is established here. |
| Worker/recovery — saved config, watchdogs, tablebases | **Retain.** `worker.py:1453` calls `save_worker_config`; header-only takeover has arena/match consumers. `train.sh:191–192` launches `watchdog_loop.sh`, which invokes watchdog and `recover_stall.sh`. Manual/alternate paths require callee marker checks; `tests/test_recover_stall_guard.py` exercises them. Tablebase callers exist in `selfplay/finalize.py:183`, `manager.py:82`, `bt4_outcome.py:125`. | Recovery rarity and caller-side checks do not establish redundancy of callee guards. No runtime-control changes are authorized by this audit. |
| Shell tooling — ratchet and pause libraries | **Retain.** `daily_gate_ratchet.sh:75` and `ratchet_loop.sh:21` source `ratchet_common.sh`; `pause_window.sh:118` sources `pause_drain_lib.sh`. | Source directives are callers even without Python imports. |
| Validation — CPU, capped, PEXT and canary jobs | **Retain distinct contracts.** CI and `scripts/validate.py:23–51,139–160` distinguish ordinary CPU, fixed two-thread cap, separately built PEXT backend and intentionally upgraded advisory tools. | Similar job names or overlapping tests do not prove interchangeable environments. No CI job removal justified. |
| Diagnostics — `NeuralWorkLedger` | **Unresolved retirement; retain now.** Tests-only callers were found, but `docs/neural_work.md:30–35` documents a bounded-run public helper. Shared `WorkCounters` is active in `inference.py:1831` when `CAE_NEURAL_WORK=1`. | Establish supported public usage before removing the class; never delete the whole module based on that class's direct caller count. |
| Historical toolchains/configs | **Unresolved use; preserve.** `tools/ceres_cpu_oracle/operator.py:73–80` dispatches pinned stage argv; its README and historical experiment link define reproducibility use. Twenty YAMLs include production/reference/experiment/snapshot roles. | Obtain owner disposition and exact running/queued manifest references. Preserve experiment evidence and coordinate research migration. No historical driver or config is removed solely from its date. |
| Replay performance — eager default allocations | **Unresolved optimization.** At base `shard.py:1575–1586`, `arrs.get('priority', np.ones(...))`, `has_policy` and optional flag defaults are eagerly constructed even for present keys. Similar defaults exist in `prune_storage_arrays`. This is structurally unnecessary allocation on those branches, not an obsolete feature flag. | Separate representative measured cost from production frequency, then test present/missing keys, dtype/shape/error behavior and mapping contract before a focused performance patch. No change made here. |

## Configuration and CPU evidence

These are checked-in **template values at the base**, not realized production values. The live checkout may differ, checkpoint architecture carries its own identity, and queued/runtime copies were not inspected here.

| Surface | Checked-in value / evidence | Realized production value |
| --- | --- | --- |
| Main template history fix | `configs/pbt2_small.yaml:222`: `history_rep_fix: true` | Unknown |
| Main template search | `configs/pbt2_small.yaml:446`: `mcts: gumbel` | Unknown; does not retire optional PUCT |
| Reference search | `configs/default.yaml:71`: `mcts: puct` | Reference, not deployment evidence |
| Worker FP8 | `configs/pbt2_small.yaml:1022`: `distributed_worker_inference_fp8: false` | Unknown; default-off is not removal authority |
| Training compile | `configs/pbt2_small.yaml:944–945`: `use_compile: true`, `max-autotune` | Unknown |
| View-rate key | `configs/pbt2_small.yaml:677`: `train_views_per_ingested_position: 5.0` | Unknown; old name is a rejected different-denominator contract |

| Candidate | Supported CPU conclusion | Measured overhead / saving |
| --- | --- | --- |
| R1, T1 unused definitions | No checked-in invocation found; removing function objects is import-time cleanup. | Unknown import-time effect; no hot-path savings claimed. |
| V1 redundant guard | One shell condition removed once per lint invocation. | Unmeasured; no meaningful production CPU saving claimed. |
| Eager replay defaults | Present-key evaluation allocates scratch defaults; frequency and importance need a representative run. | Unknown; no profile/admission performed. |
| Native/recovery/config guards | Actual consumers exist; not established as useless. | Unknown; do not infer savings from deleting them. |

`docs/sf_cpu_cost_split.md` contains dated measurements of Stockfish budget/yield, not a measurement of these cleanup patches. Those figures are not reused as current telemetry or cleanup savings. No expensive useless runtime check was established by this audit.

## Validation and review status

Source checks: pinned base, clean starting worktree, whole tracked-content Statsig/symbol searches, declaration/export/dynamic-dispatch inspection, overlap PR path checks, exact diffs and `git diff --check`. A standard-library AST/name inventory was used as a candidate screen only; name frequency was not treated as proof of non-use.

Independent Sol reviewed the complete code diff `32370b099..bb0a35a3176190a015f7f472b3f5cb0fa4cfcabb`: no actionable findings, with explicit external-consumer and no-runtime-measurement limits. Grok cleared R1 alone; its full-patch review timed out without a verdict. Details are recorded below.

Existing regression targets are `tests/test_replay_shard_npz.py`, `tests/test_replay_shard_validation.py`, `tests/test_bt4_raw_corpus_sidecar.py`, and the whole-repo lint gate. No source-spelling tests or bespoke validation framework were added.

No local tests, imports, builds, lint, compiler or profiling were launched. Integration owner confirmed that the available CPU/native records bind another tree, not this cleanup. Local validation would require a separate exact source/runtime/selection/output admission on the original guard; previous product counts cannot be credited here. Existing hosted CI is the intended validation route for the draft PR. No profile is warranted to claim nonexistent hot-path benefit from these removals.

## Remaining limits and follow-up workflow

1. Read this audit as a repository source assessment with scoped implemented removals, not a certificate that all 6,103 paths are necessary or that all external consumers were seen.
2. Runtime owners can supply exact deployed revision/configuration/queued-manifest pins and usage evidence for the unresolved compatibility/config/driver candidates. Unavailable evidence stays unknown; no replacement telemetry is invented.
3. Performance work needs a concrete candidate, immutable source/runtime, existing validation tooling, distinct off-C output, CPU31 exclusion, nice19, resource headroom and fresh integration-owner admission. Do not launch new competing tests, guards or controllers from this report.
4. Review R1, T1 and V1 independently, using the relevant existing tests. Keep research migration and owned Bend/V10 work separate. Draft PR only; no merge, deployment or live adoption is authorized by this report.

## Completion addendum

Code commits (all based on the pinned main):

- R1: `a05b4764fc817faeb650f6ad21ada60fc26cfaea`.
- T1: `dee191896d2eef0c9235b0c8fb6c13db83d1f5af`.
- V1 / final code head: `bb0a35a3176190a015f7f472b3f5cb0fa4cfcabb`.

The full code diff is three files, one insertion and 13 deletions. This report is a separate documentation commit; it changes no executing code. Independent Sol reviewed the full exact code range and the report draft, finding no actionable code issues. The draft's inventory typo was corrected from 47 to 27 and the eager-allocation location corrected to base lines1575–1586.

Grok R1 review returned `BEGIN_FINDINGS / NONE / END_FINDINGS`, wrapper exit0, source unchanged. Raw receipt: `/home/josh/chess-artifacts/unused-code-audit-20261015/grok/review.0LPmOJwU/raw.txt`, SHA256 `fed786b27f266f9503fe135baf866f2b7c6842dfd2e0c0036475c8d6b8a75982`.

Grok full three-file review hit its 300-second limit (`grok` exit124, wrapper command exit1) and has **no valid full-patch verdict**. Raw partial receipt: `/home/josh/chess-artifacts/unused-code-audit-20261015/grok-final/review.3v4s7XNa/raw.txt`. The wrapper confirmed source unchanged. No retry or budget increase was launched; this is not credited as a clean review.

Source evidence: `source-inventory-v1.json` SHA256 `96075a252dd7402420b2f054f405a706ef1ac9a869eb98d67aa94aba00b55128`; reversible `cleanup-code.patch` SHA256 `81876dd206f827912f633d7a4580e48a4e003f3c623c8b6a801179f09b4429b1`; saved full-code Sol review `independent-sol-review-v1.md` SHA256 `081c1feddd768729eccca5a3758a1881f66725831a4ca1f062cbfbda846d8493`. All reside under `/home/josh/chess-artifacts/unused-code-audit-20261015/`. Local paths are supporting evidence, not portable runtime authority.

At report-commit preparation, hosted checks have not yet run, and local regressions/lint/builds remain unrun. [Draft PR #1098](https://github.com/jjoshua2/DeepFin/pull/1098) is open against the pinned base. Its [existing CI check records](https://github.com/jjoshua2/DeepFin/pull/1098/checks) are authoritative for its actual head; there is no borrowed qualification credit. Final hosted results and PR identity will also be preserved in the task's off-C report/receipts after publication. Any uncompleted check remains a blocker to declaring validation complete.

A direct status message to originating thread `01a13814-9200-71fa-88a8-3cef5dd090d8` was rejected by automatic approval review because destination-specific messaging authorization was not established. It was not retried or routed indirectly. Coordination with the explicitly named integration and research owners succeeded. A training-owner query failed with `invalid turn/start params: AbsolutePathBuf deserialized without a base path`; no delivery or production-consumer evidence is inferred from it. The platform final handoff remains available.
