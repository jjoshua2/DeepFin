# A39 versus pinned eight-thread Stockfish: first timed baseline

Status at registration: **COMPLETE PROSPECTIVE PROTOCOL; SOURCE/DATA PREREQUISITES ACCEPTED; ENGINEERING AND BASELINE GAMES UNRUN.** Main publication and fresh phase admission remain required before play.

This study asks how unchanged donor A39 performs against one pinned full-strength Stockfish deployment under equal wall-clock controls. It establishes an initial descriptive external baseline. It does not test a new learner, establish a gain, claim parity, or identify the latest or strongest Stockfish build. The broader goal of beating multicore Rofchade and approaching Stockfish/Lc0 remains open.

## Fixed model and measured instrument

Keep A39 SHA `a39f900c944477c4ecf7f56cf0e1a6dd9486b67e5c2abb00973254e4bb6749b0`. Use frozen R5 source `f914f51eb349681300fe50147029e98b880de63e` and its existing compiled reduce-overhead cache lineage. The public UCI model/search implementation, checkpoint, arguments and environment remain fixed: two walkers, gather one, maximum batch 1024, evaluation cache zero, c-puct 1.75, factor 3.89, base 38739, FPU .33, policy temperature 1 and Gumbel scale zero. Preserve the selected legacy pending mode and single visible GPU. Actual route and single-device behavior must be observed rather than inferred from a flag alone.

R5's UCI Threads setting actively changes its worker; keep Threads 2 together with walkers 2. DeepFin uses CPU affinity 16,18. Its existing Hash 1024 and indexed 3–6-piece Syzygy paths/rule-50 settings are retained. This is distinct from the internal 64-simulation Gumbel instrument and from newer main revisions. No source/runtime adoption from PR1315 or PR1299 is involved.

The historical unwrapped R5/A39/Rofchade profile supplies reuse and cost evidence. The new source-pinned observational child wraps the unchanged public R5 entrypoint through runpy to obtain fresh model-state, AMP, shape, graph, native-origin and health evidence. This is a declared physical launcher/observation change, requiring qualification in the engineering pair. It is not asserted equivalent to the historical unwrapped instrument. Full state hashing and asset/network exports occur at off-clock readiness/closure boundaries; any lightweight shape/AMP counters are part of the actual measured wrapper profile.

Fresh owners must authenticate actual source/native/dependency/assets/cache custody, complete 494-entry model loading and served-state integrity, actual input/AMP/transport settings, warm shapes and graph health, non-tablebase neural work and bounded closure. No cache migration, cold-cache replacement, implicit rebuild, alternative route or suppression/fallback may rehabilitate a failure. The old compiled numeric mode remains distinct and lacks an equivalence tolerance to eager/internal instruments.

## Opponent and clocks

Use installed Stockfish dev-20260810-5062aee5, binary SHA `f4451f72ff871602b851769479948197d952b195d65951ce819b42bd850099d6`. Configure Threads 8, Hash 1024 MiB, Skill Level 20, UCI_LimitStrength false, nodestime 0, Move Overhead 10 ms, SyzygyProbeDepth 1, ProbeLimit 6 and rule-50 support with the same indexed 3–6-piece tables. Its affinity is 0,2,4,6,8,10,12,14; verify the current physical-core mapping. Configured eight threads does not imply continuous use of eight cores.

Only the binary's declared default `nn-ab28990d4ea3.nnue` is permitted. No alternative net, download, override search or strength-based selection is included. A source-qualified off-clock export from the same live configured SF process must bind the actual current-default content by a successful export, ready barrier and full exported-file SHA. Repeat in the fresh baseline process and require the engineering-accepted content SHA before warmup/scored play. Provenance is the authenticated binary, live process, configured default and observed export. Embedded-versus-external storage origin and source-versus-binary build equivalence remain unknown unless observed; neither is mislabeled proven. Filename, help or readyok alone cannot establish actual live network identity. Ambiguity about that identity is UNQUALIFIED.

The pinned python-chess API manages Ponder and MultiPV; they must not be passed through configure. The existing play path is nonponder and resets the advertised MultiPV default to one. Observe the actual command path with no go ponder or background search. There is no node, Elo or skill handicap.

Both engines receive **60 seconds + 1 second per move**, zero clock grace. The driver charges elapsed move wall time and applies a clock forfeit before increment. Retain full starting histories, ucinewgame/per-game reset semantics and at most 450 additional plies. Keep off-clock warmup at 64 DeepFin nodes and 1000 SF nodes, warmup timeout 120 seconds and restored command timeout 300 seconds. The scored clock mode has no node-enforcement limit. The pinned public match driver deterministically schedules openings and colors and has no RNG seed option; the public UCI has no seed CLI. Earlier unrun design seed labels were withdrawn before execution. No seed control is claimed, and no new seeding code is injected. Disable theoretical harness tablebase adjudication; engine tablebase support remains enabled. These settings do not pool with the earlier Rofchade 30+.1 baseline.

## Phases, panel and deciding rules

1. **Engineering color pair:** two games from the already exposed g1f3 d7d5 readiness input, fixed public color schedule, GPU owner at most 1800 seconds. This is an instrument qualification input, not a fresh strength panel. Qualify actual default NNUE content, compiled model/runtime, both-color history/clock/result paths and cleanup. The separate CPU audit has at most 300 seconds. Closed root and different-author actual technical acceptance is required to advance; W/D/L cannot choose whether the fixed baseline runs.
2. **Descriptive baseline:** eight games / four opening color pairs in fixed book/color order, GPU owner at most 7200 seconds and CPU audit at most 600 seconds. Freeze the book before publication: first four eligible complete-parent-disjoint groups in the accepted 16,384-entry cached order, with full 16-ply histories and eight distinct raw/mirror identities. Protect all old/retired/reserved roles, all 884 native-.2 pilot roles including unused confirmation, and all 500 A512/A39 diagnostic roles. No new ranking, encoding, alternative corpus or outcome/model-based selection.

Report all games, colors, histories, elapsed clocks, terminal reasons, faults and costs. Retain genuine clock losses as recorded losses with qualification flags; never hide unfavorable games. A cap or unresolved game is not imputed as a clean chess draw. A technical failure stops the affected advancement, preserves partial evidence and receives an honest UNQUALIFIED readout. No automatic retry, resume, reseed, extension, replacement opening/model/source/net/cache or favorable pooling.

Use the existing R5 driver's paired means, unbiased pair variance and approximate normal 95% interval. Its zero-variance unavailable policy remains unchanged; do not import the internal arena's Hoeffding fallback. Four pairs give limited precision and do not establish parity, a gain, regression, Elo or a milestone. There is no learning confirmation or adoption mechanism.

The panel's exclusion scope covers named retained metadata and protected starts, complete parents, histories/EPDs and raw/mirror identities. Later played trajectories and unretained search outside those indexes, external/unrecorded inference, pretraining and broader opening-family exposure remain unknown. No universal unseen-position claim is made.

## Budgets, controls and publication

Scientific GPU owner-wall maximum is **9000 seconds**, including failures, startup, network export, warmup, games and cleanup. CPU readers total at most **900 seconds**. A separate metadata qualification scope permits one selector owner at most 300 seconds and one combined synthetic-guard owner at most 60 seconds, total at most 360; freeze the complete selector/runtime control count before that single control owner. Source reviews do not count as executed controls.

Root is the sole numerical launcher. Use the three existing leases, finite owned-process cleanup, unchanged STOP/PAUSE/CANCEL/YIELD controls and fresh independent GPU scope. The week-one deadline remains November 19, 2026, 16:09:20 UTC and global GPU owner-wall cap 172800 seconds. Bounded GPU work alongside COD is authorized; record the shared-machine condition. A capacity label is not a resident-memory or latency qualification.

Publish the complete source/data/protocol preregistration on main before both engineering and baseline games. Scientific choices are fixed; only declared receipts from actual engineering may later bind the observed default-network/runtime identities before the baseline. These derived identities cannot select another network or route. No new-engine source fix, live deployment or model adoption is authorized by this record.

## Qualified prerequisites and immutable evidence

Study UUID `65df8c1b-3970-47bc-a4d3-20bd5caf820a`. The complete scientific freeze changes only five declared identity/status/data-adoption leaves from the reviewed V3 template; all clocks, budgets, phases, schedules and model/engine choices are unchanged. Later phase bindings may fill only their declared publication/source/science receipts and, for the baseline, the accepted role/book and actual engineering/network receipts. These fields cannot change scientific choices. Actual engineering acceptance is technical only, independent of the observed score.

The single combined synthetic owner passed **39 controls (19 selector + 20 runtime), zero failures/errors/skips**, in **1.2098453280050308 seconds**. The separate selector owner passed in **5.318102055985946 seconds**: all 16,384 cached candidates were validated and ranks 1384–1387 supplied the first four eligible groups. They form four complete-parent groups, eight raw/mirror identities, full 16-ply histories and zero collisions across all six protected indexes. The protected union includes 6236 prior/current groups and 404496 history identities; 1053188 raw/mirror index entries were retained. The exact R5 public opening parser and ordered-panel digest were verified against its source. No encoding, reranking, neural inference, tablebase query or model selection ran in these prerequisites.

Total prerequisite CPU owner wall is **6.527947383990977 / 360 seconds**. Root and different-author actual reviews accepted the closed 39-control result and four-role result. The independent role receipt at the canonical bindings path is an exact byte copy of the original independent receipt. Reviews reconciled compact reports, source and closure evidence; they did not reread the bulk book, role rows or cached pool. The new games remain unrun.

Unrun runtime V1 source review found a clock-increment reader mismatch, a missing synthetic-control namespace import and an obsolete model-config attribute. V2 corrected those; a source inspection then caught missing normal module registration for a Python 3.13 dataclass import. V3 corrects that registration. All versions remain preserved. These were source corrections before the single successful 39-control owner, not failed neural games. They provide no playing-strength evidence.

External artifacts use `~/chess-artifacts/operations/rl-week-one-20261112-v1/`. Bulk model, book, role and game data remain outside Git. The complete runtime, role selection, phase templates, commands and closure contracts are source pinned by the following compact records.

| Artifact | SHA-256 |
| --- | --- |
| `core-A39-SF20260810-timed-baseline-direction-v1/decision.json` | `048d8a6346cfe809d7d497d60c58a2ecd8c2477c7573673f5877582c0fd057c5` |
| `core-A39-SF20260810-timed-baseline-direction-independent-review-v1/review.json` | `44d8a4ae9e020be03adbd14a2db69cbfda66910d85a3758203d9362525acbbd0` |
| `core-A39-SF20260810-observational-launcher-selection-v1/decision.json` | `ff1bdd4f7c2a88dbda570994ef458a7fca27914ce5cb998135c1ebd264a67ee8` |
| `core-A39-SF20260810-observational-launcher-independent-review-v1/review.json` | `3f269f8f043fb420a66dd6d4be3a70f904c0a13232d27126830937c0e0a515ec` |
| `core-A39-SF20260810-schedule-selection-v1/decision.json` | `df989ace793f9962af34a920399aaf343b64b5b2e0ff1b36da0fc3d8158491ff` |
| `core-A39-SF20260810-schedule-independent-review-v1/review.json` | `b6a9a46b952da293d8ea1172e8eeb133b2f6eaceeab5219b3e62e147c552312f` |
| `core-A39-SF20260810-timed-baseline-scientific-freeze-v1/protocol.json` | `9dbbe567e5f90483bcd3b10eff4d96a4394e14e8568a713b52c55734b674b8d6` |
| `core-A39-SF20260810-timed-baseline-scientific-freeze-v1/root-freeze.json` | `d518fd54c17831eda254f15723ec86b2347d9d5d1e24d1c62315bbfcf74af4bf` |
| `core-A39-SF20260810-timed-baseline-runtime-preparation-v3/manifest.json` | `87bedd7becc0165ef2c7c3d5022d076318e20d9dbee69625e57ef5407b5c18fa` |
| `core-A39-SF20260810-timed-baseline-runtime-independent-source-review-v3/review.json` | `3ce5b3cbb08c46600d5231f0c617268788a94704cd1b6166638be6e505753097` |
| `core-A39-SF20260810-role-selector-preparation-v1/manifest.json` | `82a964a60bca21bafccf36016854de2ebecb121e1294f4fbf370fa29448c0e1e` |
| `core-A39-SF20260810-role-selector-independent-source-review-v1/review.json` | `8f7d9af993d11d51ec4a794a7a05566fea545b4f044d319f977a44301b4422a6` |
| `core-A39-SF20260810-timed-baseline-bindings-v1/combined-controls-root-acceptance.json` | `0af11b5ac408323aa69bd86799b926afc923f60be165f6d9e6caabe213cad556` |
| `core-A39-SF20260810-guard-controls-independent-actual-review-v1/review.json` | `d53e78cc967b6b132599d7c98b4cbd47e10783f4b1d326d1bbf3bf64eac68c84` |
| `core-A39-SF20260810-role-selector-results-v1/manifest.json` | `ac28d05e98b87bc61832abd311a3fa4f5b72a1dc7b1ec44c228fefa9c59b4a8e` |
| `core-A39-SF20260810-timed-baseline-bindings-v1/roles/root-acceptance.json` | `f6e1591ca5cb7d41ddb56fa61d20856f7c507d7ccedc5d6e737ac78af91c29cb` |
| `core-A39-SF20260810-timed-baseline-bindings-v1/roles/independent-review.json` | `c15d61ea17c0cd8ee7a9618fae35d702e27b548770c33923114ed97beb0719ec` |
| `core-A39-SF20260810-timed-baseline-bindings-v1/GPU-scope-plan.json` | `e9ad7594aeca7d6f7f798598819c7b34d9602f6b020a7b0dc88cc6c2780e7a0a` |
| `core-A39-SF20260810-timed-baseline-bindings-v1/resource-selection.json` | `ca3fcde4e4110151d78c83b960ca3a11fccc732e1d23a3ab51ec2547bffbb295` |

The accepted baseline book SHA is `5f7cf2f5eba9fea78be61edc3ca30a128184cad6147e1f20e8e8d9fec7c53f27`; role rows SHA is `1d99a227226eb5a08703cbad1ba9c2390f3f165bde2ffe5ee62e9bfabb618d5e`. The actual R5 ordered-panel digest is `ab9093dcd022dc528f6116360789aa767484a7706cdff45a0d55b0aae9031e76`. These identities are copied from accepted selector reports; publication preparation does not reread payloads.
