# Selected-E sparse E source audit (NO-LAUNCH)

This is a CPU-only dependency and feasibility audit of the missing input adapter
for the [Selected-E label-cost kernel](2026-09-27-selected-e-label-cost-kernel.md).
It does not freeze a 12,288-row sample, authenticate selected row payloads,
open model sessions, run a labeler, or measure throughput. The historical E
bank remains a lower-assurance source for a cost screen only.

## Frozen dependency chain

| Boundary | Frozen path and SHA-256 | What it establishes |
| --- | --- | --- |
| E storage receipt | `/home/josh/chess-artifacts/operations/factorial58-sffree-preparation-20260921/execution_v1/qualified.json` `4325d6bb319d278d5c844b0e16a82a3211dab1224d9c6443d0bbc25b82af1f03` | 35 roots, 7,108 overlay shards, 58,090,688 rows; its declared scope is **storage only**. |
| E manifest roster | `/home/josh/chess-artifacts/operations/factorial58-sffree-preparation-20260921/manifests.json` `ca4e61bbae176ac0b6d1f32698aad9de299bc4f48de3035f2a430dac60ccf34c` | Pins the 35 E cohort manifests. |
| Example routing cohort | `/home/josh/chess-artifacts/operations/factorial58-sffree-preparation-20260921/cohort00.json` `697623fbde145eea79ae0382a977ad7c45958d9fcdd7607cb9f35441ab20a186` | The `factorial58-sffree-v1` manifest used as the routing digest; its entries align a base shard with a Ceres sidecar and bind the original factorial manifest. |
| Original cohort manifest | `/home/josh/projects/chess/scratchpad/bt4_joint20/factorial58_20260919/preparation/cohorts/cohort00/manifest.json` `71bfef65938b2d25dbd7bc4c35c4d03bbb12e26d12003904dacb5fd73b5d9a35` | Original factorial shard/teacher roster, linked by the E manifest. |
| Example base seal | `/home/josh/projects/chess/scratchpad/bt4_joint20/factorial58_20260919/preparation/cohorts/cohort00/base-seal.json` `7ab832944fdac145ae0a20d5d6eac9684e8b417dc72156596b76af8dce8060ec` | Base content and row-history identity; shard 0 binds 8,192 rows, 175 input planes and compact policy encoding. |
| Example E overlay | `/home/josh/chess-artifacts/labels/factorial58_sffree_20260921/cohort00/E/shard_000000.zarr/target_overlay.json` `0ed5381894418dffa000aa657b0179a9b7e75bd6da812d816f2fbe1c746a8a06` | Binds base content `56a2ffb40b86da4c235a9a17229b0ab92aefdd8f0f7ee26f50198b0aef3c3b1b`, row identity `1feb72872fe553df2cbb4c01a645883c2f1deba83b915bac9ca3c337c4937cd2`, and replacement policy/WDL content hashes `50db182cd31a5faa479acc659a6901588380b287d189c068f1acd3fdfab9f17f`/`a96401a64cc45fbcc927a50090f75772e1b9e68fb6179bfb6b5e78bbb83354b6`. |
| Historical E training | `/home/josh/chess-artifacts/operations/factorial58-sffree-training-20260922/E/complete.json` `300e0c18bb7435cb27068a3fc7050cb943c751eba8e3360f605215597e2d7e21` | Completed seed-121 `game_epoch` receipt: 58,090,688 policy and WDL objective rows, zero realized auxiliary objective rows. It is a same-recipe reference, not an authenticated production control. |

The base shard in this example is
`/home/josh/projects/chess/data/nnue_derived/armB/qtemp_0.0005_hist_20m_bt4_global_B100T05_value50/shard_000000.zarr`.
Its BT4 policy sidecar is
`/home/josh/projects/chess/data/lc0/bt4_policy_sidecars/armB_qtemp0005_hist20m/shard_000000.zarr`;
its Ceres sidecar is
`/home/josh/projects/chess/data/lc0/ceres_compact_sidecars/original20m_fixed32_dual_first16_guard_cadence_v2/shard_000000.zarr`.
The native BT4 WDL sidecar is
`/home/josh/projects/chess/data/lc0/bt4_wdl_sidecars/original20m_stored_v1/shard_000000.zarr`;
its example E-overlay attrs pin `2cea7dad67257193280b40f7a5bdace1dff58ee924c83ae4746468a16ecd6ed5`.
The same overlay pins Ceres attrs
`8d988c0d974954bc2395a6d30e597e26817af525c4193fa7a71adfde76bb05f3`.
The model digests are BT4
`1d3c0bd28ebfb42b015d18f67831cb1d6d15ad5d358b25b8a8cf500786262fc0`
and Ceres
`44aa02c775456f18ed464e33fc37b8e4abf58d7bf8f4cfb3ff19492e32e56df3`.
The overlay contains only `policy_target` and `search_wdl`; the base holds
`x`, `legal_mask`, `game_id`, `ply_index`, `wdl_target`, and the main-head
presence masks. The Ceres sidecar contains row/game/ply indices and native
FP16 policy, primary-value and secondary-value logits. These are distinct
contracts; the E overlay's blended targets are not independent selected
teacher component pins.

For each selected row, the adapter must join the base's exact
`(resolved base parent, game_id, ply_index, shard, offset)` and `x` bytes to
the overlay's base/row-identity declaration, the BT4 native sidecar's
`row_index/game_id/ply_index/lc0_feed_sha256`, and the Ceres sidecar's
`row_index/game_id/ply_index/tpg_feed_sha256` and legal offsets/indices.
The base's stored BT4 T=0.5 policy and native BT4 WDL, and the Ceres native
policy/value/value2 logits, provide independent component-target pins.
Their original row order and source bindings must agree before any selected
teacher call. The historical E recipe combines those components; comparing
one selected target to the already blended E `policy_target` or `search_wdl`
would be the wrong byte gate.

The base stores `has_policy`, `has_search_wdl`, `has_legal_mask`,
`has_game_id` and `has_ply_index` presence flags. The completed E training
receipt records `objective_mask_weights.policy =
objective_mask_weights.wdl = 58,090,688` and zero for every auxiliary
objective. A selected row must have all five required presence flags equal
to one, a nonempty binary compact legal mask, finite policy and WDL, and
zero auxiliary objective evidence under the real trainer consumer. The
base's `wdl_target` outcome remains a preserved nonmain field; it cannot
stand in for the `search_wdl` teacher target. Source identity, legal support,
target byte pins and the actual consumer masks must all survive saved-output
readback before any qualified-row count is reported.

The current kernel calls `bt4_derived_wdl_sidecar.stored_feed`,
`bt4_policy_dump.compact_legal_policy`, `bt4_policy_mix._tempered_bt4_policy`,
`ceres_tpg.stored_x_to_ceres_tpg_bytes`, `leela_index.leela_gather_indices`,
`ceres_target_mix.policy_target`, and `ceres_value_mix.softmax`. Its
`Row` constructor checks supplied byte hashes and legal shape, but the caller
has not yet authenticated those hashes against this receipt chain. The
historical CPU `target_contract.load_pinned_cohort` in
`/home/josh/chess-artifacts/operations/teacher-e-direct-audit-readout-20260924/dependencies/target_contract.py`
authenticates the E manifest/base-seal metadata and shard ordinal only; it
explicitly leaves full source, sidecar and overlay qualification to the caller.

## Sample feasibility and refusal boundary

An exact 12,288-row, 35-cohort *game-clustered* sample needs an authenticated
map from each `(resolved base-shard parent, game_id)` to **all** row positions
in that game. The historical E training receipt reports 299,144 games and the
same parent-plus-ID identity, but it does not publish the game-to-row index.
Reading shard 0 and shard 1 `game_id` columns in the first SF cohort and a G10
cohort found one ID shared across the boundary in each; IDs are not sorted
within either shard. Therefore selecting offsets from one shard and claiming
complete games is invalid. A bounded sparse reader can follow a frozen index,
but an index must first be derived and authenticated against the sealed source
without silently substituting a broad registered-bank scan for this CPU task.
The exact-row total also needs whole-game subset selection with a declared
cohort/phase/legal-count allocation and a deterministic failure if no exact
solution exists; truncating a game to hit 12,288 would break the sampling unit.

The smallest complete index proposal reads only `game_id` and `has_game_id`
from all 7,108 sealed base shards, with an explicit CPU budget, and writes
per-shard `(game_id, count)` records under each authenticated cohort. Those
columns are 58,090,688 × (8 + 1) = **522,816,192 decoded bytes** before
Zarr chunk overhead; this is a proposed separate metadata scan, not a scan
performed here. The index must bind the E qualification SHA, the 35 E
manifest/base-seal pins, each shard's historical content SHA and a digest of
the exact scalar chunks read. It must reject missing/nonbinary `has_game_id`,
duplicate shard identities, inconsistent row counts and changed chunk bytes.
Across every shard under one resolved base parent, sum the counts for each
game ID to establish whole-game size; candidate selection can then solve for
exact cohort quotas and 12,288 total rows. Only after freezing that roster
should a bounded reader rescan the selected shards' game-ID chunks to recover
offsets and verify every selected game's count against the index. A full
per-row offset index would require at least another 58,090,688 × 4 =
232,362,752 bytes for uint32 offsets and is unnecessary. This proposal does
not authenticate dense source contents or replace selected-chunk checks and
the existing lower-assurance qualification caveat.

Before calling `matched_cpu_kernel`, an adapter must authenticate the E
qualification digest, all selected cohort manifest and base-seal digests, each
selected overlay/base shard identity, and an immutable routing-seed digest.
For each row it must bind source namespace, cohort/shard/offset, game and ply,
exact stored `x`, legal map, independent BT4/Ceres component target bytes and
all realized objective masks. It must reject duplicate exact input/context
keys, missing or nonbinary `has_policy`/`has_search_wdl`, nonzero auxiliary
objective evidence, and changed selected chunk bytes before model use. The
E receipt's storage scope and historical WDL-only tablebase check cannot be
promoted to new native all-row qualification. BT4's raw JSON writer requires
a closed raw shard; the Ceres derived writer requires a contiguous derived
source with a matching summary. Neither can accept a fabricated sparse shard.

The existing `tests/test_selected_e_label_cost_screen.py` passed 10 CPU tests
with CUDA hidden and torch capped at two threads. This checks the already
admitted kernel only. No new adapter is claimed by this audit.
