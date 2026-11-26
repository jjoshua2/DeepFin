# 2026-11-26: Banked collections — qualification and admission

**Status: proposal for review.** This record decides which banked collections may be used as training rows for the
next recipe comparisons. Until it merges, no bank listed here is admitted. It changes nothing live and trains nothing.
Every check is CPU-only and read-only on the banks.

**Evidence** (all under `RC/banked-admission-20261126/`; `RC/` is the evidence log's prefix):
- the census script `admission_census.py` [`ce22a28f`] and its output `census.json` [`318a05e2`];
- the stock-reader check `reader_check.py` [`eefbfbd1`] and its output `reader_check.json` [`506aef15`].

The owner's rules this record applies:
- reuse compatible banked games rather than generating fresh ones per test;
- make row qualification and admission explicit before training;
- train on uniform shuffled rows, each used exactly once (at most two views including mirrors);
- change one variable at a time.

## The banks

Bank paths (`~/chess-artifacts/operations/` unless absolute):

| Name | Collection directory | Games | Rows | Seed | Openings file |
|---|---|---:|---:|---:|---|
| pilot | `rl-run1-pilot/collect/gen_0001/collect` | 8,000 | 761,979 | 2026112100 | pilot train book |
| gen2 | `rl-run1-gen2/gen2/collect/gen_0001/collect` | 8,000 | 796,341 | 2026112282 | pilot train book |
| succ_pilot | `deepfin-successor-pilot-20261124-v1/collection/gen_0001/collect` | 768 | 76,362 | 2026112444 | stage batch (book subset) |
| succ_cont | `deepfin-successor-continuation-20261124-v1/collection/gen_0001/collect` | 3,072 | 303,818 | 2026112445 | stage batch (book subset) |
| uho | `/mnt/deepfin-e-native/deepfin/uho-12ply-collect-20261124-v1/collection/gen_0001/collect` | 3,072 | 313,801 | 2026112447 | UHO 12-ply book |
| fresh_indep (HOLD) | `deepfin-fresh-collect-independent-20261124-v1/collection/gen_0001/collect` | 8,000 | 794,391 | 2026112443 | stage batch (book subset) |
| b01_r2 (HOLD) | `deepfin-fresh-collect-b01-recovery20261123r2/collection/gen_0001/collect` | 8,000 | 796,123 | 2026112442 | stage batch (book subset) |

Games and rows are each manifest's `counts`. For every bank, the game ledger (`games.jsonl`) agrees: distinct game
ids = games, the rows sum = rows, and the shard meta sidecars hold the same row count.

## Qualification checks (`census.json`)

**Provenance — identical for every bank except the pilot's actor:**
- collector pin `0c795fb7ab5a46e20848f6b0013de5a686fcb8ad`, `dirty` false;
- actor: `8dd6ac8c371d…` (`gen1_baked.pt`) for gen2 and all five banked collections, both in the manifest and as
  the sha256 of the bank's own `gen_0000/model.pt` today. The pilot's actor is the SL base `a39f900c…`.

**Recipe — no difference from generation 2's collection in any setting.** I diffed each manifest's
`config.settings` against gen2's. The diff skips only seed, game count and machine bindings (SF path, Syzygy path
strings, process count, concurrency, device, eval batch). Every bank's diff is empty. The shared recipe:
- 64 sims, topk 16, full-prob 1, c_scale 0.1, c_visit 50, tree reuse;
- temperature 0.5 → 0.1 over 10 plies, root noise scale 1.0, policy temp 1.0;
- max 450 plies;
- opening sampling `cycle`, book mix [0.95, 0.05];
- production Syzygy pair; `sf_nodes` 1 (no SF labels).

The SF and Syzygy path strings are the same production paths in every manifest. Adjudication follows from the shared
recipe; tablebase adjudication ends 53–55% of games in every bank (52.7% in the 768-game successor pilot).

**File integrity — every bank verifies.** Recomputed today:
- the sha256 of `games.jsonl` and of `starts.txt`;
- the shard tree, by the `sha256_tree` rule of `chess_anti_engine/simple_rl/common.py` at the pin.

All three equal the manifest's recorded values in all seven banks.

**Stock reader — every shard loads.** `reader_check.py` runs the pin's own `collect.gather_rows` on every shard of
every bank, all keys, one shard at a time. Checks:
- every row array has the meta sidecar's row count;
- floating arrays are finite;
- total rows read = manifest rows = ledger rows;
- the meta game ids = the ledger's game ids with rows > 0.

| Bank | Shards | Rows read | = manifest | = ledger | Game ids = ledger | Problems |
|---|---:|---:|---|---|---|---|
| pilot | 39 | 761,979 | yes | yes | yes | none |
| gen2 | 42 | 796,341 | yes | yes | yes | none |
| succ_pilot | 6 | 76,362 | yes | yes | yes | none |
| succ_cont | 18 | 303,818 | yes | yes | yes | none |
| uho | 18 | 313,801 | yes | yes | yes | none |
| fresh_indep | 42 | 794,391 | yes | yes | yes | none |
| b01_r2 | 42 | 796,123 | yes | yes | yes | none |

All 207 shards load, with the same 103 keys in every shard. One UHO game ended at its 12-ply start with 0 rows. It is
in the ledger and has no rows, which is why the game-id check counts ledger games with rows > 0.

**Opening identity.** An opening is the position after its book moves (EPD), from `starts.txt`.
- **Within a bank:** no repeated opening in any bank (`positions_repeated_within` 0).
- **Across banks:**
  - gen2 and the pilot share **430** opening positions. Both cycle through the same 153,842-line pilot train book
    under different seeds.
  - The UHO bank shares 1 position each with the pilot, gen2 and fresh_indep.
  - **Every other pair shares 0.**
  - The successor, continuation and both HOLD banks are disjoint from each other and from gen2.
- **The source book:** every opening id of all six non-UHO banks is a line of the pilot train book
  (`rl-run1-pilot/openings/train.jsonl`). They are one opening distribution. The UHO bank uses none of its lines.
- **How the disjointness was made:** each successor stage batch is a subset of that book. Its `plan.json` records the
  source book's sha (`1a98416d…`) and the exclusion sets it removed, among them the historical evaluation ids and
  every earlier bank's booked ids, each with 0 overlap. The census above re-derives the disjointness from the
  banks' own files rather than trusting the plan.
- **Game ids:** 0 shared between any two banks.
- **Against evaluation openings:** every non-smoke arena game log under the RL banks (`rl-run1-gen2`,
  `rl-run1-followup`, `rl-run1-pilot`, `rl-run1-value-anchor`): 34 logs, 71,216 games, 3,973 distinct opening
  positions, the 16-ply EVAL book. **No non-UHO bank shares any opening with them.** The UHO bank shares 1.
- **Against the frozen audit sets, as openings only:**
  - `data/audit_set_v1.jsonl` (4,000 positions): gen2 6, pilot 2, fresh_indep 3, b01_r2 1, the successor banks 0.
  - The 3,069 rl-audit positions (`RC/rl-audit/positions.npz`): 1–9 per bank.
  - rl-audit game ids shared with any bank: 0.

**The hold-out identity question.** The trainer's hold-out is per game: `is_holdout_game(game_id, 0.05)`. The
recalibration is fitted on those hold-out games.
- Game ids are unique across all banks, so pooling banks never moves a game between training and hold-out.
- An opening position in one bank's training rows can appear as the opening of another bank's hold-out game only
  through the gen2 ∩ pilot overlap (430 positions). The pilot is not admitted for training on the generation-1
  actor (below), so for the admitted pool that overlap is 0.
- Within the admitted pool, every opening is used by exactly one game.

**The legacy-ledger limitation.** `games.jsonl` records each game's opening, result, length and row count, but no
played moves (no played-move suffix). So:
- row positions cannot be reconstructed from the ledger;
- the audit-set comparison above is at the opening level only;
- a row-level check would hash the board planes of the stored `x` against the audit positions encoded the same way.
  It is not done here.

What this does and does not leave open:
- The audit sets are rulers. A row that equals an audit position would be a leak into a ruler's training data.
- At the opening level the overlaps are a handful of positions in banks of 8,000 games. Row-level is unmeasured,
  and the record says so.
- The comparison in the next record reads strength by matches on the EVAL book, which no admitted bank touches.

## The two HOLD banks: what failed, and whether the data is affected

Both collectors exited 0 and wrote a manifest. The HOLD comes from the collection wrapper's closeout guard, which
runs after the collector. Neither has a data-only receipt.

**fresh_indep** — `owner-terminal.json`: `"reason": "private output changed during disk census"`,
`lease_closed_verified` false, `own_descendants_gone` true.
- **What changed:** a leftover per-process staging directory, `collect/procs/01/`. It holds 651 MB of process 1's
  local shards (shard_00000 … 00013, some already missing their meta file). Its directory mtimes move past the
  manifest's (21:42:12): a deletion of its entries in progress during the census. No sealed file is newer than the
  manifest. The newest shard file under `shards/` is 21:39:21.
- **Whether it is sealed data:** no. `generation_shards` reads only `collect/shards/shard_*.zarr`, and `procs/` is
  outside every hash the manifest records.
- **Whether its games are lost:** no. Every game id in the 12 leftover meta files (2,416 games) is in the sealed
  shards and in the ledger. The leftover is a partially deleted copy of rows that were merged into the sealed set.

**b01_r2** — `"reason": "nonzero or descendant remains: stock-collect.log"`, `worker_rc` 0,
`own_descendants_gone` true.
- The collect log ends with the collector's normal final stats line: 8,000 games, 796,123 rows.
- Nothing under `collection/` other than the lineage cache and directory entries is newer than the manifest.
- **The failure is the wrapper's descendant check.** It is custody (was every process gone at closeout), not
  content.

**What would qualify them — evidence, not assertion.** A HOLD bank qualifies only if every item below is shown on
disk:
- **(E1)** the manifest-recorded sha256 of `games.jsonl`, `starts.txt` and the shard tree recompute equal. Shown
  for both.
- **(E2)** the stock reader loads every shard with consistent row counts and finite values, and the rows and game
  ids equal the manifest's and the ledger's. Shown for both (table above).
- **(E3)** no sealed file is newer than the manifest, and every path that is newer is identified as outside the
  sealed outputs. Shown for both: fresh_indep's `procs/01/`; for both, the lineage cache and directory entries.
- **(E4)** actor, pin and recipe are identical to the admitted banks'. Shown.
- **(E5)** the rows are not duplicated elsewhere: game ids and openings are disjoint from every other admitted bank.
  Shown.

**Proposed reading: E1–E5 hold, so both HOLD banks qualify.** The HOLD is a custody finding about the wrapper's
closeout. It does not touch the bytes the collector sealed at rc 0. Two judgements are the reviewer's:
- whether E1–E5 are enough evidence;
- whether to require the owner's word before admitting a bank whose wrapper refused it.

If either answer is no, both banks stay out and the next record runs on the other admitted banks (below).

## Admission

| Bank | Qualifies | Admitted for training on the generation-1 actor's data | Reason |
|---|---|---|---|
| gen2 | yes | **yes** | the generation-2 net was trained on it, and k = 1 of the volume comparison is that net |
| succ_pilot | yes | **yes** | clean terminal and data-only receipt; same actor, pin and recipe; disjoint openings and ids |
| succ_cont | yes | **yes** | the same |
| fresh_indep | yes, on E1–E5 (above) | **yes, if the reviewer accepts E1–E5** | custody HOLD only; data verified |
| b01_r2 | yes, on E1–E5 | **yes, if the reviewer accepts E1–E5** | the same |
| uho | yes (integrity, recipe) | **no, not in the first comparison** | a different opening distribution (UHO 12-ply, no shared book line). Mixing it in would add a second variable. It is the next single variable after volume. If it is ever admitted, its one opening shared with the EVAL book is excluded. |
| pilot | yes | **no, for generation-1-actor training** | collected by the SL base, a different actor. It stays the generation-1 training set |

**Admitted pool for training from `gen1_baked.pt` on its own data:** gen2 + succ_pilot + succ_cont = 1,176,521
rows (11,840 games, 1.48× gen2). With both HOLD banks: 2,767,035 rows (27,840 games, 3.47× gen2).

**Usage rules for admitted rows** (owner's rules; enforced by the training record's checks):
- each stored row is used once (`--presentations 1`), with uniform shuffled sampling across the pooled banks;
- at most two views including mirrors;
- the hold-out is per game and the same `is_holdout_game(game_id, 0.05)` rule in every bank;
- the recalibration is fitted on the pooled hold-out.
