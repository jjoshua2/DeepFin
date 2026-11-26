# 2026-11-26: Banked collections — qualification and admission

**Status: for merge (revision 2, after the statistics review).** This record decides which banked collections may be
used as training rows for the next recipe comparisons. The coordinator admits both HOLD banks on the data evidence
below (E1–E6). Until this record merges, no bank listed here is admitted. It changes nothing live and trains nothing.
Every check is CPU-only and read-only on the banks.

**Evidence** (all under `RC/banked-admission-20261126/`; `RC/` is the evidence log's prefix):
- the census script `admission_census.py` [`df56474d`] and its output `census.json` [`318a05e2`] (revision 2 removed
  a self-check that could not fail; the re-run output is byte-identical);
- the stock-reader check `reader_check.py` [`eefbfbd1`] and its output `reader_check.json` [`506aef15`];
- the every-row content check, the statistics reviewer's scripts re-run into `bank_check/` (E6):
  - driver `run.sh` [`5593b9a4`], running `bank_check.py` [`bfc78db3`] per bank, then `ref_keys.py` [`46237d68`],
    `cross.py` [`19c256cf`] and `control.py` [`642c042e`] (the positive control, its directory argument
    re-pointed);
  - outputs: `gen2.npz.json` [`bacf27d2`], `succ_pilot.npz.json` [`2940cff1`], `succ_cont.npz.json` [`f7d24009`],
    `fresh_indep.npz.json` [`8d0a6db8`], `b01_r2.npz.json` [`f13340b2`], `cross.out` [`9c61e104`];
  - each bank's summary is identical to the reviewer's own run (`RC/review-1523/`).

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
- **Against the match book.** Every match reads `rl-run1-pilot/openings/eval.pgn` (sha256 `067675c2…`): 4,000
  16-ply lines (the same lines as `eval.jsonl`, `1d3d13ed…`), 4,000 distinct start positions.
  - Start level, all 4,000: **0** starts shared with any of gen2, succ_pilot, succ_cont, fresh_indep, b01_r2 (or the
    pilot). Recomputed against `eval.jsonl` directly.
  - Played subset: the 34 non-smoke arena game logs under `rl-run1-gen2`, `rl-run1-followup`, `rl-run1-pilot` and
    `rl-run1-value-anchor` (71,216 games) played 3,973 of the 4,000 lines; 27 lines were never played. No non-UHO
    bank shares an opening with them. The UHO bank shares 1.
  - Row level (any stored row whose board is one of the 4,000 eval starts, E6): admitted pool gen2 + succ_pilot +
    succ_cont **2 rows in gen2 (1 exact position + 1 colour-mirror twin)**; the HOLD banks fresh_indep 1 and b01_r2 2.
    A start position never begins a banked game; these are rows reached in play.
- **Against the frozen audit sets.** Opening level (census): `data/audit_set_v1.jsonl` (4,000 positions) gen2 6,
  pilot 2, fresh_indep 3, b01_r2 1, the successor banks 0; the 3,069 rl-audit positions 1–9 per bank. Row level
  (every stored row's decoded board, colour-canonical, E6):

| Bank | Rows | Rows on an eval start | Rows on an audit_v1 position (distinct) | Rows on an rl-audit position (distinct, games) | Rows with an rl-audit `x_hash` | rl-audit game ids |
|---|---:|---:|---:|---:|---:|---:|
| gen2 | 796,341 | 2 | 12 (11) | 31 (30, 14) | 26 | 0 |
| succ_pilot | 76,362 | 0 | 0 | 2 (2, 1) | 2 | 0 |
| succ_cont | 303,818 | 0 | 3 (3) | 8 (8, 4) | 8 | 0 |
| fresh_indep | 794,391 | 1 | 9 (9) | 22 (22, 6) | 18 | 0 |
| b01_r2 | 796,123 | 2 | 10 (10) | 28 (28, 12) | 23 | 0 |

  At most 31 rows in 796k (0.004%) in any bank. A ruler that scores those positions sees a handful of
  training-seen boards; the training record lists them for exclusion where a ruler is per-position.

**The hold-out identity question.** The trainer's hold-out is per game: `is_holdout_game(game_id, 0.05)`. The
recalibration is fitted on those hold-out games.
- Game ids are unique across all banks, so pooling banks never moves a game between training and hold-out.
- An opening position in one bank's training rows can appear as the opening of another bank's hold-out game only
  through the gen2 ∩ pilot overlap (430 positions). The pilot is not admitted for training on the generation-1
  actor (below), so for the admitted pool that overlap is 0.
- Within the admitted pool, every opening is used by exactly one game.

**The legacy-ledger limitation.** `games.jsonl` records each game's opening, result, length and row count, but no
played moves (no played-move suffix), so row positions cannot be reconstructed from the ledger. The row-level
comparisons above therefore decode each row's board from its stored `x` planes (E6) instead of the ledger. The
ledger alone supports only the opening-level checks.

**Cross-bank duplicate inputs (row level, E6).** Rows whose stored-input hash (`x_hash`, the meta sidecar's) also
appears elsewhere in the pool:
- within one bank: 0.003% (succ_pilot) to 0.095% (b01_r2) of rows;
- pairwise between banks: 0.004%–0.098% of A's rows;
- **pooled admitted 3 banks: 0.097% of 1,176,521 rows** on a repeated input (585 excess rows);
- **pooled 5 banks: 0.25% of 2,767,035 rows** (3,545 excess rows).

These are common positions reached independently in different games (game ids are disjoint), not copied rows.
Exactly-once is defined per stored row, so they are kept; at 0.25% they cannot move a strength reading.

**One variable: no recipe shift between banks.** Same actor, pin and settings (above). The realised outcome mix
agrees within sampling noise:

| Bank | Games | Rows / game | Decisive (± 95%) | White share of decisive | TB-adjudicated | Mean stored-policy entropy |
|---|---:|---:|---:|---:|---:|---:|
| gen2 | 8,000 | 99.54 | 50.26% ± 1.10 | 61.3% | 55.15% | 0.6628 |
| succ_pilot | 768 | 99.43 | 50.26% ± 3.54 | 59.6% | 52.73% | 0.6638 |
| succ_cont | 3,072 | 98.90 | 50.55% ± 1.77 | 61.9% | 54.52% | 0.6660 |
| fresh_indep | 8,000 | 99.30 | 51.02% ± 1.10 | 59.9% | 53.91% | 0.6657 |
| b01_r2 | 8,000 | 99.52 | 50.01% ± 1.10 | 61.6% | 54.30% | 0.6654 |

The largest gap, the TB-adjudicated share (gen2 55.15% vs fresh_indep 53.91%), is 1.6 SE of a difference of two
8,000-game proportions.

## The two HOLD banks: what failed, and whether the data is affected

Both collectors sealed a manifest (the last write of a collect stage). The HOLD comes from the collection
wrapper's closeout guard, which runs after the collector. Neither has a data-only receipt. Only b01_r2's collector
exit code is known (0).

**fresh_indep** — `owner-terminal.json`: `"reason": "private output changed during disk census"`,
`lease_closed_verified` false, `own_descendants_gone` true.
- **The collector was terminated after the seal; its exit code is unknown.** `owner-terminal.json` records no
  `worker_rc`; `stock-collect.log` ends without the collector's final stats line; `plan.json` and `collect/procs/01/`
  still exist. The collector was stopped during its own post-seal cleanup (`rmtree` of `procs/`).
- **What `procs/01/` is: hard links, not a copy.** All 52,033 files under it have link count 2, and every one is the
  same inode as a file of sealed shards `shard_00014` … `shard_00027` (process 1's 266,337 rows, renamed into the
  sealed set). The directory mtimes past the manifest's (21:42:12) are the interrupted `rmtree` removing entries.
  No sealed file is newer than the manifest; the newest file under `shards/` is 21:39:21.
- **Whether it changes sealed data:** not as it stands. `generation_shards` reads only `collect/shards/shard_*.zarr`,
  `procs/` is outside every hash the manifest records, and the sealed tree's sha256 recomputes equal (E1).
- **The hazard it leaves.** Because those files share inodes with sealed shards, an in-place write to any file under
  `procs/01/` changes the sealed bytes. So: never run `--redo` or any writer on that collect directory, and never
  open its files for writing. Removing `procs/01/` (unlinking) would not change the sealed bytes, but removal is the
  owner's call; this record does not remove it. Every training stage re-verifies the shard sha256 (lineage), so a
  change would refuse the stage rather than train on altered rows.

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
- **(E6)** every row's content is consistent: for every stored row, the board decoded from `x` regenerates a legal
  move set equal to the stored `legal_mask`; the stored policy has no mass outside it; (game, ply) is unique and
  contiguous per game with the ledger's row count; and each played move leads to the next row's position.

| Bank | Rows | `legal_mask` equal | Policy mass outside legal | Continuity checked / breaks | Dup (game, ply) | Games rows ≠ ledger |
|---|---:|---:|---:|---:|---:|---:|
| gen2 | 796,341 | 796,341 | 0 | 776,894 / 0 | 0 | 0 |
| succ_pilot | 76,362 | 76,362 | 0 | 74,464 / 0 | 0 | 0 |
| succ_cont | 303,818 | 303,818 | 0 | 296,498 / 0 | 0 | 0 |
| fresh_indep | 794,391 | 794,391 | 0 | 774,749 / 0 | 0 | 0 |
| b01_r2 | 796,123 | 796,123 | 0 | 776,859 / 0 | 0 | 0 |

  2,767,035 / 2,767,035 rows equal. Positive control: every bank's 8,000 (or 768 / 3,072) start positions are found
  among its rows, so the decode-and-key path does find positions that are there. E6 is the statistics reviewer's
  check; it is re-run into this record's evidence directory (Evidence, above).

**Reading: E1–E6 hold, so both HOLD banks qualify, and the coordinator admits them on that evidence.** The HOLD is a
custody finding about the wrapper's closeout, not a content finding. fresh_indep keeps the `procs/01/` hazard above.

## Admission

| Bank | Qualifies | Admitted for training on the generation-1 actor's data | Reason |
|---|---|---|---|
| gen2 | yes | **yes** | the generation-2 net was trained on it, and k = 1 of the volume comparison is that net |
| succ_pilot | yes | **yes** | clean terminal and data-only receipt; same actor, pin and recipe; disjoint openings and ids |
| succ_cont | yes | **yes** | the same |
| fresh_indep | yes, on E1–E6 | **yes** | custody HOLD only (collector terminated after the seal, rc unknown); data verified row by row. Keep the `procs/01/` hard-link hazard: no writer on that directory |
| b01_r2 | yes, on E1–E6 | **yes** | custody HOLD only (descendant check, worker rc 0); data verified row by row |
| uho | yes (integrity, recipe) | **no, not in the first comparison** | a different opening distribution (UHO 12-ply, no shared book line). Mixing it in would add a second variable. It is the next single variable after volume. If it is ever admitted, its one opening shared with the EVAL book is excluded. |
| pilot | yes | **no, for generation-1-actor training** | collected by the SL base, a different actor. It stays the generation-1 training set |

**Admitted pool for training from `gen1_baked.pt` on its own data, once this record merges:** gen2 + succ_pilot +
succ_cont + fresh_indep + b01_r2 = 2,767,035 rows (27,840 games, 3.47× gen2). Without fresh_indep: 1,972,644 rows
(2.48×).

**Usage rules for admitted rows** (owner's rules; enforced by the training record's checks):
- each stored row is used once (`--presentations 1`), with uniform shuffled sampling across the pooled banks;
- at most two views including mirrors;
- the hold-out is per game and the same `is_holdout_game(game_id, 0.05)` rule in every bank;
- the recalibration is fitted on the pooled hold-out.
