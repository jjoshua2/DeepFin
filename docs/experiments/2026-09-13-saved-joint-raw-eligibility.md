# Saved joint raw-label eligibility expansion

Status: audit registered; selected raw rows have not been audited or admitted.

A stable metadata snapshot of the run06/run07 BT4 receipt indexes contains 59,264,839 policy-labeled raw rows, including 23,827,971 with named native WDL receipts. After excluding the actual 35,314,577-row combined corpus raw rosters and the newly derived 184-shard cohort, 2,579 shards / 21,404,459 raw rows remain as candidates. This is saved-label reuse, not new inference. The 35,436,868 policy-only rows are excluded.

The next audit selects the first 384 run06 and 128 run07 candidate receipts in saved index order, preserving source order run06 then run07: 512 shards / 4,249,935 raw rows (3,187,400 + 1,062,535). No growing inventory entries may enter. Source directory/config identities qualify shard names; the exact accepted-roster exclusion does not establish unique board positions or absence from every unadmitted historical experiment.

The allocation is 7,200 seconds inclusive, CPU 8–9, two threads, GPU hidden, 48 GiB available RAM at startup and 32 GiB while running, 150 GiB free-disk floor and 1 GiB metadata-output ceiling. There is no address-space cap because the inherited selector imports Torch without constructing a model. The planned internal deadline is at most 7,140 seconds and the outer timeout includes cleanup. No automatic extension, derivation, new labels, value mixture or training is allocated.

The existing audit now accepts a separately named pinned receipt selection. Every selected receipt must exactly match its source-qualified pinned JSONL snapshot; duplicate, missing or changed members fail. It does not fabricate a collector-completion receipt. The original collection path and 192-shard/1,740-second defaults remain. Explicit bounds permit this registered 512-shard audit without changing phase-zero uniform-d9 policy or latest-phase composite value selectors.

Actual execution must preserve source/physical-offset diagnostics, distinguish no-result rows from required baseline rejections, and report whole-shard collateral valid rows. Partial progress or failure is not eligibility admission. Future derivation depends on the actual audit and a separate resource decision. The prior 184-shard audit took 1,300.442 seconds for 1,527,153 raw rows; linear scaling suggests about 60 minutes here, not a throughput guarantee.

[Compact evidence](evidence/saved-joint-raw-eligibility-20260913.json) pins the inventory, exact selection and disjointness assessment without copying giant receipt rosters into main. The original raw shards, sidecars and existing training corpora remain unchanged.

The final selection manifest passed the actual metadata-only admission function for all 512 receipts / 4,249,935 raw rows. It explicitly binds source IDs to snapshots; the original draft is retained. This check did not read raw shards or establish baseline eligibility.
