# Saved joint raw-label eligibility expansion

Status: the 512-shard audit and actual exclusion-loader qualification completed. Derivation of 4,219,426 eligible rows launched; completion and training admission remain pending.

A stable metadata snapshot of the run06/run07 BT4 receipt indexes contains 59,264,839 policy-labeled raw rows, including 23,827,971 with named native WDL receipts. After excluding the actual 35,314,577-row combined corpus raw rosters and the newly derived 184-shard cohort, 2,579 shards / 21,404,459 raw rows remain as candidates. This is saved-label reuse, not new inference. The 35,436,868 policy-only rows are excluded.

The next audit selects the first 384 run06 and 128 run07 candidate receipts in saved index order, preserving source order run06 then run07: 512 shards / 4,249,935 raw rows (3,187,400 + 1,062,535). No growing inventory entries may enter. Source directory/config identities qualify shard names; the exact accepted-roster exclusion does not establish unique board positions or absence from every unadmitted historical experiment.

The allocation is 7,200 seconds inclusive, CPU 8–9, two threads, GPU hidden, 48 GiB available RAM at startup and 32 GiB while running, 150 GiB free-disk floor and 1 GiB metadata-output ceiling. There is no address-space cap because the inherited selector imports Torch without constructing a model. The planned internal deadline is at most 7,140 seconds and the outer timeout includes cleanup. No automatic extension, derivation, new labels, value mixture or training is allocated.

The existing audit now accepts a separately named pinned receipt selection. Every selected receipt must exactly match its source-qualified pinned JSONL snapshot; duplicate, missing or changed members fail. It does not fabricate a collector-completion receipt. The original collection path and 192-shard/1,740-second defaults remain. Explicit bounds permit this registered 512-shard audit without changing phase-zero uniform-d9 policy or latest-phase composite value selectors.

Actual execution must preserve source/physical-offset diagnostics, distinguish no-result rows from required baseline rejections, and report whole-shard collateral valid rows. Partial progress or failure is not eligibility admission. Future derivation depends on the actual audit and a separate resource decision. The prior 184-shard audit took 1,300.442 seconds for 1,527,153 raw rows; linear scaling suggests about 60 minutes here, not a throughput guarantee.

[Compact evidence](evidence/saved-joint-raw-eligibility-20260913.json) pins the inventory, exact selection and disjointness assessment without copying giant receipt rosters into main. The original raw shards, sidecars and existing training corpora remain unchanged.

The final selection manifest passed the actual metadata-only admission function for all 512 receipts / 4,249,935 raw rows. It explicitly binds source IDs to snapshots; the original draft is retained. This check did not read raw shards or establish baseline eligibility.

## Actual audit launch

The parent launched the frozen 512-shard audit at **2026-09-13T23:05:57.634636+00:00** (exec **89943**, sole observer **849**). The supervisor recorded its start at **23:05:57.706510 UTC**. One saved process observation records audit PID **550756**, exact requested and observed argv, the intended runtime directory, and `complete=false`. This establishes invocation of the audit stage, not its completion or eligible-row count.

The actual plan is `a138c9e7…`, command `8a38a750…`, and independent operator review `e05bdddb…`; all 14 preparation input pins passed. The source was published in PR #742 (merge `111c8c5b…`), while execution retains the frozen `/tmp/deepfin-saved-raw-audit` runtime `2d2b0215…`. The plan’s preparation-only status is historical; the separate saved launch receipts record actual execution.

The audit retains the exact **512 shards / 4,249,935 raw rows**, CPU **8–9**, two threads, hidden GPU, nice 19 and idle I/O, **48/32-GiB startup/running available RAM**, **150-GiB disk floor**, and **1-GiB aggregate allocated metadata ceiling**. The auditor’s diagnostic ceiling is separately 512 MiB. The **7,200-second inclusive** allocation uses outer TERM at 7,160 seconds with 40 seconds for cleanup and a shared internal deadline of at most 7,140 seconds. The saved child received a 7,094-second hard allowance within that deadline. Existing STOP checks and authenticated owned-process cleanup apply. Sampled RAM and output checks are bounds with sampling limitations, not measured peak-use guarantees.

[Compact launch evidence](evidence/saved-joint-raw-audit-launched-20260913.json) preserves the start receipts, source/selection pins and single saved process observation. No audit result, derivation, new teacher evaluation, value mixture or training admission is reported at this snapshot. The active training and SF generation remain separate.

## Completed audit

The audit exited **0** after **3,784.638 seconds** (2026-09-14 **00:09:02.272221 UTC**), within its original 7,200-second allocation. Independent compact review checked the exact ordered 512-shard receipt roster, all saved diagnostic identities and count arithmetic, without rereading raw payloads or rerunning the audit.

| Source | Raw shards | Physical rows | No result | Required baseline exclusions | Eligible rows | Eligible rows lost by whole-shard rejection |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| run06 | 384 | 3,187,400 | 21,622 | 54 | 3,165,724 | 205,832 |
| run07 companion4 | 128 | 1,062,535 | 8,799 | 34 | 1,053,702 | 132,149 |
| Total | 512 | 4,249,935 | 30,421 | 88 | 4,219,426 | 337,981 |

The 88 required exclusions comprise **87 rows** with both a phase-zero support `CorpusIntegrityError` and composite-value `invalid_roster`, plus **one row** that passes the policy check but fails the composite-value roster check. For the first 87, the composite error does not establish a later-phase defect. The remaining row establishes a separate composite failure, but the saved diagnostics do not identify its exact malformed block. Two no-result rows also have value errors, including one with a policy error. Thus the overlapping totals of 88 policy and 90 value errors are not 178 exclusions. No-result rows remain a separate category.

Whole-shard rejection would retain only **3,881,445** eligible rows, discarding another **337,981** eligible rows with defective neighbors. Source-qualified physical-offset exclusion drafts preserve the exact 54 run06 and 34 run07 required exclusions. At the audit-completion snapshot, the exclusion loader accepted only the legacy collector-completion route with at most 192 shards, so these drafts were not yet runnable derivation admission. The subsequent reviewed extension and actual qualification below resolve that limitation without fabricating a collector receipt or weakening a selector.

[Compact completion evidence](evidence/saved-joint-raw-audit-completed-20260914.json) pins the actual completion, diagnostics, independent review and both source-specific selection/exclusion drafts. The result establishes eligibility of physical source rows, not unique chess positions, independent games, a materialized corpus or training admission. No new teacher evaluation was performed.

## Qualified exclusion loader and actual derivation launch

PR #750 (merge `9fc2935f…`) added the explicit saved-receipt route up to 512 shards by reusing the audit’s existing metadata admission. It checks exact ordered source identities and retains source-manifest and receipt-snapshot pins through binding; the legacy collection route remains capped at 192. Independent source review passed, as did 21 focused fixtures, scoped types and Ruff. Whole-project Ruff/Vulture passed; default full type discovery lacked dependencies and reported 1,861 errors / 1,143 warnings, retained in validation evidence. The frozen runtime is `6b6893bd…`; previously running runtimes were not edited.

One actual metadata-only load of both final exclusion manifests exited **0 in 2.087432 seconds**, confirming exactly **54 / 34** source-qualified physical exclusions under the reviewed contract. The synthetic writer tests separately demonstrated preservation of clean rows. Maximum RSS was **263,308 KiB**, with zero swaps. This check read saved audit/receipt metadata, not raw payloads, and did not derive data.

The parent then launched the two-source derivation at **2026-09-14T00:36:48.338446+00:00** (exec **34792**, sole observer **935**); the supervisor recorded its start at **00:36:48.487737 UTC**. The plan `a0f8b648…`, command `ea5394d4…` and independent operator review `e9a683ed…` bind the actual loader qualification and source/exclusion manifests. This is an actual invocation, not a completed derivation.

The expected outputs are **3,165,724 run06 rows / 387 shards** and **1,053,702 run07 rows / 129 shards**, totaling **4,219,426 / 516**. Each source uses one worker, sequentially, with two threads on CPU **8–9**, nice 19 and idle I/O. The unchanged deriver uses uniform-d9 phase-zero policy, latest-phase search value, temperature 0.0005, floor 0, seed 0 and 8,192-row output shards. Original raw physical identities remain in provenance; 30,421 no-result drops are separate from the 88 audited exclusions. No GPU or teacher inference is used.

The allocation is **10,800 seconds inclusive**, with a shared 10,740-second internal deadline and outer TERM at 10,760 seconds plus 40 seconds of cleanup. Each stage gets only remaining time minus a 45-second reserve; failed outputs and any completed prefix remain recorded without automatic extension. The prior 1,517,925-row derivation took 2,016.376 seconds, suggesting roughly 5,605 seconds here, not a guarantee.

The operator requires **48/32 GiB startup/running available RAM**, retains the **150-GiB running free-disk floor**, and samples an **8-GiB aggregate output cap** including partials and metadata. Two named prior 8,192-row shard metadata samples imply roughly 2.73 GiB for the expected outputs; this is not a full-output measurement. Startup reserves 150 GiB plus 8 GiB output, 4 GiB other writers and only the remaining potential SoftSF staging growth. Actual free space was **176,176,648,192 bytes**, above **175,474,421,760 required bytes**; the supervisor recorded **1,528,246,272 bytes** of remaining SoftSF allowance. Already allocated staging bytes are not counted twice. The completed Ceres SDK lane has no additional reservation. Sampled guards are not instantaneous peak guarantees.

[Compact qualification and launch evidence](evidence/saved-joint-exclusion-derive-launched-20260914.json) preserves these receipts. A completed derivative and its later qualifications are still required before downstream use or training admission.
