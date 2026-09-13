# Bounded raw policy and native-WDL scale labels

Status: the bounded collector launched on **2026-09-13 at 17:50:11 UTC**, after the parent observed the preceding Ceres arena exit 0. This record captures launch only; no new group completion or training-ready rows are claimed. The parent owns execution 55179 and observer 609.

## Allocation and purpose

Prepare useful labels toward the larger corpus while the [registered V100 target rewrite](2026-09-13-native-bt4-value-endpoint.md) runs. This is an operational use of the existing qualified BT4 collector, not another teacher recipe or strength experiment. A metadata snapshot found **1,477,290 closed raw rows  / 178 shards** without sidecar receipts: 979,788 / 118 in run06 and 497,502 / 60 in run07. Their exact raw-name intersections with the accepted 20 G10 cohort rosters are empty within each source namespace. These are raw counts, not retained derived rows.

The allocation is **7200 seconds total, at most 192 newly recorded shard receipts, at most 16 per group**, one group at a time. Newly closed shards may enter the selection after the snapshot; actual group receipts establish coverage. There is no automatic extension. The first actual group has 1800 seconds including cleanup:1740 seconds for the collector and the existing cleanup margin. Later group entry uses its measured elapsed time per newly recorded shard, multiplied by the next count and 1.5, plus 60 seconds; teacher/batch settings remain fixed. The outer timeout is 7160 seconds TERM plus 40 seconds KILL, covering the inherited 30+5-second cleanup.

## Unchanged collector and resource controls

The runtime remains `/tmp/deepfin-bt4-label-runtime-pr580-overlay` at `8fd3940e60530aebdd7bd7f398cd04eb2679af1d`. One inference collects the original legal BT4 policy and native `/output/wdl` probabilities, batch 128, two threads and an 8 GiB ORT GPU allowance. CPU affinity is 2,3; available host RAM must be 48 GiB at startup and 32 GiB during collection, with 150 GiB free SSD. Sampled headroom and the ORT allowance are not hard total-RSS/device-memory limits.

The saved old bank contains 35,436,868 policy-only rows; this operation does **not** backfill them. Existing receipts must remain identical. Newly recorded coverage can include the unchanged collector's recovery of a published sidecar whose append receipt was missing, so receipt counts do not necessarily mean fresh inference. No `--verify-all` payload sweep is requested.

The fresh operator holds the old driver's coordination lock and checks actual same-user processes before collection; the recorded ownership check found no matching old driver or producer. The collector still enforces its writer and GPU leases. Old pause markers and PID files remain intact. A fresh parent-owned `STOP.request` yields at the next group boundary, with 15 seconds between groups to give waiting training priority. Hard faults clean the entire owned collector process group. A `.writing` remainder is preserved and refused on a later restart, requiring separate review.

## Actual launch evidence

The outer launch recorded 88,090,669,056 bytes available RAM and 214,444,892,160 bytes free disk. The operator started at 1789321811.9766617; the first collector group started at 1789321812.2052217, PID/PGID 411322, with the reviewed exact argument vector and maximum 16 shards. These are launch samples, not peak-memory or completed-throughput measurements. No active log, game bank, model or corpus payload was read for this publication.

Parent source/command review and independent review passed. Independent review `74ce1320987521e09924488519e646eaf6ebccdf54d52bb9a3a32128e9df67a1` includes the corrected outer cleanup grace. The plan is `bdca27264c2ca8d86f56b4b28a18c4a20504b30a16356a7ae024debdf72ef287`; command is `2e9964cc4a00151758a8a055cdf40f9e959f35156df14e8dc67a8ed1c83c3eca`.

[Compact evidence](evidence/2026-09-13-bounded-raw-scale-label-launch.json) preserves actual launch, ownership, first-group argv and original snapshot references. Later derivation and raw-to-derived identity qualification remain necessary before these labels can join a training corpus. Existing accepted G10 native coverage and frozen V100 implementation are unchanged.
