# Three-source selected-target diagnostic pipeline

The bounded BT4, Ceres and Stockfish-origin replay now has a complete,
independently checked **83,416-winner diagnostic pack**. This closes the
small-slice source-to-selected-target-to-trainer-format qualification requested
by the [500M practical plan](2026-09-29-500m-practical-next-tests.md). It does
not admit these rows to the production corpus, measure playing strength, or
establish a month-scale 500M throughput rate.

## Scope and result

The engineering question was whether this archived three-source slice could
pass strict history/outcome replay, exact-byte dedup, one source-independent
chosen-neural route per winner, corrected target binding and real trainer
collation. The previously banked B/C heads and production loader were the
fixed inputs; this was not a competing label-recipe or strength arm. Reviewed
one-shot packets set exact SHA-256 input pins, bounded CPU/GPU resources and
fail-closed gates before each physical stage. The deciding rule was zero
unexplained row/feed/target/array mismatches and independent all-row PASS at
each handoff. A failed output root was spent with zero credit and would require
a fresh source-reviewed successor. These are exact qualification checks, so
there is no sampling interval or Elo threshold.

The CPU replay consumed 83,991 archived rows with their source-qualified UIDs,
game histories and terminal outcomes. Exact input-byte dedup removed 575 rows,
leaving 83,416 winners: 27,305 BT4-origin, 31,468 Ceres-origin and 24,643
Stockfish-origin. The independent replay pass rechecked all source histories,
canonical input rows, winner selection and outcome provenance. There were 304
duplicate groups with outcome variation; the selected outcome remains tied to
the chosen source row.

Each winner routes to one chosen neural teacher by the same source-independent
UID rule. The 58,773 neural-origin winners joined existing banked BT4/Ceres
heads by exact UID. The 24,643 Stockfish-origin winners required new inference,
including 29 rows with the same input content as a previously banked row but a
different UID. Keeping those 29 in the missing-head roster preserved exact
source/UID/feed provenance. The resulting teacher split is 41,782 BT4 and
41,634 Ceres winners.

The new Stockfish-origin head pass used the qualified BT4 physical-128 and
Ceres physical-512 profiles, serially on the sole GPU. It made 97 BT4 calls
with 40 repeat-last pad slots and 24 Ceres calls with 21 pads. The supervised
pass took **62.252 seconds** for 24,643 rows, including charged preflight and
first-call proof work. An independent CPU pass reconstructed all physical
feeds, checked every raw-head row/offset and verified actual CUDA neural
profile events. The corrected binder then converted every chosen raw head
into sorted compact legal-policy FP16 values and the main FP16 WDL. A separate
direct all-row readback recomputed the target bytes from raw heads. These batch
profiles are numerical choices; this run does not claim byte equivalence to a
different batch profile.

The real trainer-format writer produced **163 ZIP_STORED Zarr shards** of at
most 512 rows: **49,581,976 bytes, or 594.394 bytes per unique winner**. The
physical pack took 137.049 seconds. Its independent 118.607-second readback
checked all 83,416 winner/target joins and all 15 stored arrays against the
source rows, including legal masks, selected policy, search WDL, terminal
outcome from the side to move, source-qualified game IDs, and actual CPU
collation through the production loader. Both pack and audit are diagnostic
with zero corpus/ingest/owner/outcome credit.

| Stage | Measured scope | Wall time |
| --- | --- | ---: |
| Strict archived-source CPU replay | 83,991 gross rows, 83,416 winners | 937.209 s |
| Missing chosen-neural-head GPU pass | 24,643 SF-origin rows, two serial profiles | 62.252 s |
| Independent raw-head/feed CPU audit | 24,643 rows, all 121 calls | 29.091 s |
| Corrected SF-origin target CPU conversion | 24,643 rows | 34.590 s |
| Full trainer-format pack | 83,416 winners, 163 ZIPs | 137.049 s |
| Independent full-pack readback | 83,416 rows, all 15 arrays | 118.607 s |

These are different stage denominators and should not be added into a
steady-state production throughput figure. Neural-origin heads were already
banked; the 62.252-second GPU pass covers only missing Stockfish-origin heads.
The previous local CPU loader screen consumed the **58,773-row B/C subset**
at 46,665 and 46,958 rows/s on two passes. It does not measure the new
full pack, external storage, GPU transfer or training updates. External-drive
consumption of this pack remains untested.

## Retention and scale boundary

The three selected source-archive sets occupy 261.20 MB in this slice, or
3,131 bytes per unique winner. Those are whole physical archives and include
unselected members and losing rows; they are not a winner-only observation
bank. The flat float32 staging geometry is 44,800 bytes per row and is a
diagnostic transport format, not a retained 500M design.

The selected-neural target ZIP is measured at 594.394 bytes per winner in
this small mixed-source slice. Scaling only that term unchanged to 500M
positions gives about 297.2 GB. It excludes raw observations, source proof,
indexes and a compact chosen-teacher raw legal-head/value bank needed for
future target retransform. A physical B/C-only diagnostic bank preserved
native legal logits and value heads for 58,773 winners. Its legal-head payload
was 8,468,846 bytes (144.09 bytes/row) before compression and 5,749,282
bytes (97.82 bytes/row) at zstd level 5. Its verbose JSON row index was a
further 43,931,297 bytes (747.47 bytes/row), so the two retained files
totaled 49,680,579 bytes (845.30 bytes/row). The producer reconstructed the
same corrected FP16 target bytes for every row; independent compact-bank
readback is a separate gate. The index is a diagnostic provenance format,
not a scale-ready compact index.

A worksheet's roughly 2.17 TB conditional subtotal extrapolates current
whole archives, the earlier B/C selected-ZIP rate and measured metadata
compression. It omits
the compact raw-head bank and assumes unchanged source mix and compression,
so it is **not a complete 500M storage forecast**. Extrapolating just the
observed zstd legal-head payload gives about 48.9 GB for 500M rows, before a
qualified index, headers, source-mix changes or redundancy.

The 512-row ZIPs here are a controlled diagnostic. At 500M rows this layout
would create about 976,563 files. A production 8,192-row container with
internal 512-row locality is the next format measurement; no such container
is qualified by this pack.

## Decision and evidence

The decision for this slice is **pipeline qualification PASS, corpus admission
HOLD**. Exact source replay, selected-target bytes and physical trainer-format
arrays each have independent all-row checks. No matched training run or arena
used this pack, so no Elo or source-mixture strength conclusion follows. The
next production gates are a durable compact raw observation format, a larger
shard and actual external-loader screen, and an independently registered
matched training/arena comparison on a representative unique-position bank.

The [compact evidence manifest](evidence/2026-09-29-tri-source-selected-target-pipeline.json)
records source packet, producer receipt and independent-audit SHA-256 identities,
stage geometry, measured resources and the limits of each measurement. Bulk
source archives, raw-head tapes and ZIP shards are not included in this PR.
