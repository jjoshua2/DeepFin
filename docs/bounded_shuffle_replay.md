# Bounded shuffle replay

`BoundedShuffleReplay` is an opt-in host replay backend. It reads checked sources sequentially, copies rows into a fixed-size shuffle buffer, and optionally prefetches a few FIFO batches. It does not change `StoredRowOnceReplay` or the current globally uniform learner.

## Order and row accounting

The order is explicitly `APPROXIMATE_BOUNDED_SHUFFLE_V1`. The backend fills a fixed number of slots, emits a random slot, replaces it with the next upstream row, and drains the buffer at EOF. Every stored row appears once, including the real final partial batch. This is an approximate shuffle rather than a globally uniform permutation. Producer timing, decoded sparsity and queue occupancy do not affect selection order.

Prefetch does not spend row quota. `sample_batch_arrays` calls the durable reservation callback before returning payload to the consumer. Original and mirrored inputs share one presentation. `BoundedReservationWriter` connects those records to `QuotaLedger` and a draw journal; its write-ahead reservation remains spent if later work fails.

A returned batch is borrowed. After successful use, the caller must release all NumPy and CPU tensor aliases before calling `complete_batch()`. On a preparation, forward, backward or optimizer error, call `abort_consumption()` and preserve the tainted attempt. A tainted replay cannot publish a clean checkpoint or redraw the failed batch.

## Memory limits

`ReplayLimits` bounds source/decode workspace, shuffle rows, ready batches, consumer assembly and scratch storage. Their declared byte caps must fit within `total_bytes`; there are at most four ready batches. Fixed header-derived maximum row charges keep selection independent of actual sparsity. Insufficient caps fail rather than dropping rows or silently shrinking the shuffle window.

`SequentialArraySource` caches at most one decoded source and drops it before decoding its successor. Returned rows own their storage, so a small view cannot retain an entire shard. Array ownership checks charge the underlying allocation. Consumer assembly reserves its declared workspace before allocating.

These limits describe active replay storage and declared workspaces. They do not bound whole-process RSS: parsed manifest graphs, model/framework memory, allocator caches and OS page cache are separate. The backend does not imply a 1 GiB process-memory limit.

## Checkpoint and resume

Finish and release the borrowed batch before calling `checkpoint()`. The checkpoint joins the producer at a batch boundary and records source cursor, ordered buffer IDs, FIFO ready IDs, spent-row state, reservation chain, origin identity and separate shuffle/mirror RNG states. Ready rows remain unspent. Decoded payloads are discarded at the barrier.

Before `restore`, authenticate immutable checkpoint and journal bytes, strictly parse the reservation records, and run `audit_selection`. This audit reconstructs symbolic row selection without reading payloads and checks the complete consumed and queued frontier. Use the authenticated journal count/prefix and `QuotaLedger` parent-tail check when restoring. Payload reconstruction uses saved row IDs without another selection draw or reservation. A resumed run may reopen sources; an uninterrupted pass decodes each sequential source once, while inventory checks may read its bytes more than once.

A training checkpoint must bind this replay state together with its actual model, optimizer, augmentation RNG and source identities. The existing global-permutation journal and closed-model validators do not accept the bounded-order family.

## Integration and validation

The finite learner is `scripts.core_selfplay_union_bounded_shuffle`, with passive saved-model observation in `scripts.core_selfplay_union_bounded_shuffle_proof`. Its contract/checkpoint family is `core_selfplay_union_bounded_shuffle_v1`; it uses the checked fixed-union manifest and explicit replay limits. Existing globally uniform entry points and defaults remain unchanged.

Trainer integration is opt-in through the borrowed-row-batch interface. It requires single-microbatch, serialized training and disables Trainer batch prefetch/host overlap. Successful optimizer work marks the batch complete; the batch generator releases host-byte credit after dropping its final tensor aliases. Faults abort consumption and taint the attempt. The saved boundary binds actual model/optimizer updates, immutable journals and ledger prefix, bounded cursor, source identities, and private shuffle/mirror plus training RNG states. Closed lineage validation alone does not establish a saved model; the model-delivery validator also checks the own-live saved-payload certificate.

The two test modules contain 24 synthetic cases covering row conservation, partial tails, queued resume, RNG/order continuity, byte ownership, backpressure, durable reservations, failure taint and symbolic journal audits. They run through the repository's normal CPU validation suite.

The learner integration adds 14 Tiny model/Trainer cases covering continuous-versus-resumed state, actual partial tails, queued-but-unspent rows, CPU alias lifetime, failure taint, tampered continuation and saved-model proof.
