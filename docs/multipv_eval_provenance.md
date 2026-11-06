# MultiPV evaluation provenance

A bounded rank-1 update can leave the root value at an earlier exact A40 while
the latest policy candidates are B10/A35. These are different retained evidence
sets from the same search. Comparing the value to policy row 0 rejects this
legitimate result. The policy candidate list must not be made stale just to
satisfy that comparison.

## Additive evidence contract

`StockfishResult.eval_witness` captures the actual move and CP/mate from an
unbounded, scored rank-1 observation with a PV. Each search has a fresh
accumulator. A scored rank-1 line without a PV invalidates the witness; a later
PV-only line cannot reconstruct it. Bounds never supply exact witness values.

The label attachment checks the witness move against the same legal action set
as the policy candidates. An illegal/unmappable identity remains explicitly
invalid evidence, rather than becoming an absent field. No policy row or root
scalar is modified.

`sf_eval_witness` is an optional int32 vector with its own presence flag:

1. format version, currently 1;
2. root action in the row's policy encoding;
3. CP, clipped like `sf_label_meta`, with -32768 for absent;
4. mate, clipped like `sf_label_meta`;
5. score kind: 1 for CP, 2 for mate, including observed mate zero;
6. policy width, 1858 or 4672;
7. CRC31 over the preceding six canonical little-endian int32 values;
8. CRC31 over canonical non-padding raw policy rows in stored order.

CRC31 is `zlib.crc32(bytes) & 0x7fffffff`. Checksums detect accidental corruption,
not malicious rewriting or cryptographic authenticity. The policy checksum
preserves detection of raw-row score/identity corruption even though the root
value and newest policy score can legitimately differ.

Encoding and mirror transforms verify the original witness/raw pairing before
updating action indices, encoding width and checksums. Invalid input becomes an
explicit invalid marker and is never resealed as valid. Narrowing serialization
cannot turn fractional or overflowing evidence into a valid witness.

## Readers and persistence

- The selfplay record, its NPZ resume payload, and pending shallow-result resume
  carry the evidence. Pending result metadata binds its witness and ordered PV
  list with an integrity checksum. Presence flags and integrity inputs are
  checked before narrowing or absence conversion can erase invalid evidence.
- Native sparse finalization transforms the policy block as before, then verifies
  and remaps its paired witness into the same stored encoding.
- Replay sample conversion, optional shard fields, sealed Zarr/NPZ I/O and both
  sample and batch mirror paths preserve the evidence.
- The live health predicate and `sf_eval_pv_orphan_flags` use the same witness
  validation. CAST continues to consume the shared orphan predicate.
- Only rows in the existing checked population are classified. A valid new
  witness requires scalar agreement, valid identity/encoding and both integrity
  checks. Present-but-invalid evidence rejects; absent evidence keeps the exact
  legacy top-PV comparison. Mixed legacy/new batches are supported.

Shard version 2 and resume version 3 remain unchanged: the field is optional and
additive. No existing data is rewritten or migrated. Older readers ignore the
new evidence and can still conservatively reject benign reranks; older writers
may drop it. Updating a producer alone is therefore not a claim that all
consumers are compatible. Adoption and rollout remain separate decisions.

## Limits

The witness proves CP/mate provenance within the generating code path, not that
Stockfish answered the right position. A wrong-position answer whose root move
is coincidentally legal remains a blind spot. WDL is not authenticated. A
coordinated transplant of an entire valid witness/policy pair into an equal-score
record can pass. These checks must not be described as a general corruption or
security guarantee, or used to admit an existing corpus without separate review.

The regression tests exercise synthetic observations through actual native
finalization, replay conversion, sealed Zarr write/read and CAST. They also pin
live-policy/rebuild parity, legacy fallback, resume, mirroring, corrupted values,
identities, provenance and raw policy rows. No real-engine occurrence rate,
training effect or playing-strength claim follows from these tests.
