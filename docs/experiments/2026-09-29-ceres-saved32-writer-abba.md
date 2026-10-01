# Ceres saved32 proof-reuse writer comparison

**Readout, 2026-09-29.** On the same 32 saved games and 4,751 rows, a one-use proof writer reduced mean actor-to-sink publication time from **7.312217 to 4.790848 seconds**, a **2.521369-second (34.4816%)** saving across eight CPU-only arms. All arms passed their full semantic ZIP and strict terminal readbacks. This measures the saved32 writer interval, not live source generation, complete strict/physical readback, corpus admission, or 500M throughput.

## Frozen comparison

Control A used the executed whole-shard ZIP writer: it fully verified the private archive before rename, fully verified the published archive in the ledger, and read the final tail again. Candidate B retained the first full semantic verifier, bound its result to the archive and exact receipt, then rehashed the renamed regular ZIP through a no-follow descriptor before ledger credit. The one-use proof was consumed for both normal and tail publication. The independent post-arm full reader remained in place.

The reviewed packet ran A/B/B/A followed by B/A/A/B in fresh processes. Each arm replayed the same pinned saved records through the actual buffered actor finish, queue, flush, ledger observation, and finalization. Each published two normal shards plus one tail under a 2,048-row shard cap. CUDA was hidden; the cooperative GPU lease, cores 14 and 16, nice 19, 300-second arm cap, 2,400-second total cap, 3 GiB peak RSS cap, and 128 MiB output cap per arm were enforced. The timed interval excluded saved-record reconstruction and post-arm readback; the 102.558-second supervisor wall includes that work and process overhead.

| Block | Control A mean | Proof writer B mean | A − B |
| --- | ---: | ---: | ---: |
| A/B/B/A | 7.278083 s | 4.767993 s | 2.510091 s |
| B/A/A/B | 7.346351 s | 4.813703 s | 2.532648 s |
| All arms | 7.312217 s | 4.790848 s | 2.521369 s |

All eight arm receipts report 4,751 rows, 32 completed games, zero discards, 15 natural and 17 Syzygy terminal redecisions, and full ZIP semantic readback. Their seven retained logical-array hashes, row/game metadata hashes, buffered receipts, normalized ledger receipts, and normalized terminal hashes match. The independent output audit reopened all **24 ZIPs**, checked CRC and physical SHA, decoded all seven arrays and row/game metadata, and reconciled receipts and timing. It did **not** repeat the raw-fixture comparison or real strict Syzygy redecisions performed inside each arm. The 24 physical ZIP hashes differ across fresh writes; equality here is of decoded data and normalized publication records, not compressed archive bytes.

Peak arm RSS was 884,858,880 bytes; each arm wrote about 3.49 MB, below its cap. The run passed its saved32 diagnostic gates, but eight serial arms on one small bank provide no formal uncertainty interval or representative live-stage timing. The current full-prefix source still pins the baseline writer. A production gain needs a separately reviewed integration into that source and a matched full-prefix run with complete strict and physical-ledger readback. **Source, owner, corpus, and 500M capacity credit remain zero.**

The path-free [compact evidence](evidence/ceres-saved32-writer-abba-20260929.json) has SHA-256 `1968d197419b8cef68bbd93acc01f856bad982b0db3a47d06bc5150130b32ccf`. The retained supervisor terminal is `cf531f0c3cde4263d8b23b22ee413dbf5d86584f3e45a133ecfcd97da62fda02`; the independent output audit is `daa84c8e6d0047ec6767d9df7a71fb42ac891eb053d8e67ec08b356b8cad847b`. The evidence records the exact source, plan, review, arm-result, and semantic hashes without host paths or bank payloads.
