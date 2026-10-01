# Lossless checkpoints for banked Stockfish raw observations

**CPU-only storage prototype, 2026-09-30. Zero label, target, corpus, training, Elo, external-drive throughput, or 500M-row credit.** One failed-attempt full05 raw-observation block contained 747 JSONL rows and 1,803,467 original bytes. A separately sealed Zstd frame held 289,954 bytes; its frame, claim, receipt, and lock file total 291,091 bytes, or 6.196 times smaller than the raw block alone. The original block remains retained, so this diagnostic has **not** reduced current corpus storage.

The new [frame writer](../../chess_anti_engine/source/checkpointed_uci_frames.py) takes one already pinned raw JSONL block at a time. It preserves the original bytes, including whitespace and every UCI line. A frame holds at most 2,048 rows and 8 MiB uncompressed; metadata files are capped at 16 KiB. The claim binds the source, schema, config/receipt, input-block, code, codec/version, level, checksum, content-size, and decoder-limit identities. The fsynced receipt records row count, compressed and uncompressed byte counts and SHA-256 hashes, and the claim hash. A nonblocking per-block lock and unique atomic staging files let a killed invocation recover only its own unsealed output. A resumed sealed block is rechecked against its bounded input and compressed bytes before reuse.

The decoder rejects a missing or mismatched content size, a window above 8 MiB, truncated data, and bytes after the single frame. It checks the header limits before bounded one-shot decompression. Fifteen focused Python 3.10 tests pass, including actual SIGKILL after a completed seal and during an unsealed frame stage, tampered or truncated frames even with a rewritten receipt, forged small content size, oversized window/expansion, changed source/schema/code, and an unchanged no-op receipt. A separate direct Zstd decode in the pilot operator matched all original bytes, but this remains an **author-performed** check; an independently authored all-byte audit is pending.

| Bounded input | Rows | Raw bytes | Zstd frame | Frame + claim + receipt + lock-file subtotal | Raw/subtotal |
| --- | ---: | ---: | ---: | ---: | ---: |
| Synthetic UCI-style block | 2,048 | 2,753,835 | 216,108 | 217,247 | 12.676× |
| Failed full05 source-31 block | 747 | 1,803,467 | 289,954 | 291,091 | 6.196× |

The real block is sealed failed-attempt evidence with **no label credit**. Its input SHA-256, source receipt, source identity, roster identity, and frame hashes are pinned in the [evidence index](evidence/2026-09-30-sf-raw-zstd-frames.json). The measured writer call, including input read, compression, fsync and readback, took 0.061 s wall / 0.030 s CPU on this one block; direct decode took 0.001 s. These small-block times do not estimate sustained production or external-drive throughput. The subtotals exclude the retained raw input, pilot result, directory entries, filesystem allocation, and any future index or trainer packaging.

Full Ruff and Vulture pass. Focused Pyright reports zero findings. The available full-repository basedpyright overlay reports the same pre-existing 439 errors and 7 warnings on current main and this successor, with zero normalized diagnostic differences; it is not a clean gate in this host setup. The proof bundle records the exact producer, tests, operator, synthetic measurement, direct pilot receipt, and lint comparison. Production use still needs indexed source-block inventory, an independent byte-level audit, reader and trainer integration, and representative external-drive cost and restart measurements. The live Stockfish label recipe and campaign were unchanged.

## October 1 implementation reconciliation

The September 30 measurements and their hashes above remain historical evidence
from the pinned original producer; they are not measurements of a later revision.
The original raw block and compressed pilot artifact are not available through the
repository connector, so the independently authored all-byte audit remains
**pending, with production qualification held**. Synthetic byte-for-byte tests
cannot discharge that original-artifact audit or establish a corpus storage gain.

[PR #966](https://github.com/jjoshua2/DeepFin/pull/966) reconciles the prototype with
current main while preserving its original branch ancestry. The successor compares
canonical metadata bytes, avoiding Python's equality between booleans and integers;
checks the promised checksum and dictionary-free frame recipe; preflights all
cleanup targets before deletion; binds torn initial claim stages to the intended
claim hash; and repeats parent-directory fsync barriers on resume. The added tests
exercise actual 2,048-row and 8 MiB inputs, actual over-cap input/frame files,
metadata-type substitutions, missing frame-header properties, refused foreign
stages, and resumed directory durability. Hosted checks on the exact final head
are reported in the PR; the historical 15-test/local-lint results above do not
stand in for those checks. No live files, recipes, processes or original artifacts
are changed by this reconciliation.
