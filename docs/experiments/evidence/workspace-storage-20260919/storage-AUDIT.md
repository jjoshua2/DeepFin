# Factorial storage boundary audit

PASS: Future target outputs/ ancestor symlink: real producer build, preparation verify_cohort, all B/C/D qualification and load_shard_arrays consumption. PASS: Ceres bank ancestor symlink: actual producer/verify/cache with mocked inference, plus real driver launch/completion discovery using fake Ceres worker. This is a storage-path test, not GPU qualification. Full executable fixture and output retained beside this note.

Exact boundaries:
- bootstrap_factorial_targets.py:67-87 resolves base input, preserves logical out; nonexistent cohort leaves required.
- target_overlay.py:309-318 resolves staged paths to qualified canonical output paths.
- preparation/run.py:63-79 verifies logical cohort completion and immutable original base; it passed fixture unchanged.
- ceres_collection_batches.py:300-318 preserves absolute logical output, requires existing parent/nonexistent chunk leaf, rejects canonical overlap; :420-430 checks direct invocation entries, not ancestor links.
- collection/run_chunk.py:137-139 requires nonexistent output leaf; source/environment checks remain untouched.
- ceres_derived_sidecar.py:239 binds resolved SOURCE, not output location; :444,472-492 consumes output via unchanged alias. verify_cached passes with linked ancestor.
- coverage/assemble_collected.py:38-48 uses logical bank path and original source binding; does not demand bank alias==resolve().

All fourteen bank parents were existing EMPTY ordinary directories when inventoried. Replace only those empty parents with links after rechecking no job began; leave chunk leaves absent. Do not move controls, input sources, existing shards, or active data roots. Labels/target output and repository currently share filesystem2096.

Training output leaf symlinks are NOT allowed by training/run_arm.py:32 (fresh output required). Rebind future arm out, command --out-dir, initial anchor, arenas checkpoint references, then re-register and repin queue.

Active sampler blockers: game_epoch.py:954 planned digest hashes resolved path; :1500 re-resolves future loads; :1666 rejects planned/realized mismatch. Moving active35roots can fail at epoch end despite unchanged data. Input seal bindings: target_overlay.py:121-126 canonical parent/tree stamp; :234-238 canonical seal base identity. Storage identity includes inode/dev/mtime/ctime; rsync preservation of mtime is insufficient.
