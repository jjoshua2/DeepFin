# Exact retained teacher references

The 17 `.txt` files contain 322,134 bytes copied unchanged from the reviewed retained
sources. `manifest.json` records each original source path, repository fixture
path, materialization path, SHA256 and byte count. This includes the original
owner guard, priority/readiness loop, SQLite checkpoint/recovery, Companion,
private-process cleanup, Ceres history loader/publisher, encoder and adapter.

These files are CPU test and source-review evidence. They are not installed
production helpers or an authorization to load models, acquire custody, modify a
queue, or adopt a runtime. Absolute paths in the original metadata are historical
provenance. `tests/teacher_reference_fixture.py` validates source bytes, then
materializes them in pytest-owned directories and derives explicit unqualified
CPU metadata. Readiness namespaces and native/runtime dependencies are relocated
by the fixtures. Fake-CPU subprocess tests use the same materialized reference
bytes, including the original cleanup implementation.

The repository's own built native extensions and Python dependencies are still
required. No native binaries, models, training data or credentials are embedded.

The local `.gitattributes` disables text conversion for source `.txt` files so
Windows checkout does not change their byte hashes.
