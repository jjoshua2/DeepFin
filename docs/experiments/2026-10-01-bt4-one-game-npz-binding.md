# BT4 one-game NPZ binding to the literal-UID cursor

## Scope and qualification

This is a bounded CPU diagnostic implementation, not a replay launch or corpus
admission. It connects a pinned BT4-v9 selected-game adapter to the
[literal-UID cursor](2026-09-30-tri-source-literal-uid-cursor.md).
No archive or native spool was read during the GitHub-only reconciliation.

Actual archive parity, target/trainer joins, GPU use, training, throughput and
500M-row credit remain unqualified. A successful synthetic test or repository
merge does not change those holds.

## Historical implementation evidence

[PR #975](https://github.com/jjoshua2/DeepFin/pull/975) initially added five files
at `2a2c66c9dc6065c1c194d28045608b3d7de97c26`, based on the tested
#973 head `ee3c6d2c9e35bb619186207bbb833fc4f046c0fe`.
The original report recorded 14 BT4 and six existing cursor CPU tests passing,
whole-repository Ruff and Vulture passing, and scoped Python 3.10 Basedpyright
with zero errors and warnings. The full host lint invocation could not resolve
project dependencies (2,467 errors and 1,524 warnings); that was not a clean
whole-repository type gate or a normalized base delta.

The metadata-only selected-game plan identified one 93-row game and two prior
wave witnesses. Its saved game-1388 proof line remains byte-for-byte unchanged:
`tests/fixtures/bt4_game1388_old_proof_v8.json`, 96,844 bytes,
SHA-256 `d422febc15864197a793eec210c06e8807af505a748f1ac9726bf037fcf8a018`.
The original proof lacks a Syzygy inventory field. The binding adds only the
already-open strict gate's checked inventory to a copy; the old proof and its
complete original fields remain the comparison witness.

## Reconciliation and corrections

The existing branch is reconciled with current main using a two-parent merge,
retaining the newer checkpointed sort and paired-wave guards. The prototype
still snapshots one authenticated NPZ and invokes strict replay once. It compares
all ten original bridge row fields, literal UID, 44,800-byte native input,
history/context, outcome and contract-derived teacher route with the old witness.

Review found and corrected these additional gaps:

- A per-root nonblocking no-follow regular-file lock now keeps concurrent owners
  from launching separate attempts and racing completion publication
- Frozen imports and the raw verifier now execute retained authenticated source
  snapshots rather than timestamp-valid cached bytecode. Every pinned path must
  pass the source loader before replay; custom loaders that bypass it fail closed
- A pending owner alarm could interrupt between child launch and cleanup ownership.
  Launch now defers that signal until the child handle is covered by cleanup; the
  exec guard explicitly unblocks the inherited child alarm mask before replay.
  A pre-blocked owner SIGALRM is rejected before any owned work
- Resumed result JSON and semantic output verification could re-open paths after
  hashing. Parsing and comparison now use the exact authenticated byte snapshots,
  including bounded no-follow reads for claim and completion metadata
- Equally incomplete old and new proofs could compare equal. The saved proof,
  source, terminal and history formats and complete history/trace lengths are
  checked before enrichment and witness comparison
- Expiry or an fsync failure during completion publication now removes the
  completion receipt rather than allowing a late success to be resumed

The caps remain 400 rows, 64 MiB NPZ, 128 MiB combined NPZ/replay context,
32 MiB output, four GiB observed process-tree RSS and 600 seconds owner wall
time. Other pinned files have individual caps. CUDA is hidden and CPU math
threads are capped. Parent-death protection covers the direct replay child;
the pinned replay path does not launch subprocesses.

## Validation boundary

The reconciliation adds deterministic regressions for the spawn-return alarm
window, authenticated-result snapshot use, interrupted receipt publication,
concurrent owner refusal, pre-blocked owner signals, stale bytecode, and ambient
source-path substitution;
the saved proof fixture also tests matching omissions throughout its original
format. These are repository tests, not fresh archive-parity evidence.
No local executor or local tests were used for this reconciliation. Final
validation is the exact-head GitHub CI and independent review recorded on #975;
historical results above must not be represented as checks of a later revision.
