# SF-rooted corpus rule-aware outcome mode (2026-09-23)

This is a code qualification record, not a corpus run. No Stockfish games,
derivation, training, or GPU work was launched for this change.

The generator now requires an explicit `--outcome-mode` at its CLI. The
historical `theoretical_v1` route keeps its existing Syzygy result convention.
The opt-in `rule50_match_v1` route opens one strict, caller-owned WDL+DTZ
six-man handle per worker. Natural outcomes, including claimable draws, take
precedence. At an eligible six-man root, the shared match helper treats WDL
±1/0 as draw and certifies WDL ±2 as decisive only after the halfmove clock
has reset. A positive-clock ±2 position remains unresolved and play continues;
at a ply cap its rows carry no game result. Missing eligible WDL or DTZ fails
the worker and the run. Only positions with at least seven pieces bank rows.

The mode is in the requested config hash, launch manifest, row `run` block,
worker/summary counters, derived summary and committed derived-shard attrs.
The deriver refuses a row whose mode differs from its corpus record. Older
records and rows with no mode stamp mean `theoretical_v1`; an old run can resume
only when its new invocation explicitly selects that historical mode. Strict
mode cannot append to an old theoretical corpus. A future trainer recipe that
consumes game-outcome targets should gate mixed derived outcome modes; current
teacher-search-only arms do not use that game result as their value target.

In strict mode the Stockfish wrapper requires advertised UCI support for
`SyzygyPath`, `Syzygy50MoveRule` and `SyzygyProbeLimit`, sends `true` and `6` with the named
Syzygy path, and records the advertisement, request, and subsequent `readyok`
barrier. UCI provides no per-option acknowledgement or effective-value
readback, so these stamps alone do not prove internal probing. The strict
opener checks WDL/DTZ capacity and each encountered eligible material; it does
not content-hash the tablebase files or freeze their contents across a resume.
The raw-baseline audit, ordinary and selected policy rewrites, and d9 rank
sidecar now bind strict row identity to the qualified source record or launch
manifest. They cross-check the derived summary's declared mode; rewritten
shards retain the committed mode stamp.

Validation used synthetic engines/tablebases and focused repository tests.
The scoped generator/deriver/tablebase suite passed 481 selected tests
on two low-priority CPU cores with CUDA hidden. Three older tests that hard-code
`.jsonl.zst` failed because this virtual environment lacks `zstandard` and the
writer used `.jsonl.gz`; all three failed identically on the unchanged
`0d089488` base with the same interpreter and rebuilt extension. Scoped Ruff
and Basedpyright passed. Four additional strict-corpus consumer fixtures pass
through real serialized source and derived records. Whole-repository lint was
attempted once: Ruff passed, while repository-wide Basedpyright failed with
1,949 errors and 1,231 warnings in the local dependency environment.

A [bounded real-Stockfish search receipt](evidence/sf-rule50-corpus/actual-search.json)
and its [exact qualification script](evidence/sf-rule50-corpus/actual-search.py.txt)
show one 1,024-node, one-thread, 16 MiB search of a seven-piece position. The
engine received the explicit path, 50-move rule `true`, and six-piece probe
limit, selected `b2c3` (a seven-to-six capture), and emitted `tbhits` up to 73.
The pinned binary SHA-256 is in the receipt; its UCI wrapper source SHA-256 is
the reviewed `c6274f9b…` revision. This establishes realized tablebase hits
for that fixture only. No corpus generation, throughput qualification, training,
or GPU work was launched for this change.
