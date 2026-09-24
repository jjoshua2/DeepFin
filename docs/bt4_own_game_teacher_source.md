# BT4 own-game teacher source

`scripts/bt4_own_game_teacher_source.py` reuses root teacher observations already
saved by the BT4 game worker. It performs no ONNX inference. Its output is a
source cohort for later labeling work, **not a training shard**.

Each input is a closed `bt4_root_policy_games_v1` bank, its expected
`summary.json` SHA-256, and a separately produced full-bank audit receipt with
its SHA-256. The accepted audit profiles cover the saved two-game qualification,
independent 32-game readback, and a separately rerun ordinary-bank strict
audit. The latter is produced by `scripts/bt4_ordinary_bank_audit.py` from a
completed ordinary pilot or parallel-screen stage. It imports only one of the
two SHA-reviewed frozen supervisors, calls that supervisor's full `verify_bank`
on the saved bank, and checks its facts against the original stage terminal.
The receipt binds the plan, stage, verifier source, auditor source, terminal,
summary, CUDA provider proof, accepted rows, and unchanged bank tree. It is
written only after the strict verifier returns. The adapter requires the
receipt's exact SHA. A new verifier source requires a reviewed allowlist
update; a matching status string alone is insufficient.
The adapter validates every NPZ receipt and game ID, exact input and position
keys, actor move history, legal support of the saved T=1 root policy, native
WDL probability contract, outcome POV, and model/head/history compatibility.
It rechecks source and audit pins before atomic publication. A failed write
leaves `<out>.writing/incomplete.json`; the final `<out>` path is absent.

For each input bank the output has one Zarr group with exact float32 `x`,
float32 `bt4_policy` `[N,1858]`, float32 `bt4_wdl_raw` `[N,3]`, input/position
keys, bank-qualified 32-byte row IDs, game/absolute-ply IDs, and terminal
outcome labels. A paired JSONL file retains the saved FEN, actor move, model,
game result, source NPZ path/SHA, and row reference. The manifest pins the
source summaries, audit receipts, producer source, and output array digests.
Bare game ID or position fingerprint is never used as a unique row identity.

`bt4_policy` is the teacher's **unsearched prior**, while `bt4_wdl_raw` is
its native root value. `outcome_wdl_target` is the observed/adjudicated game
result. None is silently written into replay `policy_target` or `search_wdl`.
The existing SF factorial policy/value mixers require an SF-derived policy
and search-WDL baseline with raw SF row provenance; the mixed value arm also
requires separate Ceres labels. They reject this source by their normal input
contracts. A later own-game cohort consumer must define its policy/value
recipe and additional labels explicitly.

The first adapter caps input at 100,000 accepted rows and stores exact `x` so
a later Ceres pass can use the original root input. Large-scale chunking and
training-cohort admission are separate work. The saved forced 2+32-game banks
are a schema regression only; their one-ply tablebase fixture is neither a
throughput nor label-quality sample.

The ordinary audit producer is CPU-only and requires `CUDA_VISIBLE_DEVICES=''`.
Run it only after an ordinary stage has a complete terminal, using the exact
frozen plan and supervisor SHA-256s, the stage's bank and terminal, and a fresh
audit output path outside the bank. Its full verifier reconstructs boards and
input history, checks legal policy and native WDL, and verifies natural or
rule50 Syzygy terminal results for every accepted game. The synthetic receipt
tests exercise the contract only; no ordinary game bank has been audited or
adapted by this change.
