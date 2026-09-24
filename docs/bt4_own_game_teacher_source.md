# BT4 own-game teacher source

`scripts/bt4_own_game_teacher_source.py` reuses root teacher observations already
saved by the BT4 game worker. It performs no ONNX inference. Its output is a
source cohort for later labeling work, **not a training shard**.

Each input is a closed `bt4_root_policy_games_v1` bank, its expected
`summary.json` SHA-256, and a separately produced full-bank audit receipt with
its SHA-256. The accepted audit profiles currently cover the saved two-game
qualification and independent 32-game readback. An ordinary-opening bank needs
an independent full-bank audit profile before this adapter can publish it.
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
