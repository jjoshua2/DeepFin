# Opt-in BT4 root-policy game worker — September 23, 2026

Status: implementation and CPU fake-teacher checks only. No ONNX game bank,
GPU run, replay consumer, training admission or production worker adoption is
claimed. This slice builds on the reviewed [BT4 root stepper](2026-09-23-bt4-root-policy-stepper.md)
and shared strict rule-aware tablebase helper. It answers the immediate need
for a runnable, bounded actor and a writer whose corpus is visibly separate
from the historical SF source rows and BT4 sidecars.

`scripts/bt4_root_policy_worker.py` accepts an explicit
`--outcome-mode rule50_match_v1` and opens the six-man WDL+DTZ pair strictly.
It runs a named BT4 ONNX policy and WDL head on CPU by default. An explicit,
bounded CUDA path is prepared in the separate
[GPU qualification plan](2026-09-23-bt4-root-worker-gpu-qualification.md);
no GPU game has run. The played actor samples
the raw root policy at the supplied temperature; it does **not** run C search
or consult Syzygy to choose moves at seven or more pieces. Its stored policy
is the legal-mapped compact T1 distribution from the native policy head; its
stored WDL preserves the named native tensor and dtype. Neither is corrected
by the game outcome. When a game finishes, the
stepper's rule-aware result supplies a distinct per-row `wdl_target` from the
root side to move. A six-man decision happens before another inference. A
missing eligible strict table probe aborts the invocation, while a positive
clock decisive WDL, unresolved cap, or ineligible six-man position discards
the whole game with a named reason. The worker admits only pre-move roots with
at least seven pieces into completed row payloads.

Invocation shape (select actual paths and the graph's named heads):

```bash
python3 -m scripts.bt4_root_policy_worker \
  --out /path/to/new/experimental-run \
  --onnx /path/to/bt4.onnx \
  --syzygy-path '/path/to/wdl:/path/to/dtz' \
  --outcome-mode rule50_match_v1 \
  --policy-output policy --wdl-output wdl --wdl-kind probabilities \
  --games 2 --seed 121 --max-plies 400 --parallel-games 2 \
  --temperature 1.0 --threads 2
```

The worker refuses an existing output directory and bounds both
`parallel_games * max_plies` and `games * max_plies` to 4,096 plies,
with at most 32 games per invocation. This caps the number of model/root
calls and possible banked rows; it does not impose a wall-time deadline on an
individual ONNX call. The CLI accepts one or two ONNX CPU threads. No resume or shard
rotation exists in this experimental schema. `launch.json` records the exact
model SHA-256, realized CPU provider, code-file and native encoder hashes,
input/history mode,
actor, seed, temperature, and Syzygy path, handle capacity counts and
file-stat inventory (resolved filename, size, nanosecond mtime). The table
inventory deliberately is **not** a content-hash attestation. The inventory
is built from the strict handle's requested path before writing and checked
again before completion; any path/stat change leaves the run incomplete.
Each finalized
game publishes as one atomically renamed `games/game_NNNNNNNN.npz`; no row
file is visible until the complete game validates and serializes. Its JSON
`metadata` array records game/result/termination and ordered FEN, source key,
input key, ply, true-frame UCI, input/head contract and target per row
(0=win, 1=draw, 2=loss from the root side to move). Its
`x`, `policy_t1` and `wdl_raw` arrays retain exact float32 encoded input,
compact policy and native WDL dtype. Discarded games publish no rows and
retain attempted/discarded counts. Only a fully successful invocation writes
`summary.json` with game-file SHA-256 receipts; its absence means the corpus
is incomplete and must not be admitted.

The `lc0_1858_compact` stored policy is mapped through the board-aware
Leela-to-compact conversion in the existing evaluator. `source_key` is the
history-sensitive plane fingerprint; `input_key` hashes exact float32 input.
The per-row FEN and move UCI are in the true game frame. This own schema has
no replay loader. Any future consumer must validate all game-file receipts,
source/encoding/move mapping and target semantics before training use.

CPU checks cover a seven-to-six capture with the raw teacher unchanged, a
whole-game cap discard, a missing required table probe, serialization failure
cleanup, and bounded/explicit CLI preflight. They use fake inference and
fake tablebase values, so they do not qualify an actual ONNX model, six-man
material coverage or game production throughput.
