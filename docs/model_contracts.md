# Model and training contracts

Use this reference when changing heads, targets, search values, encoding or
training-view accounting. These notes were moved from the shared root guide so
unrelated tasks do not need them. Verify the relevant implementation, effective
config and checkpoint metadata; historical evidence is not a current run setting.

- The Stockfish component of the WDL blend has been load-bearing in prior runs.
  Consult the relevant [historical evidence](experiment_ledger.md) before removing
  it; a sharper teacher-fit score is not sufficient evidence. Negative results are
  conditional on the model, data and horizon tested, not a ban on a scoped experiment.
- `policy_sf` predicts the opponent's reply at P1 after the network's move. It is
  not a teacher distribution over the network's current moves.
- MCTS uses the `wdl` value head; `sf_eval` and `categorical` are auxiliary. See
  [model heads](model_heads.md) and `chess_anti_engine/train/losses.py` for wiring.
- Higher PID `wdl_regret` permits worse Stockfish moves and makes the opponent
  weaker. Best-move-based labels differ from the handicapped move actually played.
- Search action IDs and network policy indices are different spaces. Use shared
  mappings in `chess_anti_engine/moves/torch_maps.py`; preserve encoding and history
  metadata across checkpoints, replay and evaluation.
- Count tied parameters once. A naive `state_dict` element sum counts shared
  Smolgen weights repeatedly; `tests/test_param_count.py` measures the actual model.
- Production uses `train_views_per_ingested_position`; the old
  `train_views_per_position` name is rejected and used a different denominator.

For pipeline invariants and measurement traps, consult the relevant method/stage
sections of the [loop audit](rl_loop_audit.md). Use the [experiment index](experiments/README.md)
for prior readouts, rather than treating these reminders as new assignments.
