# Toolchain entry points

Install the full development environment in [development](development.md) for these
tools. Corpus repair and production `.zst` files require `zstandard`, included in
`.[dev]`. Fresh worktrees also need their own native extension build; importing the
corpus tools uses `CBoard` even for CLI help. Normal package installation builds portable
extensions; production host builds follow the separate operations procedure.

Run scripts from the repository root with an installed package or `PYTHONPATH=.`.
Use each command's `--help` for its current arguments; examples below inspect interfaces
and do not launch experiments.

## Corpus to offline training

| Stage | Entry point | Contract to preserve |
| --- | --- | --- |
| Generate rooted Stockfish corpus | `scripts/gen_sf_rooted_corpus.py` | Frozen worker/search/resume settings, engine identity, row history and source-qualified game identity |
| Audit possible teacher labels | `scripts/audit_label_candidates.py` | Fixed banked observations and comparable teacher/search settings |
| Repair an existing corpus's history | `scripts/repair_corpus_history.py` | Distinct output, inventory coverage, unchanged labels unless explicitly relabeling, truthful completion stamp |
| Derive replay targets | `scripts/derive_corpus_targets.py` | Input corpus identity, target recipe, history/encoding stamps and completed-shard inventory |
| Train a controlled LC0-shaped comparison | `scripts/lc0_control_train.py` | Declared architecture, data lineage, objective, view/step budget and checkpoint identity |
| Offline target/loss comparison | `scripts/retarget_retrain.py` | Same-seed controls and a frozen readout appropriate to the hypothesis |
| Replay epoch reference runner | `scripts/offline_replay_epoch.py` | Distinguish its sampling contract from the game-aware exact-epoch mode |

```bash
PYTHONPATH=. python scripts/gen_sf_rooted_corpus.py --help
PYTHONPATH=. python scripts/derive_corpus_targets.py --help
PYTHONPATH=. python scripts/repair_corpus_history.py --help
PYTHONPATH=. python scripts/lc0_control_train.py --help
```

Repair is conditional on a known corpus defect; it is not an obligatory stage for a
new healthy corpus. Never mutate a populated corpus to reuse its identity for different
worker or teacher settings. Keep a companion corpus separate when scaling changes the
run's interpretation.

Game-aware exact epochs are implemented in `chess_anti_engine/replay/game_epoch.py`
and consumed by the replay/training paths. They freeze a corpus census, choose rows
per source-qualified game, preflight memory and target masks, and account for complete
epoch consumption without wrapping or silently truncating. Follow
[target rebuildability](target_rebuildability.md) for retained observations and identity.

## Playing and measurement

| Question | Entry point |
| --- | --- |
| Paired checkpoint comparison, resume and optional SPRT | `scripts/arena_standard.py` |
| Match against another UCI engine | `scripts/match_vs_uci.py` |
| Fixed handicapped-Stockfish opponent | `scripts/match_vs_handicapped_sf.py` |
| Joint PGN rating estimate | `scripts/ordo_pooled_fit.py` |
| Bank or inspect foreign BT4 policy/history behavior | `scripts/bt4_policy_dump.py`, `scripts/bt4_history_sensitivity.py` |
| Relabel/reconstruct RVG targets and shadow readout | `scripts/rvg_label_pass.py`, `scripts/nnue_shadow_label_readout.py` |

Use [evaluation](eval_protocol.md) to select the deciding measure and budget. A frozen
handicapped-opponent match is different from a production curriculum winrate whose
opponent changes under PID control. Resume only with matching schedule/settings and
preserve per-game records; an interrupted pair is not an observed draw. SPRT is an
optional precommitted stopping rule, not permission to keep extending any weak result.

## Agent tooling

`scripts/grok_review.sh` provides independently authored reviews from a disposable
snapshot; the `deepfin-grok-review` Skill documents invocation and interpretation.
`scripts/grok_fix.sh` prepares a separate worktree for explicitly requested Grok
implementation and leaves its diff for inspection. Neither workflow deploys or merges
its result. Useful changes from either route still target main under the
[branch lifecycle](branch_lifecycle.md).
