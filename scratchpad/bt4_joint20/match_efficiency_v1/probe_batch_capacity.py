"""One bounded component screen of the existing arena search at three widths.

No games, strength estimates, statistical decisions or repeated timing trials.
Run only through the registered GPU lease supervisor.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
import time


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    plan = json.loads(args.plan.read_text())
    for path, expected in plan['pins'].items():
        if digest(path) != expected:
            raise ValueError(f'changed input: {path}')
    import numpy as np
    import torch
    from scripts.arena_standard import (
        apply_search_overrides, arena_uncapped_leaf_rows, build_arena_evaluator,
        load_paired_openings, resolve_search_shape,
    )
    from chess_anti_engine.selfplay.match import pick_moves_for_boards
    from chess_anti_engine.moves import move_to_index
    from chess_anti_engine.uci.model_loader import load_model_from_checkpoint

    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.cuda.set_per_process_memory_fraction(0.75)
    side = apply_search_overrides(resolve_search_shape('training'), spec='policy_temp=1.0')
    if side.as_record() != plan['resolved_search']:
        raise ValueError('resolved search differs from registered arena')
    boards = load_paired_openings(Path(plan['book']), n_pairs=1152,
                                 max_plies=16, rng=np.random.default_rng(42))
    panel = [{'fen': b.fen(), 'root_fen': b.root().fen(),
              'moves': [m.uci() for m in b.move_stack]} for b in boards]
    (args.out / 'panel.json').write_text(json.dumps(panel) + '\n')
    models = []
    for checkpoint in plan['checkpoints']:
        model = load_model_from_checkpoint(checkpoint, device='cuda')
        if not hasattr(model, '_inference_only'):
            raise ValueError('missing arena inference-only head switch')
        model._inference_only = True
        models.append(model)
    relations = tuple(bool(getattr(m, 'use_dynamic_relations', False)) for m in models)
    # Allocate once at the largest tested arena pool's true leaf requirement.
    # This changes allocated capacity only: all cases remain uncapped.
    cap = arena_uncapped_leaf_rows(max_concurrent_games=256, sides=(side, side),
                                  relations=relations)
    models = [torch.compile(m) for m in models]
    evaluators = [build_arena_evaluator(m, device='cuda', max_batch=cap) for m in models]
    metadata = dict(plan_sha256=digest(args.plan), panel_sha256=digest(args.out / 'panel.json'),
                    python=__import__('sys').version, torch=torch.__version__,
                    cuda=torch.version.cuda, numpy=np.__version__,
                    resolved_search=side.as_record(), actual_overrides=dataclasses.asdict(side), actual_max_batch=cap,
                    dynamic_relations=relations, component_only=True)
    (args.out / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')

    def search(model, evaluator, selected, simulations, rng):
        actions = pick_moves_for_boards(
            model, selected, device='cuda', rng=rng, mcts_type='gumbel',
            mcts_simulations=simulations, temperature=0.1, c_puct=2.5,
            gumbel_add_noise=True, gumbel_overrides=side.gumbel,
            gumbel_vloss_weight=side.vloss_weight, gumbel_target_batch=side.target_batch,
            evaluator=evaluator)
        if len(actions) != len(selected):
            raise ValueError('search dropped a board')
        return actions

    with (args.out / 'cells.jsonl').open('x') as records:
        for simulations, width in plan['cells']:
            required = arena_uncapped_leaf_rows(max_concurrent_games=2 * width,
                                               sides=(side, side), relations=relations)
            if cap < required:
                raise ValueError('leaf capacity would change search')
            rng = np.random.default_rng(20260908)
            t0 = time.perf_counter()
            for model, evaluator in zip(models, evaluators, strict=True):
                search(model, evaluator, boards[:width], simulations, rng)
            torch.cuda.synchronize()
            warmup_seconds = time.perf_counter() - t0
            torch.cuda.reset_peak_memory_stats()
            cpu0, t0 = time.process_time(), time.perf_counter()
            all_actions = []
            for start in range(128, 1152, width):
                selected = boards[start:start + width]
                for model, evaluator in zip(models, evaluators, strict=True):
                    all_actions.extend(search(model, evaluator, selected, simulations, rng))
            torch.cuda.synchronize()
            seconds, cpu_seconds = time.perf_counter() - t0, time.process_time() - cpu0
            # Validate each returned search action against that board's legal
            # set outside the timed region. The timed operation is unchanged.
            offset = 0
            for start in range(128, 1152, width):
                selected = boards[start:start + width]
                for _ in models:
                    for board in selected:
                        if all_actions[offset] not in {move_to_index(move, board) for move in board.legal_moves}:
                            raise ValueError('search returned an illegal action')
                        offset += 1
            record = dict(simulations=simulations, boards_per_side_call=width,
                          equivalent_balanced_arena_pool=2 * width,
                          root_decisions=len(all_actions), seconds=seconds,
                          cpu_seconds=cpu_seconds, warmup_seconds=warmup_seconds,
                          root_decisions_per_second=len(all_actions) / seconds,
                          actual_max_batch=cap, required_max_batch=required,
                          peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                          peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                          actions_sha256=hashlib.sha256(json.dumps(all_actions).encode()).hexdigest(),
                          all_actions_legal=True)
            records.write(json.dumps(record) + '\n')
            records.flush()
            print(json.dumps(record), flush=True)
    (args.out / 'complete.json').write_text(json.dumps(dict(
        complete=True, metadata_sha256=digest(args.out / 'metadata.json'),
        cells_sha256=digest(args.out / 'cells.jsonl'), cells=len(plan['cells']),
        no_strength_claim=True), indent=2) + '\n')


if __name__ == '__main__':
    main()
