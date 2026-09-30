"""A requested tablebase probe reaches search on both arena sides."""
from __future__ import annotations

from typing import Any, cast

import chess
import numpy as np
import pytest
import torch

from chess_anti_engine.moves import move_to_index
from chess_anti_engine.selfplay import match
from chess_anti_engine.tablebase import SyzygyProbe
from chess_anti_engine import tablebase
from chess_anti_engine.utils.game_log import settings_fingerprint
from scripts import arena_standard as arena


class DummyModel(torch.nn.Module):
    pass


def test_probe_reaches_compiled_search(monkeypatch: pytest.MonkeyPatch) -> None:
    probe = cast(SyzygyProbe, object())
    seen: list[Any] = []

    def search(_model: Any, boards: list[chess.Board], **kwargs: Any) -> tuple[Any, ...]:
        seen.append(kwargs['tb_probe'])
        return ([], [int(move_to_index(next(iter(b.legal_moves)), b)) for b in boards], [], [])

    monkeypatch.setattr(match, '_HAS_GUMBEL_C', True)
    monkeypatch.setattr(match, '_run_gumbel_root_many_c', search)
    result = match.pick_moves_for_boards(
        DummyModel(), [chess.Board()], device='cpu', rng=np.random.default_rng(1),
        mcts_type='gumbel', mcts_simulations=2, temperature=0, c_puct=2.5,
        gumbel_add_noise=False, tb_probe=probe,
    )
    assert len(result) == 1
    assert seen == [probe]


@pytest.mark.parametrize('route', ['puct', 'python', 'volatility'])
def test_unsupported_search_refuses_probe(monkeypatch: pytest.MonkeyPatch, route: str) -> None:
    monkeypatch.setattr(match, '_HAS_GUMBEL_C', route != 'python')
    with pytest.raises(ValueError, match=r'[Tt]ablebase'):
        match.pick_moves_for_boards(
            DummyModel(), [chess.Board()], device='cpu', rng=np.random.default_rng(1),
            mcts_type='puct' if route == 'puct' else 'gumbel', mcts_simulations=2,
            temperature=0, c_puct=2.5, gumbel_add_noise=False,
            volatility_q_scale=1 if route == 'volatility' else 0,
            tb_probe=cast(SyzygyProbe, object()),
        )


@pytest.mark.parametrize('rolling', [False, True])
def test_both_arena_sides_receive_same_probe(monkeypatch: pytest.MonkeyPatch, rolling: bool) -> None:
    probe = object()
    seen: list[tuple[Any, Any]] = []
    candidate, reference = DummyModel(), DummyModel()

    def pick(model: Any, boards: list[chess.Board], **kwargs: Any) -> list[int]:
        seen.append((model, kwargs['tb_probe']))
        return [int(move_to_index(next(iter(b.legal_moves)), b)) for b in boards]

    monkeypatch.setattr(match, 'pick_moves_for_boards', pick)
    side = arena.resolve_search_shape('play')
    kwargs: dict[str, Any] = {
        'device': 'cpu', 'rng': np.random.default_rng(1),
        'sims_candidate': 2, 'sims_reference': 2,
        'max_plies': 1, 'temperature': 0, 'gumbel_add_noise': False,
        'search_candidate': side, 'search_reference': side, 'tb_probe': probe,
    }
    play = (arena.play_paired_games_matched_sims_rolling if rolling
            else arena.play_paired_games_matched_sims)
    if rolling:
        kwargs['pool_size'] = 2
    with pytest.raises(RuntimeError, match='unresolved game'):
        play(candidate, reference, [chess.Board()], **kwargs)
    assert {id(model) for model, _ in seen} == {id(candidate), id(reference)}
    assert all(received is probe for _, received in seen)


@pytest.mark.parametrize('rolling', [False, True])
def test_last_ply_capture_enters_tablebase_before_cap(
    monkeypatch: pytest.MonkeyPatch, rolling: bool,
) -> None:
    # Seven pieces before search, six after White captures the pawn. The
    # strict route must adjudicate the final permitted move, not score a cap.
    opening = chess.Board('4k3/8/8/8/8/8/4p3/3RKQRR w - - 0 1')
    capture = chess.Move.from_uci('e1e2')
    assert capture in opening.legal_moves
    assert len(opening.piece_map()) == 7
    events: list[dict[str, Any]] = []
    probed_counts: list[int] = []

    def rule50(board: chess.Board, _handle: Any, *, max_pieces: int) -> str | None:
        assert max_pieces == 6
        count = len(board.piece_map())
        probed_counts.append(count)
        return '1-0' if count == 6 else None

    def pick(_model: Any, boards: list[chess.Board], **_kwargs: Any) -> list[int]:
        return [int(move_to_index(capture, board)) for board in boards]

    monkeypatch.setattr(tablebase, 'rule50_match_result', rule50)
    monkeypatch.setattr(match, 'pick_moves_for_boards', pick)
    side = arena.resolve_search_shape('play')
    kwargs: dict[str, Any] = {
        'device': 'cpu', 'rng': np.random.default_rng(1),
        'sims_candidate': 2, 'sims_reference': 2,
        'max_plies': 1, 'temperature': 0, 'gumbel_add_noise': False,
        'search_candidate': side, 'search_reference': side,
        'syzygy_tablebase': object(), 'tb_max_pieces': 6,
        'tb_probe': cast(SyzygyProbe, object()),
        'pgn_sink': lambda **event: events.append(event),
    }
    if rolling:
        kwargs['pool_size'] = 2
        scores = arena.play_paired_games_matched_sims_rolling(
            DummyModel(), DummyModel(), [opening], **kwargs,
        )
    else:
        scores = arena.play_paired_games_matched_sims(
            DummyModel(), DummyModel(), [opening], **kwargs,
        )
    assert scores == [1.0]
    assert probed_counts.count(7) == 2
    assert probed_counts.count(6) == 2
    assert len(events) == 2
    assert {event['termination'] for event in events} == {'syzygy'}
    assert {event['result'] for event in events} == {'1-0'}
    assert all(event['moves'] == (capture,) for event in events)


def test_strict_protocol_changes_resume_fingerprint() -> None:
    side = arena.resolve_search_shape('play')
    settings = arena.arena_game_log_settings(
        mode='matched_sims', candidate='a', reference='b', games=2, seed=1,
        openings_path='openings.pgn', openings_kind='book', opening_plies=4,
        sims_candidate=2, sims_reference=2, ms_per_move=None, max_plies=200,
        temperature=0, gumbel_add_noise=False, search_candidate=side,
        search_reference=side, volatility_candidate=None, uci_args='',
        syzygy_path='/fake/syzygy', tb_max_pieces=6,
    )
    assert settings['syzygy_protocol'] == arena.SYZYGY_MATCH_PROTOCOL
    old_settings = dict(settings)
    del old_settings['syzygy_protocol']
    assert settings_fingerprint(settings) != settings_fingerprint(old_settings)


@pytest.mark.parametrize(
    ('override', 'message'),
    [
        ({'mode': 'matched_time'}, 'only in matched_sims'),
        ({'pgn_out': None}, 'requires --pgn-out'),
        ({'volatility_candidate': {}}, 'cannot use candidate volatility'),
        ({'tb_max_pieces': 0}, 'must be an integer'),
    ],
)
def test_strict_mode_rejects_unsupported_routes_before_io(
    override: dict[str, Any], message: str,
) -> None:
    side = arena.resolve_search_shape('play')
    kwargs: dict[str, Any] = {
        'candidate': 'a', 'reference': 'b', 'games': 2, 'openings_path': None,
        'opening_plies': 0, 'mode': 'matched_sims',
        'sims_candidate': 2, 'sims_reference': 2, 'ms_per_move': 0,
        'max_plies': 1, 'temperature': 0, 'gumbel_add_noise': False,
        'device': 'cpu', 'seed': 1, 'out_path': None,
        'syzygy_path': '/fake/syzygy', 'pgn_out': '/fake/game.pgn',
        'search_candidate': side, 'search_reference': side,
    }
    kwargs.update(override)
    with pytest.raises(SystemExit, match=message):
        arena.run_arena(**kwargs)
