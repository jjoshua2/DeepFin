"""Canonical decision samples remain independent of game completion order."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from chess_anti_engine.eval.sprt import SprtMonitor, SprtSpec
from tests.test_arena_sprt import SPEC_TIGHT, SPEC_WIDE, _drive_main, _run_chunked_arena


def test_late_first_pair_releases_each_look_and_keeps_first_crossing() -> None:
    spec = SprtSpec(0, 20, .05, .05, first_pairs=64, step_pairs=64)
    scores = [2.0] * 64 + [0.0] * 336
    monitor = SprtMonitor(spec, pairs_cap=400, granularity='pair')
    monitor.update(scores[1:], pair_ids=list(range(1, 400)))
    assert monitor.pairs == 0
    assert monitor.trajectory == []
    assert monitor.verdict is None
    monitor.update(scores[:1], pair_ids=[0])
    assert monitor.verdict == 'H1'
    assert monitor.pairs == 64
    assert [n for n, _ in monitor.trajectory] == [64]
    assert monitor.pair_scores == scores[:64]
    assert monitor.as_record()['speculative_completed_pair_ids'] == list(range(64, 400))
    # Looking only at the latest complete sample would reverse the decision.
    at_cap = SprtMonitor(SprtSpec(0, 20, .05, .05, first_pairs=400),
                         pairs_cap=400, granularity='pair')
    at_cap.update(scores)
    assert at_cap.verdict == 'H0'


def test_first_look_and_final_ragged_cap_are_declared() -> None:
    monitor = SprtMonitor(SprtSpec(0, 20, 1e-9, 1e-9, first_pairs=128, step_pairs=64),
                          pairs_cap=500, granularity='pair')
    monitor.update(([0.0, 2.0] * 250)[:127])
    assert monitor.trajectory == []
    assert monitor.verdict is None
    monitor.update([0.0, 2.0] * 250)
    assert [n for n, _ in monitor.trajectory] == [128, 192, 256, 320, 384, 448, 500]
    assert monitor.finalize(stop_reason='cap') == 'INCONCLUSIVE'


def test_deadline_between_looks_does_not_add_a_decision() -> None:
    monitor = SprtMonitor(SprtSpec(0, 20, .05, .05, first_pairs=128, step_pairs=64),
                          pairs_cap=500, granularity='pair')
    monitor.update([2.0] * 127)
    assert monitor.llr > monitor.spec.bound_h1  # descriptive, not a declared look
    assert monitor.trajectory == []
    assert monitor.finalize(stop_reason='max_seconds') == 'INCONCLUSIVE'
    assert monitor.as_record()['last_decision_look_pairs'] == 0


@pytest.mark.parametrize(('ids', 'scores'), [([0, 0], [1., 1.]), ([500], [1.]), ([-1], [1.]),
                                       ([True], [1.]), ([0], [float('nan')]), ([0], [.3])])
def test_bad_observations_do_not_enter_the_deciding_sample(ids: list[int], scores: list[float]) -> None:
    monitor = SprtMonitor(SPEC_WIDE, pairs_cap=500, granularity='pair')
    with pytest.raises(ValueError, match="SPRT"):
        monitor.update(scores, pair_ids=ids)
    assert monitor.pairs == 0
    assert monitor.verdict is None


def test_changed_completed_observation_is_refused() -> None:
    monitor = SprtMonitor(SPEC_WIDE, pairs_cap=500, granularity='pair')
    monitor.update([1.], pair_ids=[3])
    with pytest.raises(ValueError, match='changed'):
        monitor.update([2.], pair_ids=[3])


def test_actual_chunked_record_separates_bank_suffix_from_scored_prefix(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    result = _run_chunked_arena(monkeypatch, tmp_path, sprt=SPEC_TIGHT,
                                n_pairs=60, calls=[])
    rows = [json.loads(line) for line in (tmp_path/'chunked.games.jsonl').read_text().splitlines()]
    assert len([r for r in rows if r.get('kind') == 'game']) == 32
    assert result['games'] == 30
    assert result['sprt']['scored_pair_ids'] == list(range(15))
    assert result['sprt']['speculative_completed_pair_ids'] == [15]
    assert result['sprt']['not_started_games'] == 88
    assert result['game_log_agrees'] is True


def test_resume_keeps_nonprefix_completed_pairs_without_scoring_them_first(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    path = tmp_path/'gap.games.jsonl'
    _run_chunked_arena(monkeypatch, tmp_path, sprt=SPEC_WIDE, n_pairs=60,
                       calls=[], log_path=path)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    # A crash left all other pairs, but neither coloring of pair0.
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows if r.get('pair_id') != 0))
    calls: list[int] = []
    result = _run_chunked_arena(monkeypatch, tmp_path, sprt=SPEC_WIDE, n_pairs=60,
                                calls=calls, log_path=path, resume=True)
    assert calls == [1]
    assert result['sprt']['scored_pair_ids'] == list(range(60))
    assert result['sprt']['speculative_completed_pair_ids'] == []
    assert result['game_log_agrees'] is True


def test_cli_passes_explicit_look_schedule(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    call = _drive_main(monkeypatch, tmp_path,
                       ['--sprt', 'elo0=0,elo1=15,alpha=.05,beta=.1,first_pairs=128,step_pairs=64'])
    assert call['sprt'].first_pairs == 128
    assert call['sprt'].step_pairs == 64


@pytest.mark.parametrize('extra', ['first_pairs=0', 'step_pairs=1.5', 'first_pairs=nan'])
def test_invalid_cli_look_schedule(extra: str) -> None:
    with pytest.raises(ValueError, match='positive integer'):
        SprtSpec.from_cli('elo0=0,elo1=15,alpha=.05,beta=.1,'+extra)
