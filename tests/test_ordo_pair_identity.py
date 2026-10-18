"""Tiny source/run identity fixtures; no Ordo or subprocess calls."""

from pathlib import Path

import pytest

from scripts.ordo_pooled_fit import Game, group_pairs, read_games


def game(order: int, *, source: str = "one.pgn", run: str | None = None,
         half: str | None = None, reverse: bool = False,
         pair_id: str | None = "0") -> Game:
    return Game("B" if reverse else "A", "A" if reverse else "B",
                "1/2-1/2", pair_id, None, None, order,
                source=source, run_id=run, pair_half=half)


def test_same_pair_id_in_two_sources_stays_two_blocks() -> None:
    rows = [game(0), game(1, reverse=True),
            game(2, source="two.pgn"), game(3, source="two.pgn", reverse=True)]
    pairs, singles = group_pairs(rows)
    assert singles == 0
    assert [p.games for p in pairs[("A", "B")]] == [rows[:2], rows[2:]]
    assert all(p.complete for p in pairs[("A", "B")])


def test_complementary_orphans_in_different_sources_stay_incomplete() -> None:
    pairs, _ = group_pairs([game(0), game(1, source="two.pgn", reverse=True)])
    assert len(pairs[("A", "B")]) == 2
    assert not any(p.complete for p in pairs[("A", "B")])


def test_explicit_runs_resolve_appended_pair_ids() -> None:
    rows = [game(0, run="first"), game(1, run="second"),
            game(2, run="second", reverse=True), game(3, run="first", reverse=True)]
    pairs, _ = group_pairs(rows)
    assert [[g.order for g in p.games] for p in pairs[("A", "B")]] == [[1, 2], [0, 3]]


@pytest.mark.parametrize("count", [3, 4])
def test_untagged_appended_collision_refused(count: int) -> None:
    with pytest.raises(SystemExit, match=r"one.pgn.*PairId='0'"):
        group_pairs([game(i, reverse=bool(i % 2)) for i in range(count)])


def test_duplicate_orientation_refused() -> None:
    with pytest.raises(SystemExit, match="ambiguous pair"):
        group_pairs([game(0), game(1)])


@pytest.mark.parametrize("half", ["0", "2", "bad"])
def test_repeated_or_invalid_half_refused(half: str) -> None:
    with pytest.raises(SystemExit, match="ambiguous pair"):
        group_pairs([game(0, half="0"), game(1, half=half, reverse=True)])


@pytest.mark.parametrize("runs", [(None, "known"), ("known", None)])
def test_partial_run_identity_refused(runs: tuple[str | None, str | None]) -> None:
    with pytest.raises(SystemExit, match="mixes known and missing RunId"):
        group_pairs([game(0, run=runs[0]), game(1, run=runs[1], reverse=True)])


@pytest.mark.parametrize("count", [1, 2])
def test_legacy_pair_without_optional_metadata(count: int) -> None:
    pairs, singles = group_pairs([game(i, reverse=bool(i)) for i in range(count)])
    assert singles == 0
    assert len(pairs[("A", "B")]) == 1
    assert pairs[("A", "B")][0].complete == (count == 2)


def test_missing_pair_ids_remain_singletons() -> None:
    pairs, singles = group_pairs([game(0, pair_id=None), game(1, pair_id=None)])
    assert singles == 2
    assert [len(p.games) for p in pairs[("A", "B")]] == [1, 1]


def pgn(*, reverse: bool = False, run: str | None = None,
        half: str | None = None) -> str:
    tags = f'[White "{"B" if reverse else "A"}"]\n[Black "{"A" if reverse else "B"}"]\n'
    tags += '[Result "1/2-1/2"]\n[PairId "0"]\n'
    if run is not None:
        tags += f'[RunId "{run}"]\n'
    if half is not None:
        tags += f'[PairHalf "{half}"]\n'
    return tags + '\n1/2-1/2\n\n'


def test_reader_retains_source_and_explicit_identity(tmp_path: Path) -> None:
    path = tmp_path / "tiny.pgn"
    path.write_text(pgn(run="first", half="0") + pgn(reverse=True, run="first", half="1"),
                    encoding="utf-8")
    rows = read_games([path])
    assert [(g.source, g.run_id, g.pair_half) for g in rows] == [
        (str(path), "first", "0"), (str(path), "first", "1")]
    assert group_pairs(rows)[0][("A", "B")][0].complete


def test_reader_blank_identity_is_unknown(tmp_path: Path) -> None:
    path = tmp_path / "tiny.pgn"
    path.write_text(pgn(run=" ", half=" "), encoding="utf-8")
    row, = read_games([path])
    assert row.run_id is None
    assert row.pair_half is None


@pytest.mark.parametrize("tagged", [False, True])
def test_appended_pgn_requires_resolving_metadata(tmp_path: Path, tagged: bool) -> None:
    path = tmp_path / "appended.pgn"
    path.write_text("".join(pgn(reverse=bool(i % 2), run=str(i // 2) if tagged else None)
                            for i in range(4)), encoding="utf-8")
    rows = read_games([path])
    if tagged:
        pairs, _ = group_pairs(rows)
        assert len(pairs[("A", "B")]) == 2
        assert all(p.complete for p in pairs[("A", "B")])
    else:
        with pytest.raises(SystemExit, match="ambiguous pair"):
            group_pairs(rows)
