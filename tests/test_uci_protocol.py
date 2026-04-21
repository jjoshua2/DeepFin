from __future__ import annotations

import pytest

from chess_anti_engine.uci.protocol import (
    CmdGo,
    CmdIsReady,
    CmdPonderHit,
    CmdPosition,
    CmdQuit,
    CmdSetOption,
    CmdStop,
    CmdUci,
    CmdUciNewGame,
    CmdUnknown,
    InfoFields,
    format_bestmove,
    format_info,
    parse_command,
)


@pytest.mark.parametrize("line,expected_type", [
    ("uci", CmdUci),
    ("isready", CmdIsReady),
    ("ucinewgame", CmdUciNewGame),
    ("ponderhit", CmdPonderHit),
    ("stop", CmdStop),
    ("quit", CmdQuit),
])
def test_bare_commands(line: str, expected_type: type) -> None:
    assert isinstance(parse_command(line), expected_type)


def test_blank_and_unknown() -> None:
    assert isinstance(parse_command(""), CmdUnknown)
    assert isinstance(parse_command("   "), CmdUnknown)
    assert isinstance(parse_command("garbage"), CmdUnknown)


def test_position_startpos() -> None:
    cmd = parse_command("position startpos")
    assert isinstance(cmd, CmdPosition)
    assert cmd.fen is None
    assert cmd.moves == ()


def test_position_startpos_with_moves() -> None:
    cmd = parse_command("position startpos moves e2e4 e7e5 g1f3")
    assert isinstance(cmd, CmdPosition)
    assert cmd.fen is None
    assert cmd.moves == ("e2e4", "e7e5", "g1f3")


def test_position_fen() -> None:
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    cmd = parse_command(f"position fen {fen}")
    assert isinstance(cmd, CmdPosition)
    assert cmd.fen == fen
    assert cmd.moves == ()


def test_position_fen_with_moves() -> None:
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    cmd = parse_command(f"position fen {fen} moves d2d4 g8f6")
    assert isinstance(cmd, CmdPosition)
    assert cmd.fen == fen
    assert cmd.moves == ("d2d4", "g8f6")


def test_go_movetime() -> None:
    cmd = parse_command("go movetime 1500")
    assert isinstance(cmd, CmdGo)
    assert cmd.args.movetime_ms == 1500


def test_go_nodes_depth() -> None:
    cmd = parse_command("go nodes 500 depth 10")
    assert isinstance(cmd, CmdGo)
    assert cmd.args.nodes == 500
    assert cmd.args.depth == 10


def test_go_full_clock() -> None:
    cmd = parse_command("go wtime 60000 btime 55000 winc 1000 binc 1000 movestogo 20")
    assert isinstance(cmd, CmdGo)
    assert cmd.args.wtime_ms == 60000
    assert cmd.args.btime_ms == 55000
    assert cmd.args.winc_ms == 1000
    assert cmd.args.binc_ms == 1000
    assert cmd.args.movestogo == 20


def test_go_infinite() -> None:
    cmd = parse_command("go infinite")
    assert isinstance(cmd, CmdGo)
    assert cmd.args.infinite is True
    assert cmd.args.ponder is False


def test_go_ponder() -> None:
    cmd = parse_command("go ponder wtime 10000 btime 10000")
    assert isinstance(cmd, CmdGo)
    assert cmd.args.ponder is True
    assert cmd.args.wtime_ms == 10000


def test_go_searchmoves() -> None:
    cmd = parse_command("go searchmoves e2e4 d2d4 movetime 500")
    assert isinstance(cmd, CmdGo)
    assert cmd.args.searchmoves == ("e2e4", "d2d4")
    assert cmd.args.movetime_ms == 500


def test_go_malformed_int_ignored() -> None:
    cmd = parse_command("go movetime notanumber depth 3")
    assert isinstance(cmd, CmdGo)
    assert cmd.args.movetime_ms is None
    assert cmd.args.depth == 3


def test_setoption_name_only() -> None:
    cmd = parse_command("setoption name Threads")
    assert isinstance(cmd, CmdSetOption)
    assert cmd.name == "Threads"
    assert cmd.value is None


def test_setoption_name_value() -> None:
    cmd = parse_command("setoption name UCI_Hash value 128")
    assert isinstance(cmd, CmdSetOption)
    assert cmd.name == "UCI_Hash"
    assert cmd.value == "128"


def test_setoption_multiword_name() -> None:
    cmd = parse_command("setoption name Skill Level value 20")
    assert isinstance(cmd, CmdSetOption)
    assert cmd.name == "Skill Level"
    assert cmd.value == "20"


def test_debug_is_ignored() -> None:
    assert isinstance(parse_command("debug on"), CmdUnknown)
    assert isinstance(parse_command("debug off"), CmdUnknown)


def test_format_bestmove() -> None:
    assert format_bestmove("e2e4") == "bestmove e2e4"
    assert format_bestmove("e2e4", ponder="e7e5") == "bestmove e2e4 ponder e7e5"


def test_format_info_minimal() -> None:
    assert format_info(InfoFields()) == "info"


def test_format_info_full() -> None:
    out = format_info(InfoFields(
        depth=8, seldepth=12, nodes=1000, nps=5000, time_ms=200,
        score_cp=35, pv=("e2e4", "e7e5"), hashfull_per_mille=250,
    ))
    assert "depth 8" in out
    assert "seldepth 12" in out
    assert "nodes 1000" in out
    assert "nps 5000" in out
    assert "time 200" in out
    assert "score cp 35" in out
    assert "pv e2e4 e7e5" in out
    assert "hashfull 250" in out


def test_format_info_mate_overrides_cp() -> None:
    out = format_info(InfoFields(score_cp=99, score_mate=3))
    assert "score mate 3" in out
    assert "score cp" not in out


def test_format_info_string_trailing() -> None:
    out = format_info(InfoFields(string="hello world"))
    assert out.endswith("string hello world")
