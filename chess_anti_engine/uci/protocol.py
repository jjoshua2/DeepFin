"""UCI command parsing and response formatting.

Pure logic — no engine, no threads, no I/O. Parses the subset of the UCI
protocol we implement (uci/isready/ucinewgame/position/go/ponderhit/stop/quit),
ignoring unknown commands per spec.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


@dataclass(frozen=True)
class CmdUci:
    pass


@dataclass(frozen=True)
class CmdIsReady:
    pass


@dataclass(frozen=True)
class CmdUciNewGame:
    pass


@dataclass(frozen=True)
class CmdPosition:
    # fen is None for startpos
    fen: str | None
    moves: tuple[str, ...] = ()


@dataclass(frozen=True)
class GoArgs:
    ponder: bool = False
    movetime_ms: int | None = None
    nodes: int | None = None
    depth: int | None = None
    wtime_ms: int | None = None
    btime_ms: int | None = None
    winc_ms: int | None = None
    binc_ms: int | None = None
    movestogo: int | None = None
    infinite: bool = False
    searchmoves: tuple[str, ...] = ()


@dataclass(frozen=True)
class CmdGo:
    args: GoArgs


@dataclass(frozen=True)
class CmdPonderHit:
    pass


@dataclass(frozen=True)
class CmdStop:
    pass


@dataclass(frozen=True)
class CmdQuit:
    pass


@dataclass(frozen=True)
class CmdSetOption:
    name: str
    value: str | None


@dataclass(frozen=True)
class CmdUnknown:
    raw: str


Command = (
    CmdUci | CmdIsReady | CmdUciNewGame | CmdPosition | CmdGo | CmdPonderHit
    | CmdStop | CmdQuit | CmdSetOption | CmdUnknown
)


_INT_KEYS = {
    "movetime": "movetime_ms",
    "nodes": "nodes",
    "depth": "depth",
    "wtime": "wtime_ms",
    "btime": "btime_ms",
    "winc": "winc_ms",
    "binc": "binc_ms",
    "movestogo": "movestogo",
}


def _parse_go(tokens: list[str]) -> GoArgs:
    kwargs: dict[str, object] = {}
    searchmoves: list[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok == "ponder":
            kwargs["ponder"] = True
            i += 1
        elif tok == "infinite":
            kwargs["infinite"] = True
            i += 1
        elif tok in _INT_KEYS and i + 1 < len(tokens):
            try:
                kwargs[_INT_KEYS[tok]] = int(tokens[i + 1])
            except ValueError:
                pass
            i += 2
        elif tok == "searchmoves":
            i += 1
            while i < len(tokens) and tokens[i] not in _GO_KEYWORDS:
                searchmoves.append(tokens[i])
                i += 1
        else:
            i += 1
    if searchmoves:
        kwargs["searchmoves"] = tuple(searchmoves)
    return GoArgs(**kwargs)  # type: ignore[arg-type]


_GO_KEYWORDS = frozenset({
    "ponder", "infinite", "searchmoves",
    "movetime", "nodes", "depth",
    "wtime", "btime", "winc", "binc", "movestogo",
})


def _parse_position(tokens: list[str]) -> CmdPosition:
    if not tokens:
        return CmdPosition(fen=None)
    fen: str | None = None
    moves: list[str] = []
    i = 0
    if tokens[0] == "startpos":
        fen = None
        i = 1
    elif tokens[0] == "fen":
        # FEN is 6 space-separated tokens: pieces stm castling ep halfmove fullmove
        if len(tokens) >= 7:
            fen = " ".join(tokens[1:7])
            i = 7
        else:
            return CmdPosition(fen=None)
    if i < len(tokens) and tokens[i] == "moves":
        moves = list(tokens[i + 1:])
    return CmdPosition(fen=fen, moves=tuple(moves))


def _parse_setoption(tokens: list[str]) -> CmdSetOption:
    # setoption name <...> [value <...>]
    if not tokens or tokens[0] != "name":
        return CmdSetOption(name="", value=None)
    try:
        value_idx = tokens.index("value")
        name = " ".join(tokens[1:value_idx])
        value: str | None = " ".join(tokens[value_idx + 1:])
    except ValueError:
        name = " ".join(tokens[1:])
        value = None
    return CmdSetOption(name=name, value=value)


def parse_command(line: str) -> Command:
    """Parse one UCI input line. Never raises; unknown → CmdUnknown."""
    stripped = line.strip()
    if not stripped:
        return CmdUnknown(raw=line)
    tokens = stripped.split()
    head = tokens[0]
    rest = tokens[1:]
    match head:
        case "uci":
            return CmdUci()
        case "isready":
            return CmdIsReady()
        case "ucinewgame":
            return CmdUciNewGame()
        case "position":
            return _parse_position(rest)
        case "go":
            return CmdGo(_parse_go(rest))
        case "ponderhit":
            return CmdPonderHit()
        case "stop":
            return CmdStop()
        case "quit":
            return CmdQuit()
        case "setoption":
            return _parse_setoption(rest)
        case _:
            return CmdUnknown(raw=line)


# --- formatters --------------------------------------------------------------


@dataclass
class InfoFields:
    depth: int | None = None
    seldepth: int | None = None
    multipv: int | None = None
    nodes: int | None = None
    nps: int | None = None
    time_ms: int | None = None
    score_cp: int | None = None
    score_mate: int | None = None
    pv: tuple[str, ...] = field(default_factory=tuple)
    hashfull_per_mille: int | None = None
    tbhits: int | None = None
    string: str | None = None


def format_info(f: InfoFields) -> str:
    parts: list[str] = ["info"]
    if f.depth is not None:
        parts += ["depth", str(f.depth)]
    if f.seldepth is not None:
        parts += ["seldepth", str(f.seldepth)]
    if f.multipv is not None:
        parts += ["multipv", str(f.multipv)]
    if f.nodes is not None:
        parts += ["nodes", str(f.nodes)]
    if f.nps is not None:
        parts += ["nps", str(f.nps)]
    if f.time_ms is not None:
        parts += ["time", str(f.time_ms)]
    if f.hashfull_per_mille is not None:
        parts += ["hashfull", str(f.hashfull_per_mille)]
    if f.tbhits is not None:
        parts += ["tbhits", str(f.tbhits)]
    if f.score_mate is not None:
        parts += ["score", "mate", str(f.score_mate)]
    elif f.score_cp is not None:
        parts += ["score", "cp", str(f.score_cp)]
    if f.pv:
        parts.append("pv")
        parts.extend(f.pv)
    if f.string is not None:
        parts += ["string", f.string]
    return " ".join(parts)


def format_bestmove(move: str, ponder: str | None = None) -> str:
    if ponder:
        return f"bestmove {move} ponder {ponder}"
    return f"bestmove {move}"


def format_id_lines(name: str, author: str) -> Iterable[str]:
    yield f"id name {name}"
    yield f"id author {author}"


def format_uciok() -> str:
    return "uciok"


def format_readyok() -> str:
    return "readyok"
