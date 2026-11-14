"""UCI command parsing and response formatting.

Pure logic — no engine, no threads, no I/O. Parses the subset of the UCI
protocol we implement (uci/isready/ucinewgame/position/go/ponderhit/stop/quit),
ignoring unknown commands per spec.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any


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
  # Why the command itself is malformed (no `startpos`/`fen`, a FEN with fewer
  # than four or more than six fields, a stray token before `moves`), or None.
  # The engine rejects such a command instead of guessing a position.
    error: str | None = None


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
  # UCI `go mate N`: search for a mate in N moves (see `limits_from_go`).
    mate: int | None = None
  # Values dropped as unusable -- unparseable (`movetime abc`), missing
  # (`movetime` then another keyword), or out of range (`mate 0`) -- as
  # "<keyword> <value>"; the engine names them on an `info string`, and a dropped
  # BOUND keyword gets the implicit cap (see `limits_from_go`).
    errors: tuple[str, ...] = ()
  # Values used only after rounding toward zero (`movetime 999.9` -> 999), as
  # "<keyword> <sent> -> <used>"; the engine names them on an `info string`.
    truncated: tuple[str, ...] = ()


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
class CmdSearchConfig:
    """`searchconfig` — dump every realized search parameter, LIVE or INERT.

    Not a UCI-spec command; no GUI sends it. It exists for the operator who
    has to prove, from the engine's own mouth, that the values in a TCEC /
    cutechess config reached the search. UCI has no readback for options at
    all, which is precisely how a knob gets set for a whole tournament while
    doing nothing.
    """


@dataclass(frozen=True)
class CmdUnknown:
    raw: str


Command = (
    CmdUci | CmdIsReady | CmdUciNewGame | CmdPosition | CmdGo | CmdPonderHit
    | CmdStop | CmdQuit | CmdSetOption | CmdSearchConfig | CmdUnknown
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
    "mate": "mate",
}


def _parse_go_int(text: str) -> tuple[int, bool] | None:
    """``(value, exact)`` for a numeric token: an integer, or a finite float
    rounded toward zero (``exact`` False unless it was integral, so ``1000.0``
    is exact and ``999.9`` is not). None when the token is not a finite number."""
    try:
        return int(text), True
    except ValueError:
        pass
    try:
        value = float(text)
    except ValueError:
        return None
    if value != value or value in (float("inf"), float("-inf")):
        return None
    return int(value), value.is_integer()


def _parse_go(tokens: list[str]) -> GoArgs:
    kwargs: dict[str, Any] = {}
    searchmoves: list[str] = []
    errors: list[str] = []
    truncated: list[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok == "ponder":
            kwargs["ponder"] = True
            i += 1
        elif tok == "infinite":
            kwargs["infinite"] = True
            i += 1
        elif tok in _INT_KEYS:
            if i + 1 >= len(tokens) or tokens[i + 1] in _GO_KEYWORDS:
                errors.append(f"{tok} (no value)")
                i += 1
                continue
            parsed = _parse_go_int(tokens[i + 1])
            if parsed is None:
                errors.append(f"{tok} {tokens[i + 1]}")
            elif tok == "mate" and parsed[0] <= 0:
                # "Mate in 0 / -2 moves" names no search: dropped like an
                # unparseable value (so with no other bound the cap applies).
                errors.append(f"{tok} {tokens[i + 1]}")
            else:
                value, exact = parsed
                if not exact:
                    truncated.append(f"{tok} {tokens[i + 1]} -> {value}")
                kwargs[_INT_KEYS[tok]] = value
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
    if errors:
        kwargs["errors"] = tuple(errors)
    if truncated:
        kwargs["truncated"] = tuple(truncated)
    return GoArgs(**kwargs)


_GO_KEYWORDS = frozenset({
    "ponder", "infinite", "searchmoves",
    "movetime", "nodes", "depth", "mate",
    "wtime", "btime", "winc", "binc", "movestogo",
})


def _parse_fen_fields(
    tokens: list[str], *, start: int,
) -> tuple[str | None, int, str | None]:
    """``(fen, next_index, error)`` for the FEN beginning at ``tokens[start]``.

    A full FEN is six space-separated fields (pieces stm castling ep halfmove
    fullmove), but EPD-derived tooling routinely sends only the first four or
    five. Requiring six meant such a command silently selected the START
    POSITION and the engine then returned a confidently-scored move for a
    completely different board, with no diagnostic anywhere. Pad the missing
    move counters instead — they affect 50-move / repetition bookkeeping, not
    the position. Fewer than four fields, or more than six before ``moves``, is
    not a FEN: ``error`` says so and the engine rejects the command. (Both used
    to be silent: fewer than four selected the START POSITION, and a seventh
    token was dropped.)
    """
    end = tokens.index("moves", start) if "moves" in tokens[start:] else len(tokens)
    fields = tokens[start:end]
    if len(fields) < 4:
        return None, end, (
            f"FEN has {len(fields)} field(s), need 4-6: {' '.join(fields)!r}"
        )
    if len(fields) > 6:
        return None, end, (
            f"FEN has {len(fields)} fields, at most 6 before 'moves': "
            f"{' '.join(fields)!r}"
        )
    if len(fields) == 4:
        fields = [*fields, "0", "1"]
    elif len(fields) == 5:
        fields = [*fields, "1"]
    return " ".join(fields), end, None


def _parse_position(tokens: list[str]) -> CmdPosition:
    if not tokens:
        return CmdPosition(fen=None, error="no position given (expected 'startpos' or 'fen')")
    fen: str | None = None
    moves: list[str] = []
    i = 0
    if tokens[0] == "startpos":
        fen = None
        i = 1
    elif tokens[0] == "fen":
        fen, i, error = _parse_fen_fields(tokens, start=1)
        if error is not None:
            return CmdPosition(fen=None, error=error)
    elif "/" in tokens[0]:
  # Bare FEN with no `fen` keyword. Non-standard but common in EPD-derived
  # tooling; the alternative is answering for the START POSITION.
        fen, i, error = _parse_fen_fields(tokens, start=0)
        if error is not None:
            return CmdPosition(fen=None, error=error)
    else:
        return CmdPosition(
            fen=None, error=f"expected 'startpos' or 'fen', got {tokens[0]!r}",
        )
    if i < len(tokens):
        if tokens[i] != "moves":
            return CmdPosition(
                fen=None,
                error=f"unexpected {' '.join(tokens[i:])!r} after the position",
            )
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
        case "searchconfig":
            return CmdSearchConfig()
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
  # Per-mille (W, D, L) from the stm's perspective. When set, emitted as
  # `wdl W D L` after the score field per UCI_ShowWDL convention.
    wdl: tuple[int, int, int] | None = None
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
    if f.wdl is not None:
        w, d, l = f.wdl
        parts += ["wdl", str(w), str(d), str(l)]
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
