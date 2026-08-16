"""PGN output for arena matches, for pooled rating fits (Ordo/BayesElo).

``scripts/arena_standard.py`` scores a match with PAIRED openings and
pentanomial statistics, which is the right instrument for a single A/B. It is
the wrong instrument for "three arms played a round robin, fit all their
ratings jointly" — that is what Ordo does, and Ordo eats PGN.

Two things this module is deliberate about:

* **Traceability.** A pooled fit merges games from many runs into one number.
  If a game cannot say WHICH checkpoint and WHICH search shape produced it,
  merging it is how a "same name, different population" error gets in — two
  arms whose PGNs both say ``White "candidate"`` silently become one player.
  So every game carries the engine identities, ``ConfigHash``, ``GitSha`` and
  the realized search shape of BOTH sides.
* **Pair identity.** Ordo's per-game state is exactly
  ``{whiteplayer, blackplayer, score}`` (``mytypes.h:90``) — the opening never
  enters its likelihood, and its ``-s`` resampler regenerates each game
  independently from the fitted model, so it cannot exploit our mirrored
  pairs. Recording ``PairId``/``PairHalf`` keeps that information IN the PGN,
  so a pair-level block bootstrap can recover it later even though Ordo itself
  will ignore the tags.

Engine names are sanitized to ``[A-Za-z0-9_.+-]``: they are parsed back out of
Ordo's whitespace-column output and matched by its ``-A`` anchor flag, so a
name containing a space silently breaks both. Names are also never derived
from an absolute path — this repo is public.
"""

from __future__ import annotations

import datetime
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType

import chess
import chess.pgn

__all__ = [
    "ArenaGame",
    "ArenaPgnWriter",
    "engine_name_from_checkpoint",
    "sanitize_engine_name",
]

_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.+-]+")
_RESULTS = frozenset({"1-0", "0-1", "1/2-1/2", "*"})


def sanitize_engine_name(name: str, *, fallback: str = "unknown") -> str:
    """Collapse anything Ordo cannot round-trip into ``_``.

    Ordo prints ratings in whitespace-delimited columns and selects the anchor
    by exact name match, so whitespace and quotes in a player name are not a
    cosmetic problem — they corrupt the parse and silently mis-anchor the fit.
    """
    cleaned = _SAFE_NAME.sub("_", name).strip("_")
    return cleaned or fallback


def engine_name_from_checkpoint(path: str, *, fallback: str) -> str:
    """A stable, path-free engine identity for a checkpoint.

    Uses the last two path components (e.g. ``arm_A_iter100/checkpoint_000099``
    -> ``arm_A_iter100_checkpoint_000099``) because the basename ALONE is not
    unique across arms: every arm has a ``checkpoint_000099``, and collapsing
    them into one player is exactly the pooled-fit contamination this module
    exists to prevent. Directory prefixes above those two are dropped so no
    absolute path reaches the file.

    This is a DEFAULT, not a guarantee of uniqueness. Callers that know the
    experiment's labels should pass them explicitly.
    """
    parts = [p for p in Path(path).parts if p not in ("/", "")]
    tail = "_".join(parts[-2:]) if len(parts) >= 2 else (parts[-1] if parts else "")
    return sanitize_engine_name(tail, fallback=fallback)


@dataclass(frozen=True)
class ArenaGame:
    """One finished arena game, ready to serialize."""

    white: str
    black: str
    result: str
    moves: tuple[chess.Move, ...] = ()
    start_fen: str | None = None
    pair_id: int | None = None
    pair_half: int | None = None
    extra: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.result not in _RESULTS:
            raise ValueError(
                f"result must be one of {sorted(_RESULTS)}, got {self.result!r}"
            )


class ArenaPgnWriter:
    """Append-only PGN writer that flushes after every game.

    Flushing per game is the point, not a detail: it is what lets a partial or
    killed run still leave a VALID, parseable PGN that Ordo can fit. The arena
    is routinely run under ``timeout -k`` and a block-buffered file loses its
    tail on SIGKILL (the same failure that cost the ratchet its rows —
    see ``print_summary`` in ``scripts/arena_standard.py``).
    """

    def __init__(
        self,
        path: str | Path,
        *,
        event: str = "arena",
        site: str = "?",
        date: str | None = None,
        base_tags: Mapping[str, str] | None = None,
    ) -> None:
        self.path = Path(path)
        self.event = event
        self.site = site
        self.date = date or datetime.date.today().strftime("%Y.%m.%d")
        self.base_tags = dict(base_tags or {})
        self.games_written = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")

    def __enter__(self) -> ArenaPgnWriter:
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.flush()
            self._fh.close()

    def game_to_text(self, game: ArenaGame, *, round_tag: str) -> str:
        node: chess.pgn.GameNode
        pgn_game = chess.pgn.Game()
        if game.start_fen is not None:
            board = chess.Board(game.start_fen)
            # setup() writes the FEN/SetUp tags itself, and omits them when the
            # position is the standard start — which is what a book opening
            # from the initial position should look like.
            pgn_game.setup(board)
        pgn_game.headers["Event"] = self.event
        pgn_game.headers["Site"] = self.site
        pgn_game.headers["Date"] = self.date
        pgn_game.headers["Round"] = round_tag
        pgn_game.headers["White"] = game.white
        pgn_game.headers["Black"] = game.black
        pgn_game.headers["Result"] = game.result
        for key, value in self.base_tags.items():
            pgn_game.headers[key] = str(value)
        if game.pair_id is not None:
            pgn_game.headers["PairId"] = str(game.pair_id)
        if game.pair_half is not None:
            pgn_game.headers["PairHalf"] = str(game.pair_half)
        for key, value in game.extra.items():
            pgn_game.headers[key] = str(value)

        node = pgn_game
        for move in game.moves:
            node = node.add_variation(move)
        exporter = chess.pgn.StringExporter(
            headers=True, variations=False, comments=False,
        )
        return pgn_game.accept(exporter)

    def write_game(self, game: ArenaGame) -> None:
        round_tag = (
            f"{game.pair_id + 1}.{game.pair_half + 1}"
            if game.pair_id is not None and game.pair_half is not None
            else str(self.games_written + 1)
        )
        self._fh.write(self.game_to_text(game, round_tag=round_tag))
        self._fh.write("\n\n")
        self._fh.flush()
        self.games_written += 1


def read_pair_blocks(path: str | Path) -> Iterator[tuple[str, list[tuple[str, str, str]]]]:
    """Yield ``(pair_key, [(white, black, result), ...])`` from an arena PGN.

    The unit a block bootstrap must resample. Games without a ``PairId`` are
    yielded as their own singleton block, so a mixed file degrades to a plain
    game-level bootstrap for those rows rather than silently grouping them.
    """
    import chess.pgn as pgn_mod

    blocks: dict[str, list[tuple[str, str, str]]] = {}
    order: list[str] = []
    with Path(path).open(encoding="utf-8") as fh:
        idx = 0
        while True:
            game = pgn_mod.read_game(fh)
            if game is None:
                break
            headers = game.headers
            pair = headers.get("PairId")
            key = (
                f"{headers.get('White', '?')}|{headers.get('Black', '?')}|pair{pair}"
                if pair is not None
                else f"__single_{idx}"
            )
            # Pair identity must be scoped by the ENGINES too: two different
            # matches both numbering their pairs from 0 would otherwise merge.
            if pair is not None:
                names = sorted((headers.get("White", "?"), headers.get("Black", "?")))
                key = f"{names[0]}|{names[1]}|pair{pair}"
            if key not in blocks:
                blocks[key] = []
                order.append(key)
            blocks[key].append(
                (
                    headers.get("White", "?"),
                    headers.get("Black", "?"),
                    headers.get("Result", "*"),
                )
            )
            idx += 1
    for key in order:
        yield key, blocks[key]
