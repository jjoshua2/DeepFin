from __future__ import annotations

import io
import logging
import re
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import chess
import chess.pgn

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class OpeningConfig:
  # If set, sample openings from this file.
  # Supported:
  # - .bin (Polyglot)
  # - .pgn
  # - .pgn.zip (a zip containing one or more .pgn files)
    opening_book_path: str | None = None

  # How many plies to apply from the book (max). For the stockfish 2-move book,
  # this would typically be 4 (2 full moves).
    opening_book_max_plies: int = 4

  # For PGN-based books: maximum number of games to parse (0 means all; can be slow).
    opening_book_max_games: int = 200_000

  # Probability of using the opening book (vs random-start plies).
    opening_book_prob: float = 1.0

  # Optional second opening book (e.g. a deeper 8-move book).
    opening_book_path_2: str | None = None
    opening_book_max_plies_2: int = 16
    opening_book_max_games_2: int = 200_000
  # Fraction of book games that use book 2 (the remainder use book 1).
    opening_book_mix_prob_2: float = 0.0

  # If >0, play this many random legal plies from the start position.
    random_start_plies: int = 0

  # Blind-spot FEN seeding: if set, with probability opening_fen_prob the game
  # starts from a seed sampled uniformly from this file (one seed per line;
  # blanks, '#' comment lines and inline '# ...' comments are skipped) instead
  # of the book/random/startpos flow. Each line is either a plain FEN or
  # '<start_fen> | <uci moves>' (see seed_board_from_line): the latter replays
  # the moves so the seed carries real LC0 history rather than repeat-filled
  # planes. Used to replay positions the net has historically misplayed (see
  # data/blindspot_fens_v1.txt) so refutations enter the training distribution
  # — search cannot rescue value-blind positions, only data can
  # (docs/experiment_ledger.md, 2026-07-03 probe).
    opening_fen_list_path: str | None = None
    opening_fen_prob: float = 0.0

  # When a FEN start is used, force the net to play the FEN's side to move
  # (the seat that blundered there). Without this, color alternation puts the
  # net on the punisher side half the time — useful data, but the point of
  # seeding is making the net face the decision it historically got wrong.
  # Net-vs-net selfplay slots are unaffected (the net plays both seats).
    opening_fen_net_side_to_move: bool = True

  # Seed ONLY selfplay slots (never curriculum/net-vs-SF slots). Curriculum games
  # play a long SF opponent that saturates the shared StockfishPool, so at a high
  # dose the short fenlist seeds crowd curriculum completions out of the fixed
  # per-iteration window and STARVE the PID winrate sample (see the FEN-seed
  # curriculum-starvation ticket). With this on, raising opening_fen_prob only ever
  # adds net-vs-net seed games, so the curriculum/PID supply is dose-invariant and
  # the dose can go much higher. Off = legacy (seed any slot). The SF-labelled
  # blind-spot VALUE injection rides on selfplay seeds anyway, so the only thing
  # lost is the curriculum-seed refutation variant. Multiply the intended net dose
  # by ~1/selfplay_fraction when enabling (seeds land on ~selfplay_fraction of slots).
    opening_fen_selfplay_only: bool = False

  # Server-doled deterministic seeding (supersedes the probabilistic opening_fen_prob
  # path when >0). Each iteration the server hands the WHOLE seed list — each seed
  # once, repeated this many times — to exactly ONE worker poll (see the server dole
  # gate); that worker plays them as selfplay seeds while every other worker plays
  # normal games, so total seeded games == N*len(list) per iteration regardless of
  # the client count and each seed gets EVEN, variance-free coverage (unlike the
  # Poisson-variance opening_fen_prob draw). While this is >0 the probabilistic draw
  # in sample_starting_board is disabled for every slot; seeding happens only by
  # draining the dole queue (resolve_slot_opening). 0 = off (legacy prob path).
    opening_fen_dole_per_iter: int = 0

  # SF-refutation channel on top of the selfplay dole (value-blind-spot lever).
  # Each doled iteration the worker also schedules a round-robin slice of the
  # seed list as short curriculum games: SF plays the first
  # ``opening_fen_sf_refute_plies`` opponent plies (punishing the fantasy eval),
  # then the slot hands off to selfplay for the rest. 0 frac = off. Typical
  # start: frac=0.5, plies=5 (~half the list × 5 SF replies — iso-cheap vs a
  # full-length SF seed game). Fenlist outcomes stay out of the PID sample
  # (finalize: source startswith "fenlist").
    opening_fen_sf_refute_frac: float = 0.0
    opening_fen_sf_refute_plies: int = 5

  # SF-refute opp-row recording (STAGED — all default OFF; no-op until flipped).
  # Today the SF-played refute plies generate NO training rows, so the net never
  # trains its MAIN policy head on the punishing move. These three flags add
  # that, gated so the live 50%×N channel is byte-identical at defaults.
  #
  # sf_refute_full_node_moves: refute-phase SF MOVE queries skip the fast-ply
  #   0.25× node scale (sf_fast_ply_node_scale) and run at full base nodes, so
  #   the punishment is not weakened on fast plies. Move queries only; labels
  #   and ordinary curriculum behaviour untouched.
    sf_refute_full_node_moves: bool = False
  # sf_refute_record_opp_rows: each SF-played refute ply also emits a training
  #   row at the SF-to-move position — MAIN policy target = soft distribution
  #   over SF's MultiPV candidates, sf_wdl = same search's cp-logistic eval,
  #   outcome WDL filled at finalize with the SF-seat POV. All aux heads
  #   (policy_soft/future/sf, volatility, moves_left, sf_p0*) masked. Rows ride
  #   the fenlist_sf_refute* source so they stay out of the PID sample.
    sf_refute_record_opp_rows: bool = False
  # sf_refute_opp_policy_net_blend: when >0, blend the net's own MCTS visit
  #   distribution at the SF position into the policy target
  #   (target = (1-b)*SF_soft + b*net_visits). Wiring a net search at an
  #   opponent turn is architecturally invasive, so the net-visit provider is a
  #   marked seam and blend>0 is REJECTED at config validation (fail loud). The
  #   blend math + interface are implemented; only the provider is deferred.
    sf_refute_opp_policy_net_blend: float = 0.0


@dataclass(frozen=True)
class OpeningStart:
    board: chess.Board
    source: str


def _random_playout_from_start(*, rng, plies: int) -> chess.Board:
    b = chess.Board()
    for _ in range(int(plies)):
        if b.is_game_over():
            break
        moves = list(b.legal_moves)
        if not moves:
            break
        b.push(moves[int(rng.integers(0, len(moves)))])
    return b


def _iter_pgn_bytes_from_path(path: Path) -> Iterable[bytes]:
    """Yield raw PGN bytes from a file that may be a .pgn or .pgn.zip."""
    suffixes = "".join(path.suffixes).lower()

    if suffixes.endswith(".pgn"):
        yield path.read_bytes()
        return

    if suffixes.endswith((".pgn.zip", ".zip")):
        with zipfile.ZipFile(path) as z:
  # Heuristic: read all members ending in .pgn
            pgn_members = [n for n in z.namelist() if n.lower().endswith(".pgn")]
            if not pgn_members:
                raise ValueError(f"No .pgn files found in zip: {path}")
            for name in pgn_members:
                with z.open(name, "r") as f:
                    yield f.read()
        return

    raise ValueError(f"Unsupported opening book format: {path}")


@lru_cache(maxsize=8)
def _load_pgn_opening_sequences(
    path_str: str, *, max_plies: int, max_games: int
) -> tuple[list[tuple[str, ...]], list[int]]:
    """Load PGN games into a weighted list of UCI move sequences.

    We aggregate identical prefixes into counts so large PGN books remain manageable.
    Cache keys are path-based, so replacing a book in place requires a process
    restart. In practice we version book filenames (`..._v2.pgn.zip`) when
    changing assets, which keeps cache invalidation explicit.

    Returns (seqs, weights).
    """
    path = Path(path_str)
    counts: dict[tuple[str, ...], int] = {}

    games_read = 0
    for blob in _iter_pgn_bytes_from_path(path):
        pgn_io = io.StringIO(blob.decode("utf-8", errors="ignore"))
        while True:
            game = chess.pgn.read_game(pgn_io)
            if game is None:
                break
            node = game
            seq: list[str] = []
            for _ in range(int(max_plies)):
                nxt = node.variation(0) if node.variations else None
                if nxt is None:
                    break
                mv = nxt.move
                seq.append(mv.uci())
                node = nxt

            if seq:
                key = tuple(seq)
                counts[key] = counts.get(key, 0) + 1

            games_read += 1
            if int(max_games) > 0 and games_read >= int(max_games):
                break

        if int(max_games) > 0 and games_read >= int(max_games):
            break

    if not counts:
        return ([], [])

    seqs = list(counts.keys())
    weights = [counts[s] for s in seqs]
    return (seqs, weights)


def _sample_from_pgn(*, rng, path: str, max_plies: int, max_games: int) -> chess.Board:
    seqs, weights = _load_pgn_opening_sequences(path, max_plies=int(max_plies), max_games=int(max_games))
    if not seqs:
        return chess.Board()

    import numpy as np

    p = np.array(weights, dtype=np.float64)
    p /= float(p.sum())
    idx = int(rng.choice(np.arange(len(seqs)), p=p))

    b = chess.Board()
    for u in seqs[idx]:
        mv = chess.Move.from_uci(u)
        if mv not in b.legal_moves:
            break
        b.push(mv)
    return b


def _sample_from_polyglot(*, rng, path: str, max_plies: int) -> chess.Board:
    import chess.polyglot

    b = chess.Board()

    with chess.polyglot.open_reader(path) as reader:
        for _ in range(int(max_plies)):
            if b.is_game_over():
                break
            entries = list(reader.find_all(b))
            if not entries:
                break

  # Polyglot entries have a weight; sample proportional to that weight.
            import numpy as np

            ws = np.array([max(0, int(e.weight)) for e in entries], dtype=np.float64)
            s = float(ws.sum())
            if s <= 0:
                mv = entries[int(rng.integers(0, len(entries)))].move
            else:
                ws /= s
                mv = entries[int(rng.choice(np.arange(len(entries)), p=ws))].move

            if mv not in b.legal_moves:
                break
            b.push(mv)

    return b


def seed_board_from_line(line: str) -> chess.Board:
    """Build the seed (terminal) board from one FEN-list line.

    Line grammar (backward-compatible):
      ``<fen>``                    plain seed — no move history (repeat-fill).
      ``<start_fen> | <uci> ...``  start from ``start_fen`` and replay the UCI
                                   moves; the TERMINAL position is the seed and
                                   its move_stack carries real LC0 history (the
                                   ~8 preceding plies), instead of the empty
                                   stack a bare FEN gets (which the encoder
                                   repeat-fills — 84/175 planes wrong, a ~14%
                                   top-1 flip; see docs/experiment_ledger.md).

    Raises ValueError on an unparseable FEN or an illegal move in the line.
    """
    fen_part, _, moves_part = line.partition("|")
    board = chess.Board(fen_part.strip())  # raises ValueError on a bad FEN
    for tok in moves_part.split():
        move = chess.Move.from_uci(tok)
        if move not in board.legal_moves:
            raise ValueError(f"illegal move {tok!r} in seed line")
        board.push(move)
    return board


def _fen_reject_reason(line: str) -> str | None:
    """Why this seed line is unusable, or None if it's fine.

    The seed is the TERMINAL position after replaying any history moves; it must
    be parseable (FEN + legal moves), legal, non-terminal (under claim-draw,
    matching CBoard's cboard_is_game_over which treats halfmove_clock>=100 as
    over), and have >=2 legal moves (a forced position is skipped by
    _apply_forced_moves, so the net never faces the seeded decision).
    """
    try:
        board = seed_board_from_line(line)
    except ValueError as exc:
        return str(exc) if "illegal move" in str(exc) else "unparseable FEN"
    if not board.is_valid():
        return "illegal position"  # missing kings, impossible pawns, etc.
    if board.is_game_over(claim_draw=True):
        return "terminal position"
    if board.legal_moves.count() < 2:
        return "forced (single legal move)"
    return None


@lru_cache(maxsize=8)
def _load_fen_list(path_str: str) -> tuple[str, ...]:
    """Load a seed list (one seed per line, '#' comments, blanks skipped).

    Each line is a seed in the ``seed_board_from_line`` grammar — a plain FEN
    or ``<start_fen> | <uci moves>`` for real history — with an optional inline
    ``# ...`` provenance comment (src/round/ply from mining) stripped here.
    Malformed/unusable lines are SKIPPED with a logged warning rather than
    raised — one bad line in a hand-edited or future seed file must not crash
    the whole worker fleet (Codex review). We raise only when the file yields
    ZERO usable seeds, so an empty/all-comment/all-bad file fails fast at
    startup (warm_opening_book_cache) instead of mid-run. Cache keys are
    path-based like the PGN loader: version the filename to replace a list.
    """
    fens: list[str] = []
    skipped: list[str] = []
    # utf-8-sig: strip a leading BOM (editors/exporters on Windows add one) so
    # the first FEN — or a first-line '# comment' — is not corrupted.
    for lineno, raw in enumerate(Path(path_str).read_text(encoding="utf-8-sig").splitlines(), 1):
        # Strip an inline '# provenance' comment (neither FEN nor UCI contains
        # '#'), then whitespace; blank / whole-line-comment lines drop out.
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        reason = _fen_reject_reason(line)
        if reason is not None:
            skipped.append(f"{path_str}:{lineno} ({reason}): {line!r}")
            continue
        fens.append(line)
    if skipped:
        _log.warning(
            "opening FEN list %s: skipped %d unusable seed(s):\n  %s",
            path_str, len(skipped), "\n  ".join(skipped),
        )
    if not fens:
        raise ValueError(
            f"opening FEN list {path_str} has no usable seeds "
            f"(skipped {len(skipped)}); fix the file and restart",
        )
    return tuple(fens)


def _sample_fen_list_line(*, rng, path: str) -> str:
    fens = _load_fen_list(path)
    return fens[int(rng.integers(0, len(fens)))]


_WEIGHT_RE = re.compile(r"\bweight=(\d+)\b")
_MAX_SEED_WEIGHT = 16  # typo guard: one seed can never exceed 16 games/iter/dose


@lru_cache(maxsize=8)
def _load_fen_backed_bodies(path_str: str) -> frozenset[str]:
    """Stripped bodies of seed lines whose comment carries ``dropped=`` —
    blame-backed decision-point seeds (scripts/blindspot_deepsf_gate.py).

    Backed seeds open at a HOLDABLE ancestor of the lost position, so their
    outcome telemetry means the opposite of a normal fenlist seed's (the
    seeded side escaping is the GOAL, not self-confirmation) — the opening
    source distinguishes them so downstream counters never mix the two.
    Path-keyed cache like _load_fen_list: version the filename to update.
    """
    backed: set[str] = set()
    try:
        with open(path_str, encoding="utf-8") as fh:
            for raw in fh:
                body, _, comment = raw.rstrip("\n").partition("#")
                body = body.strip()
                if body and "dropped=" in comment:
                    backed.add(body)
    except OSError:
        return frozenset()
    return frozenset(backed)


@lru_cache(maxsize=8)
def _load_fen_weights(path_str: str) -> dict[str, int]:
    """Per-seed dole weights from ``weight=N`` line-comment markers.

    The dole plays every list line once per iteration by default; a seed
    tagged ``# ... weight=3`` is played 3x per iteration (stubborn motifs get
    more exposure without touching the global dose), and ``weight=0`` soft-
    retires a seed from the dole without a list rewrite. Weights cap at
    _MAX_SEED_WEIGHT so a typo cannot flood an iteration. Path-keyed cache
    like _load_fen_list: version the filename to change weights live.
    """
    weights: dict[str, int] = {}
    try:
        with open(path_str, encoding="utf-8") as fh:
            for raw in fh:
                body, _, comment = raw.rstrip("\n").partition("#")
                body = body.strip()
                m = _WEIGHT_RE.search(comment) if body else None
                if m is not None:
                    w = min(int(m.group(1)), _MAX_SEED_WEIGHT)
                    if w != 1:
                        weights[body] = w
    except OSError:
        return {}
    return weights


def rotate_slice(items: list[str], k: int, training_iteration: int) -> list[str]:
    """Deterministic round-robin window of ``k`` items, advancing each iteration.

    The window start advances by ``k`` per ``training_iteration``, so over
    ~len/k iterations every item is covered exactly once per cycle. Used for
    both the SF-refute channel and the dole cap: a plain truncation would
    permanently starve the tail of the seed list, which is precisely the
    failure the cap exists to avoid.
    """
    n = len(items)
    if n == 0 or k <= 0:
        return []
    if k >= n:
        return list(items)
    start = (int(training_iteration) * k) % n
    return [items[(start + i) % n] for i in range(k)]


def cap_dole_batch(
    queue: list[str],
    sf_queue: list[str],
    *,
    max_games: int,
    training_iteration: int,
) -> tuple[list[str], list[str]]:
    """Bound total seeded games per iteration to ``max_games``.

    The dole hands the WHOLE seed list to selfplay every iteration, so the
    seeded share of selfplay is ``pool_size / selfplay_capacity`` — an
    unmanaged consequence of pool size rather than a chosen number. On
    2026-07-24 that reached 100% (zero normal-opening selfplay games; see the
    ledger). This caps the total and splits the budget between the two seeded
    channels in proportion to their natural sizes, rotating each so no seed is
    permanently starved.

    ``max_games <= 0`` disables the cap (previous behavior).
    """
    if max_games <= 0:
        return queue, sf_queue
    total = len(queue) + len(sf_queue)
    if total <= max_games:
        return queue, sf_queue
    # Proportional split, then give any rounding remainder to the main channel.
    sf_keep = min(len(sf_queue), round(max_games * len(sf_queue) / total))
    main_keep = max(0, min(len(queue), max_games - sf_keep))
    return (
        rotate_slice(queue, main_keep, training_iteration),
        rotate_slice(sf_queue, sf_keep, training_iteration),
    )


def sample_sf_refute_batch(
    seeds: list[str],
    *,
    frac: float,
    training_iteration: int,
    path: str | None = None,
) -> list[str]:
    """Deterministic round-robin slice of ``seeds`` for the SF-refute channel.

    ``frac`` in (0, 1] selects ``round(len*frac)`` seeds (at least 1 when the
    list is non-empty and frac>0). The window start advances by K each
    ``training_iteration`` so every seed rotates through SF-refutation over
    ~1/frac iterations without requiring server-side state.

    When ``path`` is set, ``weight=0`` soft-retire markers are honored (same
    map as ``expand_dole_seeds``) so retired seeds never burn SF-refute budget.
    """
    if not seeds:
        return []
    f = float(frac)
    if f <= 0.0:
        return []
    eligible = list(seeds)
    if path:
        weights = _load_fen_weights(str(path))
        if weights:
            eligible = [s for s in seeds if int(weights.get(s, 1)) > 0]
    if not eligible:
        return []
    n = len(eligible)
    k = n if f >= 1.0 else max(1, min(n, round(n * f)))
    return rotate_slice(eligible, k, training_iteration)


def expand_dole_seeds(seeds: list[str], path: str | None) -> list[str]:
    """Expand a dole seed batch by per-seed ``weight=`` markers (default 1)."""
    if not path:
        return list(seeds)
    weights = _load_fen_weights(str(path))
    if not weights:
        return list(seeds)
    out: list[str] = []
    for s in seeds:
        out.extend([s] * weights.get(s, 1))
    return out


def _fenlist_source(line: str, path: str | None, *, sf_refute: bool = False) -> str:
    # fenlist* prefix is load-bearing: finalize excludes fenlist sources from the
    # PID winrate sample so short seed games cannot bias difficulty control.
    if path and line in _load_fen_backed_bodies(str(path)):
        return "fenlist_sf_refute_backed" if sf_refute else "fenlist_backed"
    return "fenlist_sf_refute" if sf_refute else "fenlist"


def _sample_book(*, rng, path: str, max_plies: int, max_games: int) -> chess.Board:
    p = Path(path)
    suffixes = "".join(p.suffixes).lower()
    if suffixes.endswith(".bin"):
        board = _sample_from_polyglot(rng=rng, path=str(p), max_plies=max_plies)
    elif suffixes.endswith((".pgn", ".pgn.zip", ".zip")):
        board = _sample_from_pgn(rng=rng, path=str(p), max_plies=max_plies, max_games=max_games)
    else:
        raise ValueError(f"Unknown opening book format: {p}")

    if not board.move_stack:
        raise ValueError(f"Opening book produced no usable opening moves: {p}")
    return board


def warm_opening_book_cache(cfg: OpeningConfig) -> None:
    """Preload PGN opening books before launching many selfplay threads."""
    if cfg.opening_fen_list_path:
        _load_fen_list(str(cfg.opening_fen_list_path))
    for path, max_plies, max_games in (
        (cfg.opening_book_path, cfg.opening_book_max_plies, cfg.opening_book_max_games),
        (cfg.opening_book_path_2, cfg.opening_book_max_plies_2, cfg.opening_book_max_games_2),
    ):
        if not path:
            continue
        suffixes = "".join(Path(str(path)).suffixes).lower()
        if suffixes.endswith((".pgn", ".pgn.zip", ".zip")):
            _load_pgn_opening_sequences(
                str(path),
                max_plies=int(max_plies),
                max_games=int(max_games),
            )


def sample_starting_board(*, rng, cfg: OpeningConfig, allow_fenlist: bool = True) -> OpeningStart:
    """Create a starting position and label its low-cardinality source.

    Priority:
    - with probability opening_fen_prob, use the blind-spot FEN list if provided
      (only when ``allow_fenlist`` — the caller passes False for curriculum slots
      when ``opening_fen_selfplay_only`` is set, so seeding never consumes them)
    - with probability opening_book_prob, use opening book if provided
      - among book games, use book 2 with opening_book_mix_prob_2, else book 1
    - otherwise use random_start_plies if >0
    - otherwise startpos
    """
    if (
        allow_fenlist
        and cfg.opening_fen_list_path
        and float(cfg.opening_fen_prob) > 0.0
        and float(rng.random()) < float(cfg.opening_fen_prob)
    ):
        line = _sample_fen_list_line(rng=rng, path=str(cfg.opening_fen_list_path))
        return OpeningStart(
            board=seed_board_from_line(line),
            source=_fenlist_source(line, cfg.opening_fen_list_path),
        )

    if cfg.opening_book_path and float(rng.random()) < float(cfg.opening_book_prob):
        use_book2 = (
            cfg.opening_book_path_2
            and float(cfg.opening_book_mix_prob_2) > 0.0
            and float(rng.random()) < float(cfg.opening_book_mix_prob_2)
        )
        if use_book2:
            return OpeningStart(
                board=_sample_book(
                    rng=rng,
                    path=str(cfg.opening_book_path_2),
                    max_plies=int(cfg.opening_book_max_plies_2),
                    max_games=int(cfg.opening_book_max_games_2),
                ),
                source="book2",
            )
        return OpeningStart(
            board=_sample_book(
                rng=rng,
                path=str(cfg.opening_book_path),
                max_plies=int(cfg.opening_book_max_plies),
                max_games=int(cfg.opening_book_max_games),
            ),
            source="book1",
        )

    if int(cfg.random_start_plies) > 0:
        return OpeningStart(
            board=_random_playout_from_start(rng=rng, plies=int(cfg.random_start_plies)),
            source="random",
        )

    return OpeningStart(board=chess.Board(), source="start")


def resolve_slot_opening(
    *,
    rng,
    cfg: OpeningConfig,
    is_selfplay: bool,
    fen_dole_queue: list[str] | None,
) -> OpeningStart:
    """Pick one slot's opening position, honoring server-doled seeding.

    Single source of truth shared by ``SelfplayState.create`` and
    ``recycle_slot`` so the fresh-slot and recycled-slot rules can't drift.

    Dole mode (``opening_fen_dole_per_iter > 0``):
      * a SELFPLAY slot with a non-empty ``fen_dole_queue`` pops the next queued
        seed (FIFO, deterministic even coverage) and starts there;
      * every other slot — curriculum slots, and selfplay slots once the queue
        is drained — falls back to the book/random/startpos flow with the
        probabilistic FEN draw DISABLED (``allow_fenlist=False``), because in
        dole mode seeding happens exclusively via the queue.

    Legacy mode (``opening_fen_dole_per_iter == 0``): defers entirely to
    ``sample_starting_board``; the probabilistic ``opening_fen_prob`` draw runs,
    gated by the #127 ``allow_fenlist`` rule (curriculum slots pass ``False``
    when ``opening_fen_selfplay_only`` is set).

    The seeded board carries real LC0 history exactly like the probabilistic
    ``fenlist`` path (``seed_board_from_line`` replays the ``fen | moves``
    grammar); the queue only holds lines already validated by ``_load_fen_list``.
    """
    if int(cfg.opening_fen_dole_per_iter) > 0:
        if is_selfplay and cfg.opening_fen_list_path and fen_dole_queue is not None:
  # Pop exception-safely: the threaded selfplay path shares ONE queue across N
  # threads, so a truthiness check + pop() would race (another thread could empty
  # it in between). list.pop(0) is atomic under the GIL — concurrent callers get
  # distinct seeds; an empty queue raises IndexError and falls back to a normal
  # opening. (Single-threaded path: same outcome, no contention.)
            try:
                line = fen_dole_queue.pop(0)
            except IndexError:
                line = None
            if line is not None:
                return OpeningStart(
                    board=seed_board_from_line(line),
                    source=_fenlist_source(line, cfg.opening_fen_list_path),
                )
  # Dole mode owns FEN seeding; suppress the probabilistic draw for every slot.
        return sample_starting_board(rng=rng, cfg=cfg, allow_fenlist=False)

    allow_fenlist = (not bool(cfg.opening_fen_selfplay_only)) or bool(is_selfplay)
    return sample_starting_board(rng=rng, cfg=cfg, allow_fenlist=allow_fenlist)


def try_take_sf_refute_opening(
    *,
    cfg: OpeningConfig,
    fen_sf_refute_queue: list[str] | None,
) -> OpeningStart | None:
    """Pop one SF-refutation seed if the channel is armed and the queue has work.

    Caller must open a **selfplay-tagged** slot (``selfplay_arr=1``) and set
    ``sf_refute_opp_plies_left = opening_fen_sf_refute_plies`` so SF plays the
    opponent for M plies, then net-net. Returns None when the channel is off or
    the queue is empty (thread-safe pop under the GIL).
    """
    if fen_sf_refute_queue is None:
        return None
    if float(cfg.opening_fen_sf_refute_frac) <= 0.0:
        return None
    if int(cfg.opening_fen_sf_refute_plies) <= 0:
        return None
    if not cfg.opening_fen_list_path:
        return None
    try:
        line = fen_sf_refute_queue.pop(0)
    except IndexError:
        return None
    return OpeningStart(
        board=seed_board_from_line(line),
        source=_fenlist_source(line, cfg.opening_fen_list_path, sf_refute=True),
    )


def make_starting_board(*, rng, cfg: OpeningConfig) -> chess.Board:
    return sample_starting_board(rng=rng, cfg=cfg).board
