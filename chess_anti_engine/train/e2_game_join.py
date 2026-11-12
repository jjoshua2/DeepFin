"""Finite #1237 callback join, not an upload callback or bank commit.

Use both callbacks on ONE finite play_batch call; do not install the older
natural_outcome_sink. on_joined is an owner-local shard/admission consumer, never
an uploader. Every attempt has a durable observation, including exclusions.
A joined eligible observation still requires full175 replay, physical shard,
search/policy/TB qualification and a separate durable BANK manifest.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any
from collections.abc import Callable

import chess

from chess_anti_engine.moves.encode import index_to_move_strict
from chess_anti_engine.natural_outcome import NaturalOutcome, capture_natural_outcome
from chess_anti_engine.natural_outcome_sidecar import NaturalOutcomeSidecar
from chess_anti_engine.selfplay.provenance import GameProvenance


@dataclass(frozen=True)
class JoinedAttempt:
    run_id: str
    ordinal: int
    game_id: int
    exclusion: str | None
    natural: NaturalOutcome | None
    observation_sha256: str
    candidate_sha256: str | None


def replay_observation(p: GameProvenance) -> chess.Board:
    """Strict full4672 replay; source observation is not external authentication."""
    s = p.start
    if s.root_fen != chess.STARTING_FEN:
        raise ValueError("standard root required")
    if (
        type(s.opening_moves) is not tuple
        or type(p.played_move_indices) is not tuple
        or len(s.opening_moves) + len(p.played_move_indices) > 1600
    ):
        raise ValueError("bounded immutable history required")
    b = chess.Board()
    for u in s.opening_moves:
        if type(u) is not str or len(u) not in (4, 5):
            raise ValueError("invalid opening UCI")
        move = chess.Move.from_uci(u)
        if move not in b.legal_moves or b.is_game_over(claim_draw=False):
            raise ValueError("illegal opening")
        b.push(move)
    if b.fen() != s.start_fen or b.ply() != s.start_ply:
        raise ValueError("opening identity mismatch")
    for index in p.played_move_indices:
        if type(index) is not int or not 0 <= index < 4672:
            raise ValueError("invalid full4672 action")
        if b.is_game_over(claim_draw=False):
            raise ValueError("move after terminal")
        move = index_to_move_strict(index, b)
        if move not in b.legal_moves:
            raise ValueError("illegal played action")
        b.push(move)
    # cboard_to_fen emits only legal-capture EP (native _cboard_impl.h).
    # Replay retains raw EP internally; compare this explicit serialization only.
    if b.fen(en_passant="legal") != p.final_fen or b.ply() != p.total_plies:
        raise ValueError("final history mismatch")
    return b


def _exclusive(path: Path, payload: dict[str, Any]) -> str:
    data = (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode()
    fd, tmp = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    linked = False
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.link(tmp, path)
        linked = True
        d = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(d)
        finally:
            os.close(d)
    except BaseException:
        if linked:
            path.unlink(missing_ok=True)
        raise
    finally:
        Path(tmp).unlink(missing_ok=True)
    return hashlib.sha256(data).hexdigest()


class BufferedGameJoin:
    """Trusted producer callback adapter; one pending game, at most160 attempts.

    All exceptions poison this instance. No later finish or callback may silently
    turn a failed partial collection into a complete run. Observation files are
    explicitly NOT_BANK_COMMITTED, even after successful finish.
    """

    def __init__(
        self,
        directory: Path,
        *,
        run_id: str,
        attempts: int,
        on_joined: Callable[[JoinedAttempt, tuple[Any, ...]], None],
    ):
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", run_id):
            raise ValueError("invalid run identity")
        if type(attempts) is not int or not 1 <= attempts <= 160:
            raise ValueError("finite attempt bound required")
        self.directory = Path(directory) / run_id
        self.directory.mkdir(parents=True, exist_ok=False)
        self.run_id, self.attempts, self.on_joined = run_id, attempts, on_joined
        self.pending: tuple[Any, ...] | None = None
        self.seen: set[int] = set()
        self.failed = False
        self.tb_rescore_enabled: bool | None = None

    def _check(self):
        if self.failed:
            raise ValueError("collection previously failed")

    def on_state_ready(self, state):
        self._check()
        if state.continuous or state.natural_outcome_sink is not None:
            self.failed = True
            raise ValueError("finite mode without duplicate natural sink required")
        if (
            state.game.input_history_encoding != "lc0_root_legacy_meta"
            or state.game.history_rep_fix is not True
        ):
            self.failed = True
            raise ValueError("qualified input mode required")
        if state.game.syzygy_path is not None and not isinstance(
            state.game.syzygy_path, str
        ):
            self.failed = True
            raise ValueError("actual TB path configuration required")
        self.tb_rescore_enabled = bool(state.game.syzygy_path)

    def on_game_complete(self, batch):
        self._check()
        try:
            if self.pending is not None or not 1 <= len(batch.samples) <= 1600:
                raise ValueError("unjoined or unbounded completed game")
            if (
                batch.input_history_encoding != "lc0_root_legacy_meta"
                or batch.history_rep_fix is not True
            ):
                raise ValueError("input encoding mismatch")
            if batch.positions != len(batch.samples):
                raise ValueError("position count mismatch")
            # Trusted producer objects only; copy before producer callback return.
            # Full tensor budget is enforced by the owner child resource limits.
            self.pending = tuple(copy.deepcopy(batch.samples))
        except BaseException:
            self.failed = True
            raise

    def on_game_provenance(self, p: GameProvenance):
        self._check()
        try:
            if self.tb_rescore_enabled is None:
                raise ValueError("actual producer configuration not captured")
            ordinal = p.start.start_ordinal
            if (
                type(ordinal) is not int
                or not 0 <= ordinal < self.attempts
                or ordinal in self.seen
            ):
                raise ValueError("duplicate/out-of-range start ordinal")
            samples = self.pending or ()
            if p.retained_samples != len(samples) or any(
                s.game_id != p.game_id for s in samples
            ):
                raise ValueError("sample/provenance join mismatch")
            if len({s.ply_index for s in samples}) != len(samples):
                raise ValueError("duplicate retained ply")
            records = {(r.ply_index, r.pov_color) for r in p.records if r.has_policy}
            if any(
                (s.ply_index, not bool(s.x[108, 0, 0])) not in records for s in samples
            ):
                raise ValueError("row/record crosswalk mismatch")
            b = replay_observation(p)
            for record in p.records:
                if (
                    type(record.move_offset) is not int
                    or not 0 <= record.move_offset < len(p.played_move_indices)
                    or record.ply_index != p.start.start_ply + record.move_offset
                    or type(record.pov_color) is not bool
                    or record.pov_color != (record.ply_index % 2 == 0)
                    or type(record.is_sf_refute_opp) is not bool
                ):
                    raise ValueError("record offset/ply/actor mismatch")
            if len({r.move_offset for r in p.records}) != len(p.records):
                raise ValueError("duplicate record offset")
            exclusion = None
            natural = None
            candidate_hash = None
            if not p.start.is_selfplay:
                exclusion = "not_selfplay"
            elif any(r.is_sf_refute_opp for r in p.records):
                exclusion = "sf_refute_actor"
            elif p.resolution != "natural" or p.cap_reached:
                exclusion = "censored_or_adjudicated"
            elif p.tablebase_rescore_result is not None:
                exclusion = "tablebase_rescored"
            elif b.outcome(claim_draw=False) is None:
                exclusion = "not_automatic_terminal"
            elif not samples:
                exclusion = "zero_retained_rows"
            else:
                natural = capture_natural_outcome(
                    b,
                    resolved_result=p.natural_result,
                    adjudicated=False,
                    tb_adjudicated=False,
                    resigned=False,
                    censored=False,
                )
                if p.result != natural.result or p.resolved_result != natural.result:
                    raise ValueError("natural/stored outcome mismatch")
                game_dir = self.directory / str(ordinal)
                sink = NaturalOutcomeSidecar(game_dir)
                sink(
                    natural,
                    samples,
                    post_rescore_result=p.result,
                    tb_rescore_enabled=self.tb_rescore_enabled,
                )
                candidate_hash = hashlib.sha256(
                    (game_dir / (natural.trace_sha256 + ".candidate.json")).read_bytes()
                ).hexdigest()
            payload = {
                "status": "JOINED_OBSERVATION_NOT_BANK_COMMITTED",
                "run_id": self.run_id,
                "ordinal": ordinal,
                "game_id": p.game_id,
                "exclusion": exclusion,
                "provenance": asdict(p),
                "candidate_sha256": candidate_hash,
                "tb_rescore_enabled": self.tb_rescore_enabled,
                "full175_replay_qualified": False,
            }
            digest = _exclusive(self.directory / f"{ordinal}.observation.json", payload)
            joined = JoinedAttempt(
                self.run_id,
                ordinal,
                p.game_id,
                exclusion,
                natural,
                digest,
                candidate_hash,
            )
            self.on_joined(joined, samples if exclusion is None else ())
            self.pending = None
            self.seen.add(ordinal)
        except BaseException:
            self.failed = True
            raise

    def finish(self):
        self._check()
        if self.pending is not None or self.seen != set(range(self.attempts)):
            self.failed = True
            raise ValueError("incomplete finite collection")
        return {
            "status": "ALL_ATTEMPTS_JOINED_NOT_BANK_COMMITTED",
            "run_id": self.run_id,
            "attempts": self.attempts,
        }
