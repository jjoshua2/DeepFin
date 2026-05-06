from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from chess_anti_engine.replay.buffer import ReplaySample
from chess_anti_engine.replay.shard import (
    LOCAL_SHARD_SUFFIX,
    ShardMeta,
    samples_to_arrays,
    save_local_shard_arrays,
)

log = logging.getLogger(__name__)


@dataclass
class _BufferedUpload:
    samples: list[ReplaySample] = field(default_factory=list)
    model_sha: str | None = None
    model_step: int | None = None
    games: int = 0
    positions: int = 0
    w: int = 0
    d: int = 0
    l: int = 0
    total_game_plies: int = 0
    adjudicated_games: int = 0
    tb_adjudicated_games: int = 0
    total_draw_games: int = 0
    selfplay_games: int = 0
    selfplay_adjudicated_games: int = 0
    selfplay_draw_games: int = 0
    curriculum_games: int = 0
    curriculum_adjudicated_games: int = 0
    curriculum_draw_games: int = 0
    plies_win: int = 0
    plies_draw: int = 0
    plies_loss: int = 0
    checkmate_games: int = 0
    stalemate_games: int = 0
    sf_d6_sum: float = 0.0
    sf_d6_n: int = 0
    first_buffered_at_s: float | None = None

    def reset(self) -> None:
        import dataclasses as _dc
        fresh = _BufferedUpload()
        for f in _dc.fields(self):
            setattr(self, f.name, getattr(fresh, f.name))


def _buffer_elapsed_s(*, buf: _BufferedUpload, now_s: float) -> float:
    if buf.first_buffered_at_s is None:
        return 0.0
    return max(0.0, float(now_s) - float(buf.first_buffered_at_s))


def _buffer_should_flush(
    *,
    buf: _BufferedUpload,
    now_s: float,
    last_send_s: float,
    target_positions: int,
    flush_seconds: float,
) -> bool:
    if buf.positions <= 0:
        return False
    if int(target_positions) > 0 and int(buf.positions) >= int(target_positions):
        return True
    if float(flush_seconds) > 0.0 and (float(now_s) - float(last_send_s)) >= float(flush_seconds):
        return True
    return False


def _buffer_add_completed_game(
    *,
    buf: _BufferedUpload,
    game_batch,
    now_s: float,
    model_sha: str,
    model_step: int,
    max_positions: int = 0,
) -> None:
    if getattr(game_batch, "positions", 0) <= 0 or not getattr(game_batch, "samples", None):
        return
    new_positions = int(getattr(game_batch, "positions", 0))
    if max_positions > 0 and int(buf.positions) + new_positions > int(max_positions):
  # Backstop: if flush-to-disk has been failing, drop rather than OOM.
  # Normal flow flushes at upload_target_positions; this cap only bites
  # when something is wrong (disk full, permissions, etc.).
        log.warning(
            "upload buffer at %d positions; dropping %d-position game batch (cap=%d)",
            int(buf.positions), new_positions, int(max_positions),
        )
        return
    if buf.positions > 0:
        if str(buf.model_sha or "") != str(model_sha) or int(buf.model_step or 0) != int(model_step):
            raise ValueError("buffered upload model metadata mismatch")
    else:
        buf.model_sha = str(model_sha)
        buf.model_step = int(model_step)
    if buf.first_buffered_at_s is None:
        buf.first_buffered_at_s = float(now_s)
    buf.samples.extend(game_batch.samples)
    buf.games += int(getattr(game_batch, "games", 0))
    buf.positions += int(getattr(game_batch, "positions", 0))
    buf.w += int(getattr(game_batch, "w", 0))
    buf.d += int(getattr(game_batch, "d", 0))
    buf.l += int(getattr(game_batch, "l", 0))
    buf.total_game_plies += int(getattr(game_batch, "total_game_plies", 0))
    buf.adjudicated_games += int(getattr(game_batch, "adjudicated_games", 0))
    buf.tb_adjudicated_games += int(getattr(game_batch, "tb_adjudicated_games", 0))
    buf.total_draw_games += int(getattr(game_batch, "total_draw_games", 0))
    buf.selfplay_games += int(getattr(game_batch, "selfplay_games", 0))
    buf.selfplay_adjudicated_games += int(getattr(game_batch, "selfplay_adjudicated_games", 0))
    buf.selfplay_draw_games += int(getattr(game_batch, "selfplay_draw_games", 0))
    buf.curriculum_games += int(getattr(game_batch, "curriculum_games", 0))
    buf.curriculum_adjudicated_games += int(getattr(game_batch, "curriculum_adjudicated_games", 0))
    buf.curriculum_draw_games += int(getattr(game_batch, "curriculum_draw_games", 0))
    buf.plies_win += int(getattr(game_batch, "plies_win", 0))
    buf.plies_draw += int(getattr(game_batch, "plies_draw", 0))
    buf.plies_loss += int(getattr(game_batch, "plies_loss", 0))
    buf.checkmate_games += int(getattr(game_batch, "checkmate_games", 0))
    buf.stalemate_games += int(getattr(game_batch, "stalemate_games", 0))
    buf.sf_d6_sum += float(getattr(game_batch, "sf_d6_sum", 0.0))
    buf.sf_d6_n += int(getattr(game_batch, "sf_d6_n", 0))


def _pending_elapsed_path(shard_path: Path) -> Path:
    return shard_path.with_suffix(shard_path.suffix + ".elapsed_s")


def _flush_upload_buffer_to_pending(
    *,
    pending_dir: Path,
    username: str,
    buf: _BufferedUpload,
    now_s: float,
    trial_id: str | None = None,
) -> tuple[Path | None, float]:
    if buf.positions <= 0 or not buf.samples:
        return None, 0.0
    model_sha = str(buf.model_sha or "")
    if not model_sha:
        raise ValueError("buffered upload missing model sha")
    if buf.model_step is None:
        raise ValueError("buffered upload missing model step")
    ts = int(now_s)
    shard_path = pending_dir / f"{ts}_{model_sha[:8]}_{buf.games}g_{buf.positions}p{LOCAL_SHARD_SUFFIX}"
    elapsed_s = _buffer_elapsed_s(buf=buf, now_s=now_s)
    meta = ShardMeta(
        username=str(username),
        run_id=(str(trial_id).strip() if trial_id is not None else "") or None,
        generated_at_unix=ts,
        model_sha256=str(model_sha),
        model_step=int(buf.model_step),
        games=int(buf.games),
        positions=int(buf.positions),
        wins=int(buf.w),
        draws=int(buf.d),
        losses=int(buf.l),
        total_game_plies=int(buf.total_game_plies),
        adjudicated_games=int(buf.adjudicated_games),
        tb_adjudicated_games=int(buf.tb_adjudicated_games),
        total_draw_games=int(buf.total_draw_games),
        selfplay_games=int(buf.selfplay_games),
        selfplay_adjudicated_games=int(buf.selfplay_adjudicated_games),
        selfplay_draw_games=int(buf.selfplay_draw_games),
        curriculum_games=int(buf.curriculum_games),
        curriculum_adjudicated_games=int(buf.curriculum_adjudicated_games),
        curriculum_draw_games=int(buf.curriculum_draw_games),
        plies_win=int(buf.plies_win),
        plies_draw=int(buf.plies_draw),
        plies_loss=int(buf.plies_loss),
        checkmate_games=int(buf.checkmate_games),
        stalemate_games=int(buf.stalemate_games),
        sf_d6_sum=float(buf.sf_d6_sum),
        sf_d6_n=int(buf.sf_d6_n),
    )
    arrs = samples_to_arrays(list(buf.samples))
    save_local_shard_arrays(shard_path, arrs=arrs, meta=meta)
    _pending_elapsed_path(shard_path).write_text(f"{float(elapsed_s):.6f}\n", encoding="utf-8")
    buf.reset()
    return shard_path, float(elapsed_s)


def _maybe_flush_upload_buffer(
    *,
    pending_dir: Path,
    username: str,
    buf: _BufferedUpload,
    now_s: float,
    last_send_s: float,
    target_positions: int,
    flush_seconds: float,
    force: bool = False,
    trial_id: str | None = None,
) -> tuple[Path | None, float]:
    if not force and not _buffer_should_flush(
        buf=buf,
        now_s=now_s,
        last_send_s=last_send_s,
        target_positions=target_positions,
        flush_seconds=flush_seconds,
    ):
        return None, 0.0
    return _flush_upload_buffer_to_pending(
        pending_dir=pending_dir,
        username=username,
        buf=buf,
        now_s=now_s,
        trial_id=trial_id,
    )
