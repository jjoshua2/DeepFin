"""One generation of pure self-play through the real finite ``play_batch``.

Both seats are the same net; ``selfplay_fraction`` is 1, so no Stockfish move
is ever played. ``play_batch`` still requires a Stockfish handle: every
self-play network turn submits a P1 label query (``manager._run_step``) and
the strict rule50 tablebase route refuses a handle without the explicit TB
contract (``strict_tb.validate_stockfish_contract``). There is no supported
"no labels" switch, so the collector runs the LIGHTEST valid engine -- a
one-worker pool at ``--sf-nodes`` (default 1) with MultiPV 1 -- and nothing
downstream reads what it writes: the trainer here runs with every SF loss
weight and the SF value fraction at 0.

Rows: every search-backed ply (``policy_row_retention='all_eligible'``,
playout cap off, diff-focus off; single-legal-move plies are skipped by the
search path as everywhere else), with ``GameConfig.record_root_stats`` on so
each row also carries the log prior, completed Q, visits, root value,
halving survivor and played move its target came from.
"""
from __future__ import annotations

import hashlib
import json
import math
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import chess
import numpy as np

from chess_anti_engine.simple_rl.common import (
    append_jsonl,
    input_ref,
    seed_for,
    sha256_file,
    write_json_atomic,
    write_manifest,
)
from chess_anti_engine.simple_rl.openings import load_lines

ROW_META = "row_meta.npz"
GAMES = "games.jsonl"
STATS = "stats.json"


@dataclass(frozen=True)
class CollectSettings:
    games: int
    concurrency: int = 32
    sims: int = 64
    # Playout cap randomisation (KataGo): with probability full_prob a ply
    # searches `sims` and stores a policy row; otherwise it searches
    # `fast_sims`, is played, and stores no policy target (a value-only row
    # when fast_value_rows, else no row). full_prob 1.0 = off.
    full_prob: float = 1.0
    fast_sims: int = 16
    fast_value_rows: bool = False
    tree_reuse: bool = True
    topk: int = 16
    c_scale: float = 0.1
    c_visit: float = 50.0
    target_q_rescale: bool = True
    policy_temp: float = 1.0
    target_untempered_prior: bool = False
    vloss_weight: int = 1
    temp_start: float = 0.5
    temp_floor: float = 0.1
    temp_decay_plies: int = 10
    root_noise_scale: float = 1.0
    root_noise_after: float = 0.5
    root_noise_decay_start_move: int = 12
    root_noise_decay_moves: int = 3
    max_plies: int = 450
    syzygy: str | None = None
    sf_path: str = ""
    sf_nodes: int = 1
    sf_hash_mb: int = 16
    sf_timeout_s: float = 30.0
    book_mix: tuple[float, ...] = (0.95, 0.05)
    rows_per_shard: int = 20000
    device: str = "cpu"
    eval_max_batch: int = 512
    seed: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


def _arch(checkpoint: Path) -> dict[str, Any]:
    import torch

    payload = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    arch = payload.get("arch") if isinstance(payload, dict) else None
    if not isinstance(arch, dict):
        raise ValueError(f"{checkpoint} has no embedded arch")
    return dict(arch)


def build_configs(settings: CollectSettings, arch: dict[str, Any], starts_path: Path) -> dict[str, Any]:
    """The exact six dataclasses handed to ``play_batch`` (kwargs by group)."""
    from chess_anti_engine.selfplay.config import (
        DiffFocusConfig,
        GameConfig,
        OpponentConfig,
        SearchConfig,
        TemperatureConfig,
    )
    from chess_anti_engine.selfplay.opening import OpeningConfig

    strict = settings.syzygy is not None
    game = GameConfig(
        max_plies=int(settings.max_plies),
        selfplay_fraction=1.0,
        policy_row_retention="all_eligible",
        record_fast_ply_value=bool(settings.fast_value_rows),
        record_dense_sf_policy=False,
        record_prior_top1=True,
        record_root_stats=True,
        # Strict route: a game reaching the cap is UNRESOLVED and keeps no rows
        # (finalize), so Stockfish never decides an outcome. Without tablebases
        # the cap is scored a draw: no SF probability can exceed 1.0.
        timeout_adjudication_threshold=0.99 if strict else 1.0,
        policy_encoding=str(arch.get("policy_encoding", "lc0_1858")),
        input_history_encoding=str(arch["input_history_encoding"]),
        input_extra_features=str(arch["input_extra_features"]),
        history_rep_fix=bool(arch.get("history_rep_fix", False)),
        # The F12 collection's tablebase route: strict rule50 TB in search,
        # adjudication of every TB-eligible game, TB-correct policy rescoring.
        syzygy_outcome_mode="rule50_match_v1" if strict else "theoretical_wdl",
        syzygy_path=settings.syzygy,
        stockfish_syzygy_path=settings.syzygy,
        syzygy_rescore_policy=strict,
        syzygy_adjudicate=strict,
        syzygy_adjudicate_fraction=1.0,
        syzygy_in_search=strict,
    )
    search = SearchConfig(
        simulations=int(settings.sims),
        fast_simulations=int(settings.fast_sims) if float(settings.full_prob) < 1.0 else int(settings.sims),
        playout_cap_fraction=float(settings.full_prob),
        gumbel_tree_reuse=bool(settings.tree_reuse),
        mcts_type="gumbel",
        gumbel_topk=int(settings.topk),
        gumbel_c_scale=float(settings.c_scale),
        gumbel_c_visit=float(settings.c_visit),
        gumbel_target_q_rescale=bool(settings.target_q_rescale),
        gumbel_policy_temp=float(settings.policy_temp),
        gumbel_target_untempered_prior=bool(settings.target_untempered_prior),
        gumbel_vloss_weight=int(settings.vloss_weight),
        gumbel_scale=float(settings.root_noise_scale),
        gumbel_scale_after=float(settings.root_noise_after),
        gumbel_scale_decay_start_move=int(settings.root_noise_decay_start_move),
        gumbel_scale_decay_moves=int(settings.root_noise_decay_moves),
    )
    temp = TemperatureConfig(
        temperature=float(settings.temp_start), selfplay_temperature=float(settings.temp_start),
        decay_start_move=0, selfplay_decay_start_move=0,
        decay_moves=int(settings.temp_decay_plies), selfplay_decay_moves=int(settings.temp_decay_plies),
        endgame=float(settings.temp_floor), selfplay_endgame=float(settings.temp_floor),
        after=float(settings.temp_floor), schedule_plies_from_start=True,
    )
    opening = OpeningConfig(
        opening_book_path=None, opening_book_prob=0.0, random_start_plies=0,
        opening_fen_list_path=str(starts_path), opening_fen_prob=0.0,
        opening_fen_dole_per_iter=1, opening_fen_selfplay_only=True,
    )
    return {"game": game, "search": search, "temp": temp, "opening": opening,
            "opponent": OpponentConfig(), "diff_focus": DiffFocusConfig(enabled=False)}


def sample_openings(train_jsonl: Path, *, games: int, book_mix: tuple[float, ...], seed: int,
                    exclude_ids: set[str] | None = None) -> list[dict[str, Any]]:
    """``games`` TRAIN lines, without replacement within the generation.

    Each game first draws its book by ``book_mix`` and then an unused line of
    that book; a book that runs dry falls back to the others.
    """
    rows = load_lines(train_jsonl)
    by_book: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        if exclude_ids and row["id"] in exclude_ids:
            continue
        by_book.setdefault(int(row["book"]), []).append(row)
    rng = np.random.default_rng(seed)
    for book in by_book:
        order = rng.permutation(len(by_book[book]))
        by_book[book] = [by_book[book][int(i)] for i in order]
    books = sorted(by_book)
    weights = np.asarray([book_mix[b - 1] if b - 1 < len(book_mix) else 0.0 for b in books], dtype=np.float64)
    if games > sum(len(v) for v in by_book.values()):
        raise ValueError(f"{games} games requested but only {sum(len(v) for v in by_book.values())} TRAIN lines")
    picked: list[dict[str, Any]] = []
    cursor = dict.fromkeys(books, 0)
    for _ in range(games):
        avail = np.asarray([cursor[b] < len(by_book[b]) for b in books])
        w = np.where(avail, weights, 0.0)
        if w.sum() <= 0:
            w = avail.astype(np.float64)
        book = books[int(rng.choice(len(books), p=w / w.sum()))]
        picked.append(by_book[book][cursor[book]])
        cursor[book] += 1
    return picked


def x_hash(x: np.ndarray) -> np.uint64:
    """Identity of an encoded input (position AND its history planes)."""
    digest = hashlib.blake2b(np.ascontiguousarray(x, dtype=np.float16).tobytes(), digest_size=8).digest()
    return np.uint64(int.from_bytes(digest, "little"))


def _open_stockfish(settings: CollectSettings) -> Any:
    from chess_anti_engine.selfplay.strict_tb import validate_stockfish_contract
    from chess_anti_engine.stockfish.pool import StockfishPool

    if not settings.sf_path:
        raise ValueError("--sf-path is required: play_batch cannot run without a Stockfish handle "
                         "(see chess_anti_engine/simple_rl/collect.py)")
    strict = settings.syzygy is not None
    pool = StockfishPool(
        path=settings.sf_path, nodes=int(settings.sf_nodes), num_workers=1, multipv=1,
        hash_mb=int(settings.sf_hash_mb), syzygy_path=settings.syzygy,
        syzygy_50_move_rule=True if strict else None, syzygy_probe_limit=6 if strict else None,
        read_timeout_s=float(settings.sf_timeout_s),
    )
    if strict:
        assert settings.syzygy is not None
        validate_stockfish_contract(pool, settings.syzygy)
    return pool


def _evaluator(model: Any, settings: CollectSettings) -> Any:
    if not str(settings.device).startswith("cuda"):
        return None  # play_batch builds a LocalModelEvaluator on CPU
    from chess_anti_engine.inference import DirectGPUEvaluator

    return DirectGPUEvaluator(model, device=settings.device, max_batch=int(settings.eval_max_batch),
                              use_amp=True, amp_dtype="bfloat16", n_slots=1, input_bf16=False,
                              legal_bf16=False, pad_to_compiled_buckets=False)


def collect_generation(*, gen_dir: Path, generation: int, checkpoint: Path, train_jsonl: Path,
                       settings: CollectSettings) -> dict[str, Any]:
    """Play one generation into ``gen_dir`` (a fresh, empty directory)."""
    import torch

    from chess_anti_engine.replay.shard import ShardMeta, samples_to_arrays, save_local_shard_arrays
    from chess_anti_engine.selfplay import play_batch
    from chess_anti_engine.uci.model_loader import load_model_from_checkpoint

    gen_dir = Path(gen_dir)
    if gen_dir.exists():
        shutil.rmtree(gen_dir)  # an incomplete attempt (no manifest) is redone whole
    (gen_dir / "shards").mkdir(parents=True)
    arch = _arch(checkpoint)
    seed = seed_for(settings.seed, "collect", generation)
    openings = sample_openings(train_jsonl, games=settings.games, book_mix=settings.book_mix,
                               seed=seed_for(settings.seed, "openings", generation))
    starts_path = gen_dir / "starts.txt"
    starts_path.write_text("".join(row["line"] + "\n" for row in openings))
    by_line = {row["line"]: row for row in openings}
    cfg = build_configs(settings, arch, starts_path)

    torch.manual_seed(seed & 0xFFFFFFFF)
    model = load_model_from_checkpoint(checkpoint, device=settings.device).eval()
    evaluator = _evaluator(model, settings)
    pool = _open_stockfish(settings)

    pending: dict[int, list[Any]] = {}
    buffered: list[Any] = []
    buffered_meta: list[tuple[int, int, str, bool, np.uint64]] = []
    shard_index = 0
    games_done = 0
    t0 = time.perf_counter()
    model_sha = sha256_file(checkpoint)

    def flush() -> None:
        nonlocal buffered, buffered_meta, shard_index
        if not buffered:
            return
        arrays = samples_to_arrays(buffered)
        meta = ShardMeta(run_id=f"simple_rl_gen{generation:04d}", generated_at_unix=int(time.time()),
                         model_sha256=model_sha, games=None, positions=len(buffered),
                         input_history_encoding=cfg["game"].input_history_encoding,
                         history_rep_fix=bool(cfg["game"].history_rep_fix),
                         policy_encoding=cfg["game"].policy_encoding, sf_nodes=int(settings.sf_nodes))
        save_local_shard_arrays(gen_dir / "shards" / f"shard_{shard_index:05d}.zarr", arrs=arrays, meta=meta)
        np.savez(gen_dir / "shards" / f"shard_{shard_index:05d}.meta.npz",
                 game_id=np.asarray([m[0] for m in buffered_meta], dtype=np.int64),
                 ply_index=np.asarray([m[1] for m in buffered_meta], dtype=np.int32),
                 opening_id=np.asarray([m[2] for m in buffered_meta]),
                 tb_policy_override=np.asarray([m[3] for m in buffered_meta], dtype=np.bool_),
                 x_hash=np.asarray([m[4] for m in buffered_meta], dtype=np.uint64))
        shard_index += 1
        buffered, buffered_meta = [], []

    def on_samples(batch: Any) -> None:
        if not batch.samples:
            return
        pending[int(batch.samples[0].game_id)] = list(batch.samples)

    def on_game(audit: Any) -> None:
        nonlocal games_done
        samples = pending.pop(int(audit.game_id), [])
        if len(samples) != int(audit.retained_samples):
            raise ValueError("sample callback and provenance disagree on retained rows")
        line = audit.start.root_fen + " | " + " ".join(audit.start.opening_moves)
        row = by_line.get(line.rstrip())
        if row is None:
            raise ValueError(f"game started from an opening that was not doled: {line!r}")
        tb_plies = {int(p) for p in audit.tablebase_policy_applied_plies}
        final = chess.Board(audit.final_fen)
        for s in samples:
            buffered.append(s)
            buffered_meta.append((int(audit.game_id), int(s.ply_index), row["id"],
                                  int(s.ply_index) in tb_plies, x_hash(s.x)))
        append_jsonl(gen_dir / GAMES, {
            "game_id": int(audit.game_id), "opening_id": row["id"], "book": int(row["book"]),
            "result": audit.result, "resolution": audit.resolution,
            "total_plies": int(audit.total_plies), "start_ply": int(audit.start.start_ply),
            "rows": len(samples), "mate": bool(final.is_checkmate()),
            "tb_policy_rows": sum(1 for s in samples if int(s.ply_index) in tb_plies),
            "cap_reached": bool(audit.cap_reached),
        })
        games_done += 1
        if len(buffered) >= int(settings.rows_per_shard):
            flush()

    queue = [row["line"] for row in openings]
    try:
        _, batch_stats = play_batch(
            model, device=settings.device, rng=np.random.default_rng(seed), stockfish=pool,
            evaluator=evaluator, games=min(int(settings.concurrency), int(settings.games)),
            target_games=int(settings.games), on_game_complete=on_samples, on_game_provenance=on_game,
            fen_dole_queue=queue, **cfg,
        )
    finally:
        pool.close()
    flush()
    wall = time.perf_counter() - t0
    if games_done != settings.games or queue or pending:
        raise ValueError(f"incomplete generation: {games_done}/{settings.games} games, "
                         f"{len(queue)} undoled openings, {len(pending)} pending")
    stats = generation_stats(gen_dir, previous=[p for p in sorted(gen_dir.parent.parent.glob("gen_*/collect"))
                                                if p != gen_dir and _gen_number(p) < generation])
    stats.update(wall_seconds=wall, games_per_hour=3600.0 * games_done / max(wall, 1e-9),
                 rows_per_second=stats["rows"] / max(wall, 1e-9), sf_nodes=int(settings.sf_nodes))
    write_json_atomic(gen_dir / STATS, stats)
    config_record = {"settings": asdict(settings),
                     "dataclasses": {k: asdict(v) for k, v in cfg.items()}, "arch": arch}
    write_manifest(
        gen_dir, stage="collect", config=config_record,
        inputs={"checkpoint": input_ref(checkpoint), "train_openings": input_ref(train_jsonl)},
        outputs={"shards": input_ref(gen_dir / "shards"), "games": input_ref(gen_dir / GAMES),
                 "starts": input_ref(starts_path)},
        counts={"games": games_done, "rows": stats["rows"], "shards": shard_index,
                "play_batch_games": int(batch_stats.games)},
        seeds={"collect": seed, "openings": seed_for(settings.seed, "openings", generation), "base": settings.seed},
        extra={"generation": generation, "stats": stats},
    )
    return stats


def _gen_number(collect_dir: Path) -> int:
    return int(Path(collect_dir).parent.name.split("_")[1])


def load_generation_rows(collect_dir: Path, *, keys: tuple[str, ...] | None = None) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """All shards of one generation, concatenated, plus the row-meta sidecar."""
    from chess_anti_engine.replay.shard import load_shard_arrays

    shard_dir = Path(collect_dir) / "shards"
    parts: list[dict[str, np.ndarray]] = []
    metas: list[dict[str, np.ndarray]] = []
    for shard in sorted(shard_dir.glob("shard_*.zarr")):
        arrays, _ = load_shard_arrays(shard, lazy=False, validate=True)
        if keys is not None:
            arrays = {k: v for k, v in arrays.items() if k in keys or str(k).startswith("_")}
        parts.append(arrays)
        with np.load(shard.with_name(shard.name.replace(".zarr", ".meta.npz"))) as z:
            metas.append({k: z[k] for k in z.files})
    if not parts:
        raise ValueError(f"no shards in {shard_dir}")
    return concat_arrays(parts), {k: np.concatenate([m[k] for m in metas]) for k in metas[0]}


def concat_arrays(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Row-concatenate shard arrays; a field absent from a part is zero-filled
    with the schema's own default (flag 0), metadata keys must agree."""
    from chess_anti_engine.replay.shard import zeros_for_storage_field

    keys: list[str] = []
    for part in parts:
        for k in part:
            if k not in keys:
                keys.append(k)
    out: dict[str, np.ndarray] = {}
    x_planes = int(parts[0]["x"].shape[1])
    policy_size = int(parts[0]["policy_target"].shape[1])
    for k in keys:
        if str(k).startswith("_"):
            values = {str(np.asarray(p[k]).tolist()) for p in parts if k in p}
            if len(values) != 1 or any(k not in p for p in parts):
                raise ValueError(f"metadata {k} differs across shards: {sorted(values)}")
            out[k] = np.asarray(parts[0][k])
            continue
        cols = []
        for p in parts:
            n = int(p["x"].shape[0])
            cols.append(np.asarray(p[k]) if k in p else zeros_for_storage_field(
                k, n=n, x_planes=x_planes, policy_size=policy_size))
        out[k] = np.concatenate(cols, axis=0)
    return out


def _entropy(p: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.where(p > 0, np.log(np.where(p > 0, p, 1.0)), 0.0)
    return -(p * logp).sum(axis=-1)


def policy_stats(arrays: dict[str, np.ndarray]) -> dict[str, float]:
    """Target/prior entropy, KL(target||prior) and top-1 agreement, from root stats."""
    target = np.asarray(arrays["policy_target"], dtype=np.float64)
    target = target / np.maximum(target.sum(axis=1, keepdims=True), 1e-30)
    policy_rows = np.asarray(arrays.get("has_policy", np.ones(len(target))), dtype=bool)
    out: dict[str, float] = {
        "target_entropy": float(_entropy(target[policy_rows]).mean()) if policy_rows.any() else math.nan,
        "policy_rows": int(policy_rows.sum()),
    }
    has = np.asarray(arrays.get("has_root_log_prior", np.zeros(len(target))), dtype=bool)
    out["root_stats_rows"] = int(has.sum())
    # Exact root shortcuts (solved / tablebase / single-move / #1410's pruned-draw
    # fallback) carry no statistics; their value target falls back (learn.py).
    out["rows_without_root_stats"] = int((~has).sum())
    # Value diagnostics over every searched row; policy diagnostics (below)
    # over policy rows only -- a fast ply's stored target is masked.
    if not has.any():
        return out
    for name in ("root_net_value", "root_survivor_q", "root_node_q"):
        if name in arrays:
            vals = np.asarray(arrays[name], dtype=np.float64)[has]
            out[f"mean_{name}"] = float(np.nanmean(vals))
    if "root_survivor_q" in arrays and "root_node_q" in arrays:
        gap = np.asarray(arrays["root_survivor_q"], np.float64)[has] - np.asarray(arrays["root_node_q"], np.float64)[has]
        out["mean_survivor_minus_node_q"] = float(np.nanmean(gap))
    if "root_start_visits" in arrays:
        sv = np.asarray(arrays["root_start_visits"], dtype=np.float64)[has]
        out.update(mean_root_start_visits=float(sv.mean()), reused_root_share=float((sv > 0).mean()))
    has &= policy_rows
    if not has.any():
        return out
    support = np.asarray(arrays["root_child_visits"])[has] >= 0
    logp = np.where(support, np.asarray(arrays["root_log_prior"], dtype=np.float64)[has], -np.inf)
    logp = logp - logp.max(axis=1, keepdims=True)
    prior = np.where(support, np.exp(logp), 0.0)
    prior /= prior.sum(axis=1, keepdims=True)
    t = target[has]
    with np.errstate(divide="ignore", invalid="ignore"):
        kl = np.where(t > 0, t * (np.log(np.where(t > 0, t, 1.0)) - np.log(np.where(prior > 0, prior, 1e-300))), 0.0).sum(axis=1)
    out.update(prior_entropy=float(_entropy(prior).mean()), kl_target_prior=float(kl.mean()),
               top1_agreement=float((t.argmax(axis=1) == prior.argmax(axis=1)).mean()))
    return out


def generation_stats(collect_dir: Path, *, previous: list[Path]) -> dict[str, Any]:
    collect_dir = Path(collect_dir)
    games = [json.loads(r) for r in (collect_dir / GAMES).read_text().splitlines() if r.strip()]
    keys = ("x", "policy_target", "wdl_target", "has_policy", "root_log_prior", "has_root_log_prior",
            "root_child_visits", "has_root_child_visits", "root_net_value", "root_survivor_q", "root_node_q",
            "root_start_visits", "search_budget", "has_search_budget")
    arrays, meta = load_generation_rows(collect_dir, keys=keys)
    n = int(arrays["x"].shape[0])
    results = [g["result"] for g in games]
    w, d, l_ = results.count("1-0"), results.count("1/2-1/2"), results.count("0-1")
    hashes = meta["x_hash"]
    prev: set[int] = set()
    for p in previous:
        for m in sorted((p / "shards").glob("shard_*.meta.npz")):
            with np.load(m) as z:
                prev.update(int(h) for h in z["x_hash"])
    stats: dict[str, Any] = {
        "games": len(games), "rows": n,
        "white_wins": w, "draws": d, "black_wins": l_,
        "decisive_rate": (w + l_) / max(1, len(games)),
        "draw_rate": d / max(1, len(games)),
        "mean_game_plies": float(np.mean([g["total_plies"] for g in games])) if games else math.nan,
        "mean_plies_after_book": float(np.mean([g["total_plies"] - g["start_ply"] for g in games])) if games else math.nan,
        "mate_count": sum(1 for g in games if g["mate"]),
        "tb_adjudicated_share": sum(1 for g in games if g["resolution"] == "tablebase_adjudication") / max(1, len(games)),
        "cap_reached_games": sum(1 for g in games if g["cap_reached"]),
        "unresolved_games": sum(1 for g in games if g["resolution"] == "unresolved"),
        "tb_policy_override_rows": int(np.asarray(meta["tb_policy_override"]).sum()),
        "repeat_rate_within": 1.0 - len({int(h) for h in hashes}) / max(1, n),
        "repeat_rate_vs_previous": (sum(1 for h in hashes if int(h) in prev) / max(1, n)) if previous else 0.0,
        "previous_generations_compared": len(previous),
        "unique_openings": len({g["opening_id"] for g in games}),
    }
    stats.update(policy_stats(arrays))
    if "search_budget" in arrays:
        has_b = np.asarray(arrays["has_search_budget"], dtype=bool)
        budgets = np.asarray(arrays["search_budget"])[has_b]
        stats["rows_by_search_budget"] = {str(int(b)): int((budgets == b).sum()) for b in np.unique(budgets)}
    return stats
