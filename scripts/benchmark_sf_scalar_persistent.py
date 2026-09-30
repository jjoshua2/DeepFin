"""Cold-TT persistent Stockfish scalar cost screen on a frozen history-bearing bank.

``inspect`` is metadata only; ``prepare`` builds a new 10k-row roster from a
closed-game corpus; ``run`` owns one then eight one-thread engines. No command
changes a training target or grants corpus credit.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import heapq
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any

import chess

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chess_anti_engine.stockfish.uci import StockfishUCI, _parse_info_fields  # noqa: E402
from chess_anti_engine.stockfish.wdl import cp_to_wdl  # noqa: E402
from scripts import derive_corpus_targets as derive  # noqa: E402
from scripts import gen_sf_rooted_corpus as corpus  # noqa: E402

SCHEMA = "sf_scalar_persistent_d6_d8_d10_v1"
DEPTHS = (6, 8, 10)
ARMS = (1, 8)
MAX_ROWS = 10_000
MAX_PREP_WALL = 900
MAX_RUN_WALL = 7200
MAX_OUTPUT = 128 * 2**20
SHARD_RE = re.compile(r"w(\d\d)-(\d{5})\.jsonl\.zst\Z")


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def write_x(path: Path, obj: Any) -> None:
    with path.open("xb") as f:
        f.write(canonical(obj) + b"\n")
        f.flush()
        os.fsync(f.fileno())


def load_plan(path: Path, expected_sha: str) -> dict[str, Any]:
    require(sha(path) == expected_sha, "plan SHA differs")
    p = json.loads(path.read_bytes())
    require(p["schema"] == SCHEMA, "plan schema differs")
    require(p["depths"] == list(DEPTHS) and p["arms"] == list(ARMS), "arm grid differs")
    require(type(p["rows"]) is int and p["rows"] == MAX_ROWS, "roster count differs")
    require(type(p["hash_mb"]) is int and 4 <= p["hash_mb"] <= 64, "Hash must be 4..64 MB")
    require(p["syzygy_50_move_rule"] is True and type(p["syzygy_probe_limit"]) is int and p["syzygy_probe_limit"] == 6, "strict Syzygy differs")
    require(p["retain_syzygy_on_new_game"] is True and type(p["threads_per_engine"]) is int and p["threads_per_engine"] == 1, "engine profile differs")
    require(p["cp_slope"] == 0.006 and p["cp_draw_width"] == 120.0, "D-style calibration differs")
    require(0 < p["prepare_wall_seconds"] <= MAX_PREP_WALL, "prepare wall cap differs")
    require(0 < p["run_wall_seconds"] <= MAX_RUN_WALL, "run wall cap differs")
    require(0 < p["output_bytes"] <= MAX_OUTPUT, "output cap differs")
    require(type(p["rss_cap_bytes"]) is int and 0 < p["rss_cap_bytes"] <= 8 * 2**30, "RSS cap differs")
    require(0 < p["search_timeout_seconds"] <= 15 and 0 < p["read_timeout_seconds"] <= 60, "timeout differs")
    require(len(p["affinity"]) == 8 and len(set(p["affinity"])) == 8, "8 distinct CPUs required")
    require(p["source_result_rows"] == 38615 and p["source_closed_shards"] == 106, "source geometry differs")
    require(sys.version_info[:3] == (3, 13, 15), "locked Python version differs")
    require(Path(sys.executable) == Path(p["python_executable"]), "locked Python executable differs")
    require(sha(Path(sys.executable)) == p["python_executable_sha256"], "locked Python bytes differ")
    require(sha(Path(__file__)) == p["script_sha256"], "bench source differs")
    require(subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip() == p["runtime_head"], "runtime HEAD differs")
    for item in p["pins"]:
        require(sha(Path(item["path"])) == item["sha256"], "pin differs: " + item["path"])
    return p


def listed_shards(source: Path) -> list[tuple[Path, int, set[tuple[int, int]]]]:
    """Use producer progress, including namespaced game IDs, never a file glob as authority."""
    entries: list[tuple[Path, int, set[tuple[int, int]]]] = []
    seen: set[str] = set()
    for progress in sorted(source.glob("w*.progress.jsonl")):
        worker = int(progress.name[1:3])
        for line in progress.read_text().splitlines():
            obj = json.loads(line)
            if obj.get("path") is None or obj.get("rows", 0) == 0:
                continue
            path = Path(obj["path"])
            if not path.is_absolute():
                path = source / path
            require(path.parent.resolve() == source.resolve(), "foreign shard")
            match = SHARD_RE.fullmatch(path.name)
            require(match is not None and int(match[1]) == worker, "shard worker mismatch")
            require(path.name not in seen and path.is_file() and not path.is_symlink(), "missing/duplicate shard")
            seen.add(path.name)
            games = {(worker, int(game)) for game in obj["games"]}
            entries.append((path, int(obj["rows"]), games))
    extra = {x.name for x in source.glob("w*.jsonl.zst")} - seen
    # The interrupted generation screen left one unlisted partial shard. It
    # carries no closed-game credit and must never enter this frame.
    require(extra == {"w01-00026.jsonl.zst"}, "unexpected unlisted shard inventory")
    require((source / "w01-00026.jsonl.zst").stat().st_size == 162082, "partial shard stat differs")
    return sorted(entries, key=lambda e: e[0].name)


def reference_of(row: dict[str, Any], board: chess.Board) -> dict[str, Any]:
    """Full-width d8 rank-1 proxy, not scalar d8 or deep truth."""
    phase = row["phases"][0]
    legal = {move.uci() for move in board.legal_moves}
    require(phase["index"] == 0 and phase["depth_requested"] == 8, "reference phase differs")
    require(phase["searchmoves"] is None and phase["width_realized"] == len(legal), "not full width")
    blocks = [b for b in phase["per_depth"] if b["depth"] == 8]
    require(len(blocks) == 1 and blocks[0]["complete"] is True, "missing complete d8")
    lines = blocks[0]["lines"]
    require(len(lines) == len(legal) and [x[0] for x in lines] == list(range(1, len(legal) + 1)), "rank roster differs")
    require({x[1] for x in lines} == legal, "reference move support differs")
    require(all(math.isfinite(float(x[2])) for x in lines), "nonfinite reference score")
    score = float(lines[0][2])
    wdl = cp_to_wdl(score, None, slope=0.006, draw_width_cp=120.0).tolist()
    return {"move": lines[0][1], "effective_cp": score, "d_style_wdl": wdl}


def compact_row(row: dict[str, Any], shard: str, line: int) -> dict[str, Any]:
    require(type(row.get("schema")) is int and row["schema"] == 3, "row schema differs")
    require(type(row["result"]) in (int, float) and row["result"] in (-1, 0, 1), "source outcome differs")
    require(row["run"]["run_id"] == "throughput_d8_c4", "source run differs")
    board = derive.board_from_row(row)
    require(board.is_valid() and not board.is_game_over(), "invalid/terminal board")
    require(int(row["piece_count"]) == len(board.piece_map()), "piece count differs")
    return {
        "source": "generation_throughput_20260920/d8_c4", "shard": shard, "line": line,
        "worker_id": row["worker_id"], "game_id": row["game_id"], "ply": row["ply"],
        "input_key": row["input_key"], "row_sha256": hashlib.sha256(canonical(row)).hexdigest(),
        "fen": row["fen"], "history_root_fen": row["history_root_fen"],
        "history_uci": row["history_uci"], "history_root_reason": row["history_root_reason"],
        "piece_count": row["piece_count"], "game_phase": row["game_phase"],
        "banked_full_width_d8": reference_of(row, board),
    }


def row_key(row: dict[str, Any]) -> int:
    identity = [row["source"], row["worker_id"], row["game_id"], row["ply"]]
    return int.from_bytes(hashlib.sha256(canonical(identity)).digest(), "big")


def prepare(plan: dict[str, Any], out: Path) -> None:
    require(not out.exists(), "fresh prepare output required")
    started = time.monotonic()
    out.mkdir(parents=False)
    write_x(out / "claim.json", {"schema": SCHEMA, "mode": "prepare", "started_utc": time.time()})
    try:
        source = Path(plan["source"])
        entries = listed_shards(source)
        require(len(entries) == plan["source_closed_shards"], "closed-shard count differs")
        compressed_bytes = sum(path.stat().st_size for path, _, _ in entries)
        require(compressed_bytes <= 128 * 2**20, "source compressed-byte cap")
        heap: list[tuple[int, int, dict[str, Any]]] = []
        total = 0
        games: set[tuple[int, int]] = set()
        archive_hashes = []
        for path, expected, declared_games in entries:
            require(time.monotonic() - started < plan["prepare_wall_seconds"], "prepare wall cap")
            archive_hashes.append({"name": path.name, "bytes": path.stat().st_size, "sha256": sha(path)})
            seen = 0
            observed_games: set[tuple[int, int]] = set()
            for line, row in enumerate(corpus.iter_shard_rows(path), 1):
                seen += 1
                total += 1
                require(row["run"]["config_sha256"] == plan["source_config_sha256"], "source config differs")
                pair = (int(row["worker_id"]), int(row["game_id"]))
                require(pair in declared_games, "row game absent from producer progress")
                observed_games.add(pair)
                item = compact_row(row, path.name, line)
                key = row_key(item)
                node = (-key, -total, item)
                if len(heap) < plan["rows"]:
                    heapq.heappush(heap, node)
                elif node > heap[0]:
                    heapq.heapreplace(heap, node)
                if total % 256 == 0:
                    require(time.monotonic() - started < plan["prepare_wall_seconds"], "prepare wall cap")
            require(seen == expected and observed_games <= declared_games, "shard rows/games differ")
            games |= declared_games
        require(total == plan["source_result_rows"] and len(games) == 200, "source frame differs")
        items = [node[2] for node in sorted(heap, key=lambda x: (-x[0], -x[1]))]
        require(len(items) == plan["rows"], "selected count differs")
        require(len({(x["worker_id"], x["game_id"], x["ply"]) for x in items}) == len(items), "duplicate source identity")
        seen_input: Counter[str] = Counter(x["input_key"] for x in items)
        phases = Counter(str(x["game_phase"]) for x in items)
        materials = Counter(str(x["piece_count"]) for x in items)
        plies = Counter(str(min(4, int(x["ply"]) // 40)) for x in items)
        with (out / "roster.jsonl").open("xb") as f:
            for item in items:
                f.write(canonical(item) + b"\n")
            f.flush()
            os.fsync(f.fileno())
        require(sum(p.stat().st_size for p in out.iterdir()) < plan["output_bytes"] - 1024 * 1024, "prepare output cap reserve")
        write_x(out / "prepared.json", {
            "status": "PREPARED_ZERO_CREDIT", "schema": SCHEMA, "source_rows": total,
            "closed_games": len(games), "selected_rows": len(items),
            "selected_games": len({(x["worker_id"], x["game_id"]) for x in items}),
            "duplicate_selected_input_keys": sum(n-1 for n in seen_input.values()),
            "sampling": "uniform_10000_smallest_SHA256(source,worker,game,ply); no oversampling weights",
            "game_phase_counts": phases, "piece_count_counts": materials, "ply_40bin_counts": plies,
            "archive_hashes": archive_hashes, "roster_sha256": sha(out / "roster.jsonl"),
            "unlisted_partial_excluded": {"name": "w01-00026.jsonl.zst", "bytes": 162082},
            "elapsed_seconds": time.monotonic() - started,
            "scope": "Closed-game source-qualified d8_c4 rows only; proxy reference is full-width d8 rank1; no corpus credit.",
        })
        require(sum(p.stat().st_size for p in out.iterdir()) <= plan["output_bytes"], "prepare output cap")
    except BaseException as exc:
        write_x(out / "failed.json", {"status": "FAILED_PREPARE", "error": repr(exc)})
        raise


def first_depth_score(lines: list[str], depth: int, parsed: Any) -> dict[str, Any]:
    block, full = corpus.deepest_block_with_width(parsed.blocks, want=1)
    require(full and block.complete and block.depth == depth and len(block.lines) == 1, "incomplete scalar depth")
    require(parsed.re_emissions_disagreeing == 0, "disagreeing re-emission")
    for line in lines:
        fields = line.split()
        if not fields or fields[0] != "info" or "upperbound" in fields or "lowerbound" in fields:
            continue
        rank, nodes, d, cp, mate, wdl, move = _parse_info_fields(fields)
        if d == depth and (rank is None or rank == 1) and move is not None and (cp is not None or mate is not None):
            require(move == block.lines[0].move and nodes == block.lines[0].nodes, "raw/parser line differs")
            require((cp is None) != (mate is None), "ambiguous raw score")
            require(wdl is None or (len(wdl) == 3 and sum(wdl) == 1000), "native WDL malformed")
            return {"move": move, "cp": cp, "mate": mate, "native_wdl_permille": wdl,
                    "effective_cp": block.lines[0].effective_cp, "nodes": nodes,
                    "d_style_wdl": cp_to_wdl(cp, mate, slope=0.006, draw_width_cp=120.0).tolist()}
    raise ValueError("requested raw depth absent")


def search_one(searcher: Any, row: dict[str, Any], depth: int, arm: int) -> dict[str, Any]:
    history = corpus.RowHistory(row["fen"], row["history_root_fen"], tuple(row["history_uci"]), row["history_root_reason"])
    board = chess.Board(history.root_fen)
    for token in history.uci:
        move = chess.Move.from_uci(token)
        require(move in board.legal_moves, "illegal history move")
        board.push(move)
    require(board.fen() == history.fen and board.legal_moves.count() > 0, "history does not reproduce position")
    t0 = time.monotonic()
    searcher.new_game()  # cold TT per row/depth, while qualified binary retains TB cache
    t1 = time.monotonic()
    lines = searcher.stream(history, depth=depth, multipv=1)
    t2 = time.monotonic()
    parsed = corpus.parse_depth_blocks(lines, expected_lines=1)
    score = first_depth_score(lines, depth, parsed)
    require(score["move"] in {m.uci() for m in board.legal_moves}, "illegal PV move")
    return {"arm_engines": arm, "depth": depth, "source": row["source"],
            "worker_id": row["worker_id"], "game_id": row["game_id"], "ply": row["ply"],
            "row_sha256": row["row_sha256"], "input_key": row["input_key"],
            "banked_full_width_d8": row["banked_full_width_d8"],
            "reset_seconds": t1-t0, "search_seconds": t2-t1, "parse_seconds": time.monotonic()-t2,
            "score": score}


def depth_order(arm: int) -> tuple[int, ...]:
    """Reverse fixed-depth pass order across engine-count arms to expose drift."""
    require(arm in ARMS, "unknown engine-count arm")
    return DEPTHS if arm == 1 else tuple(reversed(DEPTHS))


def rss_bytes(pids: list[int]) -> int:
    total = 0
    for pid in pids:
        status = Path(f"/proc/{pid}/status").read_text()
        match = re.search(r"^VmRSS:\s*(\d+) kB$", status, re.MULTILINE)
        if match is None:
            raise ValueError("live RSS unavailable")
        total += int(match[1]) * 1024
    return total


def load_roster(path: Path, plan: dict[str, Any]) -> list[dict[str, Any]]:
    prepared = json.loads((path / "prepared.json").read_bytes())
    require(prepared["status"] == "PREPARED_ZERO_CREDIT" and prepared["selected_rows"] == plan["rows"], "roster status/count differs")
    require(sha(path / "roster.jsonl") == prepared["roster_sha256"], "roster SHA differs")
    rows = [json.loads(line) for line in (path / "roster.jsonl").read_bytes().splitlines()]
    require(len(rows) == plan["rows"], "roster length differs")
    return rows


def run_depth_pass(plan: dict[str, Any], rows: list[dict[str, Any]],
                   searchers: list[Any], arm: int, depth: int,
                   out: Path, deadline: float) -> dict[str, Any]:
    """An inclusive observed wall for one depth on every row, one owner/lane."""
    require(depth in DEPTHS and len(searchers) == arm, "depth pass geometry differs")
    began = time.monotonic()
    output = out / f"arm{arm}_d{depth}.jsonl"
    sums: dict[str, float] = {
        "serialization_seconds": 0.0,
        "reset_seconds": 0.0,
        "search_seconds": 0.0,
        "parse_seconds": 0.0,
        "absolute_q_error_sum": 0.0,
        "same_move": 0.0,
        "rows": 0.0,
    }
    with output.open("xb") as f, ThreadPoolExecutor(max_workers=arm) as pool:
        for base in range(0, len(rows), 16):
            require(time.monotonic() < deadline, "run wall cap")
            pids = [os.getpid(), *(searcher.engine.proc.pid for searcher in searchers)]
            require(rss_bytes(pids) < plan["rss_cap_bytes"], "RSS cap")
            batch = list(enumerate(rows[base:base+16], base))
            partitions = [batch[lane::arm] for lane in range(arm)]

            def lane_work(lane: int, partitions_: list[list[tuple[int, dict[str, Any]]]] = partitions) -> list[tuple[int, dict[str, Any]]]:
                records = []
                for index, row in partitions_[lane]:
                    reserve = plan["read_timeout_seconds"] + plan["search_timeout_seconds"] + 2
                    require(time.monotonic() + reserve < deadline, "insufficient run wall reserve")
                    records.append((index, search_one(searchers[lane], row, depth, arm)))
                return records

            futures = [pool.submit(lane_work, lane) for lane in range(arm)]
            records = sorted((item for future in futures for item in future.result()), key=lambda x: x[0])
            for _, record in records:
                t0 = time.monotonic()
                f.write(canonical(record) + b"\n")
                sums["serialization_seconds"] += time.monotonic()-t0
                for key in ("reset_seconds", "search_seconds", "parse_seconds"):
                    sums[key] += record[key]
                ref = record["banked_full_width_d8"]
                score = record["score"]
                ref_q = ref["d_style_wdl"][0] - ref["d_style_wdl"][2]
                score_q = score["d_style_wdl"][0] - score["d_style_wdl"][2]
                sums["absolute_q_error_sum"] += abs(score_q - ref_q)
                sums["same_move"] += score["move"] == ref["move"]
                sums["rows"] += 1
            require(f.tell() < plan["output_bytes"] // 6 - 1024 * 1024, "depth output cap")
            f.flush()
        os.fsync(f.fileno())
    digest = sha(output)
    elapsed = time.monotonic() - began
    require(sums["rows"] == len(rows) and elapsed > 0, "depth pass row count differs")
    return {"depth": depth, "rows": len(rows), "depth_wall_seconds": elapsed,
            "observed_rows_per_depth_wall_second": len(rows) / elapsed,
            "reset_seconds_sum_across_engines": sums["reset_seconds"],
            "search_seconds_sum_across_engines": sums["search_seconds"],
            "parse_seconds_sum_across_engines": sums["parse_seconds"],
            "serialization_seconds": sums["serialization_seconds"],
            "mean_absolute_q_delta_vs_banked_full_width_d8": sums["absolute_q_error_sum"] / len(rows),
            "same_best_move_fraction_vs_banked_full_width_d8": sums["same_move"] / len(rows),
            "raw_sha256": digest}


def run(plan: dict[str, Any], roster: Path, out: Path) -> None:
    require(not out.exists(), "fresh run output required; resume refused")
    rows = load_roster(roster, plan)
    require(sha(Path(plan["stockfish"])) == plan["stockfish_sha256"], "engine binary differs")
    require(all(Path(p).is_dir() for p in plan["syzygy_path"].split(":")), "Syzygy path absent")
    require(set(plan["affinity"]) <= os.sched_getaffinity(0), "CPU affinity unavailable")
    os.sched_setaffinity(0, set(plan["affinity"]))
    os.nice(19 - os.getpriority(os.PRIO_PROCESS, 0))
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["LD_LIBRARY_PATH"] = plan["ld_library_path"]
    start = time.monotonic()
    deadline = start + plan["run_wall_seconds"]
    out.mkdir(parents=False)
    write_x(out / "claim.json", {"schema": SCHEMA, "mode": "run", "roster_sha256": sha(roster / "roster.jsonl"), "started_utc": time.time()})
    all_engines: list[StockfishUCI] = []
    try:
        arm_stats = []
        for arm in ARMS:
            arm_start = time.monotonic()
            engines = []
            for _ in range(arm):
                require(time.monotonic() < deadline, "run wall cap during startup")
                engine = StockfishUCI(plan["stockfish"], multipv=1, hash_mb=plan["hash_mb"], threads=1,
                    nice=19, syzygy_path=plan["syzygy_path"], syzygy_50_move_rule=True,
                    syzygy_probe_limit=6, retain_syzygy_on_new_game=True,
                    read_timeout_s=plan["read_timeout_seconds"])
                require(engine.syzygy_ready_after_requests and engine.retain_syzygy_option_sent, "engine option request failed")
                engines.append(engine)
                all_engines.append(engine)
                require(rss_bytes([os.getpid(), *(e.proc.pid for e in engines)]) < plan["rss_cap_bytes"], "RSS cap during startup")
            startup = time.monotonic() - arm_start
            searchers = [corpus.StaircaseSearcher(engine=e, staircase=corpus.parse_staircase("1:8"),
                cp_slope=0.006, cp_draw_width=120.0, search_timeout_s=plan["search_timeout_seconds"]) for e in engines]
            by_depth = [run_depth_pass(plan, rows, searchers, arm, depth, out, deadline)
                        for depth in depth_order(arm)]
            for engine in engines:
                engine.close()
                all_engines.remove(engine)
            arm_wall = time.monotonic()-arm_start
            arm_stats.append({"engines": arm, "rows": len(rows), "observations": len(rows)*len(DEPTHS),
                "depth_order": depth_order(arm), "startup_seconds": startup,
                "arm_wall_seconds": arm_wall, "by_depth": by_depth})
        require(sum(p.stat().st_size for p in out.iterdir()) < plan["output_bytes"] - 1024 * 1024, "run output cap reserve")
        write_x(out / "complete.json", {"status": "COMPLETE_SCALAR_COST_ZERO_CREDIT", "schema": SCHEMA,
            "roster_sha256": sha(roster / "roster.jsonl"), "rows": len(rows), "depths": DEPTHS,
            "arms": arm_stats, "wall_seconds": time.monotonic()-start,
            "limits": "Banked full-width d8 rank1 is a proxy only. Native UCI WDL is separate from D-style calibrated WDL. No Elo, production-wide or corpus credit."})
        require(sum(p.stat().st_size for p in out.iterdir()) <= plan["output_bytes"], "total output cap")
    except BaseException as exc:
        write_x(out / "failed.json", {"status": "FAILED_SCALAR_COST", "error": repr(exc)})
        raise
    finally:
        for engine in all_engines:
            engine.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("inspect", "prepare", "run"))
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--roster", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    plan = load_plan(args.plan, args.plan_sha256)
    if args.mode == "inspect":
        print(json.dumps({"status": "SOURCE_VALID_NOT_LAUNCHED", "rows": plan["rows"]}))
    elif args.mode == "prepare":
        if args.out is None or args.roster is not None:
            raise ValueError("prepare args differ")
        prepare(plan, args.out)
    else:
        if args.out is None or args.roster is None:
            raise ValueError("run args missing")
        run(plan, args.roster, args.out)


if __name__ == "__main__":
    main()
