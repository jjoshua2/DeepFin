#!/usr/bin/env python3
"""Certify one no-resume B100/H20 sequential cell or its fixed 128-pair probe.

Consumes pinned launch evidence and completed banks. Does not qualify training,
launch jobs, or change the historical fixed-N bt4_joint_readout API.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, NoReturn

import chess

from chess_anti_engine.eval.sprt import BIAS_CAVEAT, SprtMonitor, SprtSpec
from chess_anti_engine.utils.game_log import read_game_log, settings_fingerprint
from scripts.arena_standard import pentanomial_counts, summarize_pentanomial

SPEC = SprtSpec(elo0=0, elo1=15, alpha=0.05, beta=0.10, first_pairs=128, step_pairs=64)


class InvalidCell(ValueError):
    """Operationally invalid evidence; never an unfavorable scientific verdict."""


def require(ok: bool, message: str) -> None:
    if not ok:
        raise InvalidCell(message)


def pinned(item: dict[str, Any]) -> Path:
    path = Path(item["path"])
    require(path.is_absolute(), "pins require absolute paths")
    require(
        hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"],
        f"changed pin: {path}",
    )
    return path


def read_json(item: dict[str, Any]) -> Any:
    return json.loads(pinned(item).read_text())


def same(actual: Any, expected: Any, name: str) -> None:
    # JSON comparison preserves booleans versus integer identities and rejects NaNs.
    require(
        json.dumps(actual, sort_keys=True, allow_nan=False)
        == json.dumps(expected, sort_keys=True, allow_nan=False),
        f"{name} differs",
    )


def opening_panel(item: dict[str, Any]) -> list[dict[str, Any]]:
    panel = read_json(item)
    require(
        isinstance(panel, list) and len(panel) == 500,
        "require canonical 500-opening panel",
    )
    for entry in panel:
        require(
            set(entry) == {"root_fen", "moves", "fen"},
            "unexpected opening panel fields",
        )
        board = chess.Board(entry["root_fen"])
        require(
            board.is_valid()
            and isinstance(entry["moves"], list)
            and len(entry["moves"]) == 16,
            "invalid opening root/history",
        )
        for move in entry["moves"]:
            board.push_uci(move)
        require(
            board.fen() == entry["fen"]
            and board.is_valid()
            and not board.is_game_over()
            and board.legal_moves.count() >= 2,
            "opening endpoint/history mismatch or unusable board",
        )
    require(
        len({entry["fen"] for entry in panel}) == 500, "duplicate opening endpoints"
    )
    return panel


def check_command(
    command: Any,
    settings: dict[str, Any],
    execution: dict[str, Any],
    bank: Path,
    result: Path,
    low: bool,
) -> None:
    require(
        isinstance(command, list) and all(isinstance(x, str) for x in command),
        "invalid command",
    )
    flags = {
        "--candidate": settings["candidate"],
        "--reference": settings["reference"],
        "--games": str(settings["games"]),
        "--mode": "matched_sims",
        "--sims": str(settings["sims_candidate"]),
        "--seed": "42",
        "--openings": settings["openings"],
        "--opening-plies": "16",
        "--max-plies": "300",
        "--temperature": "0.1",
        "--search-shape": "training",
        "--cand-gumbel": "policy_temp=1.0",
        "--ref-gumbel": "policy_temp=1.0",
        "--compile": "on",
        "--games-out": str(bank),
        "--out": str(result),
        "--max-concurrent-games": str(execution["max_concurrent_games"]),
        "--eval-max-batch": str(execution["eval_max_batch"]),
        "--max-seconds": str(execution["max_seconds"]),
    }
    entrypoints = [
        i for i, token in enumerate(command) if Path(token).name == "arena_standard.py"
    ]
    require(len(entrypoints) == 1, "unknown arena entrypoint")

    class CommandParser(argparse.ArgumentParser):
        def error(self, message: str) -> NoReturn:
            raise InvalidCell(f"unsupported arena command: {message}")

    parser = CommandParser(allow_abbrev=False, add_help=False)
    for flag in flags:
        parser.add_argument(flag, action="append", required=True)
    for flag in (
        "--device",
        "--sprt",
        "--label",
        "--report-every",
        "--compile-cache-dir",
        "--pgn-out",
    ):
        parser.add_argument(flag, action="append")
    parser.add_argument("--no-rolling", action="count", default=0)
    parsed = vars(parser.parse_args(command[entrypoints[0] + 1 :]))
    for key, values in parsed.items():
        if key != "no_rolling" and values is not None:
            require(len(values) == 1, f"duplicate command --{key.replace('_', '-')}")
    for flag, value in flags.items():
        actual = parsed[flag[2:].replace("-", "_")][0]
        if flag == "--max-seconds":
            require(float(actual) == float(value), "command deadline differs")
        else:
            require(actual == value, f"command {flag} differs")
    require(parsed["device"] in (None, ["cuda"]), "command device differs")
    require(
        parsed["no_rolling"] == int(execution["loop"] == "chunked"),
        "command loop differs",
    )
    require(bool(parsed["sprt"]) == low, "command stopping mode differs")
    if low:
        same(
            SprtSpec.from_cli(parsed["sprt"][0]).as_record(),
            SPEC.as_record(),
            "command SPRT",
        )


def summary(scores: list[float]) -> dict[str, Any]:
    s = summarize_pentanomial(pentanomial_counts(scores))
    return {
        "games": s.games,
        "pairs": s.pairs,
        "score": s.score,
        "score_se": s.score_se,
        "elo": s.elo,
        "elo_ci95": list(s.elo_ci95),
        "pentanomial": dict(zip(("WW", "WD_DW", "DD_WL", "LD_DL", "LL"), s.counts)),
    }


def read_cell(manifest: dict[str, Any]) -> dict[str, Any]:
    require(
        manifest["schema"] == 1 and manifest["mode"] in ("low_sprt", "high_fixed128"),
        "unknown reader profile",
    )
    low = manifest["mode"] == "low_sprt"
    cap, sims = (500, 100) if low else (128, 400)
    bank, result_path = pinned(manifest["bank"]), pinned(manifest["result"])
    process, launch = read_json(manifest["process"]), read_json(manifest["launch"])
    settings, execution = manifest["expected_settings"], manifest["expected_execution"]
    same(launch["settings"], settings, "launch settings")
    same(launch["execution"], execution, "launch execution")
    same(launch["opening_panel"], manifest["opening_panel"], "launch opening panel")
    identities = launch["identities"]
    for name in ("candidate", "reference", "book", "runtime", "preregistration"):
        identity = identities[name]
        require(
            isinstance(identity["sha256"], str)
            and len(identity["sha256"]) == 64
            and all(c in "0123456789abcdef" for c in identity["sha256"]),
            "invalid content identity",
        )
        if name in ("runtime", "preregistration"):
            pinned(identity)
    for name, field in [
        ("candidate", "candidate"),
        ("reference", "reference"),
        ("book", "openings"),
    ]:
        require(
            identities[name]["path"] == settings[field],
            f"{name} launch identity differs",
        )
    require(settings["candidate"] != settings["reference"], "candidate is reference")
    require(
        launch["candidate_role"] == "B100" and launch["reference_role"] == "H20",
        "recipe direction differs",
    )
    fixed = {
        "games": 2 * cap,
        "mode": "matched_sims",
        "sims_candidate": sims,
        "sims_reference": sims,
        "seed": 42,
        "opening_plies": 16,
        "openings_kind": "book",
        "max_plies": 300,
        "temperature": 0.1,
        "gumbel_add_noise": True,
    }
    for key, value in fixed.items():
        same(settings.get(key), value, f"protocol {key}")
    search = settings["search_candidate"]
    same(search, settings["search_reference"], "matched search")
    require(
        search["shape"] == "training" and search["gumbel"]["policy_temp"] == 1.0,
        "registered search/prior missing",
    )
    require(
        execution["loop"] in ("rolling", "chunked") and execution["compile"] == "on",
        "execution mode differs",
    )
    require(
        type(execution["eval_max_batch"]) is int
        and execution["eval_max_batch"] > 0
        and str(execution["eval_max_batch"]) == execution["eval_hoist"]
        and 0 < execution["eval_leaf_cap_uncapped"] <= execution["eval_max_batch"],
        "binding/unknown leaf cap",
    )
    require(
        0
        < execution["arena_pool_size"]
        <= min(2 * cap, execution["max_concurrent_games"]),
        "invalid arena pool",
    )
    require(
        all(
            type(execution[key]) in (int, float) and math.isfinite(execution[key])
            for key in ("max_seconds", "hard_seconds")
        )
        and 0 < execution["max_seconds"] < execution["hard_seconds"],
        "invalid deadline",
    )
    require(
        process.get("process_complete") is True and process.get("exit_code") == 0,
        "operational stage did not complete successfully",
    )
    same(process["command"], launch["command"], "terminal command")
    same(process["hard_seconds"], execution["hard_seconds"], "supervisor cap")
    for key in ("started_unix", "ended_unix", "stage_seconds", "gpu_seconds"):
        require(
            type(process[key]) in (int, float) and math.isfinite(process[key]),
            f"invalid process {key}",
        )
    elapsed = process["stage_seconds"]
    wall_elapsed = process["ended_unix"] - process["started_unix"]
    require(
        0 < elapsed <= execution["hard_seconds"]
        and 0 < wall_elapsed <= execution["hard_seconds"]
        and abs(wall_elapsed - elapsed) <= 1.0,
        "process elapsed time exceeds cap or disagrees with clock",
    )
    same(process["gpu_seconds"], elapsed, "process GPU charge")
    check_command(process["command"], settings, execution, bank, result_path, low)
    entrypoints = [
        i
        for i, token in enumerate(process["command"])
        if Path(token).name == "arena_standard.py"
    ]
    require(len(entrypoints) == 1, "unknown arena entrypoint")
    panel = opening_panel(manifest["opening_panel"])
    log = read_game_log(bank)
    require(not log.truncated_tail, "torn game bank")
    require(
        log.header.get("driver") == "arena_standard" and log.header.get("version") == 1,
        "unsupported header",
    )
    same(log.settings, settings, "bank settings")
    require(
        log.fingerprint == settings_fingerprint(settings), "invalid header fingerprint"
    )
    same(log.info.get("sprt"), SPEC.as_record() if low else None, "header SPRT")
    rows: dict[tuple[int, int], dict[str, Any]] = {}
    for row in log.games:
        pair, half = row.get("pair_id"), row.get("half")
        if not (
            type(pair) is int
            and 0 <= pair < cap
            and type(half) is int
            and half in (0, 1)
        ):
            raise InvalidCell("invalid pair/half")
        key = (pair, half)
        require(key not in rows, "duplicate/replayed finished half; no-resume profile")
        require(
            row.get("a_is_white") is (half == 0)
            and type(row.get("opening_index")) is int
            and row["opening_index"] == pair,
            "color/opening identity differs",
        )
        require(
            row.get("opening_fen") == row.get("start_fen") == panel[pair]["fen"],
            "canonical opening mapping differs",
        )
        white = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}.get(str(row.get("result")))
        if white is None:
            raise InvalidCell("unfinished/unknown result")
        score = white if half == 0 else 1.0 - white
        require(
            type(row.get("score_candidate")) in (int, float)
            and row["score_candidate"] == score,
            "score/result mismatch",
        )
        same(row.get("seed"), 42, "game seed")
        for name in ("loop", "compile", "eval_hoist"):
            same(row.get(name), execution[name], f"game {name}")
        rows[key] = row
    complete = {
        i: rows[i, 0]["score_candidate"] + rows[i, 1]["score_candidate"]
        for i in range(cap)
        if (i, 0) in rows and (i, 1) in rows
    }
    require(
        all(i in complete for i in range(128)),
        "missing complete first128 core; invalid package",
    )
    result_lines = result_path.read_text().splitlines()
    require(len(result_lines) == 1, "require one terminal result, no aggregate/replay")
    result = json.loads(result_lines[0])
    require(
        result.get("game_log_agrees") is True
        and result.get("game_log") == str(bank)
        and result.get("game_log_fingerprint") == log.fingerprint,
        "terminal bank binding differs",
    )
    same(result.get("argv"), process["command"][entrypoints[0] :], "terminal argv")
    require(result.get("device") == "cuda", "terminal device differs")
    require(
        result.get("resumed_pairs") == result.get("resumed_orphan_pairs") == 0,
        "resume not supported",
    )
    require(
        result.get("git_sha") == identities["runtime"]["git_sha"],
        "terminal runtime commit differs",
    )
    for key in (
        "candidate",
        "reference",
        "mode",
        "openings",
        "openings_kind",
        "opening_plies",
        "sims_candidate",
        "sims_reference",
        "temperature",
        "gumbel_add_noise",
        "max_plies",
        "seed",
        "search_candidate",
        "search_reference",
    ):
        same(result.get(key), settings[key], f"terminal {key}")
    for key in (
        "compile",
        "eval_hoist",
        "eval_max_batch",
        "eval_leaf_cap_uncapped",
        "max_concurrent_games",
        "arena_pool_size",
        "max_seconds",
    ):
        same(result.get(key), execution[key], f"terminal {key}")
    require(
        result.get("eval_leaf_cap_bound") is False
        and result.get("mixed_compile") is False
        and result.get("mixed_eval_hoist") is False,
        "mixed/binding execution",
    )
    same(result.get("compile_values"), [execution["compile"]], "compile values")
    same(result.get("eval_hoist_values"), [execution["eval_hoist"]], "hoist values")
    orphan = sorted([list(k) for k in rows if k[0] not in complete])
    if low:
        observed = result["sprt"]
        monitor = SprtMonitor(
            SPEC,
            pairs_cap=500,
            granularity="pair" if execution["loop"] == "rolling" else "chunk",
        )
        monitor.update(list(complete.values()), pair_ids=list(complete))
        reason = observed["stop_reason"]
        require(
            reason in ("boundary", "cap", "max_seconds"),
            "invalid/incomplete stop reason",
        )
        if not monitor.crossed():
            require(
                reason != "boundary" and (reason != "cap" or len(complete) == 500),
                "unsupported terminal stop",
            )
            require(
                reason != "max_seconds"
                or (monitor.pairs < 500 and elapsed >= execution["max_seconds"]),
                "deadline claimed after cap or before actual stage deadline",
            )
        monitor.finalize(stop_reason=reason)
        reconstructed = monitor.as_record()
        # Consultation count is loop-frequency telemetry, not inferable from final rows.
        skip = {"looks", "inflight_games", "not_started_games"}
        for key, value in reconstructed.items():
            if key not in skip:
                same(observed.get(key), value, f"SPRT {key}")
        require(
            type(observed.get("looks")) is int
            and observed["looks"] >= len(monitor.trajectory),
            "invalid consultation count",
        )
        inflight = observed["inflight_games"]
        require(
            isinstance(inflight, list)
            and all(
                isinstance(k, list)
                and len(k) == 2
                and type(k[0]) is int
                and 0 <= k[0] < cap
                and type(k[1]) is int
                and k[1] in (0, 1)
                for k in inflight
            ),
            "invalid inflight identities",
        )
        pending = {tuple(k) for k in inflight}
        require(
            len(pending) == len(inflight) and not pending.intersection(rows),
            "duplicate/finished inflight game",
        )
        unstarted = sorted(
            {(i, h) for i in range(cap) for h in (0, 1)} - set(rows) - pending
        )
        same(observed["not_started_games"], len(unstarted), "unstarted accounting")
        require(
            execution["loop"] == "rolling" or not (orphan or inflight),
            "chunked orphan/inflight work",
        )
        scores = monitor.pair_scores
        sequential = {
            **reconstructed,
            "inflight_games": inflight,
            "not_started_games": len(unstarted),
            "looks": observed["looks"],
            "looks_verified": False,
        }
    else:
        require(
            "sprt" not in result and len(rows) == 256 and not orphan,
            "incomplete/sequential fixed probe",
        )
        scores, sequential, unstarted, inflight = list(complete.values()), None, [], []
    measured = summary(scores)
    for key in ("games", "pairs", "pentanomial"):
        same(result.get(key), measured[key], f"terminal {key}")
    for key, digits in [("score", 5), ("score_se", 5), ("elo", 2)]:
        want = None if measured[key] is None else round(measured[key], digits)
        same(result.get(key), want, f"terminal {key}")
    same(
        result.get("elo_ci95"),
        [None if x is None else round(x, 2) for x in measured["elo_ci95"]],
        "terminal Elo interval",
    )
    same(result.get("games_requested"), 2 * cap, "terminal requested cap")
    same(result.get("truncated"), len(scores) < cap, "terminal truncation")
    contrast = None
    if not low:
        low_manifest = read_json(manifest["low_manifest"])
        require(
            low_manifest["mode"] == "low_sprt",
            "high requires one directly pinned low manifest",
        )
        low_report = read_cell(low_manifest)
        same(
            low_manifest["opening_panel"],
            manifest["opening_panel"],
            "low/high panel identity",
        )
        omitted = {"games", "sims_candidate", "sims_reference"}
        same(
            {
                k: v
                for k, v in low_manifest["expected_settings"].items()
                if k not in omitted
            },
            {k: v for k, v in settings.items() if k not in omitted},
            "low/high protocol",
        )
        same(
            read_json(low_manifest["launch"])["identities"],
            identities,
            "low/high content identities",
        )
        differences = [
            complete[i] / 2 - low_report["fixed_core_pair_scores"][i]
            for i in range(128)
        ]
        mean = statistics.mean(differences)
        half_width = 1.96 * statistics.stdev(differences) / math.sqrt(128)
        contrast = {
            "fixed_pairs": 128,
            "score_advantage_400_minus_100": mean,
            "paired_normal_ci95": [mean - half_width, mean + half_width],
            "low_manifest": manifest["low_manifest"],
        }
        pinned(manifest["low_manifest"])
    for key in ("bank", "result", "process", "launch", "opening_panel"):
        pinned(manifest[key])
    return {
        "schema": 1,
        "status": "VALID_CELL",
        "mode": manifest["mode"],
        "candidate_role": "B100",
        "reference_role": "H20",
        "result": measured,
        "sprt": sequential,
        "fixed_core_cross_budget": contrast,
        "fixed_core128": summary([complete[i] for i in range(128)]),
        "fixed_core_pair_scores": [complete[i] / 2 for i in range(128)],
        "opening_panel": manifest["opening_panel"],
        "complete_pair_ids": sorted(complete),
        "raw_finished_games": len(rows),
        "orphan_finished_halves": orphan,
        "inflight_games": inflight,
        "unstarted_game_ids": [list(k) for k in unstarted],
        "evidence": {k: manifest[k] for k in ("bank", "result", "process", "launch")},
        "limitations": [
            BIAS_CAVEAT
            if low
            else "Fixed128-pair probe; ordinary paired interval only.",
            "Panel root/history is reconstructed and bank endpoints checked; game rows do not bank the consumed initial history. Actual history consumption depends on frozen launcher/input-generation evidence.",
            "Launch pins attest checkpoint/book/runtime identities; upstream training and original payload qualification are not repeated.",
            "Same-seed development comparison, no recipe promotion or independent confirmation.",
            "Finished-bank identities validate accounting; inflight membership and consultation count remain terminal producer telemetry.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    args = parser.parse_args()
    try:
        item = {
            "path": str(args.manifest.resolve()),
            "sha256": args.expected_manifest_sha256,
        }
        report = read_cell(read_json(item))
        pinned(item)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "OPERATIONALLY_INVALID", "error": str(exc)}))
        raise SystemExit(2) from exc
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
