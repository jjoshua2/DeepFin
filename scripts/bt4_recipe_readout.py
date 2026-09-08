#!/usr/bin/env python3
"""Certify a qualified recipe-pair sequential cell or its fixed 128-pair probe.

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
import numpy as np

from chess_anti_engine.eval.sprt import BIAS_CAVEAT, SprtMonitor, SprtSpec
from chess_anti_engine.utils.game_log import read_game_log, settings_fingerprint
from scripts.arena_standard import pentanomial_counts, summarize_pentanomial

SPEC = SprtSpec(elo0=0, elo1=15, alpha=0.05, beta=0.10, first_pairs=128, step_pairs=64)
MATCHED_PROFILE = "matched_original_epoch"
CANONICAL_EPOCH = "dc687fc333295dee565d19bb4f20da5aa95479dba3aacc5499c22a4004acc64f"
# Historical training protocol, not the arena overlay or current-main trainer.
TRAINING_PINS = {
    "scripts/lc0_control_train.py": "52d1132689c1cd53a23b63c9274226b467bd9aafdc34548a18db121a03bf9337",
    "configs/lc0_positive_control.yaml": "413dbea9dcde2774eafc2fde706e639fef9e944e301717b938b39b4729633de2",
    "chess_anti_engine/replay/game_epoch.py": "621e5d0764e62cee492688e63e4099ff8cbc0d39ea094b252c3cae31cd74fde3",
    "data/nnue_derived/armB/qtemp_0.0005_hist_20m/derive_targets_summary.json": "391837e49773465edced77bfd13f4084edc60feeff0484078280873d942e50ef",
}
SCHEDULE_VERIFIER = "4e8e27e861021dd4c75a1a21404ff9aed0d55775e0212def585bc21b1d8cde18"


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


def same_likelihood(actual: Any, expected: float, name: str) -> None:
    # Python 3.12 changed float sum precision. The qualified 3.10 producer and
    # 3.13 reader can differ by a few ulps with identical counts and source.
    # This absolute tolerance applies ONLY to computed likelihoods, never to
    # protocol fields or identities, and cannot change either boundary side.
    require(type(actual) is float and math.isfinite(actual), f"{name} must be a finite float")
    require(math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12), f"{name} differs")
    require(
        (actual <= SPEC.bound_h0, actual >= SPEC.bound_h1)
        == (expected <= SPEC.bound_h0, expected >= SPEC.bound_h1),
        f"{name} decision boundary differs",
    )


def same_sprt_field(actual: Any, expected: Any, name: str) -> None:
    if name in ("llr", "llr_first"):
        same_likelihood(actual, expected, f"SPRT {name}")
    elif name == "llr_trajectory":
        require(type(actual) is list and len(actual) == len(expected), "SPRT trajectory length differs")
        for observed, reconstructed in zip(actual, expected):
            require(type(observed) is list and len(observed) == 2, "SPRT trajectory entry differs")
            same(observed[0], reconstructed[0], "SPRT trajectory look")
            same_likelihood(observed[1], reconstructed[1], "SPRT trajectory likelihood")
    else:
        same(actual, expected, f"SPRT {name}")


def matched_training_pair(evidence: dict[str, Any]) -> tuple[str, str]:
    """Bind roles to completed original-corpus epochs, without loading weights.

    Consumes existing completion and schedule proofs; it does not reconstruct
    training, decode a corpus, or extend qualification to another runtime.
    """
    roles = []
    for side in ("candidate", "reference"):
        checkpoint = evidence[side]
        role = checkpoint["role"]
        require(isinstance(role, str) and bool(role) and role.isascii()
                and all(c.isalnum() or c in "_-" for c in role), "invalid recipe role")
        receipt = read_json(evidence[side + "_training"])
        require(receipt["complete"] is True, "training incomplete")
        charge = receipt["training_charge_seconds"]
        require(type(charge) in (int, float) and math.isfinite(charge) and 0 < charge <= 16200,
                "training charge exceeds qualified cap")
        same(receipt["role"], role, "training role")
        same(receipt["checkpoint"], checkpoint, "training checkpoint")
        run = Path(receipt["run"])
        require(run.is_absolute() and Path(checkpoint["path"]) == run / "checkpoint.pt",
                "training checkpoint path differs")
        same(receipt["canonical_plan_sha256"], CANONICAL_EPOCH, "training canonical epoch")
        for suffix, digest in TRAINING_PINS.items():
            matches = [v for k, v in receipt["input_pins"].items()
                       if Path(k).is_absolute() and k.endswith("/" + suffix)]
            same(matches, [digest], "training protocol pin " + suffix)
        summary = read_json({"path": str(run / "summary.json"), "sha256": receipt["summary_sha256"]})
        schedule = read_json(receipt["schedule"])
        same(schedule["verifier_sha256"], SCHEDULE_VERIFIER, "schedule verifier")
        same(schedule["seed"], 0, "schedule seed")
        same(schedule["batch_size"], 512, "schedule batch size")
        runtime = schedule["runtime"]
        require(runtime["python"].startswith("3.10.12") and runtime["torch"] == "2.11.0+cu128"
                and runtime["numpy"] == "1.26.2", "unqualified training schedule runtime")
        source_pin = "data/nnue_derived/armB/qtemp_0.0005_hist_20m/derive_targets_summary.json"
        source_summary = str(Path(schedule["source"]) / "derive_targets_summary.json")
        require(Path(source_summary).is_absolute() and source_summary.endswith("/" + source_pin),
                "unqualified original corpus")
        same(schedule["pins"][source_summary], TRAINING_PINS[source_pin], "schedule source identity")
        for key, value in {"plan_sha256": CANONICAL_EPOCH, "rows_planned": 18910484,
                           "batches_planned": 36935, "seed": 0, "batch_size": 512}.items():
            same(schedule["source_plan"].get(key), value, "source plan " + key)
        same(list(schedule["arms"]), [role], "completed schedule role")
        arm = schedule["arms"][role]
        require(arm["metadata_matches_source"] is True and arm["training_completion_verified"] is True
                and arm["staging"] == "verified actual", "unqualified realized schedule")
        same(arm["canonical_plan_sha256"], CANONICAL_EPOCH, "realized canonical epoch")
        same(arm["summary_sha256"], receipt["summary_sha256"], "realized summary")
        physical = receipt["physical_plan_sha256"]
        same(arm["physical_plan_sha256"], physical, "realized physical epoch")
        same(summary["corpus"]["shard_dirs"], [arm["corpus"]], "training corpus")
        expected = {"mode": "game_epoch", "complete": True, "seed": 0, "batch_size": 512,
                    "rows_planned": 18910484, "rows_realized": 18910484,
                    "batches_planned": 36935, "batches_realized": 36935, "shards": 2309,
                    "games": 97968, "plan_workers": 16, "load_workers": 16,
                    "same_game_repeats_max": 0, "decoded_rows_resident": 0,
                    "plan_sha256": physical, "realized_sha256": physical}
        for key, value in expected.items():
            same(summary["sampling"].get(key), value, "training sampling " + key)
        for key, value in {"seed": 0, "batch_size": 512, "warmup_steps": 1000, "train_window_steps": 88,
                           "steps_realized": 36935, "compute_loss_calls": 36935}.items():
            same(summary.get(key), value, "training " + key)
        windows = summary["train_window_metrics"]
        same(summary["train_windows"], 420, "training windows")
        require(len(windows) == 420, "incomplete training windows")
        for index, window in enumerate(windows, 1):
            steps = min(88, 36935 - (index - 1) * 88)
            for key, value in {"window_index": index, "steps_requested": steps,
                               "train_steps_done": steps, "steps_cumulative": min(index * 88, 36935)}.items():
                same(window[key], value, "training window " + key)
            require(window["grad_nonfinite_skip_rate"] == 0 and window["transient_cuda_retry_batches"] == 0
                    and math.isfinite(window["loss"]) and math.isfinite(window["grad_norm_mean"]),
                    "training skipped/retried or nonfinite window")
        same(sum(w["train_samples_seen"] for w in windows), 18910484, "training sample total")
        require(any(c["role"] == "last" and c["path"] == checkpoint["path"]
                    and c["sha256"] == checkpoint["sha256"] for c in summary["checkpoints"]),
                "summary checkpoint differs")
        same(receipt["historical_valid_control"], summary["valid_control"], "historical validity")
        same(receipt["historical_validity_problems"], summary["validity_problems"], "historical limitations")
        roles.append(role)
    require(roles[0] != roles[1], "recipe roles must differ")
    require(evidence["candidate"]["path"] != evidence["reference"]["path"]
            and evidence["candidate"]["sha256"] != evidence["reference"]["sha256"],
            "candidate is reference")
    return roles[0], roles[1]


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
            and not board.is_game_over(),
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
    profile = launch.get("profile", "B100_H20")
    candidate_role, reference_role = "B100", "H20"
    if profile == MATCHED_PROFILE:
        same(manifest.get("profile"), profile, "reader recipe profile")
        training = launch["training"]
        candidate_role, reference_role = matched_training_pair(training)
        for side in ("candidate", "reference"):
            same({k: training[side][k] for k in ("path", "sha256")}, identities[side], "trained " + side)
            same(training[side + "_training"], identities[side + "_training"], "training proof " + side)
    else:
        require(profile == "B100_H20" and manifest.get("profile", "B100_H20") == profile,
                "unknown recipe profile")
    same([launch["candidate_role"], launch["reference_role"]], [candidate_role, reference_role],
         "recipe direction")
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
                same_sprt_field(observed.get(key), value, key)
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
        same([low_report["candidate_role"], low_report["reference_role"]],
             [candidate_role, reference_role], "low/high recipe direction")
        same(low_manifest.get("profile", "B100_H20"), profile, "low/high recipe profile")
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
        # Registered aligned opening-pair percentile bootstrap; one paired draw
        # retains covariance between the same low/high opening outcomes.
        rng = np.random.Generator(np.random.PCG64(20260903))
        means = np.empty(10000)
        delta = np.asarray(differences, dtype=np.float64)
        for start in range(0, 10000, 1000):
            indices = rng.integers(0, 128, size=(1000, 128))
            means[start : start + 1000] = delta[indices].mean(axis=1)
        contrast = {
            "fixed_pairs": 128,
            "score_advantage_400_minus_100": mean,
            "paired_bootstrap_ci95": np.percentile(means, [2.5, 97.5]).tolist(),
            "bootstrap": {
                "unit": "aligned opening pair",
                "samples": 10000,
                "seed": 20260903,
                "generator": "PCG64",
                "method": "percentile",
            },
            "low_manifest": manifest["low_manifest"],
        }
        pinned(manifest["low_manifest"])
    for key in ("bank", "result", "process", "launch", "opening_panel"):
        pinned(manifest[key])
    return {
        "schema": 1,
        "status": "VALID_CELL",
        "mode": manifest["mode"],
        "candidate_role": candidate_role,
        "reference_role": reference_role,
        **({"profile": profile} if profile == MATCHED_PROFILE else {}),
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
            ("Pinned completed training and schedule receipts are checked; original payload qualification is inherited, without corpus decoding or retraining."
             if profile == MATCHED_PROFILE else
             "Launch pins attest checkpoint/book/runtime identities; upstream training and original payload qualification are not repeated."),
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
