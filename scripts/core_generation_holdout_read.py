"""CPU-only reader for closed HOLD observations, paired parent health veto only.

Never imports a model, Torch, replay decoder or native extension. Observations are
streamed only after caller-trusted root/independent closed-GPU acceptance. No
raw shards, checkpoints or dense policy caches are opened by this entrypoint.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
import time
from collections.abc import Iterator
from typing import Any, cast

from scripts import core_generation_holdout_metrics as metrics
from scripts import core_generation_holdout_ruler as ruler

require = ruler.require


def closed_observations(m: ruler.Metadata, p: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    binding = p["scoring"]
    require(set(binding) == {"root", "independent"}, "invalid scoring acceptance binding")
    root, review = m.read(binding["root"]), m.read(binding["independent"])
    require(set(root) == {"status", "protocol", "readout", "owner"} and root["status"] == ruler.SCORE_ROOT, "unaccepted scoring observations")
    require(set(review) == {"status", "root_acceptance", "protocol", "readout", "findings"}
            and review["status"] == ruler.SCORE_REVIEW and review["root_acceptance"] == binding["root"] and review["findings"] == []
            and review["protocol"] == root["protocol"] and review["readout"] == root["readout"], "scoring independent review differs")
    protocol = m.read(root["protocol"])
    report = m.read(root["readout"])
    require(protocol["schema"] == ruler.SCORE_SCHEMA and protocol["outer_seconds"] == 1200
            and all(protocol[key] == p[key] for key in ("study_uuid", "design", "holdout", "parents", "models", "source", "runtime", "limits", "native_basis")), "scorer/reader identities differ")
    # Admission ref is authenticated by the closed owner and exact actual argv.
    command = root["owner"]["command"]
    require(type(command) is list and command.count("--admission") == command.count("--admission-sha256") == 1, "scorer admission argv ambiguous")
    command = cast(list[str], command)
    admission = {"path": command[command.index("--admission") + 1], "sha256": command[command.index("--admission-sha256") + 1]}
    suffix = ["-m", "scripts.core_generation_holdout_score", "--protocol", root["protocol"]["path"],
              "--protocol-sha256", root["protocol"]["sha256"], "--admission", admission["path"], "--admission-sha256", admission["sha256"]]
    terminal = ruler.closed_owner(m, root["owner"], resource="gpu", suffix=suffix, seconds=1200, _ruler_phase=True)
    ad = m.read(admission)
    require(root["owner"]["admission"] == admission and ad["status"] == ruler.ADMISSION and ad["protocol"] == root["protocol"]
            and ad["source"] == protocol["source"] and ad["stage"] == protocol["stage"] and ad["owner"] == protocol["owner"], "scorer admission projection differs")
    output = ruler.core.external(protocol["output"])
    require(root["readout"]["path"] == str(output / "readout.json") and not output.with_name(output.name + ".failed.json").exists(), "failed/substituted score output")
    require(report["status"] == ruler.SCORE_PENDING and report["protocol"] == root["protocol"]
            and report["source"] == p["source"] and report["study_uuid"] == p["study_uuid"]
            and report["holdout"] == p["holdout"] and report["parents"] == p["parents"] and report["model_deliveries"] == p["models"]
            and report["optimizer_constructed"] is False and report["optimizer_restored"] is False
            and report["optimizer_updates"] == report["sampler_draws"] == 0
            and report["closed_owner_attested_by_child"] is False, "invalid scoring receipt")
    require(set(report["row_logs"]) == set(metrics.LABELS), "missing model observations")
    for label, descriptor in report["row_logs"].items():
        require(descriptor["path"] == str(output / (label + ".rows.jsonl")), "substituted row log")
    return report, terminal


def read_rows(descriptor: dict[str, Any], cap: int) -> Iterator[dict[str, Any]]:
    require(set(descriptor) == {"path", "sha256", "bytes", "rows"}, "invalid row log descriptor")
    path = ruler.core.external(descriptor["path"])
    before = ruler.bank.stamp(path)
    ruler.bank.sha(descriptor["sha256"])
    require(type(descriptor["bytes"]) is int and 0 < descriptor["bytes"] == before[2] <= cap
            and type(descriptor["rows"]) is int and descriptor["rows"] > 0, "row log bound differs")
    digest = hashlib.sha256()
    count = 0
    with path.open("rb") as stream:
        while True:
            line = stream.readline(16385)
            if not line:
                break
            require(len(line) <= 16384 and line.endswith(b"\n"), "oversized/incomplete row observation")
            digest.update(line)
            count += 1
            row = json.loads(line, object_pairs_hook=ruler.bank.pairs, parse_constant=ruler.bank.nonfinite, parse_float=ruler.bank.finite_float)
            require(type(row) is dict, "row observation must be object")
            yield row
    require(count == descriptor["rows"] and ruler.bank.stamp(path) == before and digest.hexdigest() == descriptor["sha256"], "row log custody/coverage differs")


def reconcile_rows(report: dict[str, Any], view: Any, parents: dict[str, str], cap: int) -> list[dict[str, Any]]:
    """Stream four paired logs and independently reconstruct every game sum."""
    expected = ruler.plain(view.games)
    expected.sort(key=lambda game: (game["physical_game_id"].rsplit(":", 1)[0], int(game["physical_game_id"].rsplit(":", 1)[1])))
    by_id = {game["physical_game_id"]: game for game in expected}
    require(len(by_id) == len(expected) and report["eligible_games"] == len(expected)
            and report["eligible_rows"] == sum(game["rows"] for game in expected), "accepted/observed coverage differs")
    grouped: list[dict[str, Any]] = []
    row_ordinal = 0
    game_ordinal = 0
    sums: dict[str, dict[str, list[float]]] = {label: {metric: [] for metric in metrics.METRICS} for label in metrics.LABELS}
    identity = hashlib.sha256()
    plies: set[int] = set()
    streams = [read_rows(report["row_logs"][label], cap) for label in metrics.LABELS]
    for paired in itertools.zip_longest(*streams):
        require(all(row is not None for row in paired) and game_ordinal < len(expected), "dropped/extra model rows")
        game = expected[game_ordinal]
        common = None
        for label, row in zip(metrics.LABELS, paired, strict=True):
            if not isinstance(row, dict):
                raise ValueError("dropped/invalid row observation")
            require(set(row) == {"model", "physical_game_id", "game_id", "parent_id", "row_index", "ply_index", "outcome",
                    "producer_proof", "shard_inventory_sha256", "wdl_log_probs", "metrics"}, "invalid row schema")
            require(row["model"] == label and row["physical_game_id"] == game["physical_game_id"]
                    and row["parent_id"] == parents[game["physical_game_id"]] and type(row["game_id"]) is int and row["game_id"] == game["game_id"]
                    and type(row["row_index"]) is int and row["row_index"] == row_ordinal and type(row["ply_index"]) is int and row["ply_index"] >= 0
                    and type(row["outcome"]) is int and 0 <= row["outcome"] <= 2
                    and row["producer_proof"] == game["producer_proof"]
                    and row["shard_inventory_sha256"] == hashlib.sha256(ruler.canonical(game["shard"]["files"])).hexdigest(), "row membership/source/outcome differs")
            shared = {key: value for key, value in row.items() if key not in ("model", "wdl_log_probs", "metrics")}
            require(common is None or shared == common, "four-model paired row identity differs")
            common = shared
            raw_lp, raw_value = row["wdl_log_probs"], row["metrics"]
            require(type(raw_lp) is list and len(raw_lp) == 3, "invalid WDL log probabilities")
            lp_items = cast(list[Any], raw_lp)
            require(all(type(v) in (int, float) and math.isfinite(v) and v <= 0 for v in lp_items), "invalid WDL log probabilities")
            lp = cast(list[float], lp_items)
            require(abs(math.fsum(math.exp(v) for v in lp) - 1) <= 2e-5, "invalid WDL log probabilities")
            require(type(raw_value) is dict and set(raw_value) == set(metrics.METRICS), "missing/nonfinite/inconsistent metric")
            value_items = cast(dict[str, Any], raw_value)
            require(all(type(v) in (int, float) and math.isfinite(v) for v in value_items.values()), "missing/nonfinite/inconsistent metric")
            value = cast(dict[str, float], value_items)
            require(value["wdl_ce"] == -lp[row["outcome"]], "missing/nonfinite/inconsistent metric")
            if label == "A39":
                require(value["reference_forward_kl"] == 0, "reference self KL differs")
            for metric in metrics.METRICS:
                sums[label][metric].append(value[metric])
        if common is None:
            raise ValueError("missing paired row identity")
        require(common["ply_index"] not in plies, "duplicate physical row")
        plies.add(common["ply_index"])
        identity.update(ruler.canonical([game["physical_game_id"], game["game_id"], row_ordinal, common["ply_index"], common["outcome"]]) + b"\n")
        row_ordinal += 1
        if row_ordinal == game["rows"]:
            grouped.append({"physical_game_id": game["physical_game_id"], "parent_id": parents[game["physical_game_id"]], "rows": row_ordinal,
                "row_identity_sha256": identity.hexdigest(), "models": {label: {metric: math.fsum(values) for metric, values in values_by_metric.items()}
                    for label, values_by_metric in sums.items()}})
            game_ordinal += 1
            row_ordinal = 0
            identity = hashlib.sha256()
            plies = set()
            sums = {label: {metric: [] for metric in metrics.METRICS} for label in metrics.LABELS}
    require(game_ordinal == len(expected) and row_ordinal == 0 and grouped == report["games"], "incomplete/different game sums")
    require(metrics.summarize(grouped) == report["aggregate"], "aggregate reconstruction differs")
    return grouped


def execute(protocol_ref: dict[str, str], admission_ref: dict[str, str]) -> dict[str, Any]:
    p, m, output = ruler.phase_start(protocol_ref, admission_ref, reader=True)
    started = time.monotonic()
    try:
        require(not any(name == "torch" or name.startswith(("torch.", "chess_anti_engine.")) for name in sys.modules), "reader must be model/native free")
        view, parents, models = ruler.bindings(m, p)
        report, terminal = closed_observations(m, p)
        for label in metrics.LABELS:
            evidence = report["models"][label]
            require(evidence["checkpoint"] == models[label]["checkpoint"] and evidence["step"] == models[label]["step"]
                    and evidence["model_state_sha256"] == evidence["post_state_sha256"] == models[label]["model_state_sha256"]
                    and evidence["counts"] == {"state_entries": 494, "unique_parameters": 61444448, "naive_state_elements": 77173088}
                    and evidence["rng_unchanged_across_forwards"] is True and evidence["real_decodes"] == evidence["real_constructors"] == 1
                    and evidence["rows"] == report["eligible_rows"] and evidence["batch_size"] == 32
                    and evidence["forwards"] == sum((game["rows"] + 31) // 32 for game in view.games), "model scoring evidence incomplete")
        games = reconcile_rows(report, view, parents, p["limits"]["row_log_bytes"])
        import numpy as np

        require(np.__version__ == p["runtime"]["numpy_version"] and Path(np.__file__).resolve().is_relative_to(Path(p["runtime"]["managed_site"])), "reader numerical origin/version differs")
        health = metrics.parent_health(games)
        require(not any(name == "torch" or name.startswith(("torch.", "chess_anti_engine.")) for name in sys.modules), "reader imported model/native code")
        final_view, final_parents, final_models = ruler.bindings(m, p)
        require(final_view == view and final_parents == parents and final_models == models, "accepted metadata changed during read")
        ruler.source_guard(p["source"])
        m.finish()
        result = {"status": ruler.READ_PENDING, "protocol": protocol_ref, "study_uuid": p["study_uuid"], "scoring": p["scoring"],
            "source": p["source"], "health": health, "games": games, "row_observations_reconciled": True,
            "GPU_owner_wall_seconds": terminal["wall_seconds"], "body_elapsed_seconds": time.monotonic() - started,
            "model_or_shard_payloads_opened": False, "closed_owner_attested_by_child": False,
            "limits": "VETO_ONLY, conditional resolved HOLD population; no efficacy/safety/playing authority"}
        ruler.write(output / "readout.json", result)
        ruler.core.seal(output)
        return result
    except BaseException as error:
        ruler.write(output.with_name(output.name + ".failed.json"), {"status": "UNREAD", "protocol": protocol_ref, "error": repr(error)})
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("protocol", "protocol-sha256", "admission", "admission-sha256"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    result = execute({"path": args.protocol, "sha256": args.protocol_sha256}, {"path": args.admission, "sha256": args.admission_sha256})
    print(ruler.canonical({"status": result["status"], "health": result["health"]}).decode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
