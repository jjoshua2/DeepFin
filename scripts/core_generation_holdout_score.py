"""Owned deterministic all-row HOLD scoring; observations await root closure.

No sampling, augmentation, teacher, optimizer or TRAIN-bank acceptance. The
public strict model loader is used once per model. Fixed32 scoring batches may
have one deterministic final remainder; there is no capacity/batch fallback.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import math
import os
from pathlib import Path
import random
import time
from typing import Any
from unittest.mock import patch

from scripts import core_generation_holdout_ruler as ruler
from scripts import core_generation_holdout_metrics as metrics

require = ruler.require


def snapshot_rng() -> dict[str, Any]:
    import numpy as np
    import torch

    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch_cpu": torch.get_rng_state().clone(),
            "torch_cuda": [state.clone() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_initialized() else []}


def restore_rng(value: dict[str, Any]) -> None:
    import numpy as np
    import torch

    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch_cpu"])
    if value["torch_cuda"]:
        torch.cuda.set_rng_state_all(value["torch_cuda"])


def same_rng(left: dict[str, Any], right: dict[str, Any]) -> bool:
    import numpy as np
    import torch

    return (left["python"] == right["python"] and left["numpy"][0] == right["numpy"][0]
        and np.array_equal(left["numpy"][1], right["numpy"][1]) and left["numpy"][2:] == right["numpy"][2:]
        and torch.equal(left["torch_cpu"], right["torch_cpu"]) and len(left["torch_cuda"]) == len(right["torch_cuda"])
        and all(torch.equal(a, b) for a, b in zip(left["torch_cuda"], right["torch_cuda"], strict=True)))


def load_model(binding: dict[str, Any], *, device: str, complete: bool = False) -> tuple[Any, dict[str, Any]]:
    """One real decode and public constructor; validate tag BEFORE construction."""
    import torch
    from chess_anti_engine.uci import model_loader
    from scripts.core_finite_collect import state_sha256

    checkpoint = binding["checkpoint"]
    path = ruler.core.external(checkpoint["path"])
    before = ruler.bank.stamp(path)
    require(ruler.core.file_sha(path) == checkpoint["sha256"], "checkpoint hash differs")
    load, build = torch.load, model_loader.build_model
    decoded: list[int] = []
    built: list[int] = []
    rng = snapshot_rng()

    def capture(*args: Any, **kwargs: Any) -> Any:
        require(not decoded and Path(args[0]) == path and kwargs.get("weights_only") is True, "unexpected checkpoint decode")
        payload = load(*args, **kwargs)
        decoded.append(1)
        require(type(payload) is dict and ruler.canonical(payload.get("arch")) == ruler.canonical(binding["architecture"])
                and type(payload.get("step")) is int and payload["step"] == binding["step"], "loaded architecture/step differs")
        require(type(payload.get("model")) is dict and state_sha256(payload["model"]) == binding["model_state_sha256"], "loaded model state differs")
        if complete:
            require(len(payload["model"]) == 494, "not complete494 checkpoint")
        if binding["tag"] is not None:
            require(payload.get(ruler.ANCHOR_KEY) == binding["tag"] and "core_generation_bank_learner_v1" not in payload
                    and payload.get("frozen_reference_policy_anchor_v1") == binding["tag"]["policy_anchor"], "loaded model family/reference differs")
        return payload

    def constructor(config: Any) -> Any:
        require(decoded == [1], "constructor before authenticated decode")
        from chess_anti_engine.model import ARCH_SCHEMA_VERSION

        require(ruler.canonical({"_schema_version": ARCH_SCHEMA_VERSION, **asdict(config)}) == ruler.canonical(binding["architecture"]), "loaded architecture would default/drop fields")
        model = build(config)
        built.append(id(model))
        return model

    try:
        with patch.object(torch, "load", capture), patch.object(model_loader, "build_model", constructor):
            model = model_loader.load_model_from_checkpoint(path, device="cpu", require_complete=True,
                model_config=model_loader.model_config_from_arch(binding["architecture"]))
        counts = {"state_entries": len(model.state_dict()), "unique_parameters": sum(parameter.numel() for parameter in model.parameters()),
                  "naive_state_elements": sum(tensor.numel() for tensor in model.state_dict().values())}
        if complete:
            require(counts == {"state_entries": 494, "unique_parameters": 61444448, "naive_state_elements": 77173088}, "not complete175/494 model")
        model.to(device).eval()
    finally:
        restore_rng(rng)  # Honest constructor initialization is temporary only.
    require(decoded == [1] and built == [id(model)] and same_rng(rng, snapshot_rng()), "model load/RNG custody differs")
    require(state_sha256(model.state_dict()) == binding["model_state_sha256"]
            and ruler.bank.stamp(path) == before and ruler.core.file_sha(path) == checkpoint["sha256"], "loaded checkpoint custody differs")
    require(all(str(parameter.device) == device for parameter in model.parameters()) and not model.training, "model device/eval differs")
    counts = {"state_entries": len(model.state_dict()), "unique_parameters": sum(parameter.numel() for parameter in model.parameters()),
              "naive_state_elements": sum(tensor.numel() for tensor in model.state_dict().values())}
    return model, {"counts": counts, "checkpoint": checkpoint, "step": binding["step"], "model_state_sha256": binding["model_state_sha256"],
                   "checkpoint_stamp": list(before), "real_decodes": len(decoded), "real_constructors": len(built),
                   "initialization_rng_restored": True, "optimizer_constructed_or_restored": False}


def load_games(view: Any, limits: dict[str, int]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Actual strict shard decode in fixed physical order, no replay draws."""
    import numpy as np
    from chess_anti_engine.model import infer_input_planes
    from chess_anti_engine.replay.shard import load_shard_arrays, history_rep_fix_from_arrays
    from scripts.core_finite_admit import shard_decoded_bound

    architecture = ruler.plain(view.identity["architecture"])
    games = ruler.plain(view.games)
    games.sort(key=lambda game: (game["physical_game_id"].rsplit(":", 1)[0], int(game["physical_game_id"].rsplit(":", 1)[1])))
    result = []
    decoded = 0
    rows = 0
    for game in games:
        require(game["bank"] == "holdout" and game["label_class"] in ("NATURAL", "TB_ADJUDICATED"), "not resolved HOLD rows")
        path, digest = ruler.core.verify_inventory(game["shard"])
        shard_decoded_bound(path, game["rows"], limits["game_decoded_bytes"])
        arrays, _ = load_shard_arrays(path, lazy=False, validate=True)
        n = game["rows"]
        require(arrays["x"].shape == (n, infer_input_planes(architecture["input_extra_features"]), 8, 8)
                and arrays["policy_target"].shape == arrays["legal_mask"].shape == (n, 1858)
                and arrays["x"].dtype == arrays["policy_target"].dtype == np.float16, "loaded HOLD shape/precision differs")
        for key in ("has_policy", "has_legal_mask", "is_network_turn", "has_is_network_turn", "is_selfplay", "has_is_selfplay", "has_game_id", "has_ply_index"):
            require(key in arrays and arrays[key].shape == (n,) and np.all(arrays[key] == 1), "missing active HOLD identity: " + key)
        for key in ("game_id", "ply_index", "wdl_target"):
            require(arrays[key].shape == (n,) and np.issubdtype(arrays[key].dtype, np.integer), "genuine outcome/row identity must be integer")
        require(type(game["game_id"]) is int and 0 <= game["game_id"] < 2**63 and np.all(arrays["game_id"] == game["game_id"])
                and np.all(arrays["ply_index"] >= 0) and len(np.unique(arrays["ply_index"])) == n
                and np.all((arrays["wdl_target"] >= 0) & (arrays["wdl_target"] <= 2)), "genuine HOLD game/ply/outcome differs; no draw fallback")
        legal, target = arrays["legal_mask"], arrays["policy_target"]
        require(np.all(np.isfinite(arrays["x"])) and np.all(np.isfinite(target)) and np.all(np.isfinite(legal))
                and np.all((legal == 0) | (legal == 1)) and np.all(legal.any(axis=-1)) and np.all(target >= 0)
                and np.all(target.astype(np.float32).sum(-1) > 0) and not np.any(target[legal == 0] != 0), "invalid HOLD legal target/input")
        require(str(np.asarray(arrays["_input_history_encoding"]).item()) == architecture["input_history_encoding"]
                and str(np.asarray(arrays["_policy_encoding"]).item()) == architecture["policy_encoding"]
                and history_rep_fix_from_arrays(arrays) == architecture["history_rep_fix"], "HOLD encoding/history differs")
        if architecture["use_dynamic_relations"] or architecture["policy_dynamic_relations"]:
            require("relations" in arrays and "has_relations" in arrays and np.all(arrays["has_relations"] == 1), "required relations missing")
        count = sum(array.nbytes for array in arrays.values())
        decoded += count
        rows += n
        require(count <= limits["game_decoded_bytes"] and decoded <= limits["total_decoded_bytes"]
                and rows <= limits["max_rows"], "decoded HOLD byte/row cap")
        require(ruler.core.verify_inventory(game["shard"])[1] == digest, "shard changed during decode")
        for array in arrays.values():
            array.setflags(write=False)
        result.append((game, arrays))
    return result


class RowLog:
    """Bounded JSONL, source-qualified scalar observations, never dense policies."""
    def __init__(self, path: Path, cap: int, stream: Any):
        self.path = path
        self.cap = cap
        self.stream = stream
        self.digest = hashlib.sha256()
        self.bytes = 0
        self.rows = 0

    def add(self, row: dict[str, Any]) -> None:
        raw = ruler.canonical(row) + b"\n"
        require(len(raw) <= 16384 and self.bytes + len(raw) <= self.cap, "row observation byte cap")
        self.stream.write(raw)
        self.digest.update(raw)
        self.bytes += len(raw)
        self.rows += 1

    def close(self) -> dict[str, Any]:
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.stream.close()
        self.path.chmod(0o444)
        require(ruler.core.file_sha(self.path) == self.digest.hexdigest(), "row log changed")
        return {"path": str(self.path), "sha256": self.digest.hexdigest(), "bytes": self.bytes, "rows": self.rows}


def score_games(games: list[tuple[dict[str, Any], dict[str, Any]]], parents: dict[str, str], models: dict[str, Any],
                output: Path, *, device: str, autocast: bool, limits: dict[str, int], complete: bool = False) -> dict[str, Any]:
    """The same concrete scoring body serves small CPU software fixtures.

    The public phase fixes CUDA/BF16 and complete494 models; this lower-level API
    has no authority and cannot issue a qualified readout by itself.
    """
    import numpy as np
    import torch
    from chess_anti_engine.replay.dataset import collate_arrays
    from chess_anti_engine.utils.amp import inference_autocast
    from scripts.core_finite_collect import state_sha256

    require(games and set(models) == set(metrics.LABELS), "missing games/models")
    count = sum(game["rows"] for game, _ in games)
    require(count <= limits["max_rows"] and count * 1858 * 4 <= limits["reference_cache_bytes"], "reference cache cap")
    # CPU resident, internal only: never serialized or included in observations.
    cache = np.empty((count, 1858), dtype=np.float32)
    observations: list[dict[str, Any]] = []
    for game, arrays in games:
        identity = hashlib.sha256()
        for index in range(game["rows"]):
            identity.update(ruler.canonical([game["physical_game_id"], game["game_id"], index, int(arrays["ply_index"][index]), int(arrays["wdl_target"][index])]) + b"\n")
        observations.append({"physical_game_id": game["physical_game_id"], "parent_id": parents[game["physical_game_id"]],
            "rows": game["rows"], "row_identity_sha256": identity.hexdigest(), "models": {}})
    model_evidence = {}
    logs = {}
    for label in metrics.LABELS:
        model, loading = load_model(models[label], device=device, complete=complete)
        pre = state_sha256(model.state_dict())
        require(pre == models[label]["model_state_sha256"], "pre-forward state differs")
        rng = snapshot_rng()
        row_path = output / (label + ".rows.jsonl")
        with row_path.open("xb") as stream:
            log = RowLog(row_path, limits["row_log_bytes"], stream)
            offset = 0
            forwards = 0
            dtypes: dict[str, str] = {}
            linear_dtypes: set[str] = set()
            def linear_hook(_module: Any, _args: Any, value: Any, _observed: set[str] = linear_dtypes) -> None:
                _observed.add(str(value.dtype))
            head = getattr(model, "policy_own", None)
            hook = head.q.register_forward_hook(linear_hook) if hasattr(head, "q") else None
            try:
                with torch.no_grad():
                    for ordinal, (game, arrays) in enumerate(games):
                        numerators: dict[str, list[float]] = {key: [] for key in metrics.METRICS}
                        n = game["rows"]
                        for start in range(0, n, 32):
                            stop = min(n, start + 32)
                            # Fixed source rows, no random, mirror, dropout or rebuild.
                            sliced = {key: value[start:stop] for key, value in arrays.items() if value.ndim > 0 and value.shape[0] == n}
                            batch = collate_arrays(sliced, device=device)
                            with inference_autocast(device, enabled=autocast, dtype="bf16"):
                                if device.startswith("cuda"):
                                    require(torch.is_autocast_enabled("cuda") and torch.get_autocast_dtype("cuda") == torch.bfloat16, "BF16 autocast inactive")
                                outputs = model(batch["x"], **({"relations": batch["relations"]} if "relations" in batch else {}))
                            require(not torch.is_grad_enabled() and not model.training, "evaluation mode changed")
                            logits = outputs["policy"] if "policy" in outputs else outputs["policy_own"]
                            dtypes = {"policy": str(logits.dtype), "wdl": str(outputs["wdl"].dtype), "input": str(batch["x"].dtype)}
                            if device.startswith("cuda"):
                                require(outputs["wdl"].is_cuda and outputs["wdl"].dtype == torch.bfloat16
                                        and logits.is_cuda and logits.dtype == torch.float32, "CUDA eager BF16/raw FP32-policy profile differs")
                            reference = None if label == "A39" else torch.from_numpy(cache[offset + start:offset + stop]).to(device)
                            values, policy_log, wdl_log = metrics.row_metrics(outputs, batch["legal_mask"], batch["policy_t"], batch["wdl_t"], reference)
                            if label == "A39":
                                cache[offset + start:offset + stop] = policy_log.cpu().numpy()
                            scalars = {key: tensor.cpu().tolist() for key, tensor in values.items()}
                            wdl_rows = wdl_log.cpu().tolist()
                            for local, index in enumerate(range(start, stop)):
                                row = {"model": label, "physical_game_id": game["physical_game_id"], "game_id": game["game_id"],
                                    "parent_id": parents[game["physical_game_id"]], "row_index": index, "ply_index": int(arrays["ply_index"][index]),
                                    "outcome": int(arrays["wdl_target"][index]), "producer_proof": game["producer_proof"],
                                    "shard_inventory_sha256": hashlib.sha256(ruler.canonical(game["shard"]["files"])).hexdigest(),
                                    "wdl_log_probs": wdl_rows[local], "metrics": {key: scalars[key][local] for key in metrics.METRICS}}
                                log.add(row)
                                for key in metrics.METRICS:
                                    numerators[key].append(scalars[key][local])
                            forwards += 1
                        observations[ordinal]["models"][label] = {key: math.fsum(values) for key, values in numerators.items()}
                        offset += n
                require(offset == count and same_rng(rng, snapshot_rng()) and pre == state_sha256(model.state_dict()), "scoring changed state/RNG or dropped rows")
                logs[label] = log.close()
            finally:
                if not log.stream.closed:
                    log.stream.close()
                if hook is not None:
                    hook.remove()
        require(ruler.bank.stamp(Path(models[label]["checkpoint"]["path"])) == tuple(loading["checkpoint_stamp"])
                and ruler.core.file_sha(Path(models[label]["checkpoint"]["path"])) == models[label]["checkpoint"]["sha256"], "checkpoint changed during scoring")
        model_evidence[label] = {**loading, "post_state_sha256": pre, "rng_unchanged_across_forwards": True,
            "forwards": forwards, "rows": count, "observed_last_batch_dtypes": dtypes, "policy_q_linear_dtypes": sorted(linear_dtypes),
            "batch_size": 32, "last_remainder_rule": "fixed final game remainder, no fallback"}
        del model
    return {"games": observations, "row_logs": logs, "models": model_evidence, "aggregate": metrics.summarize(observations)}


def execute(protocol_ref: dict[str, str], admission_ref: dict[str, str]) -> dict[str, Any]:
    p, m, output = ruler.phase_start(protocol_ref, admission_ref)
    started = time.monotonic()
    try:
        view, parents, models = ruler.bindings(m, p)
        native_before = ruler.native_custody(m, p["native_basis"], p["source"]["files"], imports=False)
        import numpy as np
        import torch

        require(str(torch.__version__) == p["runtime"]["torch_version"] and np.__version__ == p["runtime"]["numpy_version"], "numerical versions differ")
        for module in (np, torch):
            require(Path(module.__file__).resolve().is_relative_to(Path(p["runtime"]["managed_site"])), "foreign numerical origin")
        torch.set_num_threads(2)
        require(not torch.cuda.is_initialized(), "CUDA initialized before admission/setup")
        torch.manual_seed(2026111673)
        np.random.seed(2026111673)
        random.seed(2026111673)
        torch.cuda.init()
        torch.cuda.manual_seed_all(2026111673)
        require(ruler.core.backend_profile(torch) == models["A"]["boundary"]["backend"], "scoring/training backend profile differs")
        games = load_games(view, p["limits"])
        for binding in models.values():
            require(binding["architecture"]["input_extra_features"] == "v2_threats", "not selected175 architecture")
        result = score_games(games, parents, models, output, device="cuda:0", autocast=True, limits=p["limits"], complete=True)
        require(all(item["real_constructors"] == 1 and item["counts"] == {"state_entries": 494,
                "unique_parameters": 61444448, "naive_state_elements": 77173088} and item["policy_q_linear_dtypes"] == ["torch.bfloat16"]
                for item in result["models"].values()), "full494 model/precision construction differs")
        for game, _ in games:
            ruler.core.verify_inventory(game["shard"])
        ruler.core.import_custody()
        ruler.source_guard(p["source"])
        final_view, final_parents, final_models = ruler.bindings(m, p)
        require(final_view == view and final_parents == parents and final_models == models, "metadata/data deliveries changed during scoring")
        require(native_before == ruler.native_custody(m, p["native_basis"], p["source"]["files"], imports=True), "native bracket changed")
        m.finish()
        result.update(status=ruler.SCORE_PENDING, protocol=protocol_ref, study_uuid=p["study_uuid"], source=p["source"],
            holdout=p["holdout"], parents=p["parents"], model_deliveries=p["models"], native=native_before,
            backend=ruler.core.backend_profile(torch), body_elapsed_seconds=time.monotonic() - started,
            planned_attempts=64, attempt_class_counts=ruler.plain(view.evidence["class_counts"]),
            unresolved_attempts=view.evidence["class_counts"]["UNRESOLVED"],
            resolved_empty_attempts=64 - view.evidence["class_counts"]["UNRESOLVED"] - len(games), eligible_games=len(games), eligible_rows=sum(game["rows"] for game, _ in games),
            resolved_population_scope="only qualified resolved nonempty HOLD rows; censored/empty attempts never imputed",
            scoring_seed=2026111673, precision="eager CUDA BF16 autocast; raw policy FP32, WDL BF16; F32 metric arithmetic",
            optimizer_constructed=False, optimizer_restored=False, optimizer_updates=0, sampler_draws=0,
            torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(0), torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(0),
            mirror_or_feature_dropout=False, native_scope="fresh named-file/record/source/freshness/import bracket, not mapped-byte identity",
            closed_owner_attested_by_child=False)
        ruler.write(output / "readout.json", result)
        ruler.core.seal(output)
        return result
    except BaseException as error:
        ruler.write(output.with_name(output.name + ".failed.json"), {"status": "FAILED_UNREAD", "protocol": protocol_ref, "error": repr(error)})
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("protocol", "protocol-sha256", "admission", "admission-sha256"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    result = execute({"path": args.protocol, "sha256": args.protocol_sha256}, {"path": args.admission, "sha256": args.admission_sha256})
    print(ruler.canonical({"status": result["status"], "rows": result["eligible_rows"]}).decode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
