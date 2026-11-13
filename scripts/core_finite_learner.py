"""One finite core-learning generation, committed only at its closed boundary.

No collector, promotion policy, in-window recovery, or Worker/Tune restore route.
The trusted data admission interface is documented in docs/core_finite_learner.md.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import random
import re
import sys
from types import SimpleNamespace
from typing import Any, cast
import uuid


REPO = Path(__file__).resolve().parents[1]
CORE_KWARGS: dict[str, Any] = {
    "optimizer": "nadamw", "weight_decay_mode": "weight_decay",
    "matrix_optimizer_scope": "default", "grad_clip_mode": "none",
    "zclip_max_norm": None, "feature_dropout_p": 0.0,
    "fdp_king_safety": 0.0, "fdp_pins": 0.0, "fdp_pawns": 0.0,
    "fdp_mobility": 0.0, "fdp_outposts": 0.0,
    "resid_channel_dropout": 0.0, "resid_channel_balance_weight": 0.0,
    "rebuild_sf_targets": False, "sf_policy_sparse_ce": False,
    "rebuild_categorical_target": False, "swa_start": -1, "accum_steps": 1,
    "w_policy": 1.0, "w_wdl": 1.0,
    "w_soft": 0.0, "w_future": 0.0, "w_sf_own": 0.0,
    "w_sf_own_regret": 0.0, "w_sf_policy_floor": 0.0, "w_sf_shape": 0.0,
    "w_sf_move": 0.0, "w_sf_eval": 0.0, "w_categorical": 0.0,
    "w_volatility": 0.0, "w_sf_volatility": 0.0, "w_moves_left": 0.0,
    "policy_target_temp": 1.0, "soft_policy_min_tv": 0.0,
    "sf_wdl_frac": 0.0, "search_wdl_frac": 0.0, "sf_wdl_conf_power": 0.0,
    "sf_wdl_draw_scale": 1.0, "sf_wdl_temperature": 1.0,
    "sf_search_dampen_sf_low": 0.0, "sf_search_dampen_sf_high": 0.0,
    "use_adjusted_wdl_target": False, "wdl_terminal_outcome_plies": 0,
    "wdl_terminal_outcome_sf_frac": 0.0, "moves_left_max_plies": 450.0,
    "use_compile": False, "prefetch_batches": False,
    "prefetch_split_timing": False,
}
SOURCE_FILES = (
    "scripts/core_finite_learner.py", "scripts/bootstrap_checkpoint_resume.py",
    "scripts/offline_epoch_resume.py", "chess_anti_engine/train/trainer.py",
    "chess_anti_engine/train/losses.py", "chess_anti_engine/config_keys.py",
    "chess_anti_engine/tune/fresh_rl_donor.py",
    "chess_anti_engine/replay/generation.py", "chess_anti_engine/replay/shard.py",
)
METADATA_LIMIT = 4 << 20
GENERATION_METADATA_LIMIT = 32 << 20


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def require(condition: Any, message: str) -> None:
    if not condition:
        raise ValueError(message)


def integer(value: Any, name: str, minimum: int = 1) -> int:
    require(type(value) is int and value >= minimum, f"invalid {name}")
    return value


def number(value: Any, name: str) -> float:
    require(type(value) in (int, float) and math.isfinite(value), f"invalid {name}")
    return float(value)


def external(path: str) -> Path:
    p = Path(path)
    require(p.is_absolute() and p == p.resolve(), "artifact path must be absolute without symlinks")
    require(not any((parent / ".git").exists() for parent in (p, *p.parents)),
            "artifacts must be outside every checkout")
    return p


def file_sha(path: Path) -> str:
    require(path.is_file() and not path.is_symlink(), f"not a regular file: {path}")
    def identity() -> tuple[int, ...]:
        s = path.stat()
        return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns, s.st_mode
    before = identity()
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    require(before == identity(), f"file changed during hashing: {path}")
    return h.hexdigest()


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def read_ref(ref: dict[str, Any], *, max_bytes: int = METADATA_LIMIT,
             readonly: bool = True) -> tuple[Path, dict[str, Any]]:
    require(set(ref) == {"path", "sha256"}, "invalid file reference")
    path = external(ref["path"])
    require(not readonly or path.stat().st_mode & 0o222 == 0, f"input is writable: {path}")
    require(path.stat().st_size <= max_bytes, f"metadata exceeds {max_bytes} byte cap")
    require(file_sha(path) == ref["sha256"], f"metadata hash mismatch: {path}")
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns, before.st_mode)
            == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns, after.st_mode),
            "metadata changed during read")
    require(hashlib.sha256(raw).hexdigest() == ref["sha256"], "metadata changed")
    value = json.loads(raw, object_pairs_hook=_object,
                       parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))
    require(isinstance(value, dict), "metadata must be an object")
    return path, value


def validate_contract(c: dict[str, Any]) -> None:
    require(set(c) == {"schema", "run_uuid", "source_revision", "source_files", "architecture", "donor",
                       "batch_size", "games_per_generation", "updates_per_generation", "replay_generations",
                       "seed", "runtime", "lr", "mirror_prob", "limits"}, "unknown or missing contract fields")
    require(c["schema"] == "core_finite_learning_v1", "unsupported learning contract")
    require(str(uuid.UUID(c["run_uuid"])) == c["run_uuid"], "invalid run UUID")
    require(re.fullmatch("[0-9a-f]{40}", c["source_revision"]) is not None, "invalid source revision")
    require(set(SOURCE_FILES) <= set(c["source_files"]), "missing source pins")
    for name, sha in c["source_files"].items():
        path = REPO / name
        require(path.resolve().is_relative_to(REPO) and not path.is_symlink(), "invalid source path")
        require(file_sha(path) == sha, f"source mismatch: {name}")
    require(c["replay_generations"] == 4, "core replay must retain four whole generations")
    integer(c["batch_size"], "batch size")
    integer(c["games_per_generation"], "games per generation")
    integer(c["seed"], "seed", 0)
    require(c["seed"] < 2**32, "seed exceeds NumPy legacy RNG range")
    require(set(c["donor"]) == {"path", "sha256"}, "invalid donor reference")
    donor = external(c["donor"]["path"])
    require(donor.stat().st_mode & 0o222 == 0 and file_sha(donor) == c["donor"]["sha256"], "donor identity mismatch")
    require(isinstance(c["updates_per_generation"], list) and c["updates_per_generation"], "missing horizon")
    total = sum(integer(n, "update count") for n in c["updates_per_generation"])
    lr = c["lr"]
    require(set(lr) == {"peak", "floor", "warmup_steps", "warmup_start", "first_period"}, "invalid LR contract")
    peak, floor, start = (number(lr[k], k) for k in ("peak", "floor", "warmup_start"))
    warm = integer(lr["warmup_steps"], "warmup", 0)
    period = integer(lr["first_period"], "first cosine period")
    require(0 < peak and 0 <= floor <= peak and 0 <= start <= peak and warm < total, "invalid LR bounds")
    require(period > max(0, total - 1 - warm), "cosine restart would occur in the learning horizon")
    require(warm > 0 or start == peak, "zero warmup must start at peak")
    require(0 <= number(c["mirror_prob"], "mirror probability") <= 1, "invalid mirror probability")
    runtime = c["runtime"]
    require(set(runtime) == {"device", "use_amp", "torch_version", "numpy_version", "torch_threads", "cuda_devices"},
            "invalid runtime contract")
    require(runtime["device"] in ("cpu", "cuda:0") and type(runtime["use_amp"]) is bool, "invalid device/AMP")
    require(runtime["torch_threads"] == 2, "finite caller requires two Torch threads")
    require(runtime["cuda_devices"] == (0 if runtime["device"] == "cpu" else 1), "invalid CUDA device contract")
    require(set(c["limits"]) == {"max_rows", "max_decoded_window_bytes"}, "invalid data limits")
    for key, value in c["limits"].items():
        integer(value, key)


def verify_inventory(shard: dict[str, Any]) -> tuple[Path, str]:
    require(set(shard) == {"path", "files"}, "invalid shard reference")
    root = external(shard["path"])
    require(root.exists() and not root.is_symlink(), "missing shard")
    files = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
    require(bool(files), "empty shard")
    if root.is_dir():
        require(root.stat().st_mode & 0o222 == 0, "writable shard directory")
        for path in root.rglob("*"):
            require(not path.is_symlink(), "linked shard entry")
            require(path.is_file() or path.is_dir(), "nonregular shard entry")
            require(path.stat().st_mode & 0o222 == 0, "writable shard entry")
    actual = []
    for path in files:
        require(path.stat().st_mode & 0o222 == 0, "writable shard payload")
        actual.append({"path": "." if root.is_file() else path.relative_to(root).as_posix(),
                       "bytes": path.stat().st_size, "sha256": file_sha(path)})
    require(actual == shard["files"], "shard inventory mismatch")
    return root, digest(actual)


def failure_marker(output: Path) -> Path:
    # Deliberately outside the output tree: sealing that tree must not prevent
    # recording an unsuccessful completion-publication attempt.
    return output.with_name(output.name + ".failed.json")


def closed_parent(binding: dict[str, Any], c: dict[str, Any], contract_ref: dict[str, str],
                  stage: dict[str, Any]) -> dict[str, Any]:
    require(set(binding) == {"boundary", "owner_terminal", "owner_sha256", "command"},
            "parent requires an independently closed owner binding")
    boundary_path, parent = read_ref(binding["boundary"])
    _, original_stage = read_ref(parent["stage"])
    prior_output = external(original_stage["output"])
    require(boundary_path == prior_output / "boundary.json" and not failure_marker(prior_output).exists(),
            "parent publication is failed or misplaced")
    require(not (prior_output / "boundary.pending.json").exists(), "parent publication is incomplete")
    require(original_stage["schema"] == "core_finite_stage_v1"
            and original_stage["contract_sha256"] == contract_ref["sha256"]
            and original_stage["generation"] == stage["generation"] - 1
            and original_stage["generations"] == stage["generations"][:-1]
            and original_stage["parent"] == parent["parent"], "parent original stage mismatch")
    command = binding["command"]
    suffix = ["-m", "scripts.core_finite_learner", "--contract", contract_ref["path"],
              "--contract-sha256", contract_ref["sha256"], "--stage", parent["stage"]["path"],
              "--stage-sha256", parent["stage"]["sha256"]]
    require(isinstance(command, list) and len(command) > len(suffix)
            and all(isinstance(arg, str) and arg for arg in command)
            and command[-len(suffix):] == suffix, "parent owner command/stage mismatch")
    require(re.fullmatch("[0-9a-f]{64}", binding["owner_sha256"]) is not None, "invalid owner source SHA")
    # The existing owner emits 0644 terminal JSON only after cleanup. Root pins
    # its stable bytes after closure; do not chmod that independently owned file.
    _, terminal = read_ref(binding["owner_terminal"], readonly=False)
    require(terminal.get("status") == "PASS" and type(terminal.get("returncode")) is int
            and terminal["returncode"] == 0 and terminal.get("owned_processes_closed") is True
            and terminal.get("custody") == "OWNED_PROCESSES_CLOSED"
            and terminal.get("supervisor_sha256") == binding["owner_sha256"]
            and terminal.get("command") == command, "parent owner did not successfully close the exact command")
    # Its wrapper checkout may differ from the actual child package checkout.
    # Authenticate child source using the static contract and boundary instead.
    require(parent["source_revision"] == c["source_revision"], "parent child source mismatch")
    return parent


def metadata(c: dict[str, Any], stage: dict[str, Any], contract_ref: dict[str, str]) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    contract_sha = contract_ref["sha256"]
    require(set(stage) == {"schema", "contract_sha256", "generation", "generations", "parent", "output"},
            "unknown or missing stage fields")
    require(stage["schema"] == "core_finite_stage_v1" and stage["contract_sha256"] == contract_sha,
            "stage contract mismatch")
    g = integer(stage["generation"], "generation")
    require(g <= len(c["updates_per_generation"]) and len(stage["generations"]) == g, "generation horizon mismatch")
    generations = []
    physical_ids: set[str] = set()
    for i, ref in enumerate(stage["generations"], 1):
        _, gen = read_ref(ref, max_bytes=GENERATION_METADATA_LIMIT)
        require(gen["schema"] == "ROOT_ACCEPTED_CORE_TRAIN_GENERATION_V1", "unadmitted generation")
        require((gen["run_uuid"], gen["source_revision"], gen["generation"]) ==
                (c["run_uuid"], c["source_revision"], i), "generation identity mismatch")
        require(len(gen["games"]) == c["games_per_generation"], "incomplete game generation")
        _, serving = read_ref(gen["actor_serving"])
        served = serving["pre_state_sha256"]
        require(re.fullmatch("[0-9a-f]{64}", served) is not None
                and serving["actor_sha256"] == gen["actor_sha256"]
                and served == serving["post_state_sha256"] == serving["checkpoint_model_state_sha256"],
                "actor serving state mismatch")
        for game in gen["games"]:
            pid = game["physical_game_id"]
            require(isinstance(pid, str) and pid and pid not in physical_ids, "duplicate physical game")
            physical_ids.add(pid)
            _, proof = read_ref(game["producer_proof"])
            expected = {"status": "VERIFIED_RESOLVED_CORE_TRAIN_GAME_V1", "bank": "train",
                        "physical_game_id": pid, "game_id": game["game_id"], "rows": game["rows"],
                        "source_revision": c["source_revision"], "actor_sha256": gen["actor_sha256"],
                        "resolved": True, "full_history_verified": True, "raw_targets_verified": True,
                        "label_class": game["label_class"], "shard_inventory_sha256": digest(game["shard"]["files"])}
            require(all(proof.get(k) == v and type(proof.get(k)) is type(v) for k, v in expected.items()),
                    "TRAIN proof/identity predicates mismatch")
            require(game["bank"] == "train" and game["label_class"] in ("NATURAL", "TB_ADJUDICATED"),
                    "HOLD or unresolved game refused")
            integer(game["rows"], "game rows")
            integer(game["game_id"], "game ID", 0)
            # These are original producing evidence, not fresh booleans issued
            # by the learner. Root's trusted admission qualifies their contents.
            require({"outcome_proof", "validation", "provenance"} <= set(proof["evidence"]),
                    "missing original producing evidence")
            for kind, evidence in proof["evidence"].items():
                _, original = read_ref(evidence)
                identity = {"source_revision": c["source_revision"], "run_uuid": c["run_uuid"],
                            "generation": i, "physical_game_id": pid, "game_id": game["game_id"],
                            "rows": game["rows"], "actor_sha256": gen["actor_sha256"], "served_state_sha256": served}
                require(all(original.get(k) == v and type(original.get(k)) is type(v)
                            for k, v in identity.items()),
                        f"original {kind} source/game/actor mismatch")
                if kind == "outcome_proof":
                    verification = original["verification"]
                    require(original["schema"] == "CORE_PRODUCING_OUTCOME_PROOF_V1"
                            and original["proof_sha256"] == digest(verification)
                            and all(verification.get(k) == v for k, v in identity.items()
                                    if k not in ("rows", "generation", "run_uuid"))
                            and verification["resolved"] is True
                            and verification["retained_rows"] == game["rows"],
                            "original issued proof linkage mismatch")
                    require(original["scientific_class"] == (
                        "STRICT_TB_ADJUDICATED" if game["label_class"] == "TB_ADJUDICATED" else "NATURAL"
                    ) and verification["scientific_class"] == original["scientific_class"],
                            "original resolved outcome class mismatch")
            verify_inventory(game["shard"])
        generations.append(gen)
    require(generations[0]["actor_sha256"] == c["donor"]["sha256"], "first actor is not donor")
    parent = None
    if g == 1:
        require(stage["parent"] is None, "first generation cannot resume")
    else:
        parent = closed_parent(stage["parent"], c, contract_ref, stage)
        require(parent["status"] == "COMPLETED_CORE_GENERATION_V1" and parent["generation"] == g - 1,
                "only closed consecutive boundaries may resume")
        require(parent["contract_sha256"] == contract_sha and parent["source_revision"] == c["source_revision"],
                "parent contract/source mismatch")
        require(parent["run_uuid"] == c["run_uuid"] and parent["donor"] == c["donor"], "parent donor/run mismatch")
        require(parent["generation_manifests"] == stage["generations"][:-1], "parent data history changed")
        require(parent["step"] == sum(c["updates_per_generation"][:g - 1]), "parent dose mismatch")
        require(generations[-1]["actor_sha256"] == parent["checkpoint"]["sha256"], "actor/parent checkpoint mismatch")
        checkpoint = external(parent["checkpoint"]["path"])
        require(checkpoint.stat().st_mode & 0o222 == 0
                and file_sha(checkpoint) == parent["checkpoint"]["sha256"], "parent checkpoint changed or writable")
    return generations, parent


def make_trainer(c: dict[str, Any], log_dir: Path) -> Any:
    import torch
    from chess_anti_engine.model import ARCH_SCHEMA_VERSION, build_model, model_config_from_manifest_dict
    from chess_anti_engine.train import Trainer

    class NoRetryTrainer(Trainer):
        def _run_optimizer_step(self, **kwargs: Any) -> tuple[int, float]:
            try:
                result = super()._run_optimizer_step(**kwargs)
            except RuntimeError as exc:
                # Trainer's replacement-replay loop catches RuntimeError("CUDA").
                # ValueError makes this attempt terminal, without changing its math.
                raise ValueError("finite core update failed; retries forbidden") from exc
            require(not kwargs["step_opt_stats"].get("nonfinite_grad"), "finite core nonfinite update refused")
            tensors = list(self.model.parameters())
            for state in self.opt.state.values():
                tensors.extend(value for value in state.values() if isinstance(value, torch.Tensor))
            checks: dict[Any, list[Any]] = {}
            for tensor in tensors:
                checks.setdefault(tensor.device, []).append(torch.isfinite(tensor.detach()).all())
            require(all(bool(torch.stack(flags).all()) for flags in checks.values()),
                    "finite core nonfinite trained state refused")
            return result

    arch = c["architecture"]
    require(arch["_schema_version"] == ARCH_SCHEMA_VERSION, "unsupported architecture schema")
    cfg = model_config_from_manifest_dict(arch)
    require(canonical({"_schema_version": ARCH_SCHEMA_VERSION, **asdict(cfg)}) == canonical(arch),
            "architecture fields would be defaulted or dropped")
    lr = c["lr"]
    return NoRetryTrainer(
        build_model(cfg), model_config=cfg, log_dir=log_dir,
        **CORE_KWARGS, device=c["runtime"]["device"], use_amp=c["runtime"]["use_amp"],
        lr=lr["peak"], lr_eta_min=lr["floor"], warmup_steps=lr["warmup_steps"],
        warmup_lr_start=lr["warmup_start"], lr_schedule="cosine", lr_T0=lr["first_period"],
        lr_T_mult=1, mirror_prob=c["mirror_prob"],
    )


def import_custody() -> None:
    """A correct Git pin cannot authenticate imports from another checkout."""
    for name, module in tuple(sys.modules.items()):
        if name == "chess_anti_engine" or name.startswith("chess_anti_engine."):
            filename = getattr(module, "__file__", None)
            require(filename is not None and Path(filename).resolve().is_relative_to(REPO / "chess_anti_engine"),
                    f"import came from another checkout: {name}")


def backend_profile(torch: Any) -> dict[str, Any]:
    return {"matmul_precision": torch.get_float32_matmul_precision(),
            "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "torch_threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads()}


def load_replay(c: dict[str, Any], generations: list[dict[str, Any]], rng: Any, on_draw: Any) -> Any:
    import numpy as np
    from chess_anti_engine.model import infer_input_planes
    from chess_anti_engine.replay.generation import GenerationChunk, GenerationReplay
    from chess_anti_engine.replay.shard import load_shard_arrays
    from chess_anti_engine.replay.shard import history_rep_fix_from_arrays

    replay = GenerationReplay(generations=4, rng=rng, on_draw=on_draw)
    byte_counts: list[int] = []
    for gen in generations:
        chunks = []
        decoded = 0
        for game in gen["games"]:
            path, sha = verify_inventory(game["shard"])
            arrays, _ = load_shard_arrays(path, lazy=False, validate=True)
            require(arrays["x"].shape == (game["rows"], infer_input_planes(c["architecture"]["input_extra_features"]), 8, 8),
                    "loaded input shape mismatch")
            require(arrays["x"].dtype == np.float16 and arrays["policy_target"].dtype == np.float16,
                    "loaded input/target precision mismatch")
            for key in ("has_policy", "has_legal_mask", "is_network_turn", "has_is_network_turn", "is_selfplay", "has_is_selfplay", "has_game_id", "has_ply_index"):
                require(key in arrays and np.all(arrays[key] == 1), f"missing active TRAIN identity: {key}")
            require(not np.any(arrays["policy_target"][arrays["legal_mask"] == 0] != 0), "illegal policy target mass")
            require(all(np.issubdtype(arrays[key].dtype, np.integer) for key in ("game_id", "ply_index", "wdl_target")),
                    "loaded physical/result indices are not integers")
            require(np.all(arrays["game_id"] == game["game_id"]), "loaded game ID mismatch")
            require(len(np.unique(arrays["ply_index"])) == game["rows"], "duplicate physical rows")
            require(str(np.asarray(arrays["_input_history_encoding"]).item()) == c["architecture"]["input_history_encoding"],
                    "loaded input history mismatch")
            require(str(np.asarray(arrays["_policy_encoding"]).item()) == c["architecture"]["policy_encoding"],
                    "loaded policy encoding mismatch")
            require(history_rep_fix_from_arrays(arrays) == c["architecture"]["history_rep_fix"],
                    "loaded repetition encoding mismatch")
            require(verify_inventory(game["shard"])[1] == sha, "shard changed during eager load")
            for array in arrays.values():
                array.setflags(write=False)
            decoded += sum(a.nbytes for a in arrays.values())
            chunks.append(GenerationChunk(sha, arrays))
        byte_counts.append(decoded)
        require(sum(byte_counts[-4:]) <= c["limits"]["max_decoded_window_bytes"], "decoded window budget exceeded")
        replay.admit_generation(gen["generation"], chunks)
        require(len(replay) <= c["limits"]["max_rows"], "replay row budget exceeded")
        # No list retains all historical arrays; GenerationReplay drops complete
        # evicted generations. Rebuilding creates only this generation's input.
        del chunks, arrays
    return replay


def write_json(path: Path, value: Any) -> None:
    with path.open("x") as stream:
        stream.write(canonical(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def seal(path: Path, *, root: bool = True) -> None:
    for file in path.rglob("*"):
        if file.is_file():
            file.chmod(0o444)
    for directory in sorted((p for p in path.rglob("*") if p.is_dir()), reverse=True):
        directory.chmod(0o555)
    if root:
        path.chmod(0o555)


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_boundary(output: Path, boundary: dict[str, Any]) -> None:
    pending = output / "boundary.pending.json"
    write_json(pending, boundary)
    # Freeze all payloads before exposing the final name. The child still cannot
    # attest its eventual owner return code or closure; parent binding does that.
    seal(output, root=False)
    fsync_directory(output)
    os.link(pending, output / "boundary.json", follow_symlinks=False)
    pending.unlink()
    output.chmod(0o555)
    fsync_directory(output)
    fsync_directory(output.parent)


def execute(contract_ref: dict[str, str], stage_ref: dict[str, str]) -> dict[str, Any]:
    _, c = read_ref(contract_ref)
    _, stage = read_ref(stage_ref)
    validate_contract(c)
    # A Git read cannot refresh the pinned worktree index.
    import subprocess
    git_env = dict(os.environ, GIT_OPTIONAL_LOCKS="0")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, env=git_env, text=True).strip()
    require(head == c["source_revision"], "source revision mismatch")
    require(not subprocess.check_output(["git", "diff", "HEAD", "--name-only"], cwd=REPO, env=git_env), "dirty tracked source")
    generations, parent = metadata(c, stage, contract_ref)
    output = external(stage["output"])
    require(not failure_marker(output).exists(), "failed attempt output cannot be replayed")
    output.mkdir(parents=False, exist_ok=False)
    trainer = None
    drawn = 0
    applied = 0
    try:
        import numpy as np
        import torch
        from scripts.offline_epoch_resume import _restore_rng, _rng_state
        require(Path(inspect.getfile(_rng_state)).resolve() == REPO / "scripts/offline_epoch_resume.py",
                "RNG helper came from another checkout")
        require((torch.__version__, np.__version__) == (c["runtime"]["torch_version"], c["runtime"]["numpy_version"]),
                "runtime versions mismatch")
        torch.set_num_threads(2)
        backend = backend_profile(torch)
        if parent is not None:
            require(backend == parent["backend"], "continuation backend settings changed")
        require((torch.cuda.device_count() if torch.cuda.is_available() else 0) == c["runtime"]["cuda_devices"],
                "CUDA visibility differs from runtime contract")
        random.seed(c["seed"])
        np.random.seed(c["seed"])
        torch.manual_seed(c["seed"])
        rng = np.random.default_rng(c["seed"])
        trainer = make_trainer(c, output / "tb")
        import_custody()
        if parent is None:
            from chess_anti_engine.model import ARCH_SCHEMA_VERSION
            from chess_anti_engine.tune.fresh_rl_donor import load_fresh_donor
            from chess_anti_engine.tune.startup_intent import SCHEMA, contract_digest
            settings = {"lr": c["lr"]["peak"], "optimizer": "nadamw", "lr_schedule": "cosine",
                        "batch_size": c["batch_size"], "accum_steps": 1}
            config = {**settings, "rl_startup_intent": "fresh_bootstrap",
                      "bootstrap_handoff_mode": "fresh_rl_complete_v1", "bootstrap_checkpoint": c["donor"]["path"],
                      "bootstrap_checkpoint_sha256": c["donor"]["sha256"], "bootstrap_model_contract": c["architecture"],
                      "rl_startup_contract": {"schema": SCHEMA, "run_uuid": c["run_uuid"],
                      "donor_sha256": c["donor"]["sha256"], "model_contract_sha256": contract_digest(c["architecture"]), "settings": settings}}
            external(c["donor"]["path"])
            load_fresh_donor(config=config, trainer=trainer, checkpoint=None,
                             restore=SimpleNamespace(seed_warmstart_used=False), torch_module=torch,
                             schema_version=ARCH_SCHEMA_VERSION)
            require(trainer.step == 0 and not trainer.opt.state and trainer._swa_model is None, "fresh state not empty")
        else:
            from scripts.bootstrap_checkpoint_resume import resume_bootstrap
            require(Path(inspect.getfile(resume_bootstrap)).resolve() == REPO / "scripts/bootstrap_checkpoint_resume.py",
                    "resume helper came from another checkout")
            resume_bootstrap(trainer, external(parent["checkpoint"]["path"]), parent["checkpoint"]["sha256"],
                             parent["step"], c["seed"])

        with (output / "draws.jsonl").open("x") as journal:
            def on_draw(rows: Any) -> None:
                nonlocal drawn
                journal.write(canonical({"draw_batch": drawn // c["batch_size"], "rows": [asdict(row) for row in rows]}) + "\n")
                journal.flush()
                os.fsync(journal.fileno())
                drawn += len(rows)

            replay = load_replay(c, generations, rng, on_draw)
            import_custody()
            # Restore AFTER model construction/load and source reconstruction.
            # The reused helper's 'mask' field is NumPy's global RandomState here.
            if parent is not None:
                _restore_rng(parent["rng"], rng, cast(Any, np.random))
            start = trainer.step
            count = c["updates_per_generation"][stage["generation"] - 1]
            def counted(_optimizer: Any, _args: Any, _kwargs: Any) -> None:
                nonlocal applied
                applied += 1
            hook = trainer.opt.register_step_post_hook(counted)
            try:
                metrics = trainer.train_steps(replay, batch_size=c["batch_size"], steps=count)
            finally:
                hook.remove()
            require(trainer.step == start + count and applied == count, "applied update count mismatch")
            require(metrics.grad_nonfinite_skip_rate == 0 and metrics.transient_cuda_retry_batches == 0, "skips/retries refused")
            require(metrics.train_steps_done == count, "Trainer applied count mismatch")
            require(drawn == count * c["batch_size"] == metrics.train_samples_seen, "drawn/applied views mismatch")
        if c["runtime"]["device"].startswith("cuda"):
            torch.cuda.synchronize()
        checkpoint = output / "learner.pt"
        trainer.save(checkpoint)
        trainer.writer.close()
        with checkpoint.open("rb") as stream:
            os.fsync(stream.fileno())
        boundary = {"status": "COMPLETED_CORE_GENERATION_V1", "run_uuid": c["run_uuid"],
                    "source_revision": c["source_revision"], "contract_sha256": contract_ref["sha256"],
                    "stage": stage_ref, "parent": stage["parent"], "donor": c["donor"],
                    "generation": stage["generation"], "generation_manifests": stage["generations"],
                    "checkpoint": {"path": str(checkpoint), "sha256": file_sha(checkpoint)},
                    "final_checkpoint": {"path": str(checkpoint), "sha256": file_sha(checkpoint)},
                    "final_step": trainer.step,
                    "step": trainer.step, "applied_updates": applied, "applied_views": applied * c["batch_size"],
                    "drawn_rows": drawn, "draws_sha256": file_sha(output / "draws.jsonl"),
                    "membership": replay.membership, "metrics": asdict(metrics),
                    "rng": _rng_state(rng, cast(Any, np.random)), "backend": backend,
                    "champion_changed": False,
                    "recovery_scope": "closed generation boundary only; interrupted output is terminal/unqualified"}
        publish_boundary(output, boundary)
        return boundary
    except BaseException as exc:
        failure = {"status": "FAILED_CORE_GENERATION_NO_RECOVERY",
                   "error": repr(exc), "contract": contract_ref, "stage": stage_ref,
                   "cause": repr(exc.__cause__),
                   "drawn_rows": drawn, "completed_optimizer_calls": applied}
        # Always attempt an external marker, even if a final boundary name was
        # exposed before a chmod/fsync failure. Never invent the owner's terminal.
        try:
            marker = failure_marker(output)
            write_json(marker, failure)
            marker.chmod(0o444)
            fsync_directory(marker.parent)
        except BaseException as marker_error:
            print(f"Failed to publish failure marker: {marker_error!r}", file=sys.stderr)
        try:
            if trainer is not None:
                trainer.writer.close()
            if not (output / "failure.json").exists() and output.stat().st_mode & 0o200:
                write_json(output / "failure.json", failure)
            seal(output)
        except BaseException as cleanup_error:
            print(f"Failed to seal failed output: {cleanup_error!r}", file=sys.stderr)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--stage-sha256", required=True)
    args = parser.parse_args(argv)
    result = execute({"path": args.contract, "sha256": args.contract_sha256},
                     {"path": args.stage, "sha256": args.stage_sha256})
    print(canonical({"status": result["status"], "checkpoint": result["checkpoint"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
