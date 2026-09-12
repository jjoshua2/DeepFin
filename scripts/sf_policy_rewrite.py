#!/usr/bin/env python3
"""Rewrite only policy on a pinned legacy uniform-d9 corpus, without inference.

The default q/.0005 mode is an identity control. Effective-cp/10 uses original
float64 scores (including mate encoding), never rounded rank-sidecar gaps.
Source history/value lineage is inherited, not freshly reconstructed or promoted.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chess
import numpy as np
import zarr

from chess_anti_engine.moves.encode import COMPACT_POLICY_SIZE
from chess_anti_engine.moves.leela_index import compact_index_for_move
from chess_anti_engine.stockfish.wdl import SF_CP_CLAMP_CP, mate_to_effective_cp
from scripts import derive_corpus_targets as derive
from scripts import sf_d9_rank_sidecar as rank
from scripts.bt4_policy_dump import file_sha256

SUMMARY = "sf_policy_rewrite_summary.json"
DOWNSIDE_ALGORITHM = "stored-b100-allmove-sf-gapgt300-weight0.5-ordinary-v1"
DOWNSIDE_SUMMARY = "bt4_sf_downside_policy_summary.json"
TACTICAL_SUMMARY = "bt4_sf_tactical_policy_summary.json"
TACTICAL_ALGORITHM = "stored-b100-sf-gap100-decay100-floor0.1-categorical-mates-v1"
MATE_SCORE_VALUES = np.array([abs(mate_to_effective_cp(i)) for i in range(501)])
ARRAYS = frozenset(
    {
        "x",
        "policy_target",
        "legal_mask",
        "game_id",
        "ply_index",
        "wdl_target",
        "search_wdl",
        "priority",
        "is_selfplay",
        "is_network_turn",
        "has_game_id",
        "has_ply_index",
        "has_policy",
        "has_legal_mask",
        "has_search_wdl",
        "has_is_selfplay",
        "has_is_network_turn",
    }
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass
class Observation:
    game: int
    ply: int
    indices: np.ndarray
    scores: np.ndarray
    input_key: str


def observation(row: dict[str, Any], config_sha: str) -> Observation:
    """Full legal phase-zero roster, preserving original score/order precision."""
    derive._check_row_identity(row, config_sha)
    require(derive.row_schema_of(row) == 3, "raw row is not history schema3")
    derive.require_row_regime(row)
    key = row.get("input_key")
    if not isinstance(key, str) or len(key) != 32:
        raise ValueError("missing original history key")
    bytes.fromhex(key)
    board = chess.Board(row["fen"])
    require(row.get("stm") == ("w" if board.turn else "b"), "raw stm differs from FEN")
    require(
        row.get("piece_count") == chess.popcount(board.occupied),
        "raw piece count differs",
    )
    require(len(row["phases"]) == 1, "requires original single-phase d9 source")
    phase = row["phases"][0]
    legal_moves = list(board.legal_moves)
    width = len(legal_moves)
    require(
        width > 0
        and type(phase.get("index")) is int
        and phase["index"] == 0
        and phase.get("width_requested") == "all"
        and phase.get("searchmoves") is None,
        "phase0 is not full-width",
    )
    require(
        all(
            type(phase.get(k)) is int and phase[k] == width
            for k in ("width_realized", "width_streamed")
        ),
        "raw full-width counts differ",
    )
    require(phase.get("depth_requested") == 9, "source requested depth differs")
    blocks = [block for block in phase["per_depth"] if block.get("depth") == 9]
    require(
        len(blocks) == 1
        and type(blocks[0]["depth"]) is int
        and blocks[0].get("complete") is True,
        "malformed d9 completion",
    )
    lines = rank.d9_lines(row)
    require(
        len(lines) == width
        and all(
            len(line) == 4
            and type(line[0]) is int
            and line[0] == i
            and type(line[2]) in (int, float)
            for i, line in enumerate(lines, 1)
        ),
        "malformed full d9 roster",
    )
    # Existing rank validation checks order, finiteness, range and duplicates.
    ranked = rank.rank_observation(row, top_k=width)
    require(
        {str(line[1]) for line in lines} == {m.uci() for m in legal_moves},
        "incomplete legal d9 support",
    )
    indices = np.asarray(
        [compact_index_for_move(board, chess.Move.from_uci(line[1])) for line in lines],
        dtype=np.int64,
    )
    require(
        np.array_equal(indices, ranked.indices), "compact map differs from rank mapping"
    )
    scores = np.asarray([line[2] for line in lines], dtype=np.float64)
    require(bool(np.isfinite(scores).all()), "nonfinite raw score")
    return Observation(int(row["game_id"]), int(row["ply"]), indices, scores, key)


def target(obs: Observation, score_space: str, temperature: float) -> np.ndarray:
    require(score_space in {"q", "effective-cp"}, "unsupported policy score space")
    temperature = derive.validate_temp(temperature)
    values = (
        derive.gate.q_from_effective_cp(obs.scores, slope=0.006, draw_width_cp=120.0)
        if score_space == "q"
        else obs.scores
    )
    probabilities = derive.shard_stored(
        derive.softmax_at_temp(values, temp=temperature)
    )
    result = np.zeros(COMPACT_POLICY_SIZE, dtype=np.float16)
    result[obs.indices] = probabilities
    require(
        bool(np.isfinite(result).all() and np.all(result >= 0)), "invalid stored policy"
    )
    require(
        abs(float(result.astype(np.float64).sum()) - 1.0) <= 2**-10,
        "stored policy mass error",
    )
    return result


def tactical_target(
    obs: Observation, stored: np.ndarray, *, downside: bool = False
) -> tuple[np.ndarray, dict[str, Any]]:
    """Attenuate actual stored B100 mass; never turn mate distances into cp gaps."""
    require(
        stored.shape == (COMPACT_POLICY_SIZE,) and stored.dtype == np.float16,
        "B100 policy schema differs",
    )
    require(
        bool(np.isfinite(stored).all() and (stored >= 0).all()), "invalid B100 policy"
    )
    require(
        obs.indices.ndim == 1
        and np.issubdtype(obs.indices.dtype, np.integer)
        and len(set(obs.indices.tolist())) == len(obs.indices)
        and bool(((obs.indices >= 0) & (obs.indices < COMPACT_POLICY_SIZE)).all()),
        "invalid tactical compact indices",
    )
    legal = np.zeros(COMPACT_POLICY_SIZE, dtype=bool)
    legal[obs.indices] = True
    require(not bool(np.any(stored[~legal] != 0)), "B100 illegal policy mass")
    require(
        abs(float(stored.astype(np.float64).sum()) - 1) <= 2**-10,
        "B100 policy mass differs",
    )
    scores = obs.scores
    require(
        scores.shape == obs.indices.shape
        and len(scores) > 0
        and bool(np.isfinite(scores).all()),
        "invalid tactical scores",
    )
    magnitude = np.abs(scores)
    mates = magnitude > SF_CP_CLAMP_CP
    # The generator preserves raw cp; accept only the disjoint documented cp
    # domain or exact values of the shared mate map, not an invented gray band.
    require(
        bool(np.isin(magnitude[mates], MATE_SCORE_VALUES).all()),
        "effective score outside cp or shared mate domain",
    )
    wins, losses = scores > SF_CP_CLAMP_CP, scores < -SF_CP_CLAMP_CP
    weights = np.ones(len(scores), dtype=np.float64)
    if downside:
        category = "mate_domain_unchanged" if bool(mates.any()) else "ordinary_downside"
        if not bool(mates.any()):
            weights[scores.max() - scores > 300.0] = 0.5
    elif bool(wins.any()):
        category = "winning_mate_available"
        weights[~wins] = 0.1
    elif bool(losses.all()):
        category = "all_forced_losses"
    else:
        category = "losing_mate_alternatives" if bool(losses.any()) else "no_mate"
        nonmate = ~losses
        deficits = scores[nonmate].max() - scores[nonmate]
        weights[nonmate] = np.maximum(
            0.1, np.exp(-np.maximum(0, deficits - 100.0) / 100.0)
        )
        weights[losses] = 0.1
    base = stored[obs.indices].astype(np.float64)
    base /= base.sum()
    ideal = base * weights
    ideal /= ideal.sum()
    unchanged = bool((weights == 1).all()) or (
        downside and len(np.unique(weights[base > 0])) == 1
    )
    result = stored.copy() if unchanged else np.zeros_like(stored)
    if not unchanged:
        result[obs.indices] = ideal.astype(np.float32).astype(np.float16)
    require(
        bool(np.isfinite(result).all())
        and abs(float(result.astype(np.float64).sum()) - 1) <= 2**-10,
        "tactical stored policy mass differs",
    )
    normalized = result[obs.indices].astype(np.float64)
    normalized /= normalized.sum()
    positive = base > 0
    relative = np.zeros_like(base)
    relative[positive] = np.abs(normalized[positive] / ideal[positive] - 1)
    return result, {
        "category": category,
        "winning_mate_zero_base_mass": int(
            bool(wins.any()) and not bool((base[wins] > 0).any())
        ),
        "support_losses": int(np.count_nonzero(positive & (result[obs.indices] == 0))),
        "ideal_to_stored_relative_error_max": float(relative.max()),
        "ideal_to_stored_TV": float(np.abs(normalized - ideal).sum() / 2),
    }


def recipe_for_summary(*, downside: bool = False) -> dict[str, Any]:
    if downside:
        return {
            "gap_cp_strictly_greater_than": 300.0,
            "flagged_relative_weight": 0.5,
            "mate_handling": "any-mate-domain-row-unchanged",
            "cp_domain": [-SF_CP_CLAMP_CP, SF_CP_CLAMP_CP],
            "base": "normalized stored B100 float16 policy",
            "storage": "float64 weighting -> float32 -> float16; zero flagged mass preserves bytes",
            "roster": "all original legal d9 moves; no next-best gate",
        }
    return {
        "gap_cp": 100.0,
        "decay_cp": 100.0,
        "relative_floor": 0.1,
        "mate_handling": "categorical-v1",
        "cp_domain": [-SF_CP_CLAMP_CP, SF_CP_CLAMP_CP],
        "base": "normalized stored B100 float16 policy",
        "storage": "float64 attenuation -> float32 -> float16; all-one weights preserve bytes",
    }


def tactical_source(
    args: argparse.Namespace,
    original: dict[str, Any],
    sf_root: Path,
) -> tuple[Path, dict[Path, str]] | None:
    """Admit the pinned B100 parent, retaining original SF/history lineage."""
    root_arg = getattr(args, "tactical_bt4_source", None)
    pins_args = [
        getattr(args, key, None)
        for key in ("expected_bt4_summary_sha256", "expected_bt4_mix_sha256")
    ]
    require(
        bool(root_arg) == all(bool(v) for v in pins_args)
        and (bool(root_arg) or not any(pins_args)),
        "tactical B100 source requires both pins",
    )
    if not root_arg:
        return None
    require(
        args.score_space == "q" and args.temperature == 0.0005,
        "tactical recipe cannot override score-space/temperature",
    )
    root = Path(root_arg).resolve()
    require(
        root != sf_root
        and not root.with_name(root.name + ".writing").exists()
        and not (root / "failed.json").exists(),
        "invalid B100 source",
    )
    pins = {
        root / derive.SUMMARY_NAME: str(pins_args[0]),
        root / "bt4_policy_mix_summary.json": str(pins_args[1]),
    }
    for path, digest in pins.items():
        require(file_sha256(path) == digest, "B100 summary pin differs")
    base = json.loads((root / derive.SUMMARY_NAME).read_text())
    mix = json.loads((root / "bt4_policy_mix_summary.json").read_text())
    # Historical summaries contain NaN diagnostics, so compare their JSON form.
    require(
        json.dumps(
            {k: v for k, v in base.items() if k != "policy_target_postprocess"},
            sort_keys=True,
        )
        == json.dumps(original, sort_keys=True)
        and json.dumps(base.get("policy_target_postprocess"), sort_keys=True)
        == json.dumps(mix, sort_keys=True),
        "B100 original source lineage differs",
    )
    expected = {
        "kind": "global",
        "algorithm": "legal-normalized-global-arithmetic-v1",
        "alpha": 1.0,
        "bt4_temperature": 0.5,
        "rows": original["realized"]["rows_written"],
        "expected_shards": len(original["shards"]),
        "source_dir": str(sf_root),
        "source_derive_summary_sha256": args.expected_source_summary_sha256,
        "mutated_arrays": ["policy_target"],
    }
    require(
        all(mix.get(k) == v for k, v in expected.items()),
        "requires B100 global T.5 recipe",
    )
    require(
        [p.name for p in sorted(root.glob("shard_*.zarr"))]
        == [x["path"] for x in original["shards"]],
        "B100 shard inventory differs",
    )
    return root, pins


def source_contract(summary: dict[str, Any], record: derive.CorpusRecord) -> None:
    rows = summary["realized"]["rows_written"]
    require(
        record.corpus_complete and record.facts["row_schema"] == 3,
        "requires closed original schema3 corpus",
    )
    require(
        summary["realized"]["rows_read"] == summary["limit_requested"]
        and type(summary["limit_requested"]) is int
        and summary["limit_requested"] > 0
        and type(summary["rows_per_shard"]) is int
        and summary["rows_per_shard"] > 0,
        "invalid raw prefix/shard bounds",
    )
    require(
        summary["corpus"]["dir"] == record.shards[0].parent.name,
        "raw source directory differs",
    )
    require(
        summary.get("policy_target_postprocess") is None,
        "requires unmodified original SF policy",
    )
    require(
        summary["scheme"]["canonical"] == "uniform-d9"
        and summary["scheme"]["kind"] == "uniform"
        and summary["scheme"]["depth"] == 9
        and summary["scheme"].get("value_depth") is None
        and summary["scheme"]["value_source"] == "deepest_phase_covering",
        "source policy/value scheme differs",
    )
    require(
        summary["temp_requested"] == 0.0005 and summary["floor_requested"] == 0.0,
        "source q temperature/floor differs",
    )
    require(
        summary["cp_map"]["cp_slope"] == 0.006
        and summary["cp_map"]["cp_draw_width"] == 120.0,
        "source cp map differs",
    )
    require(summary["value_scheme"]["name"] == "search", "source value scheme differs")
    require(
        summary["input"]["input_history_encoding"] == "lc0_root_legacy_meta"
        and summary["input"]["history_rep_fix"] is True
        and summary["input"]["input_extra_features"] == "v2_threats"
        and summary["input"]["zero_history"] is False,
        "source history regime differs",
    )
    require(
        summary["realized"]["input_key_verified"] == rows
        and summary["realized"]["support_checks"] == rows
        and summary["realized"]["phases_per_row"] == {"1": rows},
        "missing original full-row identity/support proof",
    )
    require(
        summary["realized"].get("rows_dropped_envelope", 0) == 0
        and summary["realized"].get("rows_dropped_policy_support", 0) == 0,
        "unsupported source omissions",
    )
    require(
        summary.get("source_selection") is None and not summary.get("row_provenance"),
        "only legacy ordered source is supported",
    )
    require(
        summary["seed_effect"]
        == "permutes rows WITHIN each shard; changes no target value",
        "unsupported shuffle contract",
    )
    require(
        summary["corpus"]["config_sha256"] == record.facts["config_sha256"],
        "source/raw configuration differs",
    )
    require(
        record.facts["staircase_parsed"] == [{"depth": 9, "width": "all"}],
        "raw staircase differs",
    )
    require(
        summary["corpus"]["staircase_parsed"] == record.facts["staircase_parsed"]
        and summary["corpus"]["run_id"] == record.facts["run_id"]
        and summary["corpus"]["corpus_complete"] is True
        and summary["corpus"]["corpus_record_detail"]["shards_adopted"]
        == len(record.shards)
        and summary["corpus"]["corpus_record_detail"]["rows_claimed_by_inventory"]
        == record.rows_claimed,
        "source/raw inventory lineage differs",
    )


def shard_contract(attrs: dict[str, Any], summary: dict[str, Any], rows: int) -> None:
    expected = {
        "derive_state": "committed",
        "derive_run_finalized": True,
        "derive_corpus_config_sha256": summary["corpus"]["config_sha256"],
        "derive_scheme": "uniform-d9",
        "derive_temp": 0.0005,
        "derive_floor": 0.0,
        "derive_cp_slope": 0.006,
        "derive_cp_draw_width": 120.0,
        "derive_value_scheme": "search",
        "derive_value_source": "deepest_phase_covering",
        "derive_corpus_row_schema": 3,
        "derive_history_rep_fix": True,
        "history_rep_fix": True,
        "zero_history": False,
        "input_history_encoding": "lc0_root_legacy_meta",
        "policy_encoding": "lc0_1858",
        "policy_size": COMPACT_POLICY_SIZE,
        "positions": rows,
    }
    require(
        all(attrs.get(k) == v for k, v in expected.items()),
        "source shard identity differs",
    )
    require(
        not any(k.startswith(("policy_target_", "bt4_")) for k in attrs),
        "source shard already postprocessed",
    )


def copy_shard(source: Path, destination: Path) -> dict[str, str]:
    """Copy regular files; verify compressed non-policy bytes without decoding x."""
    nonpolicy: dict[str, str] = {}

    def copy_file(src: str, dst: str) -> str:
        path = Path(src)
        require(not path.is_symlink() and path.is_file(), "nonregular source storage")
        before = rank._file_identity(path)
        shutil.copy2(src, dst)
        if path.relative_to(source).parts[0] != "policy_target":
            digest = file_sha256(path)
            require(file_sha256(Path(dst)) == digest, "copied bytes differ")
            nonpolicy[str(path.relative_to(source))] = digest
        require(rank._file_identity(path) == before, "source changed during copy")
        return dst

    shutil.copytree(source, destination, copy_function=copy_file)
    return nonpolicy


def rewrite(args: argparse.Namespace) -> dict[str, Any]:
    downside = getattr(args, "tactical_recipe", "legacy") == "allmove-downside300"
    pilot_shards = getattr(args, "pilot_shards", None)
    pilot_cap = getattr(args, "pilot_max_raw_rows", None)
    require(
        (pilot_shards is None) == (pilot_cap is None),
        "pilot requires shard and raw-row caps",
    )
    pilot = pilot_shards is not None
    require(
        not pilot
        or (
            downside
            and type(pilot_shards) is int
            and pilot_shards > 0
            and type(pilot_cap) is int
            and pilot_cap > 0
        ),
        "invalid downside pilot limits",
    )
    temperature = derive.validate_temp(float(args.temperature))
    require(
        math.isfinite(args.minimum_free_gib) and args.minimum_free_gib >= 0,
        "invalid disk reserve",
    )
    raw_dir = Path(args.raw).resolve()
    source = Path(args.source).resolve()
    out = Path(args.out).resolve()
    writing = out.with_name(out.name + ".writing")
    require(
        all(
            out != root and root not in out.parents and out not in root.parents
            for root in (source, raw_dir)
        ),
        "output overlaps source",
    )
    require(
        not out.exists() and not writing.exists(), "output or partial already exists"
    )
    source_summary_path = source / derive.SUMMARY_NAME
    summary_bytes = source_summary_path.read_bytes()
    require(
        hashlib.sha256(summary_bytes).hexdigest()
        == args.expected_source_summary_sha256,
        "source summary pin differs",
    )
    summary = json.loads(summary_bytes)
    # Completed legacy corpora can predate launch manifests. Their own summary
    # remains mandatory; a manifest, when present, is still independently bound.
    manifest_path = raw_dir / "manifest.json"
    manifest_present = os.path.lexists(manifest_path)
    metadata = {raw_dir / "summary.json": file_sha256(raw_dir / "summary.json")}
    if manifest_present:
        metadata[manifest_path] = file_sha256(manifest_path)
    metadata[source_summary_path] = args.expected_source_summary_sha256
    record = derive.read_corpus_record(raw_dir)
    source_contract(summary, record)
    if manifest_present:
        manifest = derive.corpus.read_launch_manifest(raw_dir)
        require(
            all(
                manifest[k] == record.facts[k]
                for k in ("config_sha256", "row_schema", "staircase_parsed")
            ),
            "raw manifest/summary identity differs",
        )
    tactical = tactical_source(args, summary, source)
    require(
        not downside or tactical is not None,
        "downside recipe requires pinned B100 source",
    )
    parent = tactical[0] if tactical else source
    if tactical:
        metadata.update(tactical[1])
        require(
            out != parent and parent not in out.parents and out not in parent.parents,
            "output overlaps B100 source",
        )
    full_specs = summary["shards"]
    require(not pilot or pilot_shards <= len(full_specs), "pilot exceeds source shards")
    specs = full_specs[:pilot_shards] if pilot else full_specs
    require(
        [p.name for p in sorted(source.glob("shard_*.zarr"))]
        == [s["path"] for s in full_specs],
        "source shard membership differs",
    )
    source_states = {
        source / s["path"]: rank._storage_identity(source / s["path"]) for s in specs
    }
    if tactical:
        source_states.update(
            {
                parent / x["path"]: rank._storage_identity(parent / x["path"])
                for x in specs
            }
        )
    producer_hashes = {
        str(p): file_sha256(p)
        for p in (Path(__file__), Path(derive.__file__), Path(rank.__file__))
    }
    if tactical:
        from chess_anti_engine.stockfish import wdl

        for module in (wdl,):
            assert module.__file__ is not None
            producer_hashes[str(Path(module.__file__))] = file_sha256(
                Path(module.__file__)
            )
    writing.mkdir(parents=True)
    start = time.monotonic()
    raw_rows = dropped = rows_written = 0
    rng = np.random.Generator(np.random.PCG64(int(summary["seed"])))
    pending: list[Observation] = []
    outputs: list[dict[str, Any]] = []
    output_states: dict[Path, str] = {}
    raw_proofs = {}
    last_by_worker: dict[int, tuple[int, int]] = {}
    changed = 0
    max_mass_error = 0.0
    tactical_counts: dict[str, int] = {}
    support_losses = 0
    winning_mate_zero_base_mass = 0
    storage_relative_error = 0.0
    storage_tv = 0.0
    keys = hashlib.sha256()

    def guard() -> None:
        require(
            not (writing / "STOP").exists() and not (out.parent / "STOP").exists(),
            "STOP requested",
        )
        require(
            shutil.disk_usage(writing).free >= args.minimum_free_gib * 1024**3,
            "free disk reserve breached",
        )

    def flush() -> None:
        nonlocal rows_written, changed, max_mass_error, support_losses
        nonlocal storage_relative_error, storage_tv, winning_mate_zero_base_mass
        guard()
        index = len(outputs)
        require(index < len(specs), "too many source rows")
        spec = specs[index]
        src = source / spec["path"]
        dst = writing / spec["path"]
        require(len(pending) == spec["rows"], "source shard row count differs")
        order = rng.permutation(len(pending))
        aligned = [pending[int(i)] for i in order]
        g: Any = zarr.open_group(str(src), mode="r")
        require(frozenset(g.array_keys()) == ARRAYS, "expected original17-array source")
        shard_contract(dict(g.attrs), summary, len(aligned))
        require(
            np.array_equal(g["game_id"][:], [r.game for r in aligned])
            and np.array_equal(g["ply_index"][:], [r.ply for r in aligned]),
            "raw/source shuffled identity differs",
        )
        old_policy = np.asarray(g["policy_target"][:])
        legal = np.asarray(g["legal_mask"][:])
        require(
            old_policy.dtype == np.float16 and legal.shape == old_policy.shape,
            "stored source policy schema differs",
        )
        base_policy = old_policy
        copy_source = src
        if tactical:
            copy_source = parent / spec["path"]
            bg: Any = zarr.open_group(str(copy_source), mode="r")
            require(
                frozenset(bg.array_keys()) == ARRAYS, "B100 array inventory differs"
            )
            attrs = dict(bg.attrs)
            require(
                {
                    k: v
                    for k, v in attrs.items()
                    if not k.startswith("policy_target_mix_")
                }
                == dict(g.attrs),
                "B100 source attrs differ",
            )
            require(
                attrs.get("policy_target_mix_kind") == "global"
                and attrs.get("policy_target_mix_alpha") == 1.0
                and attrs.get("policy_target_mix_bt4_temperature") == 0.5,
                "B100 source recipe attrs differ",
            )
            for column in ARRAYS:
                for array in (g[column], bg[column]):
                    # A Zarr read silently fills absent chunks; verify every
                    # expected stored chunk before accepting either parent.
                    for coordinates in itertools.product(
                        *(
                            range(math.ceil(n / c))
                            for n, c in zip(array.shape, array.chunks, strict=True)
                        )
                    ):
                        require(
                            array._chunk_key(coordinates) in array.chunk_store,
                            "missing stored chunk",
                        )
                require(
                    bg[column].shape == g[column].shape
                    and bg[column].dtype == g[column].dtype,
                    "B100 array schema differs",
                )
            base_policy = np.asarray(bg["policy_target"][:])
        new_policy = np.empty_like(old_policy)
        for i, obs in enumerate(aligned):
            mask = np.zeros(COMPACT_POLICY_SIZE, dtype=np.uint8)
            mask[obs.indices] = 1
            require(np.array_equal(mask, legal[i]), "stored/raw legal support differs")
            require(
                np.array_equal(target(obs, "q", 0.0005), old_policy[i]),
                "original q-policy reconstruction differs",
            )
            if tactical:
                new_policy[i], diagnostic = tactical_target(
                    obs, base_policy[i], downside=downside
                )
                category = diagnostic["category"]
                tactical_counts[category] = tactical_counts.get(category, 0) + 1
                support_losses += diagnostic["support_losses"]
                winning_mate_zero_base_mass += diagnostic["winning_mate_zero_base_mass"]
                storage_relative_error = max(
                    storage_relative_error,
                    diagnostic["ideal_to_stored_relative_error_max"],
                )
                storage_tv = max(storage_tv, diagnostic["ideal_to_stored_TV"])
            else:
                new_policy[i] = target(obs, args.score_space, temperature)
            keys.update(bytes.fromhex(obs.input_key))
        require(
            rank._storage_identity(src) == source_states[src],
            "source changed before copy",
        )
        require(
            rank._storage_identity(copy_source) == source_states[copy_source],
            "B100 source changed before copy",
        )
        copied = copy_shard(copy_source, dst)
        if tactical:
            # Ordinary B100 copies, independently bound to the original SF
            # nonpolicy compressed files. No x/history payload decoding.
            sf_files = {
                str(p.relative_to(src)): p
                for p in src.rglob("*")
                if p.is_file()
                and p.relative_to(src).parts[0] != "policy_target"
                and str(p.relative_to(src)) != ".zattrs"
            }
            require(
                set(sf_files) == set(copied) - {".zattrs"},
                "B100 nonpolicy file inventory differs",
            )
            require(
                all(file_sha256(path) == copied[rel] for rel, path in sf_files.items()),
                "B100 changed nonpolicy bytes",
            )
        dest: Any = zarr.open_group(str(dst), mode="a")
        dest["policy_target"][:] = new_policy
        require(
            np.array_equal(dest["policy_target"][:], new_policy),
            "policy readback differs",
        )
        recipe = {
            "score_space": args.score_space,
            "temperature": temperature,
            "temperature_units": "centipawns"
            if args.score_space == "effective-cp"
            else "q",
            "source_summary_sha256": args.expected_source_summary_sha256,
            "storage": "float64 softmax -> float32 -> float16",
            "mutated_arrays": ["policy_target"],
        }
        if tactical:
            for name in list(dest.attrs):
                if name.startswith("policy_target_mix_"):
                    del dest.attrs[name]
            recipe = {
                "algorithm": TACTICAL_ALGORITHM,
                "gap_cp": 100.0,
                "decay_cp": 100.0,
                "relative_floor": 0.1,
                "mate_handling": "categorical-v1",
                "base": "normalized stored B100 float16 policy",
                "source_summary_sha256": args.expected_bt4_summary_sha256,
                "sf_summary_sha256": args.expected_source_summary_sha256,
                "storage": "float64 attenuation -> float32 -> float16; all-one weights preserve bytes",
                "mutated_arrays": ["policy_target"],
            }
        if downside:
            recipe = {
                **recipe_for_summary(downside=True),
                "algorithm": DOWNSIDE_ALGORITHM,
                "source_summary_sha256": args.expected_bt4_summary_sha256,
                "sf_summary_sha256": args.expected_source_summary_sha256,
                "mutated_arrays": ["policy_target"],
            }
        if pilot:
            recipe["pilot_only"] = True
        dest.attrs["policy_target_rewrite"] = recipe
        for rel, digest in copied.items():
            if rel != ".zattrs":
                require(file_sha256(dst / rel) == digest, "nonpolicy copy changed")
        changed += int(np.count_nonzero(np.any(new_policy != base_policy, axis=1)))
        max_mass_error = max(
            max_mass_error,
            float(np.max(np.abs(new_policy.astype(np.float64).sum(axis=1) - 1))),
        )
        rows_written += len(aligned)
        outputs.append(
            {
                "path": spec["path"],
                "rows": len(aligned),
                "source_storage_identity": source_states[copy_source],
                **(
                    {
                        "original_sf_storage_identity": source_states[src],
                        "source_policy_sha256": rank._sha_arrays(base_policy),
                    }
                    if tactical
                    else {}
                ),
                "copied_file_hashes": copied,
                "policy_sha256": rank._sha_arrays(new_policy),
            }
        )
        if tactical:
            output_states[dst] = rank._storage_identity(dst)
        pending.clear()

    try:
        for path in record.shards:
            if raw_rows >= summary["limit_requested"] or (
                pilot and len(outputs) == len(specs)
            ):
                break
            guard()
            require(not path.is_symlink() and path.is_file(), "nonregular raw storage")
            require(
                not pilot or path.stat().st_size <= 64 * 1024**2,
                "pilot raw file exceeds 64 MiB hash bound",
            )
            before = rank._file_identity(path)
            read = 0
            raw_iterator = iter(derive.iter_corpus_rows(path))
            while raw_rows < summary["limit_requested"] and not (
                pilot and len(outputs) == len(specs)
            ):
                require(
                    not pilot or (pilot_cap is not None and raw_rows < pilot_cap),
                    "pilot raw-row cap exhausted",
                )
                try:
                    raw = next(raw_iterator)
                except StopIteration:
                    break
                raw_rows += 1
                read += 1
                derive._check_row_identity(raw, str(record.facts["config_sha256"]))
                require(
                    all(
                        type(raw[k]) is int and raw[k] >= 0
                        for k in ("worker_id", "game_id", "ply")
                    ),
                    "invalid raw physical identity",
                )
                worker = int(raw["worker_id"])
                key = (int(raw["game_id"]), int(raw["ply"]))
                require(
                    worker not in last_by_worker or key > last_by_worker[worker],
                    "duplicate/nonmonotone original worker identity",
                )
                last_by_worker[worker] = key
                if raw.get("result") is None:
                    dropped += 1
                    continue
                if tactical:
                    anomalies = raw["phases"][0].get("anomalies", {})
                    require(
                        type(anomalies.get("bound_lines")) is int
                        and anomalies["bound_lines"] >= 0,
                        "missing original bound-line accounting",
                    )
                    require(
                        all(
                            line[3] is None or (type(line[3]) is int and line[3] >= 0)
                            for line in rank.d9_lines(raw)
                            if len(line) == 4
                        ),
                        "invalid d9 nodes field",
                    )
                pending.append(observation(raw, str(record.facts["config_sha256"])))
                if len(pending) == summary["rows_per_shard"]:
                    flush()
            raw_proofs[str(path)] = {
                "identity": before,
                "sha256": file_sha256(path),
                "rows_consumed": read,
            }
            require(rank._file_identity(path) == before, "raw file changed during read")
        if pending:
            flush()
        require(
            (
                pilot
                and rows_written == sum(spec["rows"] for spec in specs)
                and len(outputs) == len(specs)
                and raw_rows == rows_written + dropped
            )
            or (
                not pilot
                and raw_rows == summary["limit_requested"]
                and rows_written == summary["realized"]["rows_written"]
                and dropped == summary["realized"]["rows_dropped_no_result"]
                and len(outputs) == len(specs)
            ),
            "final source complement differs",
        )
        guard()
        if tactical:
            for root in (source, parent):
                require(
                    [p.name for p in sorted(root.glob("shard_*.zarr"))]
                    == [s["path"] for s in full_specs],
                    "final shard inventory differs",
                )
        for path, state in source_states.items():
            require(
                rank._storage_identity(path) == state,
                "source changed before publication",
            )
        for path, proof in raw_proofs.items():
            require(
                rank._file_identity(Path(path)) == proof["identity"],
                "raw source changed before publication",
            )
        require(
            os.path.lexists(manifest_path) == manifest_present,
            "raw manifest presence changed",
        )
        for path, digest in metadata.items():
            require(file_sha256(path) == digest, "source metadata changed")
        result = {
            "schema": 1,
            "status": "COMPLETE",
            "kind": "sf_policy_score_rewrite",
            "score_space": args.score_space,
            "temperature": temperature,
            "source_dir": str(source),
            "raw_dir": str(raw_dir),
            "raw_limit": raw_rows,
            "rows": rows_written,
            "shards": len(outputs),
            "rows_dropped_no_result": dropped,
            "changed_rows": changed,
            "stored_mass_error_max": max_mass_error,
            "mutated_arrays": ["policy_target"],
            "nonpolicy_arrays_copied": 16,
            "producer_sha256": producer_hashes,
            "source_derive_summary_sha256": args.expected_source_summary_sha256,
            "raw_manifest_present": manifest_present,
            "metadata_sha256": {str(p): h for p, h in metadata.items()},
            "raw_files": raw_proofs,
            "raw_history_keys_sha256_in_emitted_order": keys.hexdigest(),
            "outputs": outputs,
            "elapsed_seconds": time.monotonic() - start,
            "history_lineage": "Inherited original input_key_verified/source x; no fresh history re-encoding.",
            "limitations": "Original historical control/provenance limitations unchanged; no valid-control promotion, inference, training or strength claim.",
        }
        if tactical:
            result.update(
                kind="bt4_sf_tactical_policy_attenuation",
                algorithm=TACTICAL_ALGORITHM,
                source_dir=str(parent),
                sf_source_dir=str(source),
                source_derive_summary_sha256=args.expected_bt4_summary_sha256,
                source_policy_summary_sha256=args.expected_bt4_mix_sha256,
                sf_derive_summary_sha256=args.expected_source_summary_sha256,
                recipe=recipe_for_summary(),
                categories=tactical_counts,
                stored_support_losses=support_losses,
                winning_mate_zero_base_mass_rows=winning_mate_zero_base_mass,
                stored_relative_error_max=storage_relative_error,
                stored_TV_error_max=storage_tv,
                score_bounds="Original producer discards UCI upper/lower bounds; rows retain aggregate counts, not per-line flags.",
            )
            result.pop("score_space")
            result.pop("temperature")
        if downside:
            result.update(
                kind="bt4_sf_allmove_downside",
                algorithm=DOWNSIDE_ALGORITHM,
                recipe=recipe_for_summary(downside=True),
            )
        if pilot:
            result.update(
                status="PILOT_COMPLETE_NOT_TRAINING",
                pilot_only=True,
                pilot_max_raw_rows=pilot_cap,
                full_source_rows=summary["realized"]["rows_written"],
            )
        rank._atomic_json(
            writing
            / (
                DOWNSIDE_SUMMARY
                if downside
                else TACTICAL_SUMMARY
                if tactical
                else SUMMARY
            ),
            result,
        )
        derived = dict(summary)
        derived["policy_target_postprocess"] = {
            k: v for k, v in result.items() if k != "outputs"
        }
        if pilot:
            # Do not publish an apparently complete derived corpus for a prefix.
            rank._atomic_json(
                writing / "pilot_source_binding.json",
                {
                    "source_summary_sha256": args.expected_source_summary_sha256,
                    "selected_shards": specs,
                    "postprocess": derived["policy_target_postprocess"],
                },
            )
        else:
            rank._atomic_json(writing / derive.SUMMARY_NAME, derived)
        guard()
        for path, state in output_states.items():
            require(
                rank._storage_identity(path) == state,
                "tactical output changed before publication",
            )
        os.replace(writing, out)
        return result
    except BaseException as exc:
        rank._atomic_json(
            writing / "failed.json",
            {
                "error": repr(exc),
                "partial_output_preserved": True,
                "rows_written": rows_written,
                "raw_rows_read": raw_rows,
            },
        )
        raise


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["raw", "source", "out", "expected-source-summary-sha256"]:
        p.add_argument("--" + name, required=True)
    p.add_argument("--score-space", choices=["q", "effective-cp"], default="q")
    p.add_argument("--temperature", type=float, default=0.0005)
    p.add_argument("--minimum-free-gib", type=float, default=150.0)
    p.add_argument(
        "--tactical-bt4-source",
        help="fixed gap100/decay100/floor.1 categorical-mate recipe on B100",
    )
    p.add_argument(
        "--tactical-recipe", choices=["legacy", "allmove-downside300"], default="legacy"
    )
    p.add_argument(
        "--pilot-shards",
        type=int,
        help="Downside-only prefix; no trainable corpus summary",
    )
    p.add_argument(
        "--pilot-max-raw-rows", type=int, help="Hard cap on raw rows joined for pilot"
    )
    p.add_argument("--expected-bt4-summary-sha256")
    p.add_argument("--expected-bt4-mix-sha256")
    return p


def main(argv: list[str] | None = None) -> int:
    rewrite(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
