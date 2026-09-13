"""CPU preparation of the completed Combined35M V50/SF100 checkpoint pair; no GPU."""

import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import numpy as np
import torch
from scripts import arena_standard as arena
from chess_anti_engine.uci.model_loader import load_model_from_checkpoint


def main():
    request = json.loads(Path(sys.argv[1]).read_text())
    root = Path(request["runtime_root"])
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    assert Path(arena.__file__).resolve() == root / "scripts/arena_standard.py"
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    assert set(request["packages"]) == {"Combined35M_V50", "Combined35M_SF100"}
    assert request["cells"] == [
        {
            "name": "combined_value",
            "candidate": "Combined35M_V50",
            "reference": "Combined35M_SF100",
            "priors": [1.0, 1.0],
        }
    ]
    seed = request["arena_seed"]
    assert type(seed) is int
    assert seed == 20260913
    boards = arena.load_paired_openings(
        Path(request["book"]["path"]),
        n_pairs=256,
        max_plies=16,
        rng=np.random.default_rng(seed),
    )
    panel = [
        {
            "root_fen": b.root().fen(),
            "moves": [m.uci() for m in b.move_stack],
            "fen": b.fen(),
        }
        for b in boards
    ]
    assert panel == json.loads(Path(request["panel"]["path"]).read_text())
    models = {}
    signatures = []
    for name, ref in request["packages"].items():
        model = load_model_from_checkpoint(ref["path"], device="cpu")
        shapes = [(k, list(v.shape)) for k, v in model.state_dict().items()]
        signature = hashlib.sha256(
            json.dumps(shapes, separators=(",", ":")).encode()
        ).hexdigest()
        signatures.append(signature)
        count = sum(p.numel() for p in model.parameters())
        assert count == 61444448
        models[name] = {
            "dynamic_relations": bool(getattr(model, "use_dynamic_relations", False)),
            "parameters": count,
            "architecture_sha256": signature,
        }
        del model
    assert len(set(signatures)) == 1
    cells = {}
    for cell in request["cells"]:
        candidate = request["packages"][cell["candidate"]]
        reference = request["packages"][cell["reference"]]
        sides = [
            arena.apply_search_overrides(
                arena.resolve_search_shape("training"), spec="policy_temp=" + str(t)
            )
            for t in cell["priors"]
        ]
        leaf = arena.arena_uncapped_leaf_rows(
            max_concurrent_games=128,
            sides=tuple(sides),
            relations=tuple(
                models[k]["dynamic_relations"]
                for k in [cell["candidate"], cell["reference"]]
            ),
        )
        assert leaf <= 4096
        settings = arena.arena_game_log_settings(
            mode="matched_sims",
            candidate=candidate["path"],
            reference=reference["path"],
            games=512,
            seed=seed,
            openings_path=request["book"]["path"],
            openings_kind="book",
            opening_plies=16,
            sims_candidate=400,
            sims_reference=400,
            ms_per_move=None,
            max_plies=300,
            temperature=0.1,
            gumbel_add_noise=True,
            search_candidate=sides[0],
            search_reference=sides[1],
            volatility_candidate=None,
            uci_args="",
            syzygy_path=None,
            tb_max_pieces=0,
        )
        cells[cell["name"]] = {"settings": settings, "uncapped_leaf_rows": leaf}
    rt = {
        "python": sys.version,
        "executable": sys.executable,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda": torch.version.cuda,
        "native_extensions": {
            name: importlib.import_module(name).__file__
            for name in request["runtime"]["native_extensions"]
        },
    }
    assert not torch.cuda.is_initialized()
    out = {
        "status": "PASS_ACTUAL_COMBINED_VALUE_PAIR_CPU_PREPARATION",
        "runtime": rt,
        "models": models,
        "panel": panel,
        "cells": cells,
        "cuda_initialized": False,
    }
    with Path(sys.argv[2]).open("x") as f:
        json.dump(out, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
