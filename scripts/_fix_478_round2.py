#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
READOUT = ROOT / "scripts/nnue_gumbel_readout.py"
GEN = ROOT / "scripts/gen_random_selfplay_shards.py"
TEST = ROOT / "tests/test_nnue_gumbel_readout.py"


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one match, got {count}: {old[:120]!r}")
    path.write_text(text.replace(old, new, 1))


# 1) A missing strengthened-oracle component must fail, never hash as empty.
replace_once(
    READOUT,
    '''def searches_digest(records: list[GameRecord]) -> str:
    """One digest over each game's improved-policy/search outputs."""
    h = hashlib.sha256()
    for record in sorted(records, key=lambda r: r.game):
        h.update(f"{record.game}:{record.search_digest}\\n".encode())
    return h.hexdigest()
''',
    '''def searches_digest(records: list[GameRecord]) -> str:
    """One digest over each game's improved-policy/search outputs.

    An empty component is not a value: it means the strengthened oracle was
    never populated. Refuse it so a future constructor that forgets the field
    cannot make two unverified cells agree on the same empty digest.
    """
    h = hashlib.sha256()
    for record in sorted(records, key=lambda r: r.game):
        if not record.search_digest:
            raise ValueError(
                f"game {record.game} has no search_digest; search-output parity "
                "cannot be verified",
            )
        h.update(f"{record.game}:{record.search_digest}\\n".encode())
    return h.hexdigest()
''',
)

# 2) Matrix-level validation owns cross-arm knob legitimacy. Per-cell qsearch
# must scope the DAG-only cap away instead of re-rejecting it.
replace_once(
    READOUT,
    '''    if spec.consumes_qsearch_knobs:
        dag_cap = args.dag_node_cap
        if selected == ARM_QSEARCH and dag_cap is not None:
            raise ValueError(
                "nnue-qsearch has no DAG and cannot consume --dag-node-cap",
            )
''',
    '''    if spec.consumes_qsearch_knobs:
        dag_cap = args.dag_node_cap if selected == ARM_QSEARCH_DAG else None
        if (
            selected == ARM_QSEARCH
            and args.dag_node_cap is not None
            and strict_foreign_knobs
        ):
            raise ValueError(
                "nnue-qsearch has no DAG and cannot consume --dag-node-cap",
            )
''',
)

# 3) Capture source provenance before any cell runs, then compare an end
# snapshot. The start snapshot describes what was loaded; a changed tree voids
# attribution even if the final state looks perfectly reproducible by itself.
replace_once(
    READOUT,
    '''    pack_sha = _sha256_file(plan.pack)
    started_utc = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
''',
    '''    pack_sha = _sha256_file(plan.pack)
    git_meta_start = _git_provenance()
    started_utc = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
''',
)
replace_once(
    READOUT,
    '''    wall_s = time.perf_counter() - started

    every = [c for runs in cells.values() for c in runs]
''',
    '''    wall_s = time.perf_counter() - started
    git_meta_end = _git_provenance()
    git_changed_during_run = git_meta_start != git_meta_end

    every = [c for runs in cells.values() for c in runs]
''',
)
replace_once(
    READOUT,
    '''    mcts_binary_paths = sorted({
        str(w["mcts_ext_path"]) for c in every for w in c["workers_detail"]
    })
    git_meta = _git_provenance()
    reasons = [
''',
    '''    mcts_binary_paths = sorted({
        str(w["mcts_ext_path"]) for c in every for w in c["workers_detail"]
    })
    reasons = [
''',
)
replace_once(
    READOUT,
    '''    if file_shas != [pack_sha]:
        reasons.append(
''',
    '''    if git_changed_during_run:
        reasons.append(
            "tracked source provenance changed while the matrix was running: "
            f"start={git_meta_start}, end={git_meta_end}; the report cannot "
            "attribute all cells to one source state",
        )
    if file_shas != [pack_sha]:
        reasons.append(
''',
)
replace_once(
    READOUT,
    '''            "mcts_ext_path": mcts_binary_paths[0] if len(mcts_binary_paths) == 1 else mcts_binary_paths,
            "mcts_ext_sha256": mcts_binary_shas[0] if len(mcts_binary_shas) == 1 else mcts_binary_shas,
            **git_meta,
            "seed": plan.seed,
''',
    '''            "mcts_ext_path": mcts_binary_paths[0] if len(mcts_binary_paths) == 1 else mcts_binary_paths,
            "mcts_ext_sha256": mcts_binary_shas[0] if len(mcts_binary_shas) == 1 else mcts_binary_shas,
            **git_meta_start,
            "git_end_head": git_meta_end["git_head"],
            "git_end_tracked_dirty": git_meta_end["git_tracked_dirty"],
            "git_changed_during_run": git_changed_during_run,
            "seed": plan.seed,
''',
)

# 4) The FEN-sufficiency predicate is intentionally NOT hist_len == 0.
replace_once(
    GEN,
    '''                        "hist_len": int(board.hist_len),
                        "fen_reconstructs_full_search_state": bool(board.hash_stack_len == 0),
''',
    '''                        "hist_len": int(board.hist_len),
                        # `hash_stack_len` is cleared only at game/root start or
                        # after a pawn move/capture. Across such a zeroing move
                        # the current piece/pawn state cannot equal an earlier
                        # position, so retained `hist_hash` entries cannot make
                        # the current state a repetition. FEN already carries
                        # the halfmove clock. Requiring hist_len == 0 here would
                        # therefore reject safe rows after every zeroing move.
                        "fen_reconstructs_full_search_state": bool(board.hash_stack_len == 0),
''',
)

# 5) Existing bank fake must model the CBoard fields the writer now consumes.
replace_once(
    TEST,
    '''class _FenOnly:
    """The only thing ``_bank_batch`` asks a board for."""

    def __init__(self, fen: str) -> None:
        self._fen = fen

    def fen(self) -> str:
        return self._fen
''',
    '''class _FenOnly:
    """Minimal bank-writer board, including schema-3 history metadata."""

    def __init__(self, fen: str) -> None:
        self._fen = fen
        fields = fen.split()
        self.halfmove_clock = int(fields[4]) if len(fields) > 4 else 0
        self.hash_stack_len = 0
        self.hist_len = 0

    def fen(self) -> str:
        return self._fen
''',
)

# 6) Pin the non-vacuous search digest and update the pre-schema-4 oracle test.
replace_once(
    TEST,
    '''def test_games_digest_is_order_independent_but_content_sensitive() -> None:
    a = readout.GameRecord(game=0, plies=1, result="1-0", termination="t", digest="aa")
    b = readout.GameRecord(game=1, plies=1, result="0-1", termination="t", digest="bb")
    assert readout.games_digest([a, b]) == readout.games_digest([b, a])
    c = readout.GameRecord(game=1, plies=1, result="0-1", termination="t", digest="cc")
    assert readout.games_digest([a, b]) != readout.games_digest([a, c])


def test_the_oracle_voids_the_decomposition_when_the_digests_differ() -> None:
    same = {
        readout.ARM_QSEARCH: [{"repeat": 0, "games_digest": "same"}],
        readout.ARM_QSEARCH_DAG: [{"repeat": 0, "games_digest": "same"}],
    }
    assert readout._oracle(same)["digests_agree"] is True
    differ = {
        readout.ARM_QSEARCH: [{"repeat": 0, "games_digest": "left"}],
        readout.ARM_QSEARCH_DAG: [{"repeat": 0, "games_digest": "right"}],
    }
    assert readout._oracle(differ)["digests_agree"] is False
    # One cell alone cannot claim the comparison it did not make.
    alone = {readout.ARM_FASTQ: [{"repeat": 0, "games_digest": "x"}]}
    assert readout._oracle(alone) == {
        "arms": [readout.ARM_QSEARCH, readout.ARM_QSEARCH_DAG],
        "available": False,
        "digests_agree": None,
    }
''',
    '''def test_games_digest_is_order_independent_but_content_sensitive() -> None:
    a = readout.GameRecord(game=0, plies=1, result="1-0", termination="t", digest="aa")
    b = readout.GameRecord(game=1, plies=1, result="0-1", termination="t", digest="bb")
    assert readout.games_digest([a, b]) == readout.games_digest([b, a])
    c = readout.GameRecord(game=1, plies=1, result="0-1", termination="t", digest="cc")
    assert readout.games_digest([a, b]) != readout.games_digest([a, c])


def test_searches_digest_refuses_an_unpopulated_component() -> None:
    missing = readout.GameRecord(
        game=0, plies=1, result="1-0", termination="t", digest="aa",
    )
    with pytest.raises(ValueError, match="no search_digest"):
        readout.searches_digest([missing])


def test_the_oracle_voids_the_decomposition_when_the_digests_differ() -> None:
    same = {
        readout.ARM_QSEARCH: [{
            "repeat": 0, "games_digest": "same", "searches_digest": "search-same",
        }],
        readout.ARM_QSEARCH_DAG: [{
            "repeat": 0, "games_digest": "same", "searches_digest": "search-same",
        }],
    }
    assert readout._oracle(same)["digests_agree"] is True
    differ = {
        readout.ARM_QSEARCH: [{
            "repeat": 0, "games_digest": "left", "searches_digest": "search-same",
        }],
        readout.ARM_QSEARCH_DAG: [{
            "repeat": 0, "games_digest": "right", "searches_digest": "search-same",
        }],
    }
    assert readout._oracle(differ)["digests_agree"] is False
    # One cell alone cannot claim the comparison it did not make.
    alone = {readout.ARM_FASTQ: [{
        "repeat": 0, "games_digest": "x", "searches_digest": "sx",
    }]}
    assert readout._oracle(alone) == {
        "arms": [readout.ARM_QSEARCH, readout.ARM_QSEARCH_DAG],
        "available": False,
        "digests_agree": None,
        "game_digests_agree": None,
        "search_digests_agree": None,
    }
''',
)

# 7) Existing fake cells must publish schema-4 provenance/search outputs.
replace_once(
    TEST,
    '''            "games_digest": f"digest-{cfg.repeat}",
            "inadmissible_reasons": [],
            "nice_realized": [0],
            "workers_detail": [{
                "kernel": "scalar", "pack_source_sha256": "a" * 64,
                "pack_file_sha256": readout._sha256_file(pack),
            }],
''',
    '''            "games_digest": f"digest-{cfg.repeat}",
            "searches_digest": f"search-{cfg.repeat}",
            "inadmissible_reasons": [],
            "nice_realized": [0],
            "workers_detail": [{
                "kernel": "scalar", "pack_source_sha256": "a" * 64,
                "pack_file_sha256": readout._sha256_file(pack),
                "nnue_ext_path": "/tmp/_nnue_ext.so", "nnue_ext_sha256": "b" * 64,
                "mcts_ext_path": "/tmp/_mcts_tree.so", "mcts_ext_sha256": "c" * 64,
            }],
''',
)
replace_once(
    TEST,
    '''            "games_digest": "same",
            "inadmissible_reasons": [],
            "nice_realized": [0],
            "workers_detail": [{
                "kernel": next(kernels), "pack_source_sha256": "a" * 64,
                "pack_file_sha256": readout._sha256_file(pack),
            }],
''',
    '''            "games_digest": "same",
            "searches_digest": "search-same",
            "inadmissible_reasons": [],
            "nice_realized": [0],
            "workers_detail": [{
                "kernel": next(kernels), "pack_source_sha256": "a" * 64,
                "pack_file_sha256": readout._sha256_file(pack),
                "nnue_ext_path": "/tmp/_nnue_ext.so", "nnue_ext_sha256": "b" * 64,
                "mcts_ext_path": "/tmp/_mcts_tree.so", "mcts_ext_sha256": "c" * 64,
            }],
''',
)

# 8) Lint the regex explicitly as regex.
replace_once(
    TEST,
    'with pytest.raises(ValueError, match="no.*nnue-fastq|nnue-fastq is not selected"):',
    'with pytest.raises(ValueError, match=r"no.*nnue-fastq|nnue-fastq is not selected"):',
)

# 9) Pin the provenance timing guard with a source-changing fake matrix.
append_marker = '''def test_quality_scope_explicitly_forbids_paired_attribution_from_end_to_end_cells() -> None:
'''
insert = '''def test_source_provenance_change_during_matrix_voids_the_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    ext = FakeExt()
    pack = _pack(tmp_path)
    snapshots = iter((
        {"git_head": "start", "git_tracked_dirty": False},
        {"git_head": "end", "git_tracked_dirty": False},
    ))
    monkeypatch.setattr(readout, "_git_provenance", lambda: next(snapshots))

    def fake_cell(cfg: readout.RunConfig) -> dict[str, Any]:
        return {
            "arm": cfg.arm_config.arm,
            "repeat": cfg.repeat,
            "games_digest": "same",
            "searches_digest": "search-same",
            "inadmissible_reasons": [],
            "nice_realized": [0],
            "workers_detail": [{
                "kernel": "scalar", "pack_source_sha256": "a" * 64,
                "pack_file_sha256": readout._sha256_file(pack),
                "nnue_ext_path": "/tmp/_nnue_ext.so", "nnue_ext_sha256": "b" * 64,
                "mcts_ext_path": "/tmp/_mcts_tree.so", "mcts_ext_sha256": "c" * 64,
            }],
        }

    monkeypatch.setattr(readout, "run_cell", fake_cell)
    plan = readout.ReadoutPlan(
        arm_configs=(readout.resolve_arm_config(_args(readout.ARM_QSEARCH), ext),),
        pack=pack, games=1, workers=1, seed=1, sims=8,
        topk=gen.MAX_LEGAL_MOVES, max_plies=10, all_root_moves=True,
        cp_per_internal_unit=0.28, cp_slope=0.006, cp_draw_width=120.0,
        bank_path=None, run_id="t", nice=0,
        dag_reset_every=readout.DAG_RESET_EVERY_GAME, repeats=1,
    )
    report = readout.run(plan)
    assert report["admissible"] is False
    assert any("source provenance changed" in r for r in report["inadmissible_reasons"])
    assert report["provenance"]["git_head"] == "start"
    assert report["provenance"]["git_end_head"] == "end"
    assert report["provenance"]["git_changed_during_run"] is True


'''
text = TEST.read_text()
if text.count(append_marker) != 1:
    raise RuntimeError("quality-scope insertion marker missing or ambiguous")
TEST.write_text(text.replace(append_marker, insert + append_marker, 1))

print("#478 round-2 guarded transforms applied")
