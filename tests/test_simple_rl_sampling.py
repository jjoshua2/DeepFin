"""Opening cycling across generations and the constant-memory cross-generation
duplicate estimator."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from chess_anti_engine.simple_rl.collect import cycle_openings, sample_openings
from chess_anti_engine.simple_rl.dup_sketch import DupSketch

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def _book(tmp_path: Path, sizes: dict[int, int]) -> Path:
    path = tmp_path / "train.jsonl"
    rows = [{"id": f"b{b}l{i}", "book": b, "line": f"{START} | x{b}_{i}"} for b, n in sizes.items() for i in range(n)]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def test_cycling_reuses_no_line_before_its_book_is_exhausted(tmp_path: Path) -> None:
    sizes = {1: 23, 2: 7}
    path = _book(tmp_path, sizes)
    cursor = None
    used: dict[int, list[str]] = {1: [], 2: []}
    passes = []
    for g in range(1, 9):
        picked, cursor, log = cycle_openings(path, games=5, book_mix=(0.7, 0.3), base_seed=3, generation=g,
                                             start=cursor)
        for row in picked:
            used[int(row["book"])].append(row["id"])
        passes.append({b: c["pass"] for b, c in cursor["books"].items()})
        for line in log:
            assert "exhausted" in line
    for b, n in sizes.items():
        seq = used[b]
        for k in range(0, len(seq), n):  # every full pass is a permutation of the book
            chunk = seq[k:k + n]
            assert len(set(chunk)) == len(chunk), (b, k)
        if len(seq) >= 2 * n:
            assert seq[:n] != seq[n:2 * n]  # a new permutation per pass
        expected_pass = (len(seq) - 1) // n + 1
        assert passes[-1][str(b)] == expected_pass
    # the cursor is deterministic: replaying generation 5 from generation 4's cursor gives the same lines
    cur = None
    cursors = []
    for g in range(1, 6):
        picked, cur, _ = cycle_openings(path, games=5, book_mix=(0.7, 0.3), base_seed=3, generation=g, start=cur)
        cursors.append((cur, [r["id"] for r in picked]))
    again, after, _ = cycle_openings(path, games=5, book_mix=(0.7, 0.3), base_seed=3, generation=5,
                                     start=cursors[3][0])
    assert [r["id"] for r in again] == cursors[4][1]
    assert after == cursors[4][0]


def test_independent_redraws_reuse_lines_across_generations(tmp_path: Path) -> None:
    path = _book(tmp_path, {1: 40})
    a = {r["id"] for r in sample_openings(path, games=20, book_mix=(1.0,), seed=1)}
    b = {r["id"] for r in sample_openings(path, games=20, book_mix=(1.0,), seed=2)}
    assert a & b  # the non-default mode is what it says


def test_a_changed_book_is_refused_under_a_carried_cursor(tmp_path: Path) -> None:
    path = _book(tmp_path, {1: 10})
    _, cursor, _ = cycle_openings(path, games=3, book_mix=(1.0,), base_seed=0, generation=1, start=None)
    path.write_text(path.read_text().replace("x1_0", "x1_other"))  # same size, different lines
    with pytest.raises(ValueError, match="TRAIN openings changed"):
        cycle_openings(path, games=3, book_mix=(1.0,), base_seed=0, generation=2, start=cursor)
    _book(tmp_path, {1: 11})
    with pytest.raises(ValueError, match="changed"):
        cycle_openings(path, games=3, book_mix=(1.0,), base_seed=0, generation=2, start=cursor)


def _stream(rng: np.random.Generator, generations: int, rows: int, dup_share: float) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for _g in range(generations):
        fresh = rng.integers(0, 2**63, size=rows, dtype=np.uint64) * np.uint64(2) + rng.integers(0, 2, size=rows,
                                                                                                 dtype=np.uint64)
        if out:
            pool = np.concatenate(out)
            k = int(dup_share * rows)
            fresh[:k] = rng.choice(pool, size=k)
        out.append(fresh)
    return out


def test_the_estimator_matches_the_exact_rate_within_noise_and_respects_its_budget() -> None:
    gens = _stream(np.random.default_rng(0), 6, 20_000, 0.3)
    big, small = DupSketch(budget=10**7), DupSketch(budget=4_000)
    exact_seen: set[int] = set()
    for g, h in enumerate(gens):
        exact = float(np.mean([int(x) in exact_seen for x in h]))
        e_big, e_small = big.estimate(h), small.estimate(h)
        if g:
            assert e_big["rate"] == pytest.approx(exact)  # no sampling while under budget
            assert e_small["sampled_rows"] < len(h)  # the threshold did fall
            assert abs(e_small["rate"] - exact) < 4 * e_small["se"] + 1e-9, (g, e_small, exact)
        big.add(h)
        small.add(h)
        exact_seen.update(int(x) for x in h)
        assert small.hashes.size <= 4_000
        # the sketch holds exactly the earlier hashes below the threshold
        want = np.unique(np.concatenate(gens[:g + 1]))
        np.testing.assert_array_equal(small.hashes, want[want < small.threshold])


def test_the_sketch_round_trips_and_buckets_by_ply(tmp_path: Path) -> None:
    gens = _stream(np.random.default_rng(1), 2, 1000, 0.5)
    sk = DupSketch(budget=600)
    sk.add(gens[0])
    sk.save(tmp_path / "s.npz")
    back = DupSketch.load(tmp_path / "s.npz", budget=600)
    ply = np.arange(1000) % 50
    a, b = sk.estimate(gens[1], ply), back.estimate(gens[1], ply)
    assert a == b
    assert sum(v["sampled_rows"] for v in a["by_ply_after_book"].values()) == a["sampled_rows"]
    assert set(a["by_ply_after_book"]) == {"0-3", "4-9", "10-19", "20-39", "40+"}
