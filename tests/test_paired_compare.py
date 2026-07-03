from __future__ import annotations

import numpy as np

from scripts.paired_compare import paired_bootstrap_ci


def test_bootstrap_ci_covers_known_shift() -> None:
    rng = np.random.default_rng(7)
    deltas = rng.normal(loc=-2.0, scale=5.0, size=2000)
    lo, hi = paired_bootstrap_ci(deltas, n_boot=4000, seed=1)
    assert lo < -2.0 < hi or (lo < deltas.mean() < hi)
    # width ~ 2*1.96*5/sqrt(2000) ≈ 0.44
    assert 0.2 < (hi - lo) < 1.0
    assert hi < 0  # a real 2cp shift at n=2000 is clearly significant


def test_bootstrap_ci_null_is_not_significant() -> None:
    rng = np.random.default_rng(3)
    deltas = rng.normal(loc=0.0, scale=5.0, size=2000)
    lo, hi = paired_bootstrap_ci(deltas, n_boot=4000, seed=2)
    assert lo < 0 < hi


def test_load_dump_audit_shape(tmp_path) -> None:
    from scripts.paired_compare import load_dump

    p = tmp_path / "audit.jsonl"
    rows = [
        {"key": "k1", "phase": "middlegame",
         "cand": {"search": {"exp": 12.5, "top1": 30.0}, "net": {"exp": 20.0}}},
        {"key": "k2", "phase": 1,
         "cand": {"search": {"exp": 0.0, "top1": 0.0}}},
        {"key": "k3", "cand": {"search": {"exp": None}}},  # null metric -> dropped
        {"key": "k4", "cand": {}},                          # missing path -> dropped
    ]
    with p.open("w") as f:
        for r in rows:
            f.write(__import__("json").dumps(r) + "\n")

    d = load_dump(str(p), join_key="key", field="cand.search.exp")
    assert set(d) == {"k1", "k2"}
    assert d["k1"] == (12.5, "middlegame")
    assert d["k2"] == (0.0, "middlegame")  # int phase index mapped to name

    top1 = load_dump(str(p), join_key="key", field="cand.search.top1")
    assert top1["k1"] == (30.0, "middlegame")
