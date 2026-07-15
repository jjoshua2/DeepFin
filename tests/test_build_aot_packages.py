"""Non-GPU unit tests for scripts/build_aot_packages.py pure helpers."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.build_aot_packages import (
    _compare_bucket,
    _softmax,
    build_arg_parser,
    format_summary_line,
    package_filename,
    package_path,
    parse_buckets_arg,
    plan_build_buckets,
    select_buckets,
)


def test_package_filename_and_path() -> None:
    assert package_filename(24) == "chess_b24.pt2"
    assert package_path("data/aot", 128) == Path("data/aot/chess_b128.pt2")


def test_parse_buckets_arg_ok() -> None:
    assert parse_buckets_arg("1, 6 ,24,128") == (1, 6, 24, 128)


@pytest.mark.parametrize("bad", ["", "  ,  ", "1,foo", "1,-4", "0"])
def test_parse_buckets_arg_rejects(bad: str) -> None:
    with pytest.raises(ValueError, match=r"bucket|empty|positive"):
        parse_buckets_arg(bad)


def test_select_buckets_filters_by_max_batch() -> None:
    sel = select_buckets(max_batch=64, buckets=[1, 6, 24, 64, 128, 4096])
    assert sel == (1, 6, 24, 64)


def test_select_buckets_default_source_nonempty() -> None:
    # Uses _BATCH_BUCKETS; every element must be <= a large cap.
    sel = select_buckets(max_batch=4096)
    assert sel
    assert all(b <= 4096 for b in sel)
    assert len(set(sel)) == len(sel)


def test_select_buckets_rejects_bad_max_batch() -> None:
    with pytest.raises(ValueError, match="max-batch must be positive"):
        select_buckets(max_batch=0, buckets=[1, 2])


def test_select_buckets_rejects_all_filtered() -> None:
    with pytest.raises(ValueError, match="no buckets"):
        select_buckets(max_batch=1, buckets=[6, 24])


def test_plan_build_buckets_resume_skips_existing(tmp_path: Path) -> None:
    (tmp_path / "chess_b6.pt2").write_bytes(b"x")
    to_build, skipped = plan_build_buckets([1, 6, 24], tmp_path, resume=True)
    assert to_build == [1, 24]
    assert skipped == [6]


def test_plan_build_buckets_no_resume_rebuilds(tmp_path: Path) -> None:
    (tmp_path / "chess_b6.pt2").write_bytes(b"x")
    to_build, skipped = plan_build_buckets([1, 6], tmp_path, resume=False)
    assert to_build == [1, 6]
    assert skipped == []


def test_format_summary_line() -> None:
    line = format_summary_line(
        built=3, skipped=1, verified=4, failed=0, out_dir="data/aot_models_512",
    )
    assert line == (
        "aot_build: built=3 skipped=1 verified=4 failed=0 out=data/aot_models_512"
    )


def test_arg_parser_defaults() -> None:
    args = build_arg_parser().parse_args(["--checkpoint", "x.pt"])
    assert args.config == Path("configs/pbt2_small.yaml")
    assert args.out_dir == Path("data/aot_models_512")
    assert args.max_batch == 4096
    assert args.tol == pytest.approx(2e-2)
    assert args.wdl_tol == pytest.approx(8e-2)
    assert args.argmax_min == pytest.approx(0.90)
    assert not args.verify
    assert not args.resume


def test_arg_parser_requires_checkpoint() -> None:
    with pytest.raises(SystemExit):
        build_arg_parser().parse_args([])


def test_softmax_is_stable_and_normalized() -> None:
    # Large magnitudes (incl. a masked-slot sentinel) must not overflow.
    logits = np.array([[0.0, 1.0, -1e30], [1e6, 1e6 - 1.0, -1e6]], dtype=np.float32)
    p = _softmax(logits)
    assert np.allclose(p.sum(axis=-1), 1.0)
    assert np.all(np.isfinite(p))
    assert p[0, 2] == pytest.approx(0.0, abs=1e-12)  # sentinel -> ~0
    assert p[0, 1] > p[0, 0]  # order preserved


def _wdl(*rows: list[float]) -> np.ndarray:
    return np.array(rows, dtype=np.float32)


def test_compare_bucket_pass_on_near_identical() -> None:
    pol = np.array([[2.0, 1.0, 0.0], [0.0, 3.0, 1.0]], dtype=np.float32)
    wdl = _wdl([1.0, 0.0, -1.0], [0.0, 1.0, 0.0])
    ok, detail = _compare_bucket(
        aot_pol=pol + 1e-3, aot_wdl=wdl + 1e-3, ref_pol=pol, ref_wdl=wdl,
        pol_tol=2e-2, wdl_tol=6e-2, argmax_min=0.90,
    )
    assert ok, detail


def test_compare_bucket_fails_on_garbage_policy() -> None:
    # A broken package (wrong/unfilled constants) -> argmax collapses.
    ref_pol = np.array([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0]], dtype=np.float32)
    aot_pol = np.array([[0.0, 0.0, 5.0], [5.0, 0.0, 0.0]], dtype=np.float32)
    wdl = _wdl([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    ok, _ = _compare_bucket(
        aot_pol=aot_pol, aot_wdl=wdl, ref_pol=ref_pol, ref_wdl=wdl,
        pol_tol=2e-2, wdl_tol=6e-2, argmax_min=0.90,
    )
    assert not ok


def test_compare_bucket_fails_on_wdl_drift() -> None:
    pol = np.array([[2.0, 1.0, 0.0]], dtype=np.float32)
    ref_wdl = _wdl([4.0, 0.0, 0.0])
    aot_wdl = _wdl([0.0, 4.0, 0.0])  # flipped -> large prob delta
    ok, _ = _compare_bucket(
        aot_pol=pol, aot_wdl=aot_wdl, ref_pol=pol, ref_wdl=ref_wdl,
        pol_tol=2e-2, wdl_tol=6e-2, argmax_min=0.90,
    )
    assert not ok
