"""Non-GPU unit tests for scripts/build_aot_packages.py pure helpers."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from chess_anti_engine.moves import POLICY_SIZE
from scripts.build_aot_packages import (
    _compare_bucket,
    _real_position_batch,
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
    assert args.tv_ratio_max == pytest.approx(2.0)
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
    ok, detail, matches, rows = _compare_bucket(
        aot_pol=pol + 1e-3, aot_wdl=wdl + 1e-3, ref_pol=pol, ref_wdl=wdl,
        ctl_pol=pol + 1e-3, ctl_wdl=wdl + 1e-3,
        tv_ratio_max=1.5,
    )
    assert ok, detail
    assert (matches, rows) == (2, 2)


def test_compare_bucket_reports_a_collapsed_argmax_for_the_caller_to_gate() -> None:
    """A broken package (wrong/unfilled constants) -> argmax collapses.

    ⚑ ``_compare_bucket`` no longer gates on argmax — ``verify_packages`` does,
    on rows POOLED across trials, because per-trial the criterion is a coin flip
    at small buckets. So this asserts the COUNTS it hands back, which is the
    thing the caller's verdict is computed from.
    """
    ref_pol = np.array([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0]], dtype=np.float32)
    aot_pol = np.array([[0.0, 0.0, 5.0], [5.0, 0.0, 0.0]], dtype=np.float32)
    wdl = _wdl([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    ok, _, matches, rows = _compare_bucket(
        aot_pol=aot_pol, aot_wdl=wdl, ref_pol=ref_pol, ref_wdl=wdl,
        ctl_pol=ref_pol, ctl_wdl=wdl,
        tv_ratio_max=1e9,  # TV arm disarmed: this is purely the argmax report
    )
    assert (matches, rows) == (0, 2), "every row's argmax moved"
    assert ok, "with the TV arm disarmed _compare_bucket has nothing left to fail on"


def test_compare_bucket_fails_on_wdl_drift() -> None:
    """⚑ wdl is the ONLY value head MCTS consumes, so it needs its own arm."""
    pol = np.array([[2.0, 1.0, 0.0]], dtype=np.float32)
    ref_wdl = _wdl([4.0, 0.0, 0.0])
    aot_wdl = _wdl([0.0, 4.0, 0.0])  # flipped -> large prob delta
    ok, detail, _, _ = _compare_bucket(
        aot_pol=pol, aot_wdl=aot_wdl, ref_pol=pol, ref_wdl=ref_wdl,
        ctl_pol=pol, ctl_wdl=ref_wdl + 1e-3,
        tv_ratio_max=1.5,
    )
    assert not ok, detail


# ---------------------------------------------------------------------------
# Verify-time input encoding (audit E2)
# ---------------------------------------------------------------------------

# configs/pbt2_small.yaml:70-71 — what the deployed model is built for.
_PROD_HISTORY = "lc0_root_legacy_meta"
_PROD_EXTRA = "v2_threats"


def test_real_position_batch_honours_the_history_encoding() -> None:
    """The history keyword must reach the encoder, not be accepted and dropped.

    Mutation guard: delete ``input_history_encoding=...`` from the
    ``encode_positions_batch`` call inside ``_real_position_batch`` and both
    arms below become the legacy layout, so this test goes red. The plane COUNT
    is identical either way (112 + n_extra), which is exactly why the buffer
    shape, the package signature and the verify comparison could all succeed on
    the wrong content (encoding audit E2 measured 98/175 planes differing).
    """
    prod = _real_position_batch(
        8, input_extra_features=_PROD_EXTRA,
        input_history_encoding=_PROD_HISTORY, seed=11,
    )
    legacy = _real_position_batch(
        8, input_extra_features=_PROD_EXTRA,
        input_history_encoding="legacy", seed=11,
    )
    assert prod.shape == legacy.shape == (8, 175, 8, 8)
    assert not np.array_equal(prod, legacy)
    # Same boards, same feature version: the difference is the history layout,
    # and it is broad (not one stray plane).
    differing = int(np.sum(np.any(prod != legacy, axis=(0, 2, 3))))
    assert differing > 50, f"only {differing} planes differ — layouts too close"


class _StubPackage:
    """Stands in for a loaded AOTInductor package."""

    def __init__(self, out: dict) -> None:
        self._out = out
        self.rebinds: list[tuple[dict, bool]] = []
        self.fed: list[torch.Tensor] = []

    def get_constant_fqns(self) -> list[str]:
        return ["w"]

    def load_constants(self, constants: dict, *, check_full_update: bool = True) -> None:
        self.rebinds.append((constants, check_full_update))

    def __call__(self, xt: torch.Tensor) -> dict:
        self.fed.append(xt)
        return self._out


class _StubModel(torch.nn.Module):
    """A model that declares production's encoding and records what it is fed."""

    def __init__(self) -> None:
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))
        self.input_history_encoding = _PROD_HISTORY
        self.input_extra_features = _PROD_EXTRA
        self.policy_encoding = "lc0_1858"
        self.seen: list[torch.Tensor] = []

    def forward(self, xt: torch.Tensor) -> dict:
        self.seen.append(xt)
        n = int(xt.shape[0])
        return {
            "policy": torch.zeros((n, POLICY_SIZE), dtype=torch.float32),
            "wdl": torch.zeros((n, 3), dtype=torch.float32),
        }


def test_verify_packages_encodes_in_the_models_own_encoding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The gate must measure on the deployment input distribution.

    ``verify_packages`` reads the encoding off the model rather than taking it
    as a keyword, so there is nothing to forget. This asserts the realized
    encoder call: swap the model's declared encoding and the bytes fed to the
    package change with it.
    """
    import scripts.build_aot_packages as MOD

    calls: list[dict] = []
    real_batch = MOD._real_position_batch

    def recording(n: int, **kwargs: object) -> np.ndarray:
        calls.append(dict(kwargs))
        return real_batch(n, **kwargs)  # pyright: ignore[reportArgumentType]

    monkeypatch.setattr(MOD, "_real_position_batch", recording)
    monkeypatch.setattr(MOD, "build_aot_constants", lambda *a, **k: {})
    model = _StubModel()
    stub_out = {
        "policy": torch.zeros((2, POLICY_SIZE), dtype=torch.float32),
        "wdl": torch.zeros((2, 3), dtype=torch.float32),
    }
    monkeypatch.setattr(MOD, "_aoti_load_package", lambda p: _StubPackage(stub_out))
    (tmp_path / "chess_b2.pt2").write_bytes(b"x")

    n_pass, n_fail, rows = MOD.verify_packages(
        out_dir=tmp_path, model=model, buckets=[2], max_batch=2,
        input_planes=175, tv_ratio_max=1.5, verify_n=1, seed=3,
    )
    assert (n_pass, n_fail) == (1, 0), rows
    assert calls == [{
        "input_extra_features": _PROD_EXTRA,
        "input_history_encoding": _PROD_HISTORY,
        "seed": 3,
    }]
    # And the model really saw production-encoded planes. seen[0] is the
    # REFERENCE forward; the later calls are the batch-shape control, which by
    # construction re-runs the same rows at a different batch size.
    fed = model.seen[0].float().numpy()
    assert fed.shape == (2, 175, 8, 8)
    assert [int(t.shape[0]) for t in model.seen[1:]] == [1, 1], (
        "the eager control must re-run the batch at a DIFFERENT shape, got "
        f"{[int(t.shape[0]) for t in model.seen[1:]]}"
    )
    legacy = _real_position_batch(
        2, input_extra_features=_PROD_EXTRA,
        input_history_encoding="legacy", seed=3,
    )
    assert not np.array_equal(fed, legacy.astype(np.float32))


def test_verify_packages_refuses_a_model_that_does_not_declare_its_encoding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Guessing the layout is the failure mode; refusing is the fix."""
    import scripts.build_aot_packages as MOD

    model = _StubModel()
    del model.input_history_encoding
    (tmp_path / "chess_b2.pt2").write_bytes(b"x")
    monkeypatch.setattr(MOD, "_aoti_load_package", lambda p: _StubPackage({}))
    with pytest.raises(ValueError, match="input_history_encoding"):
        MOD.verify_packages(
            out_dir=tmp_path, model=model, buckets=[2], max_batch=2,
            input_planes=175, tv_ratio_max=1.5, verify_n=1, seed=3,
        )
