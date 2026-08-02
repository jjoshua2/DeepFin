"""The per-phase loss split must bucket by BOARD, and must publish its counts.

Until 2026-08-02 ``_phase_split_masks`` bucketed on ``moves_left``, which
``selfplay/finalize.py:924`` writes as ``(total_plies_played - ply_index) /
max_plies`` — the divisor is the CONFIGURED PLY CAP (450 in production), not
the game's own length. That is not a board property at all: a row at ply 2 of a
60-ply adjudicated game scored 0.129 and was labelled ``end``. Measured over
the whole live window (713 shards, 1,273,501 rows) the split was
0.61 % / 3.03 % / 96.37 %, so ``wdl_loss_open`` and ``wdl_loss_mid`` were
computed on ~3.6 % of the data — and because ``masked_mean`` clamps its
denominator to 1.0, a bucket holding NO rows publishes a loss of 0.0, the best
possible value, with nothing able to contradict it.

Three things are pinned here, each on the specific signal rather than on an
aggregate that a substitution could survive:

  * the PREDICATE is piece count, not ``moves_left`` — asserted on a batch
    where the two disagree by construction, so a revert to ``moves_left``
    fails rather than merely changing the numbers;
  * the DEFINITION is the same object ``eval/audit.py`` buckets its per-phase
    deep-SF regret with, checked bucket-for-bucket over every reachable piece
    count, so the training column and the audit column cannot drift apart;
  * the COUNTS reach ``TrainMetrics``, sum to the rows actually seen, and go
    to zero for an empty bucket while its loss stays 0.0 — i.e. the counts are
    the ONLY thing that can tell an empty bucket from a perfect one.
"""
from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest
import torch

from chess_anti_engine.eval.audit import PHASE_THRESHOLDS, phase_bucket
from chess_anti_engine.model.transformer import ChessNet, TransformerConfig
from chess_anti_engine.moves import POLICY_SIZE
from chess_anti_engine.replay import ReplayBuffer
from chess_anti_engine.replay.buffer import ReplaySample
from chess_anti_engine.train import Trainer
from chess_anti_engine.train.losses import _phase_split_masks, piece_counts_from_input
from chess_anti_engine.utils.architecture import DEFAULT_PHASE_PIECE_THRESHOLDS

_REPO = Path(__file__).resolve().parents[1]
_PLANES = 146


def _x_with_pieces(piece_count: int, *, n: int = 1) -> np.ndarray:
    """(n, 146, 8, 8) whose first 12 planes hold exactly ``piece_count`` ones."""
    x = np.zeros((n, _PLANES, 8, 8), dtype=np.float32)
    for i in range(int(piece_count)):
        x[:, i % 12, i // 12, i % 8] = 1.0
    assert round(float(x[0, :12].sum())) == int(piece_count)
    return x


def _bucket_of(x: np.ndarray) -> np.ndarray:
    """0=end, 1=mid, 2=open, straight off `_phase_split_masks`."""
    t = torch.from_numpy(x)
    ones = torch.ones(t.shape[0])
    masks = dict(
        _phase_split_masks(
            has_is_selfplay=ones, is_selfplay=ones,
            piece_counts=piece_counts_from_input(t),
        )
    )
    stacked = torch.stack(
        [masks["phase_end"], masks["phase_mid"], masks["phase_open"]], dim=1,
    )
    # A row must belong to EXACTLY one bucket; argmax alone would hide a row
    # claimed by two masks or by none.
    assert torch.allclose(stacked.sum(dim=1), torch.ones(t.shape[0]))
    return stacked.argmax(dim=1).numpy()


# ---------------------------------------------------------------------------
# The predicate
# ---------------------------------------------------------------------------


def test_bucket_follows_piece_count_and_not_moves_left() -> None:
    """The deciding case: piece count and `moves_left` point opposite ways.

    32 pieces with `moves_left` far below the OLD end cut (0.31), and 4 pieces
    with `moves_left` far above the OLD open cut (0.45). Under the old
    predicate the first row was `end` and the second `open`; under the correct
    one they are `open` and `end`. A revert to `moves_left` inverts both
    assertions, so this cannot pass on the old code whatever the thresholds.
    """
    x = np.concatenate([_x_with_pieces(32), _x_with_pieces(4)], axis=0)
    buckets = _bucket_of(x)
    assert buckets[0] == 2, "32 pieces is the OPENING bucket"
    assert buckets[1] == 0, "4 pieces is the ENDGAME bucket"


def test_moves_left_is_not_read_by_the_split_at_all() -> None:
    """`_phase_split_masks` must not accept a moves_left argument any more.

    Signature-level, because a parameter that still exists is a parameter a
    future edit can start reading again without any test noticing.
    """
    params = set(inspect.signature(_phase_split_masks).parameters)
    assert "piece_counts" in params
    assert "moves_left_val" not in params
    assert "has_moves_left" not in params


@pytest.mark.parametrize("piece_count", list(range(2, 33)))
def test_matches_eval_audit_phase_bucket_for_every_reachable_count(
    piece_count: int,
) -> None:
    """Bucket-for-bucket agreement with the audit's own function.

    Parametrised per count rather than asserted as a mean: a mean over the
    range passes at 30/31 and would not name the boundary that broke.
    """
    assert int(_bucket_of(_x_with_pieces(piece_count))[0]) == phase_bucket(piece_count)


def test_the_thresholds_are_one_shared_object_not_two_literals() -> None:
    """`eval/audit.py` and `train/losses.py` must read the SAME constant.

    Identity, not equality: two equal literals satisfy `==` right up to the
    moment somebody edits one of them.
    """
    assert PHASE_THRESHOLDS is DEFAULT_PHASE_PIECE_THRESHOLDS
    losses_src = (_REPO / "chess_anti_engine" / "train" / "losses.py").read_text(
        encoding="utf-8",
    )
    assert "DEFAULT_PHASE_PIECE_THRESHOLDS" in losses_src
    # The old cuts must be gone, not merely unused: a leftover constant is the
    # first thing a future edit reaches for.
    assert "_PHASE_OPEN_THRESHOLD" not in losses_src
    assert "_PHASE_END_THRESHOLD" not in losses_src


def test_boundary_rows_land_on_the_documented_side() -> None:
    """`end` is `<= 13` and `open` is `> 22`; the two boundaries are the bug bait."""
    low, high = DEFAULT_PHASE_PIECE_THRESHOLDS
    assert int(_bucket_of(_x_with_pieces(low))[0]) == 0
    assert int(_bucket_of(_x_with_pieces(low + 1))[0]) == 1
    assert int(_bucket_of(_x_with_pieces(high))[0]) == 1
    assert int(_bucket_of(_x_with_pieces(high + 1))[0]) == 2


# ---------------------------------------------------------------------------
# The counts
# ---------------------------------------------------------------------------


def _sample(piece_count: int) -> ReplaySample:
    x = _x_with_pieces(piece_count)[0]
    policy = np.zeros((POLICY_SIZE,), dtype=np.float32)
    policy[0] = 1.0
    return ReplaySample(
        x=x, policy_target=policy, wdl_target=1, priority=1.0,
        has_policy=True, is_network_turn=True,
    )


def _trainer(tmp_path: Path) -> Trainer:
    cfg = TransformerConfig(
        in_planes=_PLANES, embed_dim=32, num_layers=1, num_heads=2,
        use_smolgen=False, use_nla=False,
    )
    return Trainer(
        ChessNet(cfg), device="cpu", lr=1e-4, log_dir=tmp_path / "tb",
        use_amp=False, feature_dropout_p=0.0, swa_start=-1,
    )


def _run(tmp_path: Path, counts: list[int], *, steps: int, batch_size: int):
    rng = np.random.default_rng(0)
    buf = ReplayBuffer(len(counts) * 4, rng=rng)
    for count in counts:
        buf.add(_sample(count))
    return _trainer(tmp_path).train_steps(buf, batch_size=batch_size, steps=steps)


def test_counts_reach_train_metrics_and_sum_to_the_rows_trained(tmp_path) -> None:
    counts = [30] * 8 + [18] * 8 + [6] * 8
    metrics = _run(tmp_path, counts, steps=3, batch_size=8)
    total = (
        metrics.wdl_loss_phase_n_open
        + metrics.wdl_loss_phase_n_mid
        + metrics.wdl_loss_phase_n_end
    )
    # Every trained row lands in exactly one bucket, so the three counts must
    # reconstruct steps x batch_size exactly -- not approximately. This is what
    # fails if the counts are divided by the step count on the way out.
    assert total == pytest.approx(3 * 8)
    assert metrics.wdl_loss_phase_n_open > 0
    assert metrics.wdl_loss_phase_n_end > 0


def test_an_empty_bucket_reports_zero_count_while_its_loss_stays_zero(
    tmp_path,
) -> None:
    """The whole reason the counts exist.

    An all-endgame window leaves `open` with no rows. `masked_mean` clamps its
    denominator to 1.0, so `wdl_loss_phase_open` publishes 0.0 -- which reads as
    the BEST possible value. Only the count can say the bucket was empty, and
    the assertion below pins both halves: a loss indistinguishable from perfect,
    next to a count that is unambiguously zero.
    """
    metrics = _run(tmp_path, [6] * 16, steps=2, batch_size=8)
    assert metrics.wdl_loss_phase_n_open == 0.0
    assert metrics.wdl_loss_phase_open == 0.0
    assert metrics.wdl_loss_phase_n_end == pytest.approx(2 * 8)
    assert metrics.wdl_loss_phase_end > 0.0


def test_counts_survive_the_EVAL_pooling_path_too(tmp_path) -> None:
    """`eval_full_pass` pools differently, and that is where a count can rot.

    The training path accumulates every compute_loss scalar verbatim, so a raw
    count is safe there whatever `_RAW_SUM_LOSS_KEYS` says. `_compute_metrics`
    does NOT: it scales each scalar by the batch's row count unless the key is
    declared a row sum. Drop the counts from that declaration and the training
    row stays perfect while the `test_` twin silently reports rows x rows --
    the same column meaning two different things on two paths. Asserted on a
    RAGGED pass (14 rows, batch 4) so a batch-size-multiple coincidence cannot
    hide the scaling.
    """
    counts = [30] * 5 + [18] * 4 + [6] * 5
    rng = np.random.default_rng(0)
    buf = ReplayBuffer(len(counts), rng=rng)
    for count in counts:
        buf.add(_sample(count))
    metrics = _trainer(tmp_path).eval_full_pass(buf, batch_size=4)

    assert metrics.eval_rows == len(counts)
    assert metrics.wdl_loss_phase_n_open == pytest.approx(5)
    assert metrics.wdl_loss_phase_n_mid == pytest.approx(4)
    assert metrics.wdl_loss_phase_n_end == pytest.approx(5)


def test_counts_are_published_to_the_result_row() -> None:
    """The metric must reach progress.csv, not merely TrainMetrics.

    Read off the report module's own row builder and its zero-defaults dict:
    a field that exists on the dataclass and is never written to the row is
    exactly the shape of defect this change is repairing.
    """
    from chess_anti_engine.tune import trainable_report

    src = inspect.getsource(trainable_report)
    for name in (
        "wdl_loss_phase_n_open", "wdl_loss_phase_n_mid", "wdl_loss_phase_n_end",
    ):
        assert f'"{name}": float(metrics.{name})' in src, f"{name} not in the row"
        assert f'"{name}": 0.0' in src, f"{name} missing from the zero defaults"


# ---------------------------------------------------------------------------
# The split is REPORTING ONLY -- and that has to be proved, because moving it
# moves the frozen holdout ruler id (tests/test_holdout_ruler_identity.py).
#
# `compute_loss` takes `outputs` and `batch` separately, so the piece planes in
# `batch["x"]` can be rewritten WITHOUT touching the model's predictions. Inside
# `compute_loss`, `x` feeds exactly one thing: this split. So an intervention on
# the planes is a clean isolation of the change -- every `*_phase_*` key must
# move, and every scalar that reaches `total` must be BITWISE identical.
# ---------------------------------------------------------------------------


def _loss_batch(piece_counts: list[int]) -> tuple[dict, dict]:
    from chess_anti_engine.replay.dataset import collate

    samples = [_sample(c) for c in piece_counts]
    batch = collate(samples, device="cpu")
    net = ChessNet(
        TransformerConfig(
            in_planes=_PLANES, embed_dim=32, num_layers=1, num_heads=2,
            use_smolgen=False, use_nla=False,
        )
    )
    net.eval()
    with torch.no_grad():
        outputs = net(batch["x"])
    return outputs, batch


def test_the_phase_split_cannot_perturb_the_trained_loss() -> None:
    """Rewrite every row's phase; `total` must not move by one bit."""
    from chess_anti_engine.train.losses import compute_loss

    outputs, batch = _loss_batch([30] * 4 + [6] * 4)
    before = compute_loss(outputs, batch)

    flipped = dict(batch)
    x = batch["x"].clone()
    x[:, :12] = 0.0
    x[:, 0, 0, :4] = 1.0  # 4 pieces on every row -> all `phase_end`
    flipped["x"] = x
    after = compute_loss(outputs, flipped)

    moved = {k for k in before if not torch.equal(before[k], after[k])}
    assert moved, "the intervention did not reach the split at all"
    assert all("phase_" in k for k in moved), (
        f"the phase split perturbed a non-reporting scalar: "
        f"{sorted(k for k in moved if 'phase_' not in k)}"
    )
    assert torch.equal(before["total"], after["total"])
    # And the intervention really did empty two buckets, so `moved` is not
    # carried by a rounding wobble.
    assert float(after["wdl_rows_phase_open"]) == 0.0
    assert float(after["wdl_rows_phase_end"]) == 8.0
