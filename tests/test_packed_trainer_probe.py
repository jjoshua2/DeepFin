from types import SimpleNamespace
import json
import numpy as np
import pytest
from scripts.packed_trainer_probe import arrays_digest, observe


def test_order_digest_sees_permutation_and_dtype():
    x = np.array([1, 2], dtype=np.int64)
    assert arrays_digest({'game_id': x}) != arrays_digest({'game_id': x[::-1]})
    assert arrays_digest({'game_id': x}) != arrays_digest({'game_id': x.astype(np.int32)})


@pytest.mark.parametrize('fail', [False, True])
def test_observer_preserves_batch_and_restores_hooks(tmp_path, fail):
    import torch
    batch = {'game_id': np.array([0, 1]), 'ply_index': np.array([2, 3]),
             'has_game_id': np.ones(2, dtype=bool),
             'has_ply_index': np.ones(2, dtype=bool)}
    class Buffer:
        def sample_batch_arrays(self, *_args, **_kwargs):
            return batch
    original_sample = Buffer.sample_batch_arrays
    def build():
        return torch.nn.Linear(2, 1)
    driver = SimpleNamespace(build_model=build, GameAwareEpochBuffer=Buffer)
    def main(_argv):
        driver.build_model()
        assert Buffer().sample_batch_arrays(2) is batch
        if fail:
            raise RuntimeError('owned test failure')
        return 0
    driver.main = main
    receipt = tmp_path / 'receipt.json'
    if fail:
        with pytest.raises(RuntimeError, match='owned test failure'):
            observe(driver, receipt, [])
    else:
        observe(driver, receipt, [])
    assert driver.build_model is build
    assert Buffer.sample_batch_arrays is original_sample
    data = json.loads(receipt.read_text())
    assert data['batches'][0]['rows'] == 2
    assert len(data['initial_model_sha256']) == 64
    assert data['status'] == ('INCOMPLETE' if fail else 'TRAINER_RETURNED_SUCCESS')


def test_observer_accepts_real_epoch_sampler_schema(tmp_path):
    import torch
    from tests.test_game_aware_epoch_replay import _write, _open
    from chess_anti_engine.replay.game_epoch import GameAwareEpochBuffer
    root = _write(tmp_path / "source", [[(1, 10), (2, 20)], [(1, 11), (2, 21)]])
    original = GameAwareEpochBuffer.sample_batch_arrays
    driver = SimpleNamespace(build_model=lambda: torch.nn.Linear(2, 1),
                             GameAwareEpochBuffer=GameAwareEpochBuffer)
    seen = []
    def main(_argv):
        driver.build_model()
        buffer = _open(root, batch_size=2)
        try:
            for _ in range(buffer.plan.batches):
                batch = buffer.sample_batch_arrays(2)
                assert "ply" not in batch
                seen.append(arrays_digest({k: batch[k] for k in ("game_id", "ply_index")}))
            assert buffer.receipt()["complete"]
        finally:
            buffer.close()
    driver.main = main
    receipt = tmp_path / "real.json"
    observe(driver, receipt, [])
    data = json.loads(receipt.read_text())
    assert data["status"] == "TRAINER_RETURNED_SUCCESS"
    assert [b["order_sha256"] for b in data["batches"]] == seen
    assert sum(b["rows"] for b in data["batches"]) == 4
    assert GameAwareEpochBuffer.sample_batch_arrays is original
