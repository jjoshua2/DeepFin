import numpy as np

from chess_anti_engine.replay.shard import validate_arrays


def test_validate_rejects_wrong_policy_size():
    arrs = {
        "x": np.zeros((2, 146, 8, 8), dtype=np.float32),
        "policy_target": np.ones((2, 10), dtype=np.float32),
        "wdl_target": np.array([0, 1], dtype=np.int8),
    }
    try:
        validate_arrays(arrs)
        assert False, "expected ValueError"
    except ValueError as e:
        assert "A mismatch" in str(e)


def test_validate_rejects_negative_policy():
    arrs = {
        "x": np.zeros((1, 146, 8, 8), dtype=np.float32),
        "policy_target": np.zeros((1, 4672), dtype=np.float32),
        "wdl_target": np.array([1], dtype=np.int8),
    }
    arrs["policy_target"][0, 0] = -0.1
    try:
        validate_arrays(arrs)
        assert False, "expected ValueError"
    except ValueError as e:
        assert "negative" in str(e)
