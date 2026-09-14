"""C# shortest float32 JSON is not an exact float64 serialization of FP16."""
import numpy as np
import pytest

from tools.ceres_cpu_oracle.transport import getter_values


def test_actual_shortest_json_decimal_preserves_half_bits() -> None:
    # Actual upstream synthetic-tie getter: FP16 13653, C# serialized float32.
    row = {'bits': [13653] * 12, 'values': [0.33325195] * 12}
    values = getter_values(row)
    assert 0.33325195 != 0.333251953125
    np.testing.assert_array_equal(values, np.full(12, 0.333251953125))
    assert values.dtype == np.float64  # downstream deltas use exact half values


def test_one_float32_ulp_error_is_rejected_without_tolerance() -> None:
    wrong = float(np.nextafter(np.float32(0.33325195), np.float32(1)))
    with pytest.raises(ValueError, match='getter bit transport'):
        getter_values({'bits': [13653] * 12, 'values': [wrong] * 12})


@pytest.mark.parametrize('defect', ['wrong_bits', 'wrong_shape'])
def test_corrupted_getter_transport_is_rejected(defect: str) -> None:
    bits = [13653] * 12
    if defect == 'wrong_bits':
        bits[0] += 1
    else:
        bits.pop()
    with pytest.raises(ValueError, match='getter bit transport'):
        getter_values({'bits': bits, 'values': [0.33325195] * 12})
