from __future__ import annotations

import pytest

from scripts.bt4_batch_benchmark import provider_proof


def event(provider, op, outputs=None):
    return {'args': {'provider': provider, 'op_name': op, 'output_type_shape': outputs}}


@pytest.mark.parametrize('op', ['FusedMatMul', 'FusedGemm', 'MatMul', 'Conv'])
def test_reject_cpu_neural_even_with_cuda(op):
    with pytest.raises(ValueError, match='unapproved CPU kernel'):
        provider_proof([event('CUDAExecutionProvider', 'MatMul'),
                        event('CPUExecutionProvider', op, [{'float': [1, 512]}])])


@pytest.mark.parametrize('op', ['Cast', 'Gather'])
def test_reject_cpu_float_shape_named_op(op):
    with pytest.raises(ValueError, match='CPU floating'):
        provider_proof([event('CUDAExecutionProvider', 'MatMul'),
                        event('CPUExecutionProvider', op, [{'float': [1]}])])


@pytest.mark.parametrize('outputs', [None, [], [{'int64': [-1]}], [{'int64': [4097]}]])
def test_reject_missing_dynamic_or_large_shape(outputs):
    with pytest.raises(ValueError, match=r'unapproved CPU kernel|CPU floating|large CPU'):
        provider_proof([event('CUDAExecutionProvider', 'MatMul'),
                        event('CPUExecutionProvider', 'Gather', outputs)])


def test_require_cuda_neural_not_just_copy():
    with pytest.raises(ValueError, match='no CUDA neural'):
        provider_proof([event('CUDAExecutionProvider', 'MemcpyFromHost')])


def test_reject_unexpected_provider():
    with pytest.raises(ValueError, match='unexpected profiled'):
        provider_proof([event('CUDAExecutionProvider', 'MatMul'),
                        event('OtherProvider', 'Gather', [{'int64': [1]}])])


def test_allow_small_integer_shape_work():
    result = provider_proof([event('CUDAExecutionProvider', 'FusedMatMul'),
                             event('CPUExecutionProvider', 'Shape', [{'int64': [4]}]),
                             event('CPUExecutionProvider', 'Gather', [{'int64': []}])])
    assert result['CUDA_neural_kernel_events'] == 1
    cpu_events = result['CPU_shape_events']
    assert isinstance(cpu_events, list)
    assert len(cpu_events) == 2
