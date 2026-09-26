import torch
from scripts.bootstrap_initial_state import record_initial_state


def test_fingerprint_tracks_parameters_without_advancing_rng(tmp_path):
    torch.manual_seed(121)
    model = torch.nn.Linear(3, 2)
    before = torch.get_rng_state().clone()
    a = record_initial_state(model, tmp_path / 'a.json', seed=121)
    assert torch.equal(before, torch.get_rng_state())
    b = record_initial_state(model, tmp_path / 'b.json', seed=121)
    assert a == b
    with torch.no_grad():
        model.weight[0, 0] += 1
    c = record_initial_state(model, tmp_path / 'c.json', seed=121)
    assert a['tensor_sha256'] != c['tensor_sha256']
