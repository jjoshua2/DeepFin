"""Real tiny producer/storage fixtures; coordinator full-size admission tested separately."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts import ceres_corpus_qualification as tool
from tests.test_ceres_target_mix import fixture as policy_fixture
from tests.test_ceres_value_mix import fixture as value_fixture


def write(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data))


def reference(path: Path) -> dict[str, str]:
    return {'path': str(path), 'sha256': tool.sha(path)}


def prepared(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, profile: str) -> tuple[Path, dict[str, Any], list[str]]:
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    producer = tool.policy if profile == 'CeresB50' else tool.value
    fixture = policy_fixture if profile == 'CeresB50' else value_fixture
    args, manifest = fixture(tmp_path, monkeypatch)
    result = producer.rewrite(args)
    corpus = Path(args.out)
    sf = Path(manifest.get('sf_source', manifest['source']))
    monkeypatch.setattr(tool, 'ROWS', 32)
    monkeypatch.setattr(tool, 'SHARDS', 1)
    monkeypatch.setattr(tool.epoch, 'SOURCE', sf)
    monkeypatch.setitem(tool.epoch.CORPORA, profile, corpus)
    if profile == 'B100CeresV25':
        monkeypatch.setitem(tool.epoch.CORPORA, 'B100', Path(manifest['source']))
    monkeypatch.setitem(tool.epoch.COMMON_PINS, str(sf / producer.DERIVE_SUMMARY), tool.sha(sf / producer.DERIVE_SUMMARY))
    calls: list[str] = []

    def verify(m: dict[str, Any], actual: dict[str, Any], derived: dict[str, Any]) -> None:
        # Only the full-18.9M coordinator boundary is mocked: all storage and actual
        # producer computations above/below are real. Its admission has own tests.
        assert m == {'profile': profile, 'ceres_producer_pins': result['producer_sha256']}
        assert actual == result
        assert isinstance(derived['shards'], list)
        calls.append(profile)

    monkeypatch.setattr(tool.epoch, 'verify_ceres_recipe' if profile == 'CeresB50' else 'verify_ceres_value_recipe', verify)
    plan: dict[str, Any] = {'schema': 1, 'profile': profile, 'corpus': str(corpus),
        'producer_manifest': reference(Path(args.manifest)),
        'derive_summary': reference(corpus / producer.DERIVE_SUMMARY),
        'rewrite_summary': reference(corpus / producer.SUMMARY),
        'ceres_producer_pins': result['producer_sha256'], 'max_seconds': 180,
        'minimum_free_gib': 0, 'stop_paths': [str(tmp_path / 'STOP')]}
    terminal = {'status': 'COMPLETE', 'returncode': 0, 'profile': profile, 'corpus': str(corpus),
        'producer_manifest_sha256': plan['producer_manifest']['sha256'],
        'derive_summary_sha256': plan['derive_summary']['sha256'],
        'rewrite_summary_sha256': plan['rewrite_summary']['sha256'], 'producer_sha256': result['producer_sha256']}
    write(tmp_path / 'terminal.json', terminal)
    plan['materialization'] = reference(tmp_path / 'terminal.json')
    path = tmp_path / 'qualification_plan.json'
    write(path, plan)
    return path, plan, calls


@pytest.mark.parametrize('profile', ['CeresB50', 'B100CeresV25'])
def test_actual_producer_metadata_qualifies_without_source_changes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, profile: str) -> None:
    path, plan, calls = prepared(tmp_path, monkeypatch, profile)
    original = json.loads(Path(plan['producer_manifest']['path']).read_text())
    source = Path(original['source'])
    before = tool.policy.shared.storage_identity(source)
    out = tmp_path / 'qualified.json'
    result = tool.qualify(path, tool.sha(path), out, execute=True)
    assert result['status'] == 'PASS_REGISTERED_CORPUS_QUALIFICATION'
    assert result['rewrite_summary'] == plan['rewrite_summary']
    assert len(result['metadata'][0]['layouts']) == 17
    assert calls == [profile]
    assert tool.policy.shared.storage_identity(source) == before
    assert json.loads(out.read_text()) == result
    assert not out.with_name(out.name + '.writing').exists()
    with pytest.raises(ValueError, match='already exists'):
        tool.qualify(path, tool.sha(path), out, execute=True)


@pytest.mark.parametrize('defect', ['output_chunk', 'source_chunk', 'teacher_chunk', 'attrs', 'layout',
                                  'incomplete', 'terminal', 'pin', 'stop', 'replaced_source'])
def test_changed_or_incomplete_storage_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, defect: str) -> None:
    path, plan, _ = prepared(tmp_path, monkeypatch, 'B100CeresV25')
    corpus = Path(plan['corpus'])
    manifest = json.loads(Path(plan['producer_manifest']['path']).read_text())
    shard = corpus / 'shard_000000.zarr'
    if defect in ('output_chunk', 'source_chunk', 'teacher_chunk'):
        root = shard if defect == 'output_chunk' else (Path(manifest['source']) / shard.name if defect == 'source_chunk' else Path(manifest['entries'][0]['ceres']))
        chunk = next(p for p in root.rglob('*') if p.is_file() and not p.name.startswith('.'))
        chunk.write_bytes(chunk.read_bytes() + b'corrupt')
    elif defect in ('attrs', 'layout'):
        p = shard / ('.zattrs' if defect == 'attrs' else 'search_wdl/.zarray')
        data = json.loads(p.read_text())
        data['derive_value_source' if defect == 'attrs' else 'shape'] = 'wrong'
        write(p, data)
    elif defect == 'replaced_source':
        p = Path(manifest['source']) / shard.name / '.zattrs'
        content = p.read_bytes()
        p.unlink()
        p.write_bytes(content)
    elif defect == 'incomplete':
        corpus.with_name(corpus.name + '.writing').mkdir()
    elif defect == 'terminal':
        p = tmp_path / 'terminal.json'
        terminal = json.loads(p.read_text())
        terminal['returncode'] = 1
        write(p, terminal)
        plan['materialization'] = reference(p)
        write(path, plan)
    elif defect == 'pin':
        plan['ceres_producer_pins'] = {'/missing.py': '0' * 64}
        write(path, plan)
    else:
        (tmp_path / 'STOP').touch()
    with pytest.raises(ValueError, match=r'identity|publication|materialization|producer|STOP'):
        tool.qualify(path, tool.sha(path), tmp_path / 'qualified.json', execute=True)
    assert not (tmp_path / 'qualified.json').exists()


@pytest.mark.parametrize('profile', ['CeresB50', 'B100CeresV25'])
def test_producer_refuses_output_mutated_after_witness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, profile: str) -> None:
    producer = tool.policy if profile == 'CeresB50' else tool.value
    args, _ = (policy_fixture if profile == 'CeresB50' else value_fixture)(tmp_path, monkeypatch)
    owner = tool.policy.shared if profile == 'CeresB50' else tool.value.wdl
    original = owner.storage_identity
    output = Path(args.out).with_name(Path(args.out).name + '.writing') / 'shard_000000.zarr'
    calls = 0

    def identity(path: Path) -> str:
        nonlocal calls
        if path.resolve() == output.resolve():
            calls += 1
            if (output.parent / producer.SUMMARY).exists():
                p = path / '.zattrs'
                p.write_bytes(p.read_bytes() + b' ')
        return original(path)

    monkeypatch.setattr(owner, 'storage_identity', identity)
    with pytest.raises(ValueError, match='output changed'):
        producer.rewrite(args)
    assert calls >= 2
    assert not Path(args.out).exists()


@pytest.mark.parametrize('defect', ['attrs', 'layout', 'missing_witness'])
def test_metadata_checks_do_not_merely_trust_storage_witness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, defect: str) -> None:
    path, plan, _ = prepared(tmp_path, monkeypatch, 'B100CeresV25')
    corpus = Path(plan['corpus'])
    shard = corpus / 'shard_000000.zarr'
    summary_path = Path(plan['rewrite_summary']['path'])
    rewrite = json.loads(summary_path.read_text())
    proof = rewrite['outputs'][0]
    if defect == 'attrs':
        p = shard / '.zattrs'
        attrs = json.loads(p.read_text())
        attrs['derive_value_source'] = 'wrong teacher'
        write(p, attrs)
        proof['attrs_sha256'] = tool.sha(p)
    elif defect == 'layout':
        p = shard / 'search_wdl/.zarray'
        layout = json.loads(p.read_text())
        layout['shape'] = [32, 4]
        write(p, layout)
    proof['output_storage_identity'] = tool.policy.shared.storage_identity(shard)
    if defect == 'missing_witness':
        del proof['output_storage_identity']
    write(summary_path, rewrite)
    plan['rewrite_summary'] = reference(summary_path)
    terminal_path = Path(plan['materialization']['path'])
    terminal = json.loads(terminal_path.read_text())
    terminal['rewrite_summary_sha256'] = plan['rewrite_summary']['sha256']
    write(terminal_path, terminal)
    plan['materialization'] = reference(terminal_path)
    write(path, plan)
    with pytest.raises(ValueError, match=r'attrs|layout|identity'):
        tool.qualify(path, tool.sha(path), tmp_path / 'qualified.json', execute=True)
    assert not (tmp_path / 'qualified.json').exists()
