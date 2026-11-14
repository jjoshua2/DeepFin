import json
import pytest
from scripts.qualify_packed_trainer_remaining_worker import write_atomic


def test_stream_receipt_publish_preserves_existing_receipt(tmp_path):
    path = tmp_path / 'directory.json'
    write_atomic(path, {'rows': 12, 'sequence_sha256': 'original'})
    assert not path.with_suffix('.json.partial').exists()
    with pytest.raises(FileExistsError):
        write_atomic(path, {'rows': 99})
    assert json.loads(path.read_text()) == {'rows': 12, 'sequence_sha256': 'original'}
    assert path.with_suffix('.json.partial').exists()


def test_qualification_binding_matches_authenticated_bank(tmp_path):
    import hashlib
    from scripts.qualify_packed_trainer_remaining_worker import authenticate_bank
    source = tmp_path / 'source.zarr'
    source.mkdir()
    (source / 'chunk').write_bytes(b'x')
    archive = tmp_path / 'source.zarr.zip'
    archive.write_bytes(b'zip')
    roots = {name: tmp_path / name for name in ('directory', 'packed')}
    for root in roots.values():
        root.mkdir()
    (roots['directory'] / 'shard_000000.zarr').symlink_to(source)
    link = roots['packed'] / 'shard_000000.zarr.zip'
    link.symlink_to(archive)
    prepared = {'records': [{'source': str(source), 'zip': str(archive),
        'members': {'chunk': hashlib.sha256(b'x').hexdigest()},
        'zip_sha256': hashlib.sha256(b'zip').hexdigest()}]}
    authenticate_bank(prepared, roots)
    other = tmp_path / 'other.zip'
    other.write_bytes(b'zip')
    link.unlink()
    link.symlink_to(other)
    with pytest.raises(ValueError, match='binding drift'):
        authenticate_bank(prepared, roots)
