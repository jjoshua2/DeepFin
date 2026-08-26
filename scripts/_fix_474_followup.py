#!/usr/bin/env python3
# Temporary guarded transformer; removed by apply-474-followup.yml.
from pathlib import Path

p = Path('scripts/nnue_gumbel_readout.py')
s = p.read_text()

def one(old: str, new: str) -> None:
    global s
    if s.count(old) != 1:
        raise SystemExit(f'expected one match, got {s.count(old)} for {old[:80]!r}')
    s = s.replace(old, new, 1)

one('import sys\nimport time\n', 'import sys\nimport tempfile\nimport time\n')
one(
    '        "workers": cfg.workers,\n        "plies": plies,\n',
    '        # Active is the throughput denominator; requested is provenance.\n'
    '        # When games < workers, _build_worker_specs drops empty buckets.\n'
    '        "workers": len(results),\n'
    '        "workers_requested": cfg.workers,\n'
    '        "plies": plies,\n',
)
one(
    'def _sha256_file(path: Path) -> str:\n',
    '''def _atomic_write_text(path: Path, text: str) -> None:\n'
    '    """Replace a result artifact all-or-nothing on the same filesystem."""\n'
    '    path.parent.mkdir(parents=True, exist_ok=True)\n'
    '    tmp_name: str | None = None\n'
    '    try:\n'
    '        with tempfile.NamedTemporaryFile(\n'
    '            mode="w", encoding="utf-8", dir=path.parent,\n'
    '            prefix=f".{path.name}.", suffix=".tmp", delete=False,\n'
    '        ) as f:\n'
    '            tmp_name = f.name\n'
    '            f.write(text)\n'
    '            f.flush()\n'
    '            os.fsync(f.fileno())\n'
    '        os.replace(tmp_name, path)\n'
    '        tmp_name = None\n'
    '    finally:\n'
    '        if tmp_name is not None:\n'
    '            try:\n'
    '                Path(tmp_name).unlink()\n'
    '            except FileNotFoundError:\n'
    '                pass\n'
    '\n'
    '\n'
    'def _sha256_file(path: Path) -> str:\n'''.replace("'\n    '", "\n"),
)
one(
    '        path = Path(args.json)\n        path.parent.mkdir(parents=True, exist_ok=True)\n        path.write_text(text + "\\n", encoding="utf-8")\n',
    '        path = Path(args.json)\n        _atomic_write_text(path, text + "\\n")\n',
)
p.write_text(s)

# Add focused tests without coupling to the large worker-result fake machinery.
t = Path('tests/test_nnue_gumbel_readout_followup.py')
t.write_text('''from __future__ import annotations\n\nfrom pathlib import Path\n\nfrom scripts import nnue_gumbel_readout as readout\n\n\ndef test_worker_specs_drop_empty_requested_workers() -> None:\n    cfg = object.__new__(readout.RunConfig)\n    object.__setattr__(cfg, "workers", 8)\n    object.__setattr__(cfg, "games", 1)\n    specs = readout._build_worker_specs(cfg)\n    assert len(specs) == 1\n    assert specs[0].game_indices == (0,)\n\n\ndef test_atomic_write_replaces_existing_result(tmp_path: Path) -> None:\n    path = tmp_path / "result.json"\n    path.write_text("old\\n", encoding="utf-8")\n    readout._atomic_write_text(path, "new\\n")\n    assert path.read_text(encoding="utf-8") == "new\\n"\n    assert not list(tmp_path.glob(".result.json.*.tmp"))\n''')