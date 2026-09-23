"""Reviewed corrections to the hash-verified qualification payload, not product code."""
import hashlib


def correct(data):
    name = 'tests/test_bend_collections.py'
    text = data['files'][name]
    changes = [
        ("            for arm in arms:\n                rows.append(f'{arm} {size} {sample} 10 {expected_checksum(size)} {size}')", "            rows.extend(f'{arm} {size} {sample} 10 {expected_checksum(size)} {size}' for arm in arms)"),
        ('with pytest.raises(ValueError):\n        verify_trace(text)', "with pytest.raises(ValueError, match='owning FIFO trace differs'):\n        verify_trace(text)"),
        ('with pytest.raises(ValueError):\n        parse_benchmark', "with pytest.raises(ValueError, match='benchmark'):\n        parse_benchmark"),
        ('with pytest.raises(ValueError):\n        verify_traversal(text)', "with pytest.raises(ValueError, match='search traversal differs'):\n        verify_traversal(text)"),
    ]
    for old, new in changes:
        assert text.count(old) == 1, old
        text = text.replace(old, new)
    assert hashlib.sha256(text.encode()).hexdigest() == '5892fc97b696f9cc4c3e719a7298a888be9c1ca6364aaacf9ee093994eff1ab6'
    data['files'][name] = text
    name = 'docs/experiments/2026-09-23-bend-collections-screen.md'
    data['files'][name] += '\n\n### Preserved initial failure\n\nRun 35922817981 stopped at the first static gate: four Ruff findings in the new test file (one list-construction style issue and three broad exception assertions). Whole-repository type checking reported zero errors/warnings and Vulture had no findings. Hosted Python and native tests had not run. The corrected candidate uses list.extend and checks exception messages; no gate or expectation was suppressed. The original 17 pure Python tests passed locally both before and after this correction. A fresh complete qualification follows; this failed attempt is not counted as native coverage.\n'
    return data
