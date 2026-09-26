"""Temporary preparation correction; only the tested source edit is published."""
import hashlib
from pathlib import Path
import sys

path = Path(sys.argv[1]) / 'tests/test_bend_session_move_cache.py'
text = path.read_text()
assert hashlib.sha256(text.encode()).hexdigest() == 'c061bf151204771b18b7351ae63523d17ec4e3e9aa237668f5b8db1e48453664'
old = 'with pytest.raises(ValueError):'
assert text.count(old) == 1
text = text.replace(old, "with pytest.raises(ValueError, match=r'cache|nondecimal'):")
old = "set(('', '-1', '+1', ' 6', '6 ', '17', '4294967296', '١'))"
assert text.count(old) == 1
text = text.replace(old, "{'', '-1', '+1', ' 6', '6 ', '17', '4294967296', '١'}")
path.write_text(text)
