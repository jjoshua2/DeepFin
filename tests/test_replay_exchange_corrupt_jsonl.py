"""Regression: corrupt/partially-flushed Ray result.json files must not crash
the JSONL readers. Codex adversarial review found that a torn UTF-8 tail
would escape the inner try/except because decode errors surface during
`for ln in f` iteration, not during json.loads — and the outer except
was narrowed to OSError only.
"""
from __future__ import annotations

from pathlib import Path

from chess_anti_engine.tune.replay_exchange import _read_jsonl_rows, _read_last_jsonl_row


def test_read_last_survives_truncated_utf8_tail(tmp_path: Path) -> None:
    p = tmp_path / "result.json"
    good = b'{"iter": 1, "wdl_regret": 0.05}\n'
    # Last 2 bytes of a multi-byte codepoint, truncated mid-sequence.
    # Would raise UnicodeDecodeError during text-mode line iteration
    # pre-fix; post-fix (errors="replace") becomes U+FFFD which fails
    # json.loads and gets skipped.
    p.write_bytes(good + good + b"\xe2\x80")

    row = _read_last_jsonl_row(p)
    assert row is not None
    assert row["iter"] == 1


def test_read_all_keeps_good_rows_past_bad_utf8(tmp_path: Path) -> None:
    p = tmp_path / "result.json"
    p.write_bytes(
        b'{"iter": 1}\n'
        b'\xff\xfe not valid utf8 here\n'
        b'{"iter": 2}\n'
        b'\xe2\x80\n'  # truncated codepoint
        b'{"iter": 3}\n'
    )

    rows = _read_jsonl_rows(p)
    iters = [r["iter"] for r in rows]
    assert iters == [1, 2, 3]


def test_readers_tolerate_nonexistent_file(tmp_path: Path) -> None:
    missing = tmp_path / "nonexistent.json"
    assert _read_last_jsonl_row(missing) is None
    assert _read_jsonl_rows(missing) == []
