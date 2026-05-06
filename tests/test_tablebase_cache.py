from __future__ import annotations

import chess_anti_engine.tablebase as tablebase


class _FakeTablebase:
    instances: list["_FakeTablebase"] = []

    def __init__(self) -> None:
        self.dirs: list[str] = []
        self.closed = False
        _FakeTablebase.instances.append(self)

    def add_directory(self, path: str) -> None:
        if path.startswith("missing"):
            raise FileNotFoundError(path)
        self.dirs.append(path)

    def close(self) -> None:
        self.closed = True


def test_get_tablebase_keeps_path_swaps_from_closing_active_handles(monkeypatch):
    _FakeTablebase.instances.clear()
    monkeypatch.setattr(tablebase, "_tablebases", {})
    monkeypatch.setattr(tablebase.chess.syzygy, "Tablebase", _FakeTablebase)

    first = tablebase.get_tablebase("tb_a")
    second = tablebase.get_tablebase("tb_b")

    assert first is not None
    assert second is not None
    assert first is not second
    assert not first.closed
    assert not second.closed
    assert tablebase.get_tablebase("tb_a") is first
    assert tablebase.get_tablebase("") is None
    assert not first.closed
    assert not second.closed


def test_get_tablebase_closes_failed_new_handle(monkeypatch):
    _FakeTablebase.instances.clear()
    monkeypatch.setattr(tablebase, "_tablebases", {})
    monkeypatch.setattr(tablebase.chess.syzygy, "Tablebase", _FakeTablebase)

    assert tablebase.get_tablebase("missing_a") is None
    assert len(_FakeTablebase.instances) == 1
    assert _FakeTablebase.instances[0].closed
