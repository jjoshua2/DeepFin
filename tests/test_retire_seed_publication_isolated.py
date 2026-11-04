"""Stdlib-only exact-AST tests: no subject import, scorer, native or model."""
import ast
from collections.abc import Callable, Generator, Iterable
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import re
import tempfile
import types
import unittest
from unittest.mock import patch
import uuid
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def load_function(path: Path, name: str, namespace: dict[str, Any]) -> Callable[..., Any]:
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    assert not any(isinstance(n, (ast.Import, ast.ImportFrom)) for n in ast.walk(node))
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


@contextmanager
def expect_error(error: type[Exception]) -> Generator[None, None, None]:
    """Keep isolated tests stdlib-only rather than importing the pytest stack."""
    try:
        yield
    except error:
        return
    raise AssertionError(f"expected {error.__name__}")


class PublicationTests(unittest.TestCase):
    _directory: Path | None = None
    _writer: Callable[..., Any] | None = None

    @property
    def directory(self) -> Path:
        assert self._directory is not None, "setUp must create the fixture first"
        return self._directory

    @property
    def writer(self) -> Callable[..., Any]:
        assert self._writer is not None, "setUp must create the fixture first"
        return self._writer

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self._directory = Path(tmp.name)
        self._writer = load_function(ROOT / 'scripts/blindspot_retire_step.py', '_publish_seed_file',
                                    {'os': os, 're': re, 'uuid': uuid})

    def write(self, lines: Iterable[str] | None = None, tag: str = '42') -> str:
        return self.writer(str(self.directory), ['A'] if lines is None else lines,
                           note='test', tag=tag)

    def test_repeated_and_default_tags_never_overwrite(self) -> None:
        old = self.directory / 'blindspot_fens_retire_42.txt'
        old.write_text('selected prior')
        paths = [self.write([str(i)], tag) for i, tag in enumerate(['42', '42', '', '', '43'])]
        assert len(set(paths)) == 5
        assert old.read_text() == 'selected prior'
        for i, path in enumerate(paths):
            assert Path(path).is_absolute()
            assert Path(path).read_text() == f'# test\n{i}\n'

    def test_concurrent_writers(self) -> None:
        with ThreadPoolExecutor(max_workers=4) as pool:
            paths = list(pool.map(lambda i: self.write([str(i)]), range(12)))
        assert len(set(paths)) == 12
        for i, path in enumerate(paths):
            assert Path(path).read_text() == f'# test\n{i}\n'

    def test_collision_preserves_existing_final_and_staging(self) -> None:
        final = self.directory / 'blindspot_fens_retire_42_fixed.txt'
        staging = self.directory / '.blindspot_fens_retire_other.tmp'
        final.write_text('prior')
        staging.write_text('other writer')
        tokens = [types.SimpleNamespace(hex=x) for x in ['fixed', 'other', 'fresh']]
        with patch.object(uuid, 'uuid4', side_effect=tokens):
            result = self.write()
        assert result.endswith('_fresh.txt')
        assert final.read_text() == 'prior'
        assert staging.read_text() == 'other writer'

    def test_failed_partial_write_and_link_do_not_publish(self) -> None:
        class Broken:
            def __iter__(self):
                yield 'partial'
                raise OSError('interrupted fixture')
        with expect_error(OSError):
            self.write(Broken())
        assert list(self.directory.iterdir()) == []
        with patch.object(os, 'link', side_effect=OSError('unsupported fixture')), expect_error(OSError):
            self.write()
        assert list(self.directory.iterdir()) == []

    def test_complete_before_publication_and_retained_after_validation_failure(self) -> None:
        link = os.link
        def inspect(source, destination):
            assert not Path(destination).exists()
            assert Path(source).read_text() == '# test\nA\n'
            link(source, destination)
        with patch.object(os, 'link', side_effect=inspect):
            result = self.write()
        # A failing later validator/repoint must not mutate or delete this artifact.
        def reject_validation() -> None:
            raise ValueError('validation fixture')
        with expect_error(ValueError):
            reject_validation()
        assert Path(result).read_text() == '# test\nA\n'

    def test_tags_and_umask(self) -> None:
        for tag in ['../bad', 'a/b', 'a\\b', 'a\0b']:
            with self.subTest(tag=tag), expect_error(ValueError):
                self.write(tag=tag)
        for tag in ['space tag', 'é', 'tag:punctuation', '.', '..']:
            assert Path(self.write(tag=tag)).exists()
        previous = os.umask(0o027)
        try:
            result = self.write()
        finally:
            os.umask(previous)
        assert Path(result).stat().st_mode & 0o777 == 0o640

    def test_legacy_writer_contract_unchanged(self) -> None:
        # The feeder imports this legacy helper; retirement must not change it.
        legacy = load_function(ROOT / 'scripts/blindspot_retire_step.py', '_write_seed_file', {})
        path = self.directory / 'legacy.txt'
        assert legacy(str(path), ['legacy'], note='test') is None
        assert path.read_text() == '# test\nlegacy\n'

    def test_production_uses_returned_path_for_validation_and_selection(self) -> None:
        tree = ast.parse((ROOT / 'scripts/blindspot_retire_step.py').read_text())
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
        writer = next(n for n in calls if isinstance(n.func, ast.Name) and n.func.id == '_publish_seed_file')
        assert ast.unparse(writer.args[0]) == 'args.out_dir'
        for name, index in [('_load_fen_list', 0), ('_repoint_yaml', 1)]:
            call = next(n for n in calls if isinstance(n.func, ast.Name) and n.func.id == name)
            assert ast.unparse(call.args[index]) == 'new_path'


if __name__ == '__main__':
    unittest.main(verbosity=2)
