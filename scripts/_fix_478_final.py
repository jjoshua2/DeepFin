#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
READOUT = ROOT / "scripts/nnue_gumbel_readout.py"
TEST = ROOT / "tests/test_nnue_gumbel_readout.py"
DOC = ROOT / "docs/nnue_gumbel_readout.md"


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one match, got {count}: {old[:120]!r}")
    path.write_text(text.replace(old, new, 1))


def replace_all_exact(path: Path, old: str, new: str, expected: int) -> None:
    text = path.read_text()
    count = text.count(old)
    if count != expected:
        raise RuntimeError(f"{path}: expected {expected} matches, got {count}: {old[:120]!r}")
    path.write_text(text.replace(old, new))


# ctypes + struct are used to fingerprint the already-loaded ELF image, rather
# than reopening a pathname that may have been atomically replaced after import.
replace_once(
    READOUT,
    "import argparse\nimport hashlib\n",
    "import argparse\nimport ctypes\nimport hashlib\n",
)
replace_once(
    READOUT,
    "import subprocess\nimport sys\n",
    "import struct\nimport subprocess\nimport sys\n",
)

# Empty search traces are not evidence. This closes both a direct helper call
# and --max-plies <= 0 before either can produce SHA256(empty) as a plausible
# strengthened-oracle component.
replace_once(
    READOUT,
    '''def search_output_digest(rows: list[Any]) -> str:
    """Digest the exact improved-policy/search output for every stored ply."""
    h = hashlib.sha256()
''',
    '''def search_output_digest(rows: list[Any]) -> str:
    """Digest the exact improved-policy/search output for every stored ply."""
    if not rows:
        raise ValueError(
            "cannot hash an empty search-output trace; no improved policy was observed",
        )
    h = hashlib.sha256()
''',
)
replace_once(
    READOUT,
    '''def run_cell(cfg: RunConfig) -> dict[str, Any]:
    """One arm, one repeat."""
    if cfg.games <= 0 or cfg.workers <= 0:
        raise ValueError("games and workers must be positive")
''',
    '''def run_cell(cfg: RunConfig) -> dict[str, Any]:
    """One arm, one repeat."""
    if cfg.games <= 0 or cfg.workers <= 0 or cfg.max_plies <= 0:
        raise ValueError("games, workers, and max_plies must be positive")
''',
)
replace_once(
    READOUT,
    '''def plan_from_args(args: argparse.Namespace) -> ReadoutPlan:
    arms = list(dict.fromkeys(args.arm))
    _validate_matrix_knobs(args, arms)
''',
    '''def plan_from_args(args: argparse.Namespace) -> ReadoutPlan:
    arms = list(dict.fromkeys(args.arm))
    if int(args.max_plies) <= 0:
        raise ValueError("--max-plies must be positive")
    _validate_matrix_knobs(args, arms)
''',
)

# Replace pathname-only native provenance with:
# - a stable pathname file stamp, checked again after the worker search; and
# - GNU build-id read directly from the already mapped PT_NOTE segment through
#   dl_iterate_phdr. The latter is the authoritative identity of code executing
#   in this process even if module.__file__ was atomically replaced after import.
start = READOUT.read_text().index("def _sha256_file(path: Path) -> str:\n")
end = READOUT.read_text().index("\n\n#: This process's niceness", start)
new_identity = r'''def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


FileStamp = tuple[str, int, int, int, int, int]


def _file_stamp(path: Path) -> FileStamp:
    """Hash + inode/timestamp stamp, refusing a file that changes while read."""
    before = path.stat()
    digest = _sha256_file(path)
    after = path.stat()
    before_meta = (
        int(before.st_dev), int(before.st_ino), int(before.st_size),
        int(before.st_mtime_ns), int(before.st_ctime_ns),
    )
    after_meta = (
        int(after.st_dev), int(after.st_ino), int(after.st_size),
        int(after.st_mtime_ns), int(after.st_ctime_ns),
    )
    if before_meta != after_meta:
        raise RuntimeError(f"file changed while it was being fingerprinted: {path}")
    return (digest, *after_meta)


def _assert_file_unchanged(label: str, path: Path, expected: FileStamp) -> None:
    current = _file_stamp(path)
    if current != expected:
        raise RuntimeError(
            f"{label} changed while this worker was running: "
            f"before={expected}, after={current}"
        )


class _Elf64Phdr(ctypes.Structure):
    _fields_ = [
        ("p_type", ctypes.c_uint32),
        ("p_flags", ctypes.c_uint32),
        ("p_offset", ctypes.c_uint64),
        ("p_vaddr", ctypes.c_uint64),
        ("p_paddr", ctypes.c_uint64),
        ("p_filesz", ctypes.c_uint64),
        ("p_memsz", ctypes.c_uint64),
        ("p_align", ctypes.c_uint64),
    ]


class _DlPhdrInfo(ctypes.Structure):
    _fields_ = [
        ("dlpi_addr", ctypes.c_uint64),
        ("dlpi_name", ctypes.c_char_p),
        ("dlpi_phdr", ctypes.POINTER(_Elf64Phdr)),
        ("dlpi_phnum", ctypes.c_uint16),
    ]


_PT_NOTE = 4
_NT_GNU_BUILD_ID = 3


def _loaded_elf_build_id(path: Path) -> str:
    """GNU build-id from the mapped ELF image, not the current pathname bytes.

    The readout's production host and CI are 64-bit little-endian Linux. Failing
    closed elsewhere is preferable to publishing a pathname hash as proof of an
    image that may already have been replaced on disk.
    """
    if not sys.platform.startswith("linux") or ctypes.sizeof(ctypes.c_void_p) != 8:
        raise RuntimeError(
            "loaded native-image provenance requires 64-bit Linux dl_iterate_phdr",
        )
    target = str(path.resolve())
    found: list[str] = []
    callback_errors: list[str] = []
    callback_type = ctypes.CFUNCTYPE(
        ctypes.c_int,
        ctypes.POINTER(_DlPhdrInfo),
        ctypes.c_size_t,
        ctypes.c_void_p,
    )

    def visit(info_ptr: Any, _size: int, _data: Any) -> int:
        try:
            info = info_ptr.contents
            raw_name = info.dlpi_name
            if not raw_name:
                return 0
            loaded_name = os.path.realpath(os.fsdecode(raw_name))
            if loaded_name != target:
                return 0
            base = int(info.dlpi_addr)
            for i in range(int(info.dlpi_phnum)):
                ph = info.dlpi_phdr[i]
                if int(ph.p_type) != _PT_NOTE:
                    continue
                blob = ctypes.string_at(base + int(ph.p_vaddr), int(ph.p_memsz))
                offset = 0
                while offset + 12 <= len(blob):
                    namesz, descsz, note_type = struct.unpack_from("=III", blob, offset)
                    offset += 12
                    if namesz > len(blob) - offset:
                        break
                    name = blob[offset:offset + namesz]
                    offset += (namesz + 3) & ~3
                    if descsz > len(blob) - offset:
                        break
                    desc = blob[offset:offset + descsz]
                    offset += (descsz + 3) & ~3
                    if (
                        note_type == _NT_GNU_BUILD_ID
                        and name.rstrip(b"\0") == b"GNU"
                        and desc
                    ):
                        found.append(desc.hex())
            return 1
        except Exception as exc:  # callback exceptions cannot cross ctypes safely
            callback_errors.append(repr(exc))
            return 1

    callback = callback_type(visit)
    process: Any = ctypes.CDLL(None)
    iterate: Any = process.dl_iterate_phdr
    iterate.argtypes = [callback_type, ctypes.c_void_p]
    iterate.restype = ctypes.c_int
    iterate(callback, None)
    if callback_errors:
        raise RuntimeError(
            f"failed reading loaded ELF build-id for {target}: {callback_errors[0]}",
        )
    unique = sorted(set(found))
    if len(unique) != 1:
        raise RuntimeError(
            f"expected exactly one GNU build-id in loaded image {target}, got {unique}",
        )
    return unique[0]


def _module_identity(module: Any) -> tuple[str, str, str, FileStamp]:
    raw = getattr(module, "__file__", None)
    if not raw:
        raise RuntimeError(f"native module {module!r} has no __file__; cannot prove binary identity")
    path = Path(str(raw)).resolve()
    stamp = _file_stamp(path)
    loaded_build_id = _loaded_elf_build_id(path)
    return str(path), stamp[0], loaded_build_id, stamp


def _git_provenance() -> dict[str, object]:
    """Tracked source identity, including the actual dirty-tree contents."""
    repo = Path(__file__).resolve().parents[1]
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo, text=True, stderr=subprocess.DEVNULL,
        ).strip()
        tracked_diff = subprocess.check_output(
            ["git", "diff", "--binary", "HEAD", "--"],
            cwd=repo, stderr=subprocess.DEVNULL,
        )
        return {
            "git_head": head,
            "git_tracked_dirty": bool(tracked_diff),
            "git_tracked_diff_sha256": hashlib.sha256(tracked_diff).hexdigest(),
        }
    except (OSError, subprocess.CalledProcessError):
        return {
            "git_head": None,
            "git_tracked_dirty": None,
            "git_tracked_diff_sha256": None,
        }
'''
text = READOUT.read_text()
READOUT.write_text(text[:start] + new_identity + text[end:])

# Worker report carries authoritative loaded build-ids alongside secondary
# pathname hashes. The latter are still useful artifact checks; they are no
# longer described as proof of which image executed.
replace_once(
    READOUT,
    '''    nnue_ext_path: str = ""
    nnue_ext_sha256: str = ""
    mcts_ext_path: str = ""
    mcts_ext_sha256: str = ""
''',
    '''    nnue_ext_path: str = ""
    nnue_ext_sha256: str = ""
    nnue_ext_loaded_build_id: str = ""
    mcts_ext_path: str = ""
    mcts_ext_sha256: str = ""
    mcts_ext_loaded_build_id: str = ""
''',
)

# Worker pack: hash+inode stamp before open, after open, and after all games.
# Final check happens after the search timer is frozen. Do the same for module
# pathname files; their mapped build-id remains the primary identity.
replace_once(
    READOUT,
    '''    worker_pack_sha_before = _sha256_file(cfg.pack)
    source = ReadoutArmSource(
''',
    '''    worker_pack_stamp_before = _file_stamp(cfg.pack)
    worker_pack_sha_before = worker_pack_stamp_before[0]
    source = ReadoutArmSource(
''',
)
replace_once(
    READOUT,
    '''    worker_pack_sha_after = _sha256_file(cfg.pack)
    if worker_pack_sha_after != worker_pack_sha_before:
        raise RuntimeError(
            "NNUE pack changed while this worker was opening it: "
            f"{worker_pack_sha_before} -> {worker_pack_sha_after}"
        )
''',
    '''    _assert_file_unchanged(
        "NNUE pack while opening", cfg.pack, worker_pack_stamp_before,
    )
''',
)
replace_once(
    READOUT,
    '''    from chess_anti_engine.mcts import _mcts_tree as mcts_ext
    nnue_ext_path, nnue_ext_sha = _module_identity(_load_ext())
    mcts_ext_path, mcts_ext_sha = _module_identity(mcts_ext)
''',
    '''    from chess_anti_engine.mcts import _mcts_tree as mcts_ext
    (
        nnue_ext_path, nnue_ext_sha, nnue_ext_build_id, nnue_ext_stamp,
    ) = _module_identity(_load_ext())
    (
        mcts_ext_path, mcts_ext_sha, mcts_ext_build_id, mcts_ext_stamp,
    ) = _module_identity(mcts_ext)
''',
)
replace_once(
    READOUT,
    '''    elapsed = time.perf_counter() - started
    return WorkerResult(
''',
    '''    elapsed = time.perf_counter() - started
    # Integrity I/O is deliberately outside `elapsed`: it gates the measurement
    # and must not become part of the arm-throughput comparison.
    _assert_file_unchanged("NNUE pack", cfg.pack, worker_pack_stamp_before)
    _assert_file_unchanged(
        "_nnue_ext pathname", Path(nnue_ext_path), nnue_ext_stamp,
    )
    _assert_file_unchanged(
        "_mcts_tree pathname", Path(mcts_ext_path), mcts_ext_stamp,
    )
    return WorkerResult(
''',
)
replace_once(
    READOUT,
    '''        nnue_ext_path=nnue_ext_path,
        nnue_ext_sha256=nnue_ext_sha,
        mcts_ext_path=mcts_ext_path,
        mcts_ext_sha256=mcts_ext_sha,
''',
    '''        nnue_ext_path=nnue_ext_path,
        nnue_ext_sha256=nnue_ext_sha,
        nnue_ext_loaded_build_id=nnue_ext_build_id,
        mcts_ext_path=mcts_ext_path,
        mcts_ext_sha256=mcts_ext_sha,
        mcts_ext_loaded_build_id=mcts_ext_build_id,
''',
)

# Aggregate loaded build-id, and make it the authoritative native-image
# comparability gate in addition to the conservative current-path SHA gate.
replace_once(
    READOUT,
    '''    nnue_binary_paths = sorted({
        str(w["nnue_ext_path"]) for c in every for w in c["workers_detail"]
    })
    mcts_binary_paths = sorted({
        str(w["mcts_ext_path"]) for c in every for w in c["workers_detail"]
    })
''',
    '''    nnue_binary_paths = sorted({
        str(w["nnue_ext_path"]) for c in every for w in c["workers_detail"]
    })
    mcts_binary_paths = sorted({
        str(w["mcts_ext_path"]) for c in every for w in c["workers_detail"]
    })
    nnue_loaded_build_ids = sorted({
        str(w["nnue_ext_loaded_build_id"])
        for c in every for w in c["workers_detail"]
    })
    mcts_loaded_build_ids = sorted({
        str(w["mcts_ext_loaded_build_id"])
        for c in every for w in c["workers_detail"]
    })
''',
)
replace_once(
    READOUT,
    '''    if len(nnue_binary_shas) != 1 or len(mcts_binary_shas) != 1:
        reasons.append(
            "workers loaded different native binaries: "
            f"_nnue_ext={nnue_binary_shas}, _mcts_tree={mcts_binary_shas}"
        )
''',
    '''    if len(nnue_loaded_build_ids) != 1 or len(mcts_loaded_build_ids) != 1:
        reasons.append(
            "workers executed different loaded native images: "
            f"_nnue_ext={nnue_loaded_build_ids}, "
            f"_mcts_tree={mcts_loaded_build_ids}"
        )
    if len(nnue_binary_shas) != 1 or len(mcts_binary_shas) != 1:
        reasons.append(
            "workers observed different native-module pathname files: "
            f"_nnue_ext={nnue_binary_shas}, _mcts_tree={mcts_binary_shas}"
        )
''',
)
replace_once(
    READOUT,
    '''            "nnue_ext_path": nnue_binary_paths[0] if len(nnue_binary_paths) == 1 else nnue_binary_paths,
            "nnue_ext_sha256": nnue_binary_shas[0] if len(nnue_binary_shas) == 1 else nnue_binary_shas,
            "mcts_ext_path": mcts_binary_paths[0] if len(mcts_binary_paths) == 1 else mcts_binary_paths,
            "mcts_ext_sha256": mcts_binary_shas[0] if len(mcts_binary_shas) == 1 else mcts_binary_shas,
''',
    '''            "nnue_ext_path": nnue_binary_paths[0] if len(nnue_binary_paths) == 1 else nnue_binary_paths,
            "nnue_ext_sha256": nnue_binary_shas[0] if len(nnue_binary_shas) == 1 else nnue_binary_shas,
            "nnue_ext_loaded_build_id": (
                nnue_loaded_build_ids[0]
                if len(nnue_loaded_build_ids) == 1 else nnue_loaded_build_ids
            ),
            "mcts_ext_path": mcts_binary_paths[0] if len(mcts_binary_paths) == 1 else mcts_binary_paths,
            "mcts_ext_sha256": mcts_binary_shas[0] if len(mcts_binary_shas) == 1 else mcts_binary_shas,
            "mcts_ext_loaded_build_id": (
                mcts_loaded_build_ids[0]
                if len(mcts_loaded_build_ids) == 1 else mcts_loaded_build_ids
            ),
''',
)

# Publish the end diff fingerprint too; start dict is expanded wholesale.
replace_once(
    READOUT,
    '''            "git_end_head": git_meta_end["git_head"],
            "git_end_tracked_dirty": git_meta_end["git_tracked_dirty"],
            "git_changed_during_run": git_changed_during_run,
''',
    '''            "git_end_head": git_meta_end["git_head"],
            "git_end_tracked_dirty": git_meta_end["git_tracked_dirty"],
            "git_end_tracked_diff_sha256": git_meta_end["git_tracked_diff_sha256"],
            "git_changed_during_run": git_changed_during_run,
''',
)

# Test fakes that stand in for worker-detail provenance need the mapped-image ids.
replace_all_exact(
    TEST,
    '''                "nnue_ext_path": "/tmp/_nnue_ext.so", "nnue_ext_sha256": "b" * 64,
                "mcts_ext_path": "/tmp/_mcts_tree.so", "mcts_ext_sha256": "c" * 64,
''',
    '''                "nnue_ext_path": "/tmp/_nnue_ext.so", "nnue_ext_sha256": "b" * 64,
                "nnue_ext_loaded_build_id": "1" * 40,
                "mcts_ext_path": "/tmp/_mcts_tree.so", "mcts_ext_sha256": "c" * 64,
                "mcts_ext_loaded_build_id": "2" * 40,
''',
    expected=3,
)

# Source-provenance regression now pins the subtle dirty->dirty mutation: same
# HEAD, dirty stays true, but the tracked content digest changes.
replace_once(
    TEST,
    '''    snapshots = iter((
        {"git_head": "start", "git_tracked_dirty": False},
        {"git_head": "end", "git_tracked_dirty": False},
    ))
''',
    '''    snapshots = iter((
        {
            "git_head": "same", "git_tracked_dirty": True,
            "git_tracked_diff_sha256": "a" * 64,
        },
        {
            "git_head": "same", "git_tracked_dirty": True,
            "git_tracked_diff_sha256": "b" * 64,
        },
    ))
''',
)
replace_once(
    TEST,
    '''    assert report["provenance"]["git_head"] == "start"
    assert report["provenance"]["git_end_head"] == "end"
    assert report["provenance"]["git_changed_during_run"] is True
''',
    '''    assert report["provenance"]["git_head"] == "same"
    assert report["provenance"]["git_end_head"] == "same"
    assert report["provenance"]["git_tracked_dirty"] is True
    assert report["provenance"]["git_end_tracked_dirty"] is True
    assert report["provenance"]["git_tracked_diff_sha256"] == "a" * 64
    assert report["provenance"]["git_end_tracked_diff_sha256"] == "b" * 64
    assert report["provenance"]["git_changed_during_run"] is True
''',
)

# Replace pathname-only fake-module identity test with a live-image proof. CI
# builds both modules, and the helper must obtain their mapped GNU build-id.
replace_once(
    TEST,
    '''def test_native_module_identity_hashes_the_actual_loaded_file(tmp_path: Path) -> None:
    binary = tmp_path / "fake.so"
    binary.write_bytes(b"one build")
    module = argparse.Namespace(__file__=str(binary))
    path, digest = readout._module_identity(module)
    assert path == str(binary.resolve())
    assert digest == readout._sha256_file(binary)
    binary.write_bytes(b"different build")
    assert readout._sha256_file(binary) != digest
''',
    '''def test_native_module_identity_reads_the_loaded_elf_build_id() -> None:
    from chess_anti_engine.mcts import _mcts_tree

    for module in (_nnue_ext, _mcts_tree):
        path, digest, build_id, stamp = readout._module_identity(module)
        assert digest == readout._sha256_file(Path(path))
        assert stamp[0] == digest
        assert len(build_id) >= 16
        int(build_id, 16)
''',
)

# The intentionally dynamic bank-writer regression is a behavioral fake, not a
# type claim about private concrete fields. Mark just those two objects Any so
# basedpyright does not require StringIO/CBoard where the test intentionally
# supplies duck-typed stand-ins.
replace_once(
    TEST,
    '''    sink = io.StringIO()
    source = object.__new__(gen.NnueArmValueSource)
''',
    '''    sink = io.StringIO()
    source: Any = object.__new__(gen.NnueArmValueSource)
''',
)
replace_once(
    TEST,
    '''    board = argparse.Namespace(
        fen=lambda: "8/8/8/8/8/8/8/K6k w - - 7 9",
''',
    '''    board: Any = argparse.Namespace(
        fen=lambda: "8/8/8/8/8/8/8/K6k w - - 7 9",
''',
)

# Pin non-vacuity and file-stability helpers.
insert_before = '''def test_search_output_digest_catches_a_target_change_that_game_digest_cannot() -> None:
'''
extra = '''def test_search_output_digest_refuses_an_empty_trace() -> None:
    with pytest.raises(ValueError, match="empty search-output trace"):
        readout.search_output_digest([])


def test_file_stamp_detects_a_changed_file(tmp_path: Path) -> None:
    path = tmp_path / "stable.bin"
    path.write_bytes(b"before")
    stamp = readout._file_stamp(path)
    path.write_bytes(b"after")
    with pytest.raises(RuntimeError, match="changed while this worker was running"):
        readout._assert_file_unchanged("test file", path, stamp)


'''
text = TEST.read_text()
if text.count(insert_before) != 1:
    raise RuntimeError("search-output insertion marker missing or ambiguous")
TEST.write_text(text.replace(insert_before, extra + insert_before, 1))

# Pin the CLI-side max-plies gate without entering a worker.
insert_before = '''def test_multi_arm_matrix_scopes_explicit_knobs_to_the_consuming_cells(tmp_path: Path) -> None:
'''
extra = '''def test_plan_refuses_nonpositive_max_plies(tmp_path: Path) -> None:
    args = readout.build_parser().parse_args([
        "--arm", readout.ARM_QSEARCH,
        "--nnue-pack", str(tmp_path / "pack"),
        "--max-plies", "0",
    ])
    with pytest.raises(ValueError, match="max-plies must be positive"):
        readout.plan_from_args(args)


'''
text = TEST.read_text()
if text.count(insert_before) != 1:
    raise RuntimeError("matrix-test insertion marker missing or ambiguous")
TEST.write_text(text.replace(insert_before, extra + insert_before, 1))

# Documentation: loaded build-id is authoritative; Git diff digest is endpoint
# provenance; end-to-end banks are trace diagnostics, NOT the later paired SF
# quality population.
replace_once(
    DOC,
    '''`banking` · `dag_reset` · `repeats` · `arms` · `python` · the Git HEAD/dirty
flag · and the actual loaded `_nnue_ext` / `_mcts_tree` paths and SHA-256 hashes.
''',
    '''`banking` · `dag_reset` · `repeats` · `arms` · `python` · the Git HEAD,
tracked-diff SHA-256 and dirty flag at **both** matrix endpoints · the native
module pathname SHA-256 snapshots · and, authoritatively, the GNU build-ids read
from the already mapped `_nnue_ext` / `_mcts_tree` ELF images.
''',
)
replace_once(
    DOC,
    '''7. **Search shape/coverage**: root-budget and termination data in each worker's
   detail record, plus the banked leaf population for the standardized deep-SF
   quality readout.

The harness does not claim that similarity to qsearch is strength. The deciding
quality comparison remains the standardized deep-Stockfish target-quality
readout described by the AZ-purity framework. The point of this script is to
produce the **production-shaped leaf population and raw observations** needed to
run that decision honestly.
''',
    '''7. **Search shape/coverage**: root-budget and termination data in each worker's
   detail record, plus the banked **end-to-end trace population** for diagnosing
   how each arm changes the leaves production Gumbel actually visits.

The harness does not claim that similarity to qsearch is strength, and these
end-to-end banks are **not** the input population for a paired evaluator-quality
verdict: each arm helped choose its own later leaves. The deciding standardized
deep-Stockfish evaluator-quality comparison must be a separate frozen-driver /
shadow-arm experiment in which all candidate evaluators score the same positions
without feeding values back into MCTSTree. This readout measures production
throughput, reuse, search shape, and endogenous trace distributions; it does not
silently turn those endogenous populations into paired quality evidence.
''',
)

print("final #478 integrity staging transforms applied")
