"""Reviewed corrections to the hash-verified qualification payload, not product code."""
import hashlib
from pathlib import Path


def replace_once(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


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
        text = replace_once(text, old, new)
    assert hashlib.sha256(text.encode()).hexdigest() == '5892fc97b696f9cc4c3e719a7298a888be9c1ca6364aaacf9ee093994eff1ab6'
    data['files'][name] = text
    name = 'native/bend_engine/collections_probe/run_probe.py'
    text = replace_once(data['files'][name], 'import check_compiler', 'import compiler_digest')
    check = '''
def check_compiler(source: Path) -> None:
    """Keep the qualified 2.0.21+U64 pin; do not alter legacy probe manifests."""
    paths = [source / 'bend2' / name for name in ('base.bend', 'bend.ts', 'comp.ts', 'main.ts')]
    paths.extend(p for p in (source / 'bend2/effs').rglob('*') if p.is_file())
    if source.is_symlink() or (source / 'bend2').is_symlink() or (source / 'bend2/effs').is_symlink():
        raise ValueError('compiler root must contain regular sources')
    if len(paths) != 84 or any(p.is_symlink() or not p.is_file() for p in paths):
        raise ValueError('compiler input inventory differs from the 84-file pin')
    if compiler_digest(source) != 'd9550e30dbf17f12013aa5db89b27cf6724957fe68213c9c7e409e99f1405ef4':
        raise ValueError('compiler sources differ from the qualified 2.0.21+U64 pin')


def expected_trace()'''
    text = replace_once(text, '\ndef expected_trace()', check)
    text = replace_once(text, '    check_compiler(args.compiler_root)\n    report:', '    report:')
    text = replace_once(text, '    try:\n        with tempfile.', '    try:\n        check_compiler(args.compiler_root)\n        with tempfile.')
    text = replace_once(text, "            report['status'] = 'passed'\n    finally:", "            report['status'] = 'passed'\n    except Exception as error:\n        report['error'] = str(error)\n        raise\n    finally:")
    assert hashlib.sha256(text.encode()).hexdigest() == 'e2185dbd72825bb022be077f5b1cb9a0cc9a6365aa3d41520265ee2014342d92'
    data['files'][name] = text
    text = Path('.collections-session-check.py').read_text()
    assert hashlib.sha256(text.encode()).hexdigest() == '45f869c97c6af7827d27af6149d4a098bf1802552f5d20f05a5bedf3f6a33820'
    data['files']['native/bend_engine/collections_probe/session_check.py'] = text
    name = 'native/bend_engine/collections_probe/main.bend'
    text = replace_once(data['files'][name], 'def emit(q:', '''def emit_word(q: Q.Queue<&1, Array<U64>>, r: Array<U64> & U64) -> IO(Q.Queue<&1, Array<U64>>):
  (arr, +x) = r
  do IO<Q.Queue<&1, Array<U64>>>:
    IO.print("value " ++ U32.show(U64.high(x)) ++ " " ++ U32.show(U64.low(x)))
    return q

def emit(q:''')
    text = replace_once(text, '''      (arr, +x) = Array.get(U64, arr, 0)
      do IO<Q.Queue<&1, Array<U64>>>:
        IO.print("value " ++ U32.show(U64.high(x)) ++ " " ++ U32.show(U64.low(x)))
        return q''', '''      emit_word(q, Array.get(U64, arr, 0))''')
    assert hashlib.sha256(text.encode()).hexdigest() == '7acf9373ab842ffa4f148ef5b96c09a18e555700ec4ff3c33ce4a106ade53037'
    data['files'][name] = text
    name = 'native/bend_engine/collections_probe/benchmark.bend'
    text = replace_once(data['files'][name], '''def queue_done(size: Nat, sample: Nat, started: Nat, r: Q.Queue<&2, U32> & U32) -> IO(Unit):
  (q, sum) = r
  (q, length) = Q.length(&2, U32, q)''', '''def queue_sized(size: Nat, sample: Nat, started: Nat, sum: U32, r: Q.Queue<&2, U32> & Nat) -> IO(Unit):
  (q, length) = r''')
    text = replace_once(text, 'def list_run(size:', '''def queue_done(size: Nat, sample: Nat, started: Nat, r: Q.Queue<&2, U32> & U32) -> IO(Unit):
  (q, sum) = r
  queue_sized(size, sample, started, sum, Q.length(&2, U32, q))

def list_run(size:''')
    text = replace_once(text, 'def samples(n: Nat, first: Bool,', 'def samples(n: Nat, +first: Bool,')
    assert hashlib.sha256(text.encode()).hexdigest() == '21675aec3c9d87580f56e59c836994d5626cab35f1a98c4cf02ea894e6fc7082'
    data['files'][name] = text
    name = 'docs/experiments/2026-09-23-bend-collections-screen.md'
    data['files'][name] += '\n\n### Preserved setup failures\n\nRun 35922817981 stopped at the first static gate: four Ruff findings in the new test file (one list-construction style issue and three broad exception assertions). Whole-repository type checking reported zero errors/warnings and Vulture had no findings. Hosted Python and native tests had not run. The corrected candidate uses list.extend and checks exception messages; no gate or expectation was suppressed.\n\nRun 35923458566 passed focused Ruff/Basedpyright and all 66 Python cases, then stopped before compilation: the new probe mistakenly imported a legacy compiler guard. The existing bitboard/session manifest pins 57bc84ed with fingerprint c178489e, whereas the CI installer and this preregistration pin aaeb9bc9 with fingerprint d9550e30. Neither manifest nor compiler source was changed. The new screen now verifies the intended 84-file fingerprint itself, and session_check.py builds the same existing session/CBoard sources and invokes the unchanged Python-chess/session/root-advance oracles under that verified compiler. It does not monkeypatch the legacy guard or call its build function with a mismatched compiler. These setup failures are not native coverage. The original 17 pure Python tests also passed locally.\n\nRun 35923899467 again passed the 66 Python cases and focused static checks, then the first Bend parser rejected computed-tuple destructuring in the owning trace emitter. The emitter and the analogous benchmark length read now pass the computed tuple to dedicated parameter-destructuring helpers. No compiler, behavior expectation or test fixture changed; no native runtime was executed in that failed attempt.\n\nRun 35924258475 passed the three-mode owning FIFO, two-mode traversal and executable LIFO rejection checks, but the benchmark checker rejected reuse of its unannotated order Boolean. The shared Boolean now has a + annotation. This was not a queue/traversal behavior failure; benchmark timings and full qualification were not completed in that attempt. The final attempt reruns all checks rather than relabeling partial success.\n'
    name = 'native/bend_engine/collections_probe/README.md'
    data['files'][name] = replace_once(data['files'][name], 'python -m native.bend_engine.session_probe.root_probe', 'python -m native.bend_engine.collections_probe.session_check')
    data['files'][name] += '\n\nThe current-compiler integration wrapper compiles the existing session driver and CBoard support with the same flags and calls the unchanged independent session/root oracles. The legacy bitboard/session toolchain manifest remains untouched; this screen explicitly verifies the CI installer\'s aaeb9bc9 84-file source fingerprint.\n'
    return data
