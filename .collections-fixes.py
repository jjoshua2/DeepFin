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
    return correct_notice(data)


def correct_notice(data):
    name = 'native/bend_engine/collections_probe/run_probe.py'
    addition = '''# Existing session transport entry points and their transitive foreign callers.
# This is an explicit I/O trust boundary, never a source-proof acceptance rule.
SESSION_FOREIGN_DEFS = (
    'Job.load', 'Command.read', 'Reply.read', 'step_path_checked', 'step_traced',
    'step_read', 'step_prepared', 'step', 'run', 'begin_valid', 'begin', 'command',
    'dispatch', 'command_dispatch', 'serve', 'validated', 'start', 'main',
)


def session_notice(returncode: int, stdout: str, stderr: str) -> str:
    """Accept only the unchanged session driver's explicit foreign-I/O notice."""
    expected = 'All terms check, but 18 defs rely on unsafe or foreign code:\\n' + '\\n'.join('- ' + name for name in SESSION_FOREIGN_DEFS)
    if returncode != 0 or stdout.strip() or stderr.strip() != expected:
        raise ValueError(f'unexpected session compiler diagnostic: exit={returncode}\\n{stdout}\\n{stderr}')
    return stderr


'''
    text = replace_once(data['files'][name], 'def expected_trace() -> str:', addition + 'def expected_trace() -> str:')
    assert hashlib.sha256(text.encode()).hexdigest() == '1c723dad47969ea7c212165c199668f2bb72d988690068bc4a7b829e4349d05f'
    data['files'][name] = text
    name = 'native/bend_engine/collections_probe/session_check.py'
    text = replace_once(data['files'][name], 'import shutil\n', 'import shutil\nimport subprocess\n')
    text = replace_once(text, 'import check_compiler, command', 'import check_compiler, command, session_notice')
    text = replace_once(text, "    command([bun, str(source / 'bend2/main.ts'), str(sessions.HERE / 'main.bend'), '-o', str(generated)])", "    result = subprocess.run([bun, str(source / 'bend2/main.ts'), str(sessions.HERE / 'main.bend'), '-o', str(generated)],\n                            capture_output=True, text=True, timeout=300, check=False)\n    notice = session_notice(result.returncode, result.stdout, result.stderr)\n    (directory / 'compiler-notice.txt').write_text(notice)\n    print(notice, end='')")
    text = replace_once(text, "report = {'original_sessions':", "report = {'compiler_notice': (Path(tmp) / 'compiler-notice.txt').read_text(),\n                  'original_sessions':")
    assert hashlib.sha256(text.encode()).hexdigest() == '9166021da2301a6aa35900e4f07e9a580585afdc3fa336c5763329adec99fd64'
    data['files'][name] = text
    name = 'tests/test_bend_collections.py'
    text = replace_once(data['files'][name], '    expected_checksum,', '    SESSION_FOREIGN_DEFS,\n    expected_checksum,')
    text = replace_once(text, '    parse_benchmark,', '    parse_benchmark,\n    session_notice,')
    text += '''\n\ndef test_known_foreign_notice() -> None:
    assert len(SESSION_FOREIGN_DEFS) == 18
    text = 'All terms check, but 18 defs rely on unsafe or foreign code:\\n' + '\\n'.join('- ' + name for name in SESSION_FOREIGN_DEFS) + '\\n'
    assert session_notice(0, '', text) == text


@pytest.mark.parametrize('mutation', ['exit', 'stdout', 'missing', 'extra', 'count', 'name', 'clean'])
def test_unexpected_foreign_notice(mutation: str) -> None:
    text = 'All terms check, but 18 defs rely on unsafe or foreign code:\\n' + '\\n'.join('- ' + name for name in SESSION_FOREIGN_DEFS) + '\\n'
    code, output = 0, ''
    if mutation == 'exit':
        code = 1
    elif mutation == 'stdout':
        output = 'unexpected warning'
    elif mutation == 'missing':
        text = ''
    elif mutation == 'extra':
        text += '- new_unsafe\\n'
    elif mutation == 'count':
        text = text.replace('18 defs', '19 defs')
    elif mutation == 'name':
        text = text.replace('- Job.load', '- invented')
    else:
        text = 'All terms check.\\n'
    with pytest.raises(ValueError, match='unexpected session compiler diagnostic'):
        session_notice(code, output, text)
'''
    assert hashlib.sha256(text.encode()).hexdigest() == '924feaf8e79984a54dc2d35d38fbfb1bfa68bd89e54e9082824b234fd55feb6d'
    data['files'][name] = text
    data['files']['docs/experiments/2026-09-23-bend-collections-screen.md'] += '\nRun 35924444086 passed whole-repository lint, the 66 Python cases, all native FIFO/traversal/mutation checks and all timing checksums. Its new session-build wrapper incorrectly rejected the existing driver\'s exact 18-definition foreign-I/O dependency notice despite compiler exit zero. The session driver already imports Job.load, Command.read and Reply.read through C; this is a native integration test, not a source-proof gate. The corrected wrapper requires exactly that diagnostic and caller list, preserves it in logs/report, and still rejects every unexpected message/nonzero status. Eight added unit cases check this boundary; the pure collection/traversal compiler checks remain strict. No existing driver, oracle, compiler or foreign implementation changed.\n'
    data['files']['native/bend_engine/collections_probe/README.md'] += '\nThe session driver has existing foreign transport I/O. Its build explicitly requires and records the exact pinned compiler\'s 18-definition foreign-dependency notice; it is not labeled a pure proof. Unexpected diagnostics fail. Collection/traversal compilation still requires no stderr.\n'
    return data
