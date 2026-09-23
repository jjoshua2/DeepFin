#!/usr/bin/env python3
"""Run unchanged chess oracles against the collections screen's pinned compiler."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile

from native.bend_engine.collections_probe.run_probe import check_compiler, command
from native.bend_engine.legal_probe import run_probe as rules
from native.bend_engine.session_probe import root_probe, run_probe as sessions


def build(source: Path, directory: Path, bun: str, cc: str) -> dict[str, Path]:
    """Same sources/flags as the session driver; retain its legacy pin unchanged."""
    check_compiler(source)
    generated = directory / 'session.c'
    command([bun, str(source / 'bend2/main.ts'), str(sessions.HERE / 'main.bend'), '-o', str(generated)])
    binaries: dict[str, Path] = {}
    for mode, flags in sessions.MODES.items():
        obj, binary = directory / f'support-{mode}.o', directory / f'session-{mode}'
        command([cc, '-std=c11', '-O3', *flags, '-I', str(sessions.ROOT), '-c', str(rules.HERE / 'support.c'), '-o', str(obj)])
        command([cc, '-std=c11', '-O3', '-ffp-contract=off', *flags, '-I', str(rules.HERE), str(generated), str(obj), '-pthread', '-lm', '-o', str(binary)])
        binaries[mode] = binary
    oracle = directory / 'cboard-reference'
    command([cc, '-std=c11', '-O3', '-DLEGAL_ORACLE', '-I', str(sessions.ROOT), str(rules.HERE / 'support.c'), '-pthread', '-lm', '-o', str(oracle)])
    binaries['reference'] = oracle
    return binaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiler-root', type=Path, required=True)
    parser.add_argument('--bun', default=shutil.which('bun'))
    parser.add_argument('--cc', default=shutil.which('clang'))
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if not args.bun or not args.cc:
        parser.error('Bun and Clang are required')
    with tempfile.TemporaryDirectory(prefix='collections-sessions-') as tmp:
        binaries = build(args.compiler_root, Path(tmp), args.bun, args.cc)
        report = {'original_sessions': sessions.verify(binaries, with_python_chess=True),
                  'root_advancement': root_probe.verify_roots(binaries, check_encoding=False)}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
