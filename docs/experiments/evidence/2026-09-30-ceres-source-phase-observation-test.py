"""CPU check for a separately sealed Ceres timing-observation candidate.

Usage: python this_file.py /path/to/candidate/fresh_source_child.py
This extracts only the pure receipt checker; it never imports a GPU runtime.
"""
from __future__ import annotations

import ast
import math
from pathlib import Path
import sys
from types import SimpleNamespace


def load_checker(source: Path):
    tree = ast.parse(source.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in ('need', 'checked_actor_phases')]
    assert len(functions) == 2
    namespace = {'math': math}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), 'exec'), namespace)
    return namespace['checked_actor_phases']


def main(source: Path) -> None:
    check = load_checker(source)
    receipt = SimpleNamespace(
        actor_wall_seconds=999.9,
        root_and_outcome_seconds=330.0,
        infer_and_encode_seconds=120.0,
        session_run_seconds=410.0,
        actor_choice_and_apply_seconds=20.0,
        sink_and_readback_seconds=110.0,
    )
    phases = check(receipt, 1000.0)
    assert phases['bound_session_call_host_wall'] == 410.0
    assert phases['sink_and_readback_including_final_tail'] == 110.0
    assert phases['residual'] == 10.0
    for edit, wall in (('nan', 1000.0), ('negative', 1000.0),
                       ('gap', 1000.0), ('wall', 1200.0)):
        bad = SimpleNamespace(**vars(receipt))
        if edit == 'nan':
            bad.session_run_seconds = math.nan
        elif edit == 'negative':
            bad.infer_and_encode_seconds = -0.1
        elif edit == 'gap':
            bad.root_and_outcome_seconds = 200.0
        try:
            check(bad, wall)
        except ValueError:
            pass
        else:
            raise AssertionError(f'{edit} actor timing was admitted')
    print('PASS_FIVE_PHASES_AND_FOUR_INVALID_RECEIPTS_CPU_ONLY')


if __name__ == '__main__':
    if __debug__ is False or len(sys.argv) != 2:
        raise SystemExit('Unoptimized Python and one candidate source path required')
    main(Path(sys.argv[1]))
