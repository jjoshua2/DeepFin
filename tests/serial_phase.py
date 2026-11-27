"""Which test files run in the validator's one-process phase.

tests/conftest.py marks their tests ``serial``; scripts/validate.py selects
the phases by that marker. Standard library only.
"""

from __future__ import annotations

import os
import re

# Which files run in the validator's one-process phase instead of under xdist.
#
# A file that starts processes, installs or sends signals, or asserts on the set
# of live threads depends on state that belongs to the whole process and on the
# host not being saturated. Under xdist such files failed in a different place
# on every run: a neighbour's leftover thread or SIGINT handler, or a kill
# timeout missed on a loaded runner. The rule is read off the source so a new
# file of that kind is covered without anyone remembering a list.
_SERIAL_SOURCE = re.compile(
    r"\bsubprocess\b|\bsignal\.|\bmultiprocessing\b|ProcessPoolExecutor|os\.fork\("
    r"|threading\.enumerate\(|threading\.active_count\("
    # A real Stockfish started through production code rather than in the test.
    r"|StockfishUCI|stockfish_binary|find_stockfish"
    # Wall-clock dependence: short timeouts, measured durations, sleeps and
    # threads raced against a deadline all assume an unloaded host.
    r"|timeout(_s|_seconds)?\s*=\s*0?\.\d|time\.(monotonic|perf_counter|time|sleep)\("
    r"|threading\.Thread\(|ThreadPoolExecutor|elapsed\s*[<>]"
)
# Load-sensitive files the rule does not reach: they race threads against
# wall-clock deadlines without touching processes or signals.
SERIAL_TEST_FILES = frozenset({
    "test_cross_unit_teacher_generation.py",
    "test_lc0_control_drivers.py",
    "test_selfplay_label_overlap.py",
    "test_server_lease_assign_lock_multiwaiter.py",
})
_serial_by_path: dict[str, bool] = {}


def runs_in_serial_phase(path: os.PathLike[str]) -> bool:
    key = os.fspath(path)
    if key not in _serial_by_path:
        name = os.path.basename(key)
        with open(key, encoding="utf-8") as handle:
            _serial_by_path[key] = name in SERIAL_TEST_FILES or bool(_SERIAL_SOURCE.search(handle.read()))
    return _serial_by_path[key]
