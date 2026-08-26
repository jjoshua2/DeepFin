from __future__ import annotations

from typing import Any

from chess_anti_engine.encoding._lc0_ext import CBoard

THREAT_DIMS: int
HALFKA_DIMS: int
PACK_VERSION: int
FILE_VERSION: int
HAVE_AVX2: int

# Check-resolver / quiescence constants, exported so a test pins the C values
# rather than a Python restatement of them.
RESOLVER_EVAL_CLAMP: int
RESOLVER_MATE_BASE: int
RESOLVER_MATE_PLY_STEP: int
RESOLVER_MAX_PLIES: int
RESOLVER_MAX_DEPTH: int
QSEARCH_MAX_PLY: int
QSEARCH_CHECK_PLIES: int
# The DAG arm's node-budget tripwire ships OFF (0); pinned here against the C
# constant so a test never restates the default in Python.
QSEARCH_DAG_NODE_CAP: int

# FastQ-4+ (docs/fastq_design.md) defaults and quiet-certificate bits.
FASTQ_MAX_QPLY: int
FASTQ_NODE_CAP: int
FASTQ_DELTA_MARGIN: int
FASTQ_RECAPTURE_EXEMPT: int
CERT_COMPUTED: int
CERT_IN_CHECK: int
CERT_PROMOTION: int
CERT_GOOD_CAP: int

class InCheckError(ValueError):
    """The NNUE evaluation is undefined for a position in check.

    Callers must resolve check nodes recursively (search the evasions, which may
    themselves give check) before asking for a static evaluation; this exception
    is the enforcement backstop for that invariant.
    """

# The value-provider capsule other extensions import to install this evaluator
# on their eval seam, instead of including its header and getting a second copy
# of its kernel flag and weight cache.
value_provider_capsule: object

# The two race arms, published under the same capsule name and ABI. Both resolve
# check nodes recursively before evaluating; they differ only in what happens at
# a resolved non-check node (static NNUE vs tactical quiescence).
#
# ⚑ "nnue-qsearch-refresh" and "nnue-qsearch-dag" are not published as capsules:
# the refresh arm is a diagnostic oracle, and the DAG arm's store has a
# single-threaded construction path while the tree drives its provider from
# several search threads.
#
# ⚑ NOT PUBLISHING IS CONVENIENCE, NOT THE GUARD. MCTSTree accepts a capsule
# handed to it directly, so "we did not export one" would stop working the
# moment anyone exported it symmetrically with the two above. The DAG arm's
# vtable sets `requires_gil`, and MCTSTree refuses ANY provider that declares it
# — by name or by capsule. That is the enforcement; this is the ergonomics.
static_arm_capsule: object
qsearch_arm_capsule: object

def load(pack_path: str, /) -> object: ...
def set_simd(enabled: bool, /) -> bool: ...
def simd_active() -> bool: ...
def weight_cache_size() -> int: ...
def info(handle: object, /) -> dict[str, Any]: ...
def source_sha256(handle: object, /) -> str: ...
def evaluate(handle: object, board: CBoard, /) -> int: ...
def trace(handle: object, board: CBoard, /) -> tuple[int, tuple[int, ...], tuple[int, ...]]: ...
def active_features(
    board: CBoard, perspective: int, /
) -> tuple[tuple[int, ...], tuple[int, ...]]: ...
def benchmark(
    handle: object, boards: list[CBoard], repeats: int, threads: int, /
) -> tuple[int, float, int]: ...
def provider_eval(name: str, weights_path: str, board: CBoard, /) -> int: ...
def provider_names() -> tuple[str, ...]: ...
def set_arm_config(
    resolver_max_depth: int,
    qsearch_max_ply: int,
    qsearch_check_plies: int,
    dag_node_cap: int = 0,
    /,
) -> dict[str, int]: ...
def arm_eval(
    name: str, weights_path: str, boards: list[CBoard], /
) -> tuple[list[int], dict[str, int]]: ...
def arm_open(name: str, weights_path: str, /) -> object: ...
def arm_handle_eval(handle: object, boards: list[CBoard], /) -> list[int]: ...
def arm_stats(handle: object, /) -> dict[str, int]: ...

# The position DAG owned by a DAG-backed arm context ("nnue-qsearch-dag"). Every
# one of these raises ValueError on an arm that owns no store, rather than
# reporting a zero that could be mistaken for an idle one.
def arm_dag_stats(arm_handle: object, /) -> dict[str, int]: ...
def arm_dag_lookup(arm_handle: object, board: CBoard, /) -> int | None: ...
def arm_dag_value(arm_handle: object, node_id: int, /) -> int | None: ...
def arm_dag_reset(arm_handle: object, /) -> None: ...

# Static exchange evaluation, in internal units (pawn = 100), for the side to
# move. The same function the FastQ search orders and gates with. `promotion`
# takes python-chess's piece constants, which match the C encoding: 0 none,
# 2 knight .. 5 queen. Pins and checks are ignored by construction.
def see(
    board: CBoard, from_square: int, to_square: int, promotion: int = 0, /
) -> int: ...

# FastQ's quiet certificate (§3.1) and its knobs (§6). `fastq_certificate` takes
# NO window argument by construction; `fastq_stats` reports the CONTEXT's own
# knob snapshot, not the module globals set by `fastq_set_config`.
def fastq_certificate(board: CBoard, /) -> int: ...
def fastq_stored_certificate(arm_handle: object, board: CBoard, /) -> int | None: ...
def fastq_set_config(
    max_qply: int = ...,
    node_cap: int = ...,
    delta_margin: int = ...,
    see_recapture_exempt: int = ...,
    /,
) -> dict[str, int]: ...
def fastq_stats(arm_handle: object, /) -> dict[str, int]: ...

# Canonical structural-position graph. The object handle owns no Python objects;
# the C side retains the mmap'd NNUE weights independently of the load() capsule.
def dag_open(weights_handle: object, initial_nodes: int = 256, /) -> object: ...
def dag_intern_root(
    handle: object, board: CBoard, /
) -> tuple[int, int | None, bool]: ...
def dag_intern_child(
    handle: object, parent_id: int, action: int, child_board: CBoard, /
) -> tuple[int, int | None, bool]: ...
def dag_lookup(handle: object, board: CBoard, /) -> int | None: ...
def dag_value(handle: object, node_id: int, /) -> int | None: ...
def dag_children(handle: object, node_id: int, /) -> list[tuple[int, int]]: ...
def dag_set_root(handle: object, node_id: int, /) -> None: ...
def dag_stats(handle: object, /) -> dict[str, int]: ...
def dag_reset(handle: object, /) -> None: ...
