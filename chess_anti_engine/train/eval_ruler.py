"""Give a holdout measurement an identity, so a changed ruler is visible.

`holdout_generation` tracked which holdout SET a `test_loss` was measured on.
It did not track the MEASUREMENT applied to that set, and the two are not the
same ruler. PR #277 swapped the holdout eval from 2560 draws WITH replacement
-- WDL-rebalanced, half of them priority-weighted -- to one unweighted
deterministic pass over the same 2000 rows. The set never changed, so the
generation stayed at 1 across the swap, and the best-model comparison took its
SAME-RULER branch: iter 160-162 held `best_loss` 4.90535 at `test_size` 2560,
iter 165 beat it with 4.85326 at `test_size` 2000 and was promoted. The
-0.156-nat "improvement" was -5.70 sd on policy and +0.27 sd (flat) on WDL --
the fingerprint of dropping priority weighting, not of learning.

This module produces the missing half of the identity: a short string that
changes when the measurement changes. It is deliberately built from two
different kinds of evidence, because the two failure modes are different:

  * a **declared descriptor** -- mode, batch size, sampled-batch count,
    mirror probability, pooling. This is what catches PR #277's actual shape:
    the call site chose a different measurement while the code implementing
    both stayed identical.
  * a **semantic source digest** of the functions that actually produce the
    eval batches. This catches the opposite shape: the call site is unchanged
    but the pass itself is rewritten (an order change, a re-introduced
    augmentation, a switch back to sampling). A descriptor alone cannot see
    that, because a descriptor is a claim about the code and this is the code.

The digest is taken over the AST with docstrings removed, so comments,
docstrings, blank lines and reformatting do not move it; a change to what the
function DOES does. It is memoized per process (see ``_digest_cache``) because
``inspect.getsource`` re-reads the file when its mtime changes, and this repo
edits the live tree while a run is up -- an unloaded edit must not look like a
ruler change to a process still running the old bytes.

**Direction of error.** A false positive costs one best-model handover: the
record is adopted rather than compared, so the run keeps a valid best from the
current ruler. A false negative is the defect above -- a promotion across two
instruments, recorded as an improvement. The trade is asymmetric, so this
errs toward declaring a new ruler.
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import json
from collections.abc import Callable, Sequence
from typing import Any

# Bump when the descriptor's own shape changes in a way that should be read as
# a new ruler. Do NOT bump it to force a re-read of an unchanged measurement.
EVAL_RULER_SCHEMA = 1

_UNAVAILABLE = "nosource"

_digest_cache: dict[tuple[str, str], str] = {}


def _strip_docstrings(tree: ast.AST) -> None:
    """Drop docstring expressions in place from every body that can hold one."""
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = node.body
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            del body[0]
            if not body:
                body.append(ast.Pass())


def _dedent_definition(src: str) -> str:
    """Strip a method's own indentation without trusting ``textwrap.dedent``.

    ``textwrap.dedent`` removes the COMMON prefix, and this repo writes
    in-body comments at column 2 regardless of block depth. The common prefix
    of a method that contains one is therefore two spaces, dedent leaves the
    ``def`` indented, and ``ast.parse`` raises IndentationError -- which the
    caller would have swallowed as "no source available", silently reducing
    this module to its declared descriptor on exactly the production functions
    it exists to watch. Caught by
    ``test_the_production_batch_functions_have_readable_source``.

    Lines shallower than the ``def`` are comment-only or blank in that style,
    and the tokenizer ignores the indentation of both, so flushing them left
    is safe.
    """
    lines = src.splitlines()
    first = next((ln for ln in lines if ln.strip()), "")
    pad = first[: len(first) - len(first.lstrip())]
    if not pad:
        return src
    return "\n".join(ln[len(pad):] if ln.startswith(pad) else ln.lstrip() for ln in lines)


def digest_source(src: str) -> str:
    """Digest one definition's source, blind to comments, docstrings and layout.

    ``_UNAVAILABLE`` when the text cannot be parsed. The caller degrades to
    "the declared descriptor is the whole identity" -- the behaviour this
    module would have had without the digest at all -- rather than raising on
    the eval path.
    """
    try:
        tree = ast.parse(_dedent_definition(src))
        _strip_docstrings(tree)
        normalized = ast.unparse(tree)
    except (SyntaxError, ValueError):
        return _UNAVAILABLE
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def semantic_source_digest(fn: Callable[..., Any]) -> str:
    """``digest_source`` of *fn*, memoized for the life of the process.

    Memoized because ``inspect.getsource`` re-reads the file whenever its
    mtime moves, and this repo edits the live tree while a run is up: an
    edit that the running process has not loaded must not read as a ruler
    change. Within a process the answer is therefore fixed, and a real code
    change can only take effect at the restart that loads it.

    ``_UNAVAILABLE`` when there is no source to read -- a frozen or zipped
    install, or a function defined in an interactive session.
    """
    key = (getattr(fn, "__module__", "?"), getattr(fn, "__qualname__", repr(fn)))
    cached = _digest_cache.get(key)
    if cached is not None:
        return cached
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError):
        digest = _UNAVAILABLE
    else:
        digest = digest_source(src)
    _digest_cache[key] = digest
    return digest


def eval_ruler_id(
    *,
    mode: str,
    batch_size: int,
    steps: int,
    mirror_prob: float,
    pooling: str,
    batch_fns: Sequence[Callable[..., Any]],
) -> str:
    """The identity of one holdout measurement, e.g. ``v1:full_pass:a1b2c3d4``.

    The readable prefix is there so a log line, `best.json` and a
    `trial_meta.json` can be compared by eye; only the whole string is ever
    compared by code.

    ``steps`` is the SAMPLED batch count and is meaningless under a full pass;
    callers pass 0 there so that retuning ``test_steps`` -- which no longer
    reaches the eval -- cannot invent a ruler change.
    """
    descriptor = {
        "schema": int(EVAL_RULER_SCHEMA),
        "mode": str(mode),
        "batch_size": int(batch_size),
        "steps": int(steps),
        "mirror_prob": float(mirror_prob),
        "pooling": str(pooling),
        "batch_fns": [semantic_source_digest(fn) for fn in batch_fns],
    }
    payload = json.dumps(descriptor, sort_keys=True, separators=(",", ":"))
    return f"v{EVAL_RULER_SCHEMA}:{mode}:{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]}"
