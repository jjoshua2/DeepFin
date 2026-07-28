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
  * a **semantic source digest** of the functions that actually turn buffer
    rows into the pooled number. This catches the opposite shape: the call
    site is unchanged but the measurement itself is rewritten (an order
    change, a re-introduced augmentation, a different pooling denominator, a
    switch back to sampling). A descriptor alone cannot see that, because a
    descriptor is a claim about the code and this is the code.

**The covered set is DERIVED, not enumerated** (``call_closure``). Hashing one
function does not hash what it calls, so a hand-written list of frames is a
boundary that has to be re-drawn every time the code moves -- and it was
re-drawn wrongly twice: first the pass alone (the pooling denominator escaped),
then the pass plus three named frames (the metric-assembly tail escaped, and
`test_loss` could be DOUBLED with every test in this module's suite green).
The third answer is not a longer list. ``call_closure`` walks the call graph
statically from the eval's entry point and returns every frame it can reach
*that is defined in the same module*, so a new call on the path is covered the
day it is written.

That module boundary is the honest edge of the mechanism, and it is stated
rather than implied: `compute_loss`, the encoders, torch and numpy are outside
it, as are module-level CONSTANTS referenced by covered frames (see the
ledger's not-covered list, which names the ones that bite). A descriptor field
naming a property the digest does not reach would be the same unbacked claim
this module exists to remove -- which is why there is no `pooling` field.

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
import sys
from collections.abc import Callable, Iterable, Sequence
from typing import Any

# Bump when the descriptor's own shape changes in a way that should be read as
# a new ruler. Do NOT bump it to force a re-read of an unchanged measurement.
EVAL_RULER_SCHEMA = 1

_UNAVAILABLE = "nosource"

_digest_cache: dict[tuple[str, str], str] = {}
_closure_cache: dict[tuple[str, str, tuple[str, ...]], tuple[Callable[..., Any], ...]] = {}


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
    install, or a function defined in an interactive session. That is logged
    once per function, loudly, because a degraded digest still yields a
    perfectly well-formed id: the deploy check would pass and the handover
    line would still fire, with nothing to tell an operator that half the
    mechanism is inert. `print` rather than `logging` for the reason
    ``tune/holdout_state`` gives -- this runs inside a Ray actor whose stdout
    is what train.sh captures.
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
    if digest == _UNAVAILABLE:
        print(
            f"[eval_ruler] DEGRADED: no readable source for {key[0]}.{key[1]}, "
            f"so the holdout ruler id cannot see changes to it; the id is now "
            f"the declared descriptor alone",
            flush=True,
        )
    _digest_cache[key] = digest
    return digest


def _underlying(fn: Any) -> Any:
    """The plain function behind a bound method, staticmethod or decorator."""
    fn = getattr(fn, "__func__", fn)
    return inspect.unwrap(fn)


def _fn_key(fn: Any) -> tuple[str, str]:
    return (str(getattr(fn, "__module__", "?")), str(getattr(fn, "__qualname__", repr(fn))))


def _resolve(node: ast.AST, *, owner: type, module: Any) -> Any:
    """The function a call/attribute node refers to, or None.

    Two shapes are resolved, both against the CLASS rather than an instance:
    ``self.thing(...)`` and a bare module-level ``thing(...)``. Resolving
    against the class is what lets a property be seen as the code it is
    (``fget``) instead of being evaluated, and is why the ruler can be computed
    without building a Trainer at all.
    """
    target: Any = None
    if isinstance(node, ast.Call):
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id == "self"
        ):
            target = getattr(owner, func.attr, None)
        elif isinstance(func, ast.Name):
            target = getattr(module, func.id, None)
    elif (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    ):
  # Not a call: `self._loss_kwargs` is a property, and the dict it builds
  # decides what `compute_loss` is even given. The VALUES are out of scope
  # (see the ledger); the code that assembles them is not.
        target = getattr(owner, node.attr, None)
    if isinstance(target, property):
        target = target.fget
    return target


def call_closure(
    root: Callable[..., Any], *, owner: type,
    skip: Iterable[Callable[..., Any]] = (),
) -> tuple[Callable[..., Any], ...]:
    """Every same-module frame statically reachable from *root*, sorted.

    This is the answer to "which functions define this measurement", derived
    instead of enumerated. Two hand-written lists were wrong before it existed;
    a list is a claim about the call graph, and this reads the call graph.

    Scope rules, all three deliberate:

    * **same module only.** Recursion stops at any function defined outside
      *root*'s module. That is a real edge (for the trainer it means
      `compute_loss`, the encoders and torch are excluded) and it is what
      keeps the closure small enough to reason about -- 23 frames rather than
      the whole dependency graph. It is stated in the ledger, not implied.
    * **the branch not taken is skipped.** ``skip`` prunes the other eval
      mode's entry point, so a change to the sampled path cannot move the
      full-pass ruler. Without it the two modes would share one closure and
      lose that isolation.
    * **static, not dynamic.** A call made through a variable, a dispatch
      table or `getattr` is invisible here. The mechanism sees the code as
      written, which is the same thing the digest sees.

    Never raises: an unparseable frame is recorded (its digest degrades and
    says so) and its callees are simply not explored.
    """
    fn_root = _underlying(root)
    skip_keys = {_fn_key(_underlying(s)) for s in skip}
    cache_key = (*_fn_key(fn_root), tuple(sorted(f"{m}.{q}" for m, q in skip_keys)))
    cached = _closure_cache.get(cache_key)
    if cached is not None:
        return cached

    seen: set[tuple[str, str]] = set()
    found: list[Any] = []
    stack: list[Any] = [fn_root]
    while stack:
        fn = stack.pop()
        key = _fn_key(fn)
        if key in seen or key in skip_keys:
            continue
        seen.add(key)
        found.append(fn)
        module = sys.modules.get(str(getattr(fn, "__module__", "")))
        try:
            tree = ast.parse(_dedent_definition(inspect.getsource(fn)))
        except (OSError, TypeError, SyntaxError, ValueError):
            continue
        for node in ast.walk(tree):
            target = _resolve(node, owner=owner, module=module)
            if target is None or not (inspect.isfunction(target) or inspect.ismethod(target)):
                continue
            nxt = _underlying(target)
            if getattr(nxt, "__module__", None) == getattr(fn, "__module__", None):
                stack.append(nxt)

    out = tuple(sorted(found, key=_fn_key))
    _closure_cache[cache_key] = out
    return out


def eval_ruler_id(
    *,
    mode: str,
    batch_size: int,
    steps: int,
    mirror_prob: float,
    measured_by: Sequence[Callable[..., Any]],
) -> str:
    """The identity of one holdout measurement, e.g. ``v1:full_pass:a1b2c3d4``.

    The readable prefix is there so a log line, `best.json` and a
    `trial_meta.json` can be compared by eye; only the whole string is ever
    compared by code.

    ``steps`` is the SAMPLED batch count and is meaningless under a full pass;
    callers pass 0 there so that retuning ``test_steps`` -- which no longer
    reaches the eval -- cannot invent a ruler change.

    There is deliberately no ``pooling`` field. It was one, as the literal
    ``"row_weighted"``, and a literal is exactly the unbacked claim this
    module argues against: row-weighting lives in ``_compute_metrics``, and
    with that function unhashed the denominator could be changed with the id
    sitting still. The fix is to hash the function, not to name it -- so
    pooling is covered by ``measured_by`` now, and asserted nowhere.
    """
    descriptor = {
        "schema": int(EVAL_RULER_SCHEMA),
        "mode": str(mode),
        "batch_size": int(batch_size),
        "steps": int(steps),
        "mirror_prob": float(mirror_prob),
        "measured_by": [semantic_source_digest(fn) for fn in measured_by],
    }
    payload = json.dumps(descriptor, sort_keys=True, separators=(",", ":"))
    return f"v{EVAL_RULER_SCHEMA}:{mode}:{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]}"
