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
    mirror probability. This is what catches PR #277's actual shape: the call
    site chose a different measurement while the code implementing both stayed
    identical. It does NOT name pooling, or any other property of the code;
    see below.
  * a **semantic source digest** of the functions that actually turn buffer
    rows into the pooled number. This catches the opposite shape: the call
    site is unchanged but the measurement itself is rewritten (an order
    change, a re-introduced augmentation, a different pooling denominator, a
    switch back to sampling). A descriptor alone cannot see that, because a
    descriptor is a claim about the code and this is the code.
  * the **set of loss terms the objective contains** (``active_loss_terms``).
    This catches the third shape, which neither of the other two can see: the
    call site and every line of code are identical and a WEIGHT in the live
    yaml switched a term on. See ``active_loss_terms`` for why it is the term
    SET and deliberately not the weight values.

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

The digest is taken over a normalised TOKEN STREAM with comments, blank lines
and docstrings removed, so prose and reformatting do not move it; a change to
what the function DOES does. It is not taken over an ``ast.unparse``
round-trip, which was the first implementation and was interpreter-dependent:
unparse RENDERS an AST, and its rendering of f-strings and ``**`` splats
changed between CPython 3.10 and 3.11, so 9 of the 19 covered frames digested
differently on the two. A production interpreter upgrade would then have fired
a handover reporting a measurement change that never happened -- this module's
own failure mode, triggered by an unrelated bump. The token stream carries no
rendering decisions and is verified identical on 3.10, 3.11 and 3.12.

The digest is memoized per process (see ``_digest_cache``) because
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
import io
import json
import sys
import tokenize
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

# Bump when the descriptor's own shape changes in a way that should be read as
# a new ruler. Do NOT bump it to force a re-read of an unchanged measurement.
EVAL_RULER_SCHEMA = 1

_UNAVAILABLE = "nosource"

_digest_cache: dict[tuple[str, str], str] = {}
_closure_cache: dict[tuple[str, str, tuple[str, ...]], tuple[Callable[..., Any], ...]] = {}


_STRUCTURAL = {tokenize.NEWLINE: "NEWLINE", tokenize.INDENT: "IN", tokenize.DEDENT: "DE"}
_IGNORED = {tokenize.COMMENT, tokenize.NL, tokenize.ENCODING, tokenize.ENDMARKER}
  # 3.12 (PEP 701) splits an f-string into FSTRING_START / MIDDLE / END with the
  # embedded expressions tokenized in between, where 3.10/3.11 emit ONE STRING
  # token holding the raw text. Absent before 3.12, hence the getattr.
_FSTRING_START = getattr(tokenize, "FSTRING_START", None)
_FSTRING_END = getattr(tokenize, "FSTRING_END", None)


def _slice_source(lines: list[str], start: tuple[int, int], end: tuple[int, int]) -> str:
    """The raw source between two (row, col) token positions, rows 1-based."""
    if start[0] == end[0]:
        return lines[start[0] - 1][start[1]:end[1]]
    first = lines[start[0] - 1][start[1]:]
    middle = lines[start[0]:end[0] - 1]
    return "".join([first, *middle, lines[end[0] - 1][:end[1]]])


def _canonical_tokens(src: str) -> list[tuple[str, str]]:
    """The token stream that decides the digest, normalised across versions.

    Kept: every token that carries meaning, plus NEWLINE/INDENT/DEDENT as
    markers WITHOUT their whitespace -- nesting is semantic in Python, the
    number of spaces is not.

    Dropped: comments, blank lines, and docstrings (a STRING alone on its own
    logical line). This repo annotates heavily and reformats often; prose must
    not move a ruler.

    Normalised: an f-string is emitted as one ``STRING`` token holding its raw
    source text on EVERY version, by slicing the original lines between
    FSTRING_START and the matching FSTRING_END on 3.12+. That is exactly what
    3.10 and 3.11 already put in the single STRING token, so the stream matches
    across the split.
    """
    lines = src.splitlines(keepends=True)
    out: list[tuple[str, str]] = []
    depth = 0
    pending: tuple[int, int] | None = None
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if _FSTRING_START is not None and tok.type == _FSTRING_START:
            if depth == 0:
                pending = tok.start
            depth += 1
            continue
        if _FSTRING_END is not None and tok.type == _FSTRING_END:
            depth -= 1
            if depth == 0 and pending is not None:
                out.append(("STRING", _slice_source(lines, pending, tok.end)))
                pending = None
            continue
        if depth > 0:
            continue
        if tok.type in _IGNORED:
            continue
        marker = _STRUCTURAL.get(tok.type)
        if marker is not None:
            out.append((marker, ""))
            continue
        out.append((tokenize.tok_name[tok.type], tok.string))
    return out


def _drop_docstrings(items: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Remove the whole logical line of any STRING that stands alone on one.

    The trailing NEWLINE goes with it. Dropping only the STRING would leave a
    bare line terminator behind, so a function that gained a docstring would
    digest differently from the same function without one -- exactly the
    prose-moves-the-ruler behaviour this normalisation exists to stop, and
    caught by ``test_comments_docstrings_and_layout_do_not_change_the_digest``.
    """
    kept: list[tuple[str, str]] = []
    skip_next_newline = False
    for idx, item in enumerate(items):
        if skip_next_newline and item[0] == "NEWLINE":
            skip_next_newline = False
            continue
        skip_next_newline = False
        if item[0] != "STRING":
            kept.append(item)
            continue
        prev = kept[-1][0] if kept else "NEWLINE"
        nxt = items[idx + 1][0] if idx + 1 < len(items) else "NEWLINE"
        if prev in ("NEWLINE", "IN", "DE") and nxt == "NEWLINE":
            skip_next_newline = True
            continue
        kept.append(item)
    return kept


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

    Built on ``tokenize`` rather than an ``ast.unparse`` round-trip, and that
    is a correctness requirement, not a style choice. Unparse RENDERS an AST,
    and its rendering of f-strings and ``**`` splats changed between CPython
    3.10 and 3.11: 9 of the 19 covered frames digested differently on the two
    interpreters. The consequence was not merely a red CI on a 3.11 runner --
    a production Python upgrade would have moved the id and fired the handover,
    logging "the measurement was X and is now Y" when the measurement had not
    changed. A false ruler-change alarm inside the ruler-change detector is the
    exact failure this module exists to prevent.

    The token stream carries no rendering decisions, so the same source gives
    the same digest on 3.10, 3.11 and 3.12 (all three verified).

    ``_UNAVAILABLE`` when the text cannot be tokenized. The caller degrades to
    "the declared descriptor is the whole identity" -- the behaviour this
    module would have had without the digest at all -- rather than raising on
    the eval path.
    """
    try:
        items = _drop_docstrings(_canonical_tokens(_dedent_definition(src)))
    except (SyntaxError, ValueError, tokenize.TokenError):
        return _UNAVAILABLE
    payload = "\n".join(f"{kind}\x00{text}" for kind, text in items)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


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
      keeps the closure small enough to reason about -- 19 frames for the
      full pass rather than the whole dependency graph. It is stated in the ledger, not implied.
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


def active_loss_terms(loss_weights: Mapping[str, float]) -> tuple[str, ...]:
    """The names of the loss weights that are not zero, sorted.

    ⚑ THE RULE IS TERM MEMBERSHIP, NOT WEIGHT MAGNITUDE, AND THE ASYMMETRY IS
    THE WHOLE DESIGN. The two cases a loss weight can be in are not the same
    kind of event:

    * **magnitude** -- ``w`` moves 0.3 -> 0.6 on a term that was already in
      ``total``. The objective is REWEIGHTED. Every per-term column stays
      comparable and ``total`` moves by a bounded, signed amount.
    * **membership** -- ``w`` moves 0.0 -> anything, or back. The objective
      gains or loses a TERM. ``compute_loss`` adds these terms under an ``if``
      rather than multiplying by the weight (see ``losses.compute_loss``), so
      at 0.0 the term is not in ``total`` at all. For a term that is
      non-negative by construction -- ``sf_policy_floor`` is a sum of
      ``relu``, ``sf_own_regret`` is ``sum p*r`` -- switching it on steps
      ``test_loss`` strictly UP, with no bound and no way back down. The
      best-model record then freezes at the pre-flip checkpoint FOREVER,
      because every later measurement carries the new term's floor.

    Only the second is hashed, and hashing the first as well would be a
    REGRESSION rather than extra safety. ``sf_wdl_frac`` is a loss weight
    (`config_keys.TRAINER_WEIGHT_KEYS`) that the difficulty controller
    RECOMPUTES EVERY ITERATION from the realized ``wdl_regret``
    (`tune/trainable_metrics._dynamic_sf_wdl_weight`), so a value hash would
    move the ruler id on essentially every iteration of a production run. That
    does not merely fire the handover too often: `_update_best_model` would
    take its ADOPT branch every single iteration and the best-model comparison
    would stop existing. A defect that freezes the record is bad; a fix that
    abolishes the comparison is worse. `docs/rl_loop_audit.md` L15 already
    records the magnitude half as accepted definitional drift for exactly this
    reason, and this function keeps that decision while closing the other half.

    ⚑ THE RESIDUAL, STATED RATHER THAN IMPLIED: the boundary is at exactly
    0.0, so a controller that drove a weight THROUGH zero would move the id on
    every crossing. ``sf_wdl_frac`` cannot: `_dynamic_sf_wdl_weight`
    interpolates within ``[sf_wdl_frac_floor, sf_wdl_frac]`` and returns None
    (leaving the trainer's value alone) when the start value is <= 0, and
    production runs ``sf_wdl_frac: 0.50`` with ``sf_wdl_frac_floor: 0.45``. A
    future controller configured with a zero floor would need this
    reconsidered, which is why the floor is named here.

    Never raises and never silently drops a key: a value that will not convert
    to a float counts as ACTIVE, because this module's declared direction of
    error is toward announcing a new ruler (a false positive costs one
    best-model handover; a false negative is a promotion across two
    objectives).
    """
    names: list[str] = []
    for name in sorted(loss_weights):
        try:
            off = float(loss_weights[name]) == 0.0
        except (TypeError, ValueError):
            off = False
        if not off:
            names.append(str(name))
    return tuple(names)


def eval_ruler_id(
    *,
    mode: str,
    batch_size: int,
    steps: int,
    mirror_prob: float,
    measured_by: Sequence[Callable[..., Any]],
    loss_weights: Mapping[str, float],
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

    ``loss_weights`` is the whole live-pushable weight map, of which only the
    SET of non-zero names reaches the descriptor -- see ``active_loss_terms``
    for why the VALUES deliberately do not. Pass the map rather than a
    pre-filtered set: the rule for what counts as active then lives in exactly
    one function, so a caller cannot apply a second version of it.
    """
    descriptor = {
        "schema": int(EVAL_RULER_SCHEMA),
        "mode": str(mode),
        "batch_size": int(batch_size),
        "steps": int(steps),
        "mirror_prob": float(mirror_prob),
        "measured_by": [semantic_source_digest(fn) for fn in measured_by],
        "active_terms": list(active_loss_terms(loss_weights)),
    }
    payload = json.dumps(descriptor, sort_keys=True, separators=(",", ":"))
    return f"v{EVAL_RULER_SCHEMA}:{mode}:{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]}"
