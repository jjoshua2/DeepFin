"""Guard: a test that starts a python child with its own `env` must give it
the two variables that keep it off the real GPU, with their values
(`CUDA_VISIBLE_DEVICES=""`, `PYTORCH_NVML_BASED_CUDA_CHECK="1"`; see
`tests._child_env.gpu_free_env`), because such a child does not inherit what
`tests/conftest.py` sets. This is a static pass; it cannot see what a
helper, ``os.environ`` at run time, or a non-python launcher hands a child.

The scan is an AST pass over `tests/*.py`. It looks at process launches:
``subprocess.run`` / ``Popen`` / ``check_output`` / ``check_call`` / ``call``
(as attributes, or bare names imported with ``from subprocess import ...``),
``asyncio.create_subprocess_exec``, and ``os.execve`` / ``os.execvpe`` /
``os.posix_spawn`` / ``os.posix_spawnp``. The argv may be a list or tuple
literal, positional or ``args=``, or a name bound once (``cmd = [...]`` or
``cmd: list[str] = [...]``; a bare ``cmd: list[str]`` declaration is not a
binding) in the same function to one. A launch counts as starting python when its first program, after any
``timeout`` / ``nice`` / ``env`` / ``setsid`` / ``stdbuf`` / ``ionice`` /
``taskset`` wrapper and that wrapper's options (``-n 10``, ``30``, ``5s``,
``A=1``, a ``0x1`` mask, a ``0-3`` CPU list, and the word after an option
that takes one: ``env -u X``, ``timeout -s KILL``, ``stdbuf -o L``), is ``sys.executable`` (any
``.executable`` attribute) or a ``python*`` literal; a single string counts
when its first word is one under ``shell=True``, when the whole string is a
python path under ``shell=False`` (the default: spaces belong to the path),
and when either holds for a non-constant ``shell=``. An ``env`` wrapper's
effects on the two variables are applied: ``NAME=VALUE``, ``-u NAME`` /
``--unset=NAME``, and ``-i`` / ``--ignore-environment`` / ``-`` (an empty
environment).

FAIL CLOSED. An env passes only when the guard positively resolves its
state at the launch; anything it cannot is REPORTED as "cannot be resolved".
What resolves (the allow-list):

* Expressions: a dict literal (``**`` spreads included), ``dict(x, K=v)``,
  ``x.copy()``, ``x | y``, ``gpu_free_env(...)`` (both variables set,
  whatever it wraps), ``os.environ`` (the suite's own environment: the
  required variables it does not override come from there, so this assumes
  the suite runs with both exported, as `tests/conftest.py` arranges for the
  NVML switch),
  ``env=None`` / no ``env`` at the launch (the child inherits), and a name.
* A name: a local built from the expressions above. A PARAMETER of the
  launch's function (a caller's env, a pytest fixture's value, ``**kwargs``,
  a method's argument) is reported when its value reaches the launch, as
  the env or merged into it: its keys are unknown, and a GPU-visible env
  would pass silently. Rebinding it first (``env = gpu_free_env(env)``)
  resolves it. Every occurrence of the name in the function must be one
  of: a plain ``env = ...`` / ``env: T = ...`` binding; ``env[K] = v``;
  ``del env[K]``; ``env |= x``; a statement-level ``env.update(...)`` /
  ``pop`` / ``setdefault`` / ``clear``; or a read that cannot change it
  (``env[K]``, ``env.get`` / ``keys`` / ``items`` / ``values`` / ``copy``,
  ``K in env``, ``{**env}``, ``dict(env)``, ``env | x``,
  ``gpu_free_env(env)``, ``other.update(env)``, ``f(**env)``, another
  launch's ``env=env``). Any other occurrence reports the env: a second name
  (``e2 = env``), an unpacking / ``for`` / ``with`` / walrus / comprehension
  target, passing it to any other call (``fix(env)``), a mutator inside an
  expression (``if env.pop(...)``), any other method (``popitem``,
  ``__setitem__``), a use inside a nested function, lambda or
  comprehension, or a binding with no name node (a ``match`` capture,
  ``except ... as env``, ``import ... as env``, ``global`` / ``nonlocal``).
* Order: changes are applied in source order up to the launch, and only when
  they certainly ran exactly once before it: earlier in the same block, or
  inside a ``with`` body or a ``finally`` block that precedes it. A change in
  an ``if`` / ``try`` body / ``except`` / ``else`` / loop / ``match`` before
  the launch, after the launch inside a loop that repeats it, or in a
  different part of a ``try`` from the launch is reported. A change in the
  other branch of an ``if`` that holds the launch is skipped (it cannot run
  first), as is one after the launch outside any loop.
* Values: constant writes and removals of the two variables are tracked on
  EVERY env (inherited and ``gpu_free_env`` ones too); a write through a
  non-constant key reports both as "may be overwritten", and a later
  ``setdefault`` of one keeps it uncertain. A merge (``update``, ``|``,
  ``**``) leaves a key the source lacks as it was.

Two more rules: in a function that starts any child (or a multiprocessing
process, or a shell), an unsafe change to ``os.environ`` -- a required
variable set to another value or removed (``os.environ[...]``, ``pop``,
``del``, ``update``, ``clear``, ``monkeypatch.setenv`` / ``delenv``,
``os.putenv`` / ``unsetenv``), or any change through a non-constant key -- is
reported (annotated ``os.environ[K]: T = v`` and augmented ``os.environ[K] += v``
/ ``os.environ |= {...}`` writes included); and ``functools.partial(subprocess.Popen, env=...)`` counts as a
python launch (its program is supplied later).

A finding is waived only by ``# gpu-guard: allow: <reason>`` as a ``#``
comment alone on the line directly above the reported call; it waives that
line only. A marker inside a string, or trailing code, waives nothing
(``grep -rn "gpu-guard: allow" tests/``; a test pins the exact list).

Known false positives (each reported; the remedy is to pass
``env=gpu_free_env(...)`` directly at the launch): keys held in named
constants; any non-constant key, even on an inherited env; a required
variable set from a non-constant value; a module-level env or one built by a
helper (``launcher.environment()``), including a helper that returns
``gpu_free_env(...)``; a spread of ``GPU_FREE``; a conditional expression
of two calls; a dict comprehension over ``os.environ``; any second name for
the env; and any object's ``.run([sys.executable], env=...)``, which is
taken for ``subprocess.run``. None of these shapes occurs in tests/ today.

Still missed by this static pass (only a run-time check at the launch can
see them): a launch
through a helper whose argv is a parameter, or inside a lambda; an argv
bound more than once or assembled with ``+`` / ``*``; python reached through
a variable, ``shutil.which("python3")`` or ``str(sys.executable)``; a
``shell=True`` string whose first word is not python;
``asyncio.create_subprocess_shell``; ``taskset`` masks without ``0x``; and
any child that is not python. It scans `tests/*.py` only: launchers under
`scripts/` start production children that may be meant to use the GPU.
"""
from __future__ import annotations

from typing import Any

import ast
import io
import re
import tokenize
from pathlib import Path

from tests._child_env import GPU_FREE, gpu_free_env

TESTS = Path(__file__).resolve().parent
_SUBPROCESS_CALLS = {"run", "Popen", "check_output", "check_call", "call"}
_ASYNC_CALLS = {"create_subprocess_exec"}
# name -> (index of the program argument, index of a positional env or None)
_OS_EXEC_CALLS = {"execve": (0, 2), "execvpe": (0, 2), "posix_spawn": (0, 2), "posix_spawnp": (0, 2)}
_WRAPPERS = {"timeout", "nice", "env", "setsid", "stdbuf", "ionice", "taskset"}
# A wrapper's options that take the NEXT word as their argument (`env -u X`,
# `timeout -s KILL`, `stdbuf -o L`); that word is skipped with the option.
_WORD_OPTIONS = {
    "env": {"-u", "--unset", "-C", "--chdir"},
    "timeout": {"-s", "--signal", "-k", "--kill-after"},
    "stdbuf": {"-i", "--input", "-o", "--output", "-e", "--error"},
    "nice": {"-n", "--adjustment"},
    "ionice": {"-c", "--class", "-n", "--classdata"},
    "taskset": {"-c", "--cpu-list"},
}


def _is_python(node: ast.expr | None, *, shell: bool = False) -> bool:
    """``shell``: ``node`` is a whole command line (the string form of a
    subprocess argv), so its first word is the program; otherwise it is one
    argv element or a program path, which may itself contain spaces."""
    if isinstance(node, ast.Attribute) and node.attr == "executable":
        return True
    if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
        return False
    text = node.value
    if shell:
        words = text.split()
        text = words[0] if words else ""
    return text.rsplit("/", 1)[-1].startswith("python")


def _program(elts: list[ast.expr]) -> ast.expr | None:
    """The program a command line runs, past any wrapper and its options."""
    i = 0
    while i < len(elts):
        e = elts[i]
        word = e.value if isinstance(e, ast.Constant) and isinstance(e.value, str) else None
        if word is None:
            return e
        base = word.rsplit("/", 1)[-1]
        if base in _WRAPPERS:
            i += 1
            # the wrapper's own options / arguments: -n 10, 5s, -k 1, VAR=val, 0x1, 0-3, ...
            while i < len(elts):
                w = elts[i]
                if not (isinstance(w, ast.Constant) and isinstance(w.value, str)):
                    break
                t = w.value
                if t in _WORD_OPTIONS.get(base, ()):
                    i += 2
                    continue
                if (t.startswith("-") or "=" in t or t.replace(".", "", 1).rstrip("smhd").isdigit()
                        or re.fullmatch(r"0[xX][0-9a-fA-F]+|[0-9][0-9,-]*", t)):
                    i += 1
                    continue
                break
            continue
        return e
    return None


def _env_wrapper_problems(elts: list[ast.expr]) -> list[str]:
    """What an ``env`` wrapper in the argv does to the two variables the child
    gets: ``NAME=VALUE`` sets one, ``-u NAME`` / ``--unset=NAME`` removes one,
    ``-i`` / ``--ignore-environment`` / ``-`` starts from an empty environment."""
    out: list[str] = []
    i = 0
    while i < len(elts):
        e = elts[i]
        word = e.value if isinstance(e, ast.Constant) and isinstance(e.value, str) else None
        if word is None or word.rsplit("/", 1)[-1] not in _WRAPPERS:
            return out
        base = word.rsplit("/", 1)[-1]
        i += 1
        cleared = False
        set_: dict[str, object] = {}
        while i < len(elts):
            w = elts[i]
            if not (isinstance(w, ast.Constant) and isinstance(w.value, str)):
                break
            t = w.value
            if base == "env" and t in ("-i", "--ignore-environment", "-"):
                cleared = True
                i += 1
                continue
            if base == "env" and t in ("-u", "--unset"):
                nxt = elts[i + 1] if i + 1 < len(elts) else None
                name = nxt.value if isinstance(nxt, ast.Constant) and isinstance(nxt.value, str) else None
                if name is None:
                    out.append("an env wrapper unsets a non-constant name")
                else:
                    set_[name] = _REMOVED
                i += 2
                continue
            if base == "env" and t.startswith("--unset="):
                set_[t.split("=", 1)[1]] = _REMOVED
                i += 1
                continue
            if base == "env" and "=" in t and not t.startswith("-"):
                name, value = t.split("=", 1)
                set_[name] = value
                i += 1
                continue
            if t in _WORD_OPTIONS.get(base, ()):
                i += 2
                continue
            if (t.startswith("-") or "=" in t or t.replace(".", "", 1).rstrip("smhd").isdigit()
                    or re.fullmatch(r"0[xX][0-9a-fA-F]+|[0-9][0-9,-]*", t)):
                i += 1
                continue
            break
        for key, want in GPU_FREE.items():
            if key in set_:
                if set_[key] is _REMOVED:
                    out.append(f"the env wrapper removes {key}")
                elif set_[key] != want:
                    out.append(f"the env wrapper sets {key} to {set_[key]!r}")
            elif cleared:
                out.append(f"no {key} (the env wrapper starts from an empty environment)")
    return out


def _scopes(tree: ast.Module) -> list[ast.AST]:
    return [tree, *(n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))]


def _own_nodes(scope: ast.AST) -> list[ast.AST]:
    """The nodes of ``scope``, not descending into nested functions."""
    out: list[ast.AST] = []
    todo = list(ast.iter_child_nodes(scope))
    while todo:
        node = todo.pop()
        out.append(node)
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            todo.extend(ast.iter_child_nodes(node))
    return out


def _subprocess_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "subprocess":
            names.update(a.asname or a.name for a in node.names if a.name in _SUBPROCESS_CALLS)
    return names


_UNRESOLVED = "env at this launch cannot be resolved: "
_REQUIRED = frozenset(GPU_FREE)
_MISSING = object()
_REMOVED = object()
_NON_CONSTANT = object()
_LOOP_BODIES = {(ast.For, "body"), (ast.AsyncFor, "body"), (ast.While, "body")}
# Blocks whose statements all run, in order, whenever the block is entered and
# left normally: an event inside one, earlier than the launch, has run. (A
# context manager that swallows an exception is the documented exception.)
_STRAIGHT_LINE = {(ast.With, "body"), (ast.AsyncWith, "body"), (ast.Try, "finalbody")}
if hasattr(ast, "TryStar"):  # python 3.11+
    _STRAIGHT_LINE.add((getattr(ast, "TryStar"), "finalbody"))
_MUTATORS = {"update", "pop", "setdefault", "clear"}
_READERS = {"get", "keys", "items", "values", "copy", "__contains__", "__getitem__", "__len__"}
_STATEMENTS = (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Delete, ast.Expr)


class _Unresolved(Exception):
    """The env's state at the launch is not positively known; it is reported."""


class _Env:
    """An env's state at one point. ``inherited``: its base is the suite's own
    ``os.environ`` (directly, copied, spread, or a parameter's value), so a
    required key it does not set comes from there. Constant writes and
    removals of the required keys are tracked on EVERY env, inherited or not;
    ``maybe``: required keys a write through a non-constant key may have
    changed since they were last set."""

    def __init__(self, *, inherited: bool) -> None:
        self.inherited = inherited
        self.values: dict[str, object] = {}
        self.maybe: set[str] = set()
        # the caller's mapping, unseen: fine as the env itself, unknown as a merge source
        self.parameter: str | None = None

    def copy(self) -> _Env:
        out = _Env(inherited=self.inherited)
        out.values, out.maybe, out.parameter = dict(self.values), set(self.maybe), self.parameter
        return out

    def merge(self, other: _Env) -> None:
        """``self`` updated from ``other`` (``{**self, **other}``)."""
        if other.parameter is not None:
            raise _Unresolved(f"merges {other.parameter}, a parameter whose keys are unknown")
        for key in _REQUIRED:
            theirs = other.values.get(key, _MISSING)
            if theirs is _REMOVED:
                # absent from the source: dict.update leaves the destination's value as it
                # was. If the result now inherits, a key this env never had is still absent.
                if other.inherited and not self.inherited and key not in self.values:
                    self.values[key] = _REMOVED
            elif theirs is not _MISSING:
                self.values[key] = theirs
                self.maybe.discard(key)
            elif other.inherited:  # every key of os.environ overrides what was there
                self.values.pop(key, None)
                self.maybe.discard(key)
        self.values.update({k: v for k, v in other.values.items() if k not in _REQUIRED and v is not _REMOVED})
        self.maybe |= other.maybe
        self.inherited = self.inherited or other.inherited

    def set(self, key: ast.expr | str, value: ast.expr | None) -> None:
        if isinstance(key, ast.Constant) and isinstance(key.value, str):
            key = key.value
        if isinstance(key, str):
            self.values[key] = value.value if isinstance(value, ast.Constant) else _NON_CONSTANT
            self.maybe.discard(key)
        else:
            self.maybe |= _REQUIRED

    def remove(self, key: ast.expr) -> None:
        if isinstance(key, ast.Constant) and isinstance(key.value, str):
            self.values[key.value] = _REMOVED
            self.maybe.discard(key.value)
        else:
            self.maybe |= _REQUIRED


def _problems(env: _Env) -> list[str]:
    out = []
    for key, want in GPU_FREE.items():
        value = env.values.get(key, _MISSING)
        if value is _REMOVED or (value is _MISSING and not env.inherited):
            out.append(f"no {key}")
        elif value is not _MISSING and value != want:
            out.append(f"{key} is not {want!r}")
        elif key in env.maybe:
            out.append(f"{key} may be overwritten through a non-constant key")
    return out


def _binds(name: str, scope_nodes: list[ast.AST]) -> list[ast.Assign | ast.AnnAssign]:
    """Every ``name = ...`` / ``name: T = ...`` in the scope (a bare
    ``name: T`` declaration binds nothing)."""
    return [n for n in scope_nodes
            if (isinstance(n, ast.Assign) or (isinstance(n, ast.AnnAssign) and n.value is not None))
            and any(isinstance(t, ast.Name) and t.id == name
                    for t in (n.targets if isinstance(n, ast.Assign) else [n.target]))]


def _is_os_environ(node: ast.AST) -> bool:
    return (isinstance(node, ast.Attribute) and node.attr == "environ"
            and isinstance(node.value, ast.Name) and node.value.id == "os")


def _call_name(node: ast.AST) -> str | None:
    if not isinstance(node, ast.Call):
        return None
    f = node.func
    return f.attr if isinstance(f, ast.Attribute) else f.id if isinstance(f, ast.Name) else None


class _Scan:
    """One parsed file: parent links, so an env is resolved in source order up
    to the launch, and only through constructs whose order is certain."""

    def __init__(self, tree: ast.Module) -> None:
        self.parents: dict[ast.AST, tuple[ast.AST, str, int]] = {}
        for node in ast.walk(tree):
            for field, value in ast.iter_fields(node):
                items = value if isinstance(value, list) else [value]
                for i, child in enumerate(items):
                    if isinstance(child, ast.AST):
                        self.parents[child] = (node, field, i if isinstance(value, list) else -1)

    def chain(self, node: ast.AST, scope: ast.AST) -> list[tuple[ast.AST, str, int]]:
        """``(parent, field, index)`` from ``scope`` down to ``node``."""
        out = []
        while node is not scope:
            if node not in self.parents:
                raise _Unresolved("the launch is not inside the env's function")
            parent, field, index = self.parents[node]
            out.append((parent, field, index))
            node = parent
        return out[::-1]

    def statement(self, node: ast.AST) -> ast.AST:
        while not isinstance(node, ast.stmt):
            node = self.parents[node][0]
        return node

    def relation(self, event: ast.AST, anchor: ast.AST, scope: ast.AST) -> str:
        """``before``: the event has certainly run, once, when control reaches
        ``anchor``; ``never``: it cannot have run before it. Anything else
        raises ``_Unresolved``: the default is to report."""
        e, a = self.chain(event, scope), self.chain(anchor, scope)
        common = 0
        while common < min(len(e), len(a)) and e[common] == a[common]:
            common += 1
        shared = e[:common]
        if common == len(e) or (common < len(a) and e[common][:2] == a[common][:2]
                                and e[common][2] > a[common][2]):
            # the event statement holds the launch (its effect follows it), or
            # comes after it in the same block: never, unless a loop repeats it
            if any((type(p), f) in _LOOP_BODIES for p, f, _ in [*shared, *e[common:common + 1]]):
                raise _Unresolved("changed after the launch inside a loop that repeats it")
            return "never"
        if common == len(a):
            raise _Unresolved("changed inside the launch's own expression")
        (ep, ef, ei), (_, af, ai) = e[common], a[common]
        if ef == af and ei < ai:
            for p, f, _ in e[common + 1:]:
                if (type(p), f) not in _STRAIGHT_LINE:
                    raise _Unresolved(f"changed inside a {type(p).__name__} {f} before the launch")
            return "before"
        if isinstance(ep, ast.If) and {ef, af} == {"body", "orelse"}:
            return "never"  # the other branch of an if that holds the launch
        raise _Unresolved(f"changed in the {ef} of a {type(ep).__name__} whose {af} holds the launch")


def _binds_without_a_name_node(node: ast.AST, name: str) -> bool:
    """Bindings whose target is a string field, not an ``ast.Name``: a match
    capture (``case {'k': env}``, ``case [*env]``, ``case {**env}``),
    ``except E as env``, ``import x as env`` and ``global`` / ``nonlocal``."""
    if isinstance(node, (ast.MatchAs, ast.MatchStar)):
        return node.name == name
    if isinstance(node, ast.MatchMapping):
        return node.rest == name
    if isinstance(node, ast.ExceptHandler):
        return node.name == name
    if isinstance(node, ast.alias):
        return (node.asname or node.name.split(".")[0]) == name
    if isinstance(node, (ast.Global, ast.Nonlocal)):
        return name in node.names
    return False


class _Resolver:
    def __init__(self, scan: _Scan) -> None:
        self.scan = scan

    # -- expressions -------------------------------------------------------
    def value(self, node: ast.expr, at: ast.AST, scope: ast.AST, depth: int) -> _Env:
        """The env an expression builds, evaluated at ``at``. Only a short
        allow-list of shapes resolves; anything else is reported."""
        if depth > 8:
            raise _Unresolved("derived through too many names")
        if _is_os_environ(node):
            return _Env(inherited=True)
        if isinstance(node, ast.Dict):
            env = _Env(inherited=False)
            for k, v in zip(node.keys, node.values):
                if k is None:
                    env.merge(self.value(v, at, scope, depth + 1))
                else:
                    env.set(k, v)
            return env
        name = _call_name(node)
        if isinstance(node, ast.Call) and name == "gpu_free_env":
            env = _Env(inherited=False)  # whatever it wraps, both keys are set
            for key, want in GPU_FREE.items():
                env.values[key] = want
            return env
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and name == "dict":
            if len(node.args) > 1:
                raise _Unresolved("dict() with several positional arguments")
            env = _Env(inherited=False)
            for arg in node.args:
                env.merge(self.value(arg, at, scope, depth + 1))
            for kw in node.keywords:
                if kw.arg is None:
                    env.merge(self.value(kw.value, at, scope, depth + 1))
                else:
                    env.set(kw.arg, kw.value)
            return env
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and name == "copy" and not node.args and not node.keywords):
            return self.value(node.func.value, at, scope, depth + 1).copy()
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            env = self.value(node.left, at, scope, depth + 1).copy()
            env.merge(self.value(node.right, at, scope, depth + 1))
            return env
        if isinstance(node, ast.Name):
            return self.state(node.id, at, scope, depth + 1).copy()
        raise _Unresolved(f"built from `{ast.unparse(node)[:40]}`")

    # -- names -------------------------------------------------------------
    def _param(self, name: str, scope: ast.AST) -> ast.arg | None:
        if not isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return None
        a = scope.args
        params = [*a.posonlyargs, *a.args, *a.kwonlyargs, *(x for x in (a.vararg, a.kwarg) if x)]
        return next((p for p in params if p.arg == name), None)

    def _event(self, use: ast.Name, launch_env: ast.AST | None) -> tuple[str, ast.AST, Any] | None:
        """What one occurrence of the name does: ``(kind, statement, payload)``
        for a change, None for a read that cannot change it. Any occurrence
        outside the allow-list raises."""
        parents = self.scan.parents
        parent, field, _ = parents[use]
        if use is launch_env:
            return None
        if isinstance(use.ctx, ast.Store):
            if isinstance(parent, ast.Assign) and parent.targets == [use]:
                return ("bind", parent, parent.value)
            if isinstance(parent, ast.AnnAssign):
                # `env: T` alone binds nothing
                return ("bind", parent, parent.value) if parent.value is not None else None
            if isinstance(parent, ast.AugAssign) and isinstance(parent.op, ast.BitOr):
                return ("update", parent, ([parent.value], []))
            raise _Unresolved(f"bound by a {type(parent).__name__}, not a plain assignment")
        if isinstance(use.ctx, ast.Del):
            raise _Unresolved("deleted")
        # a load
        if isinstance(parent, ast.Subscript) and field == "value":
            grand = parents[parent][0]
            if isinstance(parent.ctx, ast.Load):
                return None
            if isinstance(parent.ctx, ast.Store) and isinstance(grand, ast.Assign) and grand.targets == [parent]:
                return ("set", grand, (parent.slice, grand.value))
            if isinstance(parent.ctx, ast.Del) and isinstance(grand, ast.Delete):
                return ("remove", grand, parent.slice)
            raise _Unresolved(f"an item changed by a {type(grand).__name__}")
        if isinstance(parent, ast.Attribute) and field == "value":
            call, cfield, _ = parents[parent]
            is_call = isinstance(call, ast.Call) and cfield == "func"
            if is_call and parent.attr in _READERS:
                return None
            if is_call and parent.attr in _MUTATORS:
                stmt, _, _ = parents[call]
                if not isinstance(stmt, ast.Expr):
                    raise _Unresolved(f".{parent.attr}() used inside an expression")
                assert isinstance(call, ast.Call)
                if parent.attr == "update":
                    return ("update", stmt, (call.args, call.keywords))
                return (parent.attr, stmt, call)
            raise _Unresolved(f"changed or passed on through .{parent.attr}")
        if isinstance(parent, ast.Dict) and field == "values" and parent.keys[parents[use][2]] is None:
            return None  # {**env}
        if isinstance(parent, ast.Call) and field == "args" and _call_name(parent) == "dict":
            return None  # dict(env, ...)
        if isinstance(parent, ast.Call) and field == "args" and _call_name(parent) == "gpu_free_env":
            return None  # gpu_free_env copies its argument
        if isinstance(parent, ast.BinOp) and isinstance(parent.op, ast.BitOr):
            return None  # env | {...}
        if isinstance(parent, ast.Call) and field == "args" and _call_name(parent) == "update":
            return None  # other.update(env) reads it
        if isinstance(parent, ast.keyword) and parent.arg is None:
            return None  # f(**env) / other.update(**env)
        if isinstance(parent, ast.AugAssign) and field == "value":
            return None  # other |= env
        if isinstance(parent, ast.Compare) and field == "comparators":
            return None  # "X" in env
        if isinstance(parent, ast.FormattedValue):
            return None
        if isinstance(parent, ast.keyword) and parent.arg == "env":
            launch = parents[parent][0]
            if isinstance(launch, ast.Call) and _launch(launch, set()) is not None:
                return None  # another launch reads it
        if isinstance(parent, ast.Call) and field == "args":
            launched = _launch(parent, set())
            if launched is not None and launched[1] is use:
                return None  # another launch's positional env (os.posix_spawn(..., env))
        raise _Unresolved(f"passed to or used by `{ast.unparse(parent)[:40]}`")

    def state(self, name: str, anchor: ast.AST, scope: ast.AST, depth: int = 0,
              launch_env: ast.AST | None = None) -> _Env:
        """``name``'s env state just before ``anchor``. (Recursion runs only
        through `value`, which bounds the depth.)"""
        own: list[ast.Name] = []
        for node in ast.walk(scope):
            if _binds_without_a_name_node(node, name):
                raise _Unresolved(f"{name} is bound by a {type(node).__name__} (match capture, except-as, "
                                  "import or global/nonlocal)")
            if isinstance(node, ast.Name) and node.id == name:
                owner = node
                while owner is not scope and not isinstance(
                        owner, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef,
                                ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                    owner = self.scan.parents[owner][0]
                if owner is not scope:
                    raise _Unresolved(f"{name} is used inside a nested function, lambda or comprehension")
                own.append(node)
        env: _Env | None = None
        param = self._param(name, scope)
        if param is not None:
            # keys unknown: reported if this value reaches the launch (`_resolve_env`)
            env = _Env(inherited=True)
            env.parameter = name
        events = []
        for use in own:
            ev = self._event(use, launch_env)
            if ev is not None:
                events.append(ev)
        events.sort(key=lambda ev: (ev[1].lineno, ev[1].col_offset))
        for kind, stmt, payload in events:
            if self.scan.relation(stmt, anchor, scope) == "never":
                continue
            if kind == "bind":
                if isinstance(payload, ast.Name):
                    raise _Unresolved(f"{name} is a second name for {payload.id}")
                env = self.value(payload, stmt, scope, depth)
                continue
            if env is None:
                raise _Unresolved(f"{name} is changed before it is bound")
            if kind == "set":
                env.set(*payload)
            elif kind == "remove":
                env.remove(payload)
            elif kind == "update":
                args, kws = payload
                for arg in args:
                    env.merge(self.value(arg, stmt, scope, depth + 1))
                for kw in kws:
                    if kw.arg is None:
                        env.merge(self.value(kw.value, stmt, scope, depth + 1))
                    else:
                        env.set(kw.arg, kw.value)
            elif kind in ("pop", "setdefault", "clear"):
                call: ast.Call = payload
                if kind == "clear":
                    env.values = dict.fromkeys(_REQUIRED, _REMOVED)
                    env.maybe.clear()
                elif not call.args:
                    raise _Unresolved(f".{kind}() without a key")
                elif kind == "pop":
                    env.remove(call.args[0])
                else:
                    key = call.args[0]
                    const = key.value if isinstance(key, ast.Constant) and isinstance(key.value, str) else None
                    present = const is not None and (
                        env.values.get(const, _MISSING) not in (_MISSING, _REMOVED)
                        or (env.inherited and const not in env.values))
                    if not present:
                        # a non-constant-key write may already have set it: setdefault
                        # then keeps that unknown value, so the key stays uncertain
                        uncertain = const is not None and const in env.maybe
                        env.set(key, call.args[1] if len(call.args) > 1 else None)
                        if uncertain and const is not None:
                            env.maybe.add(const)
        if env is None:
            raise _Unresolved(f"{name} is not built in the launch's function")
        return env


def _resolve_env(node: ast.expr | None, launch: ast.Call, scope: ast.AST, resolver: _Resolver) -> _Env | None:
    """The env a launch passes, as it stands AT the launch; None when the launch
    passes none (the child inherits the suite's environment). Raises
    ``_Unresolved`` for anything not positively resolved."""
    if node is None:
        if any(kw.arg is None for kw in launch.keywords):
            raise _Unresolved("the launch takes **keywords, which may hold env")
        return None
    if isinstance(node, ast.Constant) and node.value is None:
        return None
    if isinstance(node, ast.Name):
        env = resolver.state(node.id, launch, scope, launch_env=node)
    else:
        env = resolver.value(node, launch, scope, 0)
    if env.parameter is not None:
        # A caller's env, a fixture's value, or **kwargs: its keys are unknown
        # here, and a GPU-visible env reaches the child silently. Wrap it.
        raise _Unresolved(f"env comes from {env.parameter}, a parameter of the launch's function "
                          "(a caller's or a fixture's env, keys unknown); pass env=gpu_free_env(...)")
    return env


def _resolve_argv(node: ast.expr | None, scope_nodes: list[ast.AST]) -> list[ast.expr] | None:
    if isinstance(node, ast.Name):
        binds = _binds(node.id, scope_nodes)
        node = binds[0].value if len(binds) == 1 else None
    if isinstance(node, (ast.List, ast.Tuple)):
        return list(node.elts)
    return None


_PARTIAL = ast.Constant(value="python (program supplied later through functools.partial)")
# Calls that start a child without one of the launchers above, or start a
# shell: an unsafe write to os.environ in a function holding one is reported.
_OTHER_CHILD_STARTERS = {"Process", "Pool", "ProcessPoolExecutor", "get_context", "system", "popen",
                         "spawnv", "spawnve", "spawnl", "spawnle", "fork", "forkpty"}


def _environ_writes(scope_nodes: list[ast.AST]) -> list[tuple[int, list[str]]]:
    """Unsafe changes to the suite's own environment (``os.environ`` /
    ``monkeypatch``) in a function that also starts a child: every child that
    inherits sees them. A write the guard cannot evaluate is reported too."""
    out: list[tuple[int, list[str]]] = []

    def judge(line: int, key: ast.AST | None, value: ast.AST | None, removed: bool = False) -> None:
        if not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
            out.append((line, ["os.environ changed through a non-constant key in a function that starts a child"]))
            return
        if key.value not in GPU_FREE:
            return
        want = GPU_FREE[key.value]
        if removed:
            out.append((line, [f"os.environ loses {key.value} in a function that starts a child"]))
        elif not (isinstance(value, ast.Constant) and value.value == want):
            out.append((line, [f"os.environ {key.value} is not {want!r} in a function that starts a child"]))

    for n in scope_nodes:
        if isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Subscript) and _is_os_environ(t.value):
                    judge(n.lineno, t.slice, n.value)
        elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Subscript) and _is_os_environ(n.target.value):
            judge(n.lineno, n.target.slice, n.value)  # os.environ[K]: str = v
        elif isinstance(n, ast.AugAssign):
            t = n.target
            if isinstance(t, ast.Subscript) and _is_os_environ(t.value):
                judge(n.lineno, t.slice, None)  # os.environ[K] += v: the new value is not a constant
            elif _is_os_environ(t):
                v = n.value
                if isinstance(n.op, ast.BitOr) and isinstance(v, ast.Dict) and all(k is not None for k in v.keys):
                    for k, val in zip(v.keys, v.values):
                        judge(n.lineno, k, val)  # os.environ |= {...}
                else:
                    out.append((n.lineno, ["os.environ updated from an unknown mapping in a function that starts a child"]))
        elif isinstance(n, ast.Delete):
            for t in n.targets:
                if isinstance(t, ast.Subscript) and _is_os_environ(t.value):
                    judge(n.lineno, t.slice, None, removed=True)
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
            f, owner = n.func.attr, n.func.value
            if _is_os_environ(owner) and f == "pop" and n.args:
                judge(n.lineno, n.args[0], None, removed=True)
            elif _is_os_environ(owner) and f == "clear":
                out.append((n.lineno, ["os.environ cleared in a function that starts a child"]))
            elif _is_os_environ(owner) and f == "update":
                for a in n.args:
                    if not isinstance(a, ast.Dict) or any(k is None for k in a.keys):
                        out.append((n.lineno, ["os.environ updated from an unknown mapping in a function that starts a child"]))
                    else:
                        for k, v in zip(a.keys, a.values):
                            judge(n.lineno, k, v)
                for kw in n.keywords:
                    if kw.arg is None:
                        out.append((n.lineno, ["os.environ updated from **keywords in a function that starts a child"]))
                    else:
                        judge(n.lineno, ast.Constant(value=kw.arg), kw.value)
            elif f == "setenv" and n.args:
                judge(n.lineno, n.args[0], n.args[1] if len(n.args) > 1 else None)
            elif f == "delenv" and n.args:
                judge(n.lineno, n.args[0], None, removed=True)
            elif f in ("putenv",) and isinstance(owner, ast.Name) and owner.id == "os" and n.args:
                judge(n.lineno, n.args[0], n.args[1] if len(n.args) > 1 else None)
            elif f in ("unsetenv",) and isinstance(owner, ast.Name) and owner.id == "os" and n.args:
                judge(n.lineno, n.args[0], None, removed=True)
    return out


def _launch(node: ast.Call, bare: set[str]) -> tuple[ast.expr | None, ast.expr | None, bool] | None:
    """``(argv or program node, env node, is_subprocess_argv)`` if ``node``
    launches a process; the flag says a string first argument is a command line."""
    func = node.func
    name = func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else None
    env = next((k.value for k in node.keywords if k.arg == "env"), None)
    if name == "partial" and node.args:
        inner = node.args[0]
        iname = inner.attr if isinstance(inner, ast.Attribute) else inner.id if isinstance(inner, ast.Name) else None
        if iname in _SUBPROCESS_CALLS or iname in bare:
            # functools.partial(subprocess.Popen, env=...): the argv comes later,
            # so the program is unknown and taken to be python (fail closed)
            return _PARTIAL, env, True
    if ((isinstance(func, ast.Attribute) and name in _SUBPROCESS_CALLS)
            or (isinstance(func, ast.Name) and name in bare)):
        argv = node.args[0] if node.args else next((k.value for k in node.keywords if k.arg == "args"), None)
        return argv, env, True
    if name in _ASYNC_CALLS and isinstance(func, ast.Attribute):
        return (node.args[0] if node.args else None), env, False
    if name in _OS_EXEC_CALLS and isinstance(func, ast.Attribute):
        prog, env_at = _OS_EXEC_CALLS[name]
        if env is None and len(node.args) > env_at:
            env = node.args[env_at]
        return (node.args[prog] if len(node.args) > prog else None), env, False
    return None


def _shell_flag(launch: ast.Call) -> bool | None:
    """The launch's ``shell=``: a string argv is a command line only when it is
    true (otherwise the whole string is the program path, spaces included).
    None when it is not a constant: both readings are checked (fail closed)."""
    kw = next((k for k in launch.keywords if k.arg == "shell"), None)
    if kw is None:
        return False
    if isinstance(kw.value, ast.Constant):
        return bool(kw.value.value)
    return None


def scratch_env_children(source: str) -> list[tuple[int, list[str]]]:
    """``(line, problems)`` for each python child given a scratch env whose
    GPU-free variables are missing or wrong."""
    tree = ast.parse(source)
    bare = _subprocess_names(tree)
    resolver = _Resolver(_Scan(tree))
    found: list[tuple[int, list[str]]] = []
    for scope in _scopes(tree):
        nodes = _own_nodes(scope)
        if any(isinstance(n, ast.Call) and ((_launch(n, bare) is not None and (n.args or n.keywords))
                                            or _call_name(n) in _OTHER_CHILD_STARTERS) for n in nodes):
            found.extend(_environ_writes(nodes))
        for node in nodes:
            if not isinstance(node, ast.Call):
                continue
            launch = _launch(node, bare)
            if launch is None:
                continue
            argv_node, env_node, command_line = launch
            shell: bool | None = False
            argv: list[ast.expr] | None = None
            if isinstance(argv_node, (ast.Name, ast.List, ast.Tuple)):
                argv = _resolve_argv(argv_node, nodes)
                program = _program(argv) if argv is not None else None
            else:
                program, shell = argv_node, command_line and _shell_flag(node)
            if argv_node is _PARTIAL:
                program = _PARTIAL
            if not (_is_python(program, shell=bool(shell))
                    or (shell is None and _is_python(program, shell=True))):
                continue
            try:
                env = _resolve_env(env_node, node, scope, resolver)
            except _Unresolved as why:
                found.append((node.lineno, [f"{_UNRESOLVED}{why}"]))
                continue
            problems = _problems(env) if env is not None else []
            if argv is not None:
                problems += _env_wrapper_problems(argv)
            if problems:
                found.append((node.lineno, problems))
    allowed = _allowed_lines(source)
    return sorted(f for f in found if f[0] not in allowed)


# A finding is waived only by this marker, with a reason, in a ``#`` comment that
# is alone on the line directly above the reported call; it waives that call's
# line only. A marker inside a string, or trailing code, waives nothing.
# `grep -rn "gpu-guard: allow" tests/` lists them.
_ALLOW = re.compile(r"#\s*gpu-guard:\s*allow:\s*(\S.*)$")


def _allowed_lines(source: str) -> dict[int, str]:
    """``line -> reason`` for every line a marker waives."""
    out: dict[int, str] = {}
    lines = source.splitlines()
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, SyntaxError):
        return out
    for tok in tokens:
        if tok.type != tokenize.COMMENT:
            continue
        m = _ALLOW.search(tok.string)
        row, col = tok.start
        if m and not lines[row - 1][:col].strip():
            out[row + 1] = m.group(1).strip()
    return out


def test_no_python_child_gets_a_scratch_env_that_can_see_the_gpu() -> None:
    offenders = [
        f"{path.name}:{line}: {', '.join(problems)}"
        for path in sorted(TESTS.glob("*.py"))
        for line, problems in scratch_env_children(path.read_text(encoding="utf-8"))
    ]
    assert offenders == [], "wrap the env in tests._child_env.gpu_free_env(...):\n" + "\n".join(offenders)


_BOTH = ["no CUDA_VISIBLE_DEVICES", "no PYTORCH_NVML_BASED_CUDA_CHECK"]


def test_the_guard_sees_the_shapes_it_claims() -> None:
    src = '''
import asyncio, os, subprocess, sys
from subprocess import run as bare_run, Popen
def literal():
    subprocess.run([sys.executable, "x.py"], env={"PATH": "/bin"})
def named_env():
    env = {"PATH": "/bin"}
    subprocess.Popen([sys.executable, "-c", "pass"], env=env)
def python_literal():
    subprocess.run(["python3", "x.py"], env={"PATH": "/bin", "CUDA_VISIBLE_DEVICES": ""})
def inherits():
    subprocess.run([sys.executable], env={**os.environ, "X": "1"})
def wrapped():
    subprocess.run([sys.executable], env=gpu_free_env({"PATH": "/bin"}))
def bash_child():
    subprocess.run(["bash", "x.sh"], env={"PATH": "/bin"})
def complete():
    subprocess.run([sys.executable], env={"CUDA_VISIBLE_DEVICES": "", "PYTORCH_NVML_BASED_CUDA_CHECK": "1"})
def named_argv():
    cmd = [sys.executable, "x.py"]
    subprocess.run(cmd, env={"PATH": "/bin"})
def args_keyword():
    subprocess.run(args=[sys.executable, "x.py"], env={"PATH": "/bin"})
def bare_import():
    bare_run([sys.executable, "x.py"], env={"PATH": "/bin"})
    Popen([sys.executable, "x.py"], env={"PATH": "/bin"})
def wrapper_prefix():
    subprocess.run(["timeout", "-k", "1", "30", "nice", "-n", "10", "env", "A=1", sys.executable], env={"PATH": "/bin"})
def wrong_values():
    subprocess.run([sys.executable], env={"CUDA_VISIBLE_DEVICES": "0", "PYTORCH_NVML_BASED_CUDA_CHECK": "0"})
def mutated_complete():
    env = {"PATH": "/bin"}
    env["CUDA_VISIBLE_DEVICES"] = ""
    env.update(PYTORCH_NVML_BASED_CUDA_CHECK="1")
    subprocess.run([sys.executable], env=env)
def mutated_incomplete():
    env = {"PATH": "/bin"}
    env.update({"CUDA_VISIBLE_DEVICES": ""})
    subprocess.run([sys.executable], env=env)
def updated_from_environ():
    env = {"PATH": "/bin"}
    env.update(os.environ)
    subprocess.run([sys.executable], env=env)
def dict_call():
    subprocess.run([sys.executable], env=dict(PATH="/bin"))
async def async_exec():
    await asyncio.create_subprocess_exec(sys.executable, "x.py", env={"PATH": "/bin"})
def os_exec():
    os.execve(sys.executable, [sys.executable, "x.py"], {"PATH": "/bin"})
def other_subprocess_calls():
    subprocess.check_output([sys.executable], env={"PATH": "/bin"})
    subprocess.check_call([sys.executable], env={"PATH": "/bin"})
    subprocess.call([sys.executable], env={"PATH": "/bin"})
def tuple_argv():
    subprocess.run((sys.executable, "x.py"), env={"PATH": "/bin"})
def other_os_exec():
    os.execvpe("python3", ["python3", "x.py"], {"PATH": "/bin"})
    os.posix_spawn(sys.executable, [sys.executable], {"PATH": "/bin"})
    os.posix_spawnp("python3", ["python3"], env={"PATH": "/bin"})
def other_wrappers():
    subprocess.run(["setsid", sys.executable], env={"PATH": "/bin"})
    subprocess.run(["stdbuf", "-oL", sys.executable], env={"PATH": "/bin"})
    subprocess.run(["ionice", "-c", "3", sys.executable], env={"PATH": "/bin"})
    subprocess.run(["taskset", "0x1", sys.executable], env={"PATH": "/bin"})
    subprocess.run(["taskset", "-c", "0-3", sys.executable], env={"PATH": "/bin"})
def annotated_argv():
    cmd: list[str] = [sys.executable, "x.py"]
    subprocess.run(cmd, env={"PATH": "/bin"})
def shell_string_with_a_path():
    subprocess.run("python3 a/b.py", shell=True, env={"PATH": "/bin"})
def bash_with_a_python_path():
    subprocess.run(["bash", "python/x.sh"], env={"PATH": "/bin"})
def python_path_with_a_space():
    subprocess.run(["/opt/my env/bin/python3", "x.py"], env={"PATH": "/bin"})
def declared_then_assigned():
    cmd: list[str]
    cmd = [sys.executable, "x.py"]
    env: dict[str, str]
    env = {"PATH": "/bin"}
    subprocess.run(cmd, env=env)
def wrapper_options_with_a_word():
    subprocess.run(["env", "-u", "X", sys.executable], env={"PATH": "/bin"})
    subprocess.run(["timeout", "-s", "KILL", "30", sys.executable], env={"PATH": "/bin"})
    subprocess.run(["stdbuf", "-o", "L", sys.executable], env={"PATH": "/bin"})
    subprocess.run(["timeout", "30", "bash", "x.sh"], env={"PATH": "/bin"})
'''
    assert scratch_env_children(src) == [
        (5, _BOTH),
        (8, _BOTH),
        (10, ["no PYTORCH_NVML_BASED_CUDA_CHECK"]),
        (21, _BOTH),
        (23, _BOTH),
        (25, _BOTH),
        (26, _BOTH),
        (28, _BOTH),
        (30, ["CUDA_VISIBLE_DEVICES is not ''", "PYTORCH_NVML_BASED_CUDA_CHECK is not '1'"]),
        (39, ["no PYTORCH_NVML_BASED_CUDA_CHECK"]),
        (45, _BOTH),
        (47, _BOTH),
        (49, _BOTH),
        (51, _BOTH),
        (52, _BOTH),
        (53, _BOTH),
        (55, _BOTH),
        (57, _BOTH),
        (58, _BOTH),
        (59, _BOTH),
        (61, _BOTH),
        (62, _BOTH),
        (63, _BOTH),
        (64, _BOTH),
        (65, _BOTH),
        (68, _BOTH),
        (70, _BOTH),
        (74, _BOTH),
        (80, _BOTH),
        (82, _BOTH),
        (83, _BOTH),
        (84, _BOTH),
    ]



def test_the_guard_resolves_the_env_as_it_stands_at_the_launch() -> None:
    """Source order, up to the launch, on the launch's own path; an env whose
    state there cannot be known is reported, never passed."""
    ok = '{"CUDA_VISIBLE_DEVICES": "", "PYTORCH_NVML_BASED_CUDA_CHECK": "1"}'
    src = f'''
import os, subprocess, sys
def set_after_launch():
    env = {{}}
    subprocess.run([sys.executable], env=env)
    env.update(CUDA_VISIBLE_DEVICES="", PYTORCH_NVML_BASED_CUDA_CHECK="1")
def safe_then_unsafe():
    env = {ok}
    env["CUDA_VISIBLE_DEVICES"] = "0"
    subprocess.run([sys.executable], env=env)
def unsafe_then_safe():
    env = {{}}
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTORCH_NVML_BASED_CUDA_CHECK"] = "1"
    subprocess.run([sys.executable], env=env)
def variable_key(key):
    env = {{}}
    env[key] = "anything"
    subprocess.run([sys.executable], env=env)
def variable_key_after_complete(key):
    env = {ok}
    env[key] = "anything"
    subprocess.run([sys.executable], env=env)
def variable_key_on_inherited(key):
    env = {{**os.environ}}
    env[key] = "anything"
    subprocess.run([sys.executable], env=env)
def popped(key):
    env = {ok}
    env.pop("CUDA_VISIBLE_DEVICES")
    subprocess.run([sys.executable], env=env)
def deleted():
    env = {ok}
    del env["PYTORCH_NVML_BASED_CUDA_CHECK"]
    subprocess.run([sys.executable], env=env)
def bound_twice():
    env = {ok}
    env = {{"PATH": "/bin"}}
    subprocess.run([sys.executable], env=env)
def aliased():
    env = {{}}
    other = env
    other.update({ok})
    subprocess.run([sys.executable], env=env)
def or_assign():
    env = {{}}
    env |= {ok}
    subprocess.run([sys.executable], env=env)
def derived():
    env = {{"PATH": "/bin"}}
    a = dict(env, CUDA_VISIBLE_DEVICES="", PYTORCH_NVML_BASED_CUDA_CHECK="1")
    subprocess.run([sys.executable], env=a)
    subprocess.run([sys.executable], env={{**env, "X": "1"}})
    subprocess.run([sys.executable], env=env.copy())
    subprocess.run([sys.executable], env=env | {ok})
def on_a_branch(c):
    env = {{}}
    if c:
        env.update({ok})
    subprocess.run([sys.executable], env=env)
def other_branch(c):
    env = {{}}
    if c:
        env.update({ok})
    else:
        subprocess.run([sys.executable], env=env)
def in_a_loop(xs):
    env = {{}}
    for _ in xs:
        subprocess.run([sys.executable], env=env)
        env.update({ok})
def closure():
    env = {{}}
    def inner():
        subprocess.run([sys.executable], env=env)
    env.update({ok})
    inner()
def set_default():
    env = {{}}
    env.setdefault("CUDA_VISIBLE_DEVICES", "")
    env.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "1")
    subprocess.run([sys.executable], env=env)
'''
    may = " may be overwritten through a non-constant key"
    assert scratch_env_children(src) == [
        (5, _BOTH),
        (10, ["CUDA_VISIBLE_DEVICES is not ''"]),
        (20, _BOTH),
        (24, ["CUDA_VISIBLE_DEVICES" + may, "PYTORCH_NVML_BASED_CUDA_CHECK" + may]),
        # fail closed: a non-constant key on an inherited env may name a required variable
        (28, ["CUDA_VISIBLE_DEVICES" + may, "PYTORCH_NVML_BASED_CUDA_CHECK" + may]),
        (32, ["no CUDA_VISIBLE_DEVICES"]),
        (36, ["no PYTORCH_NVML_BASED_CUDA_CHECK"]),
        (40, _BOTH),
        (45, [_UNRESOLVED + "passed to or used by `other = env`"]),
        (54, _BOTH),
        (55, _BOTH),
        (61, [_UNRESOLVED + "changed inside a If body before the launch"]),
        (67, _BOTH),
        (71, [_UNRESOLVED + "changed after the launch inside a loop that repeats it"]),
        (76, [_UNRESOLVED + "env is not built in the launch's function"]),
    ]
    # A rebinding from the env itself reads the state before that statement.
    assert scratch_env_children(
        "import subprocess, sys\ndef f():\n    env = {'PATH': '/bin'}\n"
        f"    env = {{**env, **{ok}}}\n    subprocess.run([sys.executable], env=env)\n"
    ) == []


# The guard reviewer's adversarial snippets for round 4 (44 + 15) and the four
# Codex findings, each prefixed with _TABLE_HEADER. UNSAFE: the child can see
# the GPU, so the guard must report it (a missing / wrong variable, or
# 'cannot be resolved'). SAFE: no report, except the documented false
# positives in _TABLE_DOCUMENTED_FALSE_POSITIVES, which must stay reported so a
# change in that set is noticed.
_TABLE_HEADER = "import asyncio, functools, multiprocessing, os, subprocess, sys\nfrom tests._child_env import gpu_free_env\nOK = {'CUDA_VISIBLE_DEVICES': '', 'PYTORCH_NVML_BASED_CUDA_CHECK': '1'}\n"
_TABLE_UNSAFE = {
    'codex1_launch_before_fix': "def f():\n    env = {}\n    subprocess.run([sys.executable], env=env)\n    env.update(CUDA_VISIBLE_DEVICES='', PYTORCH_NVML_BASED_CUDA_CHECK='1')",
    'codex2_dynamic_key': "def f(key):\n    env = {}\n    env[key] = 'anything'\n    subprocess.run([sys.executable], env=env)",
    'a01_safe_then_overwrite': "def f():\n    env = {'CUDA_VISIBLE_DEVICES': '', 'PYTORCH_NVML_BASED_CUDA_CHECK': '1'}\n    env['CUDA_VISIBLE_DEVICES'] = '0'\n    subprocess.run([sys.executable], env=env)",
    'a02_helper_mutates_after_safe': "def fix(e):\n    e['CUDA_VISIBLE_DEVICES'] = '0'\ndef f():\n    env = {'CUDA_VISIBLE_DEVICES': '', 'PYTORCH_NVML_BASED_CUDA_CHECK': '1'}\n    fix(env)\n    subprocess.run([sys.executable], env=env)",
    'a03_update_nonliteral_after_safe': "def f(other):\n    env = {'CUDA_VISIBLE_DEVICES': '', 'PYTORCH_NVML_BASED_CUDA_CHECK': '1'}\n    env.update(other)\n    subprocess.run([sys.executable], env=env)",
    'a04_del_in_finally': "def f():\n    env = {'CUDA_VISIBLE_DEVICES': '', 'PYTORCH_NVML_BASED_CUDA_CHECK': '1'}\n    try:\n        pass\n    finally:\n        del env['CUDA_VISIBLE_DEVICES']\n    subprocess.run([sys.executable], env=env)",
    'a05_walrus_rebind': "def f():\n    env = {'CUDA_VISIBLE_DEVICES': '', 'PYTORCH_NVML_BASED_CUDA_CHECK': '1'}\n    if (env := {'PATH': '/bin'}):\n        pass\n    subprocess.run([sys.executable], env=env)",
    'a06_tuple_assign_rebind': "def f():\n    env = {'CUDA_VISIBLE_DEVICES': '', 'PYTORCH_NVML_BASED_CUDA_CHECK': '1'}\n    env, x = {'PATH': '/bin'}, 1\n    subprocess.run([sys.executable], env=env)",
    'a07_for_rebind': "def f():\n    env = {'CUDA_VISIBLE_DEVICES': '', 'PYTORCH_NVML_BASED_CUDA_CHECK': '1'}\n    for env in [{'PATH': '/bin'}]:\n        pass\n    subprocess.run([sys.executable], env=env)",
    'a08_with_as_rebind': "def f(cm):\n    env = {'CUDA_VISIBLE_DEVICES': '', 'PYTORCH_NVML_BASED_CUDA_CHECK': '1'}\n    with cm() as env:\n        pass\n    subprocess.run([sys.executable], env=env)",
    'a09_global_mutated_elsewhere': "ENV = {'CUDA_VISIBLE_DEVICES': '', 'PYTORCH_NVML_BASED_CUDA_CHECK': '1'}\ndef spoil():\n    ENV['CUDA_VISIBLE_DEVICES'] = '0'\ndef f():\n    subprocess.run([sys.executable], env=ENV)",
    'a10_module_level_scratch': "ENV = {'PATH': '/bin'}\ndef f():\n    subprocess.run([sys.executable], env=ENV)",
    'a11_default_arg_scratch': "def f(env={'PATH': '/bin'}):\n    subprocess.run([sys.executable], env=env)",
    'a12_alias_mutated': "def f():\n    env = {'CUDA_VISIBLE_DEVICES': '', 'PYTORCH_NVML_BASED_CUDA_CHECK': '1'}\n    e2 = env\n    e2['CUDA_VISIBLE_DEVICES'] = '0'\n    subprocess.run([sys.executable], env=env)",
    'a13_setitem_via_method': "def f():\n    env = {'CUDA_VISIBLE_DEVICES': '', 'PYTORCH_NVML_BASED_CUDA_CHECK': '1'}\n    env.__setitem__('CUDA_VISIBLE_DEVICES', '0')\n    subprocess.run([sys.executable], env=env)",
    'a14_popitem': "def f():\n    env = {'CUDA_VISIBLE_DEVICES': '', 'PYTORCH_NVML_BASED_CUDA_CHECK': '1'}\n    env.popitem()\n    subprocess.run([sys.executable], env=env)",
    'a15_env_none_explicit_scratch_before': "def f():\n    env = {'PATH': '/bin'}\n    env = None\n    subprocess.run([sys.executable], env=env)",
    'a16_partial_popen': "def f():\n    p = functools.partial(subprocess.Popen, env={'PATH': '/bin'})\n    p([sys.executable])",
    'a17_asyncio_exec_named_env': "async def f():\n    env = {'PATH': '/bin'}\n    await asyncio.create_subprocess_exec(sys.executable, 'x.py', env=env)",
    'a18_posix_spawn_named': "def f():\n    env = {'PATH': '/bin'}\n    os.posix_spawn(sys.executable, [sys.executable], env)",
    'a19_mp_spawn_context': "def f():\n    os.environ['CUDA_VISIBLE_DEVICES'] = '0'\n    ctx = multiprocessing.get_context('spawn')\n    ctx.Process(target=print).start()",
    'a20_os_environ_mutated_then_inherited': "def f():\n    os.environ['CUDA_VISIBLE_DEVICES'] = '0'\n    subprocess.run([sys.executable], env={**os.environ})",
    'a21_os_environ_pop_then_copy': "def f():\n    os.environ.pop('PYTORCH_NVML_BASED_CUDA_CHECK')\n    env = os.environ.copy()\n    subprocess.run([sys.executable], env=env)",
    'a22_branch_unsafe': "def f(c):\n    env = {'CUDA_VISIBLE_DEVICES': '', 'PYTORCH_NVML_BASED_CUDA_CHECK': '1'}\n    if c:\n        env['CUDA_VISIBLE_DEVICES'] = '0'\n    subprocess.run([sys.executable], env=env)",
    'a23_loop_after_launch': "def f():\n    env = {'CUDA_VISIBLE_DEVICES': '', 'PYTORCH_NVML_BASED_CUDA_CHECK': '1'}\n    for i in range(2):\n        subprocess.run([sys.executable], env=env)\n        env['CUDA_VISIBLE_DEVICES'] = '0'",
    'a24_while_true_break': "def f():\n    env = {'CUDA_VISIBLE_DEVICES': '', 'PYTORCH_NVML_BASED_CUDA_CHECK': '1'}\n    while True:\n        env.pop('CUDA_VISIBLE_DEVICES')\n        break\n    subprocess.run([sys.executable], env=env)",
    'a25_match_case': "def f(x):\n    env = {'CUDA_VISIBLE_DEVICES': '', 'PYTORCH_NVML_BASED_CUDA_CHECK': '1'}\n    match x:\n        case 1:\n            env.clear()\n    subprocess.run([sys.executable], env=env)",
    'a26_nested_closure_mutates': "def f():\n    env = {'CUDA_VISIBLE_DEVICES': '', 'PYTORCH_NVML_BASED_CUDA_CHECK': '1'}\n    def g():\n        env['CUDA_VISIBLE_DEVICES'] = '0'\n    g()\n    subprocess.run([sys.executable], env=env)",
    'a27_lambda_mutates': "def f():\n    env = {'CUDA_VISIBLE_DEVICES': '', 'PYTORCH_NVML_BASED_CUDA_CHECK': '1'}\n    (lambda: env.update({'CUDA_VISIBLE_DEVICES': '0'}))()\n    subprocess.run([sys.executable], env=env)",
    'a28_copy_then_mutate_source_irrelevant': "def f():\n    base = {'PATH': '/bin'}\n    env = base.copy()\n    base.update(OK)\n    subprocess.run([sys.executable], env=env)",
    'a29_ior_nonliteral_after_safe': "def f(o):\n    env = {'CUDA_VISIBLE_DEVICES': '', 'PYTORCH_NVML_BASED_CUDA_CHECK': '1'}\n    env |= o\n    subprocess.run([sys.executable], env=env)",
    'a30_attribute_env': "class C:\n    def __init__(self):\n        self.env = {'PATH': '/bin'}\n    def f(self):\n        subprocess.run([sys.executable], env=self.env)",
    'a31_subscript_env': "def f():\n    envs = {'a': {'PATH': '/bin'}}\n    subprocess.run([sys.executable], env=envs['a'])",
    'a32_conditional_expr_env': "def f(c):\n    subprocess.run([sys.executable], env={'PATH': '/bin'} if c else None)",
    'a33_gpu_free_env_then_overwrite': "def f():\n    env = gpu_free_env({'PATH': '/bin'})\n    env['CUDA_VISIBLE_DEVICES'] = '0'\n    subprocess.run([sys.executable], env=env)",
    'a34_inherit_then_set_gpu': "def f():\n    env = {**os.environ}\n    env['CUDA_VISIBLE_DEVICES'] = '0'\n    subprocess.run([sys.executable], env=env)",
    'a35_try_body_mutation_then_launch_after': "def f():\n    env = {'CUDA_VISIBLE_DEVICES': '', 'PYTORCH_NVML_BASED_CUDA_CHECK': '1'}\n    try:\n        env['CUDA_VISIBLE_DEVICES'] = '0'\n    except Exception:\n        pass\n    subprocess.run([sys.executable], env=env)",
    'a36_kwargs_dict': "def f():\n    kw = {'env': {'PATH': '/bin'}}\n    subprocess.run([sys.executable], **kw)",
    'b01_try_body_unsafe_launch_in_else': "def f():\n    env = {'CUDA_VISIBLE_DEVICES': '', 'PYTORCH_NVML_BASED_CUDA_CHECK': '1'}\n    try:\n        env['CUDA_VISIBLE_DEVICES'] = '0'\n    except Exception:\n        pass\n    else:\n        subprocess.run([sys.executable], env=env)",
    'b02_try_body_unsafe_launch_in_except': "def f(risky):\n    env = {'CUDA_VISIBLE_DEVICES': '', 'PYTORCH_NVML_BASED_CUDA_CHECK': '1'}\n    try:\n        env['CUDA_VISIBLE_DEVICES'] = '0'\n        risky()\n    except Exception:\n        subprocess.run([sys.executable], env=env)",
    'b03_try_body_unsafe_launch_in_finally': "def f():\n    env = {'CUDA_VISIBLE_DEVICES': '', 'PYTORCH_NVML_BASED_CUDA_CHECK': '1'}\n    try:\n        env['CUDA_VISIBLE_DEVICES'] = '0'\n    finally:\n        subprocess.run([sys.executable], env=env)",
    'b04_walrus_in_if_test_launch_in_body': "def f():\n    env = {'CUDA_VISIBLE_DEVICES': '', 'PYTORCH_NVML_BASED_CUDA_CHECK': '1'}\n    if (env := {'PATH': '/bin'}):\n        subprocess.run([sys.executable], env=env)",
    'b05_pop_in_if_test_launch_in_body': "def f():\n    env = {'CUDA_VISIBLE_DEVICES': '', 'PYTORCH_NVML_BASED_CUDA_CHECK': '1'}\n    if env.pop('CUDA_VISIBLE_DEVICES', None) is not None:\n        subprocess.run([sys.executable], env=env)",
    'b06_pop_in_while_test_launch_in_body': "def f():\n    env = {'CUDA_VISIBLE_DEVICES': '', 'PYTORCH_NVML_BASED_CUDA_CHECK': '1'}\n    while env.pop('PYTORCH_NVML_BASED_CUDA_CHECK', None):\n        subprocess.run([sys.executable], env=env)",
    'b07_for_rebind_launch_inside': "def f(envs):\n    env = {'CUDA_VISIBLE_DEVICES': '', 'PYTORCH_NVML_BASED_CUDA_CHECK': '1'}\n    for env in [{'PATH': '/bin'}]:\n        subprocess.run([sys.executable], env=env)",
    'b08_with_rebind_launch_inside': "def f(cm):\n    env = {'CUDA_VISIBLE_DEVICES': '', 'PYTORCH_NVML_BASED_CUDA_CHECK': '1'}\n    with cm() as env:\n        subprocess.run([sys.executable], env=env)",
    'b09_clear_in_with_header': "def f(cm):\n    env = {'CUDA_VISIBLE_DEVICES': '', 'PYTORCH_NVML_BASED_CUDA_CHECK': '1'}\n    with cm(env.clear()):\n        subprocess.run([sys.executable], env=env)",
    'b10_match_subject_clear': "def f():\n    env = {'CUDA_VISIBLE_DEVICES': '', 'PYTORCH_NVML_BASED_CUDA_CHECK': '1'}\n    match env.clear():\n        case None:\n            subprocess.run([sys.executable], env=env)",
    'b11_constant_pop_on_helper_env': "def f():\n    env = gpu_free_env({'PATH': '/bin'})\n    env.pop('CUDA_VISIBLE_DEVICES')\n    subprocess.run([sys.executable], env=env)",
    'b12_constant_del_on_inherited': "def f():\n    env = {**os.environ}\n    del env['PYTORCH_NVML_BASED_CUDA_CHECK']\n    subprocess.run([sys.executable], env=env)",
    'b13_update_kwargs_splat_after_safe': "def f(**over):\n    env = {'CUDA_VISIBLE_DEVICES': '', 'PYTORCH_NVML_BASED_CUDA_CHECK': '1'}\n    env.update(**over)\n    subprocess.run([sys.executable], env=env)",
    'b14_unsafe_set_in_if_body_launch_in_elif': "def f(a):\n    env = {'CUDA_VISIBLE_DEVICES': '', 'PYTORCH_NVML_BASED_CUDA_CHECK': '1'}\n    if a:\n        env['CUDA_VISIBLE_DEVICES'] = '0'\n    subprocess.run([sys.executable], env=env) if True else None",
    'b15_launch_in_comprehension_after_loop_var': "def f():\n    env = {'CUDA_VISIBLE_DEVICES': '', 'PYTORCH_NVML_BASED_CUDA_CHECK': '1'}\n    [subprocess.run([sys.executable], env=env) for env in [{'PATH': '/bin'}]]",
    'codex3_update_kwsplat_literal': "def f():\n    env = {}\n    env.update(**{'PATH': '/bin'})\n    subprocess.run([sys.executable], env=env)",
    'codex4_try_body_write_handled': "def f(prepare):\n    env = {}\n    try:\n        prepare()\n        env.update(CUDA_VISIBLE_DEVICES='', PYTORCH_NVML_BASED_CUDA_CHECK='1')\n    except OSError:\n        pass\n    subprocess.run([sys.executable], env=env)",
}
# The author's additions in round 5: one per fail-closed default the table above
# does not reach (each passes silently if that default is flipped to permissive).
_TABLE_UNSAFE.update({
    "r5_try_body_fix_launch_in_handler": "def f():\n    env = {}\n    try:\n        env.update({'CUDA_VISIBLE_DEVICES': '', 'PYTORCH_NVML_BASED_CUDA_CHECK': '1'})\n    except Exception:\n        subprocess.run([sys.executable], env=env)",
    "r5_for_body_fix_launch_in_else": "def f(xs):\n    env = {}\n    for _ in xs:\n        env.update({'CUDA_VISIBLE_DEVICES': '', 'PYTORCH_NVML_BASED_CUDA_CHECK': '1'})\n    else:\n        subprocess.run([sys.executable], env=env)",
    "r5_dict_two_positionals": "def f():\n    subprocess.run([sys.executable], env=dict({'CUDA_VISIBLE_DEVICES': '', 'PYTORCH_NVML_BASED_CUDA_CHECK': '1'}, {}))",
    "r5_deleted_name": "def f():\n    env = {'CUDA_VISIBLE_DEVICES': '', 'PYTORCH_NVML_BASED_CUDA_CHECK': '1'}\n    del env\n    subprocess.run([sys.executable], env=env)",
    "r5_item_augmented": "def f():\n    env = {'CUDA_VISIBLE_DEVICES': '', 'PYTORCH_NVML_BASED_CUDA_CHECK': '1'}\n    env['CUDA_VISIBLE_DEVICES'] += '0'\n    subprocess.run([sys.executable], env=env)",
    "r5_alias_changed_through_the_other_name": "def f():\n    other = {'CUDA_VISIBLE_DEVICES': '', 'PYTORCH_NVML_BASED_CUDA_CHECK': '1'}\n    env = other\n    other.pop('CUDA_VISIBLE_DEVICES')\n    subprocess.run([sys.executable], env=env)",
    "r5_module_env_written_in_the_function": "ENV = {'PATH': '/bin'}\ndef f():\n    ENV['HOME'] = '/tmp'\n    subprocess.run([sys.executable], env=ENV)",
    "r5_twelve_copies_of_an_incomplete_env": "def f():\n    e0 = {'PATH': '/bin'}\n    e1 = e0.copy()\n    e2 = e1.copy()\n    e3 = e2.copy()\n    e4 = e3.copy()\n    e5 = e4.copy()\n    e6 = e5.copy()\n    e7 = e6.copy()\n    e8 = e7.copy()\n    e9 = e8.copy()\n    e10 = e9.copy()\n    e11 = e10.copy()\n    e12 = e11.copy()\n    subprocess.run([sys.executable], env=e12)",
    # round 6: an env that reaches the launch as a parameter (the reviewer's h01-h06)
    "h01_fixture_env": "import pytest\n@pytest.fixture\ndef child_env():\n    return {'PATH': '/bin'}\ndef test_x(child_env):\n    subprocess.run([sys.executable], env=child_env)",
    "h02_fixture_env_with_a_gpu": "import pytest\n@pytest.fixture\ndef child_env():\n    return {**os.environ, 'CUDA_VISIBLE_DEVICES': '0'}\ndef test_x(child_env):\n    subprocess.run([sys.executable], env=child_env)",
    "h03_helper_parameter": "def _launch(env):\n    subprocess.run([sys.executable], env=env)",
    "h04_parameter_defaulting_to_none": "def _launch(env=None):\n    subprocess.run([sys.executable], env=env)",
    "h05_kwargs_parameter": "def _run(**env):\n    subprocess.run([sys.executable], env=env)",
    "h06_method_parameter": "class C:\n    def go(self, env):\n        subprocess.run([sys.executable], env=env)",
    "s2_rebound_to_a_second_name": "def f():\n    e2 = gpu_free_env({})\n    base = {'PATH': '/bin'}\n    e2 = base\n    subprocess.run([sys.executable], env=e2)",
    # Codex round 3 at 32566f97d
    "codex5_setdefault_after_dynamic_key": "def f(key):\n    env = {}\n    env[key] = '0'\n    env.setdefault('CUDA_VISIBLE_DEVICES', '')\n    env.setdefault('PYTORCH_NVML_BASED_CUDA_CHECK', '1')\n    subprocess.run([sys.executable], env=env)",
    "codex6_scalar_python_path_with_a_space": "def f():\n    subprocess.run('/opt/my env/bin/python3', env={})",
    "codex7_merge_source_lacks_a_key": "def f():\n    env = {'CUDA_VISIBLE_DEVICES': '0'}\n    patch = {'CUDA_VISIBLE_DEVICES': '', 'PYTORCH_NVML_BASED_CUDA_CHECK': '1'}\n    patch.pop('CUDA_VISIBLE_DEVICES')\n    env.update(patch)\n    env.setdefault('CUDA_VISIBLE_DEVICES', '')\n    subprocess.run([sys.executable], env=env)",
    "codex9_os_environ_annotated_write": "def f():\n    os.environ['CUDA_VISIBLE_DEVICES']: str = '0'\n    subprocess.run([sys.executable])",
    "codex9_os_environ_or_assign": "def f():\n    os.environ |= {'CUDA_VISIBLE_DEVICES': '0'}\n    subprocess.run([sys.executable])",
    "codex10_env_wrapper_sets_a_gpu": "def f():\n    subprocess.run(['env', 'CUDA_VISIBLE_DEVICES=0', sys.executable], env=gpu_free_env({}))",
    "codex10_env_wrapper_clears": "def f():\n    subprocess.run(['env', '-i', sys.executable], env=gpu_free_env({}))",
    "codex8_match_capture_rebinds": "def f(x):\n    env = {}\n    env.update(CUDA_VISIBLE_DEVICES='', PYTORCH_NVML_BASED_CUDA_CHECK='1')\n    match x:\n        case {'child_env': env}:\n            pass\n    subprocess.run([sys.executable], env=env)",
})
_TABLE_SAFE = {
    's01_safe_literal': "def f():\n    subprocess.run([sys.executable], env={'CUDA_VISIBLE_DEVICES': '', 'PYTORCH_NVML_BASED_CUDA_CHECK': '1'})",
    's02_fix_before_launch': "def f():\n    env = {'PATH': '/bin'}\n    env.update(CUDA_VISIBLE_DEVICES='', PYTORCH_NVML_BASED_CUDA_CHECK='1')\n    subprocess.run([sys.executable], env=env)",
    's03_gpu_free_env': "def f():\n    subprocess.run([sys.executable], env=gpu_free_env({'PATH': '/bin'}))",
    's04_inherit': "def f():\n    subprocess.run([sys.executable], env={**os.environ, 'X': '1'})",
    's05_unsafe_after_launch_no_loop': "def f():\n    env = {'CUDA_VISIBLE_DEVICES': '', 'PYTORCH_NVML_BASED_CUDA_CHECK': '1'}\n    subprocess.run([sys.executable], env=env)\n    env['CUDA_VISIBLE_DEVICES'] = '0'",
    's06_dynamic_key_on_complete': "def f(k):\n    env = {'CUDA_VISIBLE_DEVICES': '', 'PYTORCH_NVML_BASED_CUDA_CHECK': '1'}\n    env['X_' + k] = '1'\n    subprocess.run([sys.executable], env=env)",
    's07_copy_of_safe': "def f():\n    base = {'CUDA_VISIBLE_DEVICES': '', 'PYTORCH_NVML_BASED_CUDA_CHECK': '1'}\n    env = dict(base, PATH='/bin')\n    subprocess.run([sys.executable], env=env)",
    's08_or_merge': "def f():\n    env = {'PATH': '/bin'} | OK\n    subprocess.run([sys.executable], env=env)",
    # the merge fix keeps the destination's safe value for a key the source lacks
    's10_env_wrapper_benign_assignment': "def f():\n    subprocess.run(['env', 'A=1', sys.executable], env=gpu_free_env({}))",
    's11_env_wrapper_clears_then_sets_both': "def f():\n    subprocess.run(['env', '-i', 'CUDA_VISIBLE_DEVICES=', 'PYTORCH_NVML_BASED_CUDA_CHECK=1', sys.executable])",
    's09_merge_source_lacks_a_key_dest_safe': "def f():\n    env = {'CUDA_VISIBLE_DEVICES': '', 'PYTORCH_NVML_BASED_CUDA_CHECK': '1'}\n    patch = {'CUDA_VISIBLE_DEVICES': '0', 'X': '1'}\n    patch.pop('CUDA_VISIBLE_DEVICES')\n    env.update(patch)\n    subprocess.run([sys.executable], env=env)",
}
_TABLE_DOCUMENTED_FALSE_POSITIVES = {
    "s06_dynamic_key_on_complete",  # a non-constant key may name a required variable
    "s08_or_merge",  # OK is a module-level name: not built in the launch's function
}


def test_every_adversarial_unsafe_child_is_reported_and_safe_ones_pass() -> None:
    """Fail closed: nothing the guard cannot positively resolve passes."""
    silent = [k for k, body in _TABLE_UNSAFE.items() if not scratch_env_children(_TABLE_HEADER + body)]
    assert silent == []
    reported = {k for k, body in _TABLE_SAFE.items() if scratch_env_children(_TABLE_HEADER + body)}
    assert reported == _TABLE_DOCUMENTED_FALSE_POSITIVES
    assert len(_TABLE_UNSAFE) == 55 + 8 + 7 + 4 + 4
    assert len(_TABLE_SAFE) == 11


def test_a_marked_launch_is_waived_and_only_with_a_reason() -> None:
    body = "import subprocess, sys\ndef f():\n{mark}    subprocess.run([sys.executable], env={{'PATH': '/bin'}})\n"
    assert scratch_env_children(body.format(mark="")) == [(3, _BOTH)]
    assert scratch_env_children(body.format(mark="    # gpu-guard: allow: the child is a stub\n")) == []
    assert scratch_env_children(body.format(mark="    # gpu-guard: allow:\n")) == [(4, _BOTH)]
    # a trailing marker waives nothing, neither its own line nor the next
    two = ("import subprocess, sys\ndef f():\n"
           "    subprocess.run([sys.executable], env={'PATH': '/a'})  # gpu-guard: allow: the child is a stub\n"
           "    subprocess.run([sys.executable], env={'PATH': '/b'})\n")
    assert scratch_env_children(two) == [(3, _BOTH), (4, _BOTH)]
    # a marker inside a string literal is not a comment
    in_string = ("import subprocess, sys\ndef f():\n    x = '# gpu-guard: allow: the child is a stub'\n"
                 "    subprocess.run([sys.executable], env={'PATH': '/bin'})\n")
    assert scratch_env_children(in_string) == [(4, _BOTH)]


def test_every_waiver_in_the_suite_names_its_reason() -> None:
    waivers = []
    for path in sorted(TESTS.glob("*.py")):  # this file too: its samples are strings, not comments
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        waivers += [(path.name, lines[line - 1].strip(), reason) for line, reason in _allowed_lines(text).items()]
    assert [(name, line) for name, line, _ in waivers] == [
        ("test_conftest_keeps_cuda_probe_off_the_device.py", "proc = subprocess.run("),
    ]
    assert all(len(reason) > 20 for _, _, reason in waivers)


def test_gpu_free_env_adds_both_variables_and_overrides_a_visible_gpu() -> None:
    assert GPU_FREE == {"CUDA_VISIBLE_DEVICES": "", "PYTORCH_NVML_BASED_CUDA_CHECK": "1"}
    env = gpu_free_env({"PATH": "/bin", "CUDA_VISIBLE_DEVICES": "0"})
    assert env == {"PATH": "/bin", "CUDA_VISIBLE_DEVICES": "", "PYTORCH_NVML_BASED_CUDA_CHECK": "1"}
