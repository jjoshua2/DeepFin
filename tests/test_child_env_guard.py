"""Guard: a test that starts a python child with an `env` built from scratch
must give it the two variables that keep it off the real GPU, with their
values (`CUDA_VISIBLE_DEVICES=""`, `PYTORCH_NVML_BASED_CUDA_CHECK="1"`; see
`tests._child_env.gpu_free_env`), because such a child does not inherit what
`tests/conftest.py` sets.

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
``.executable`` attribute) or a ``python*`` literal; a single string (the
``shell=True`` form) counts when its first word is one.

The env may be a dict literal, ``dict(...)``, or a name; a name is resolved
AS IT STANDS AT THE LAUNCH: the function's assignments to it and changes to
it (``env[K] = v``, ``env.update(...)``, ``env |= {...}``, ``setdefault``,
``pop``, ``del env[K]``, ``clear``, a rebinding) are applied in source order
up to the launch, on the launch's own control-flow path (a change on a branch
the launch is not in is skipped; ``with`` and a ``try`` body are straight
line). A copy (``dict(env, ...)``, ``{**env, ...}``, ``env.copy()``,
``env | {...}``) takes the source env's state at that point. A dict that
spreads or is updated from anything that is not a scratch dict of the same
function (``**os.environ``) inherits from the suite and is left alone; so is
any other expression (``gpu_free_env(...)``, ``os.environ.copy()``, a
parameter). A write or removal through a non-constant key is tracked
separately from inheritance: a required variable it may have overwritten is
reported.

An env whose state at the launch cannot be resolved is REPORTED, never
passed: one changed or rebound on a branch the launch is not on, in a loop
(including a change after the launch inside the loop that holds it), bound
to a second name (``other = env``), changed by an unknown method or an
augmented assignment, or built in an enclosing function the launch's
function closes over.

Still missed: an env from a dict comprehension, or returned by a helper
function, or changed by a helper it is passed to (``fill(env)``); a launch
through a helper of its own (e.g. a ``_run`` wrapper whose argv is a
parameter), through ``functools.partial(subprocess.run, env=...)``, or
inside a lambda; an argv bound more than once, or assembled with ``+`` /
``*`` unpacking from names; python reached through a variable
(``exe = sys.executable``), ``shutil.which("python3")`` or
``str(sys.executable)``; a ``shell=True`` string whose first word is not
python (``"cd /tmp && python3 x.py"``); ``asyncio.create_subprocess_shell``;
a ``taskset`` mask written without ``0x`` (``taskset ff ...``). A guard
against the common shapes, not a proof.

Known false positives (each is reported, and the remedy is the same: wrap
the env in ``gpu_free_env(...)``): keys held in named constants; a required
variable set from a non-constant value; any second name for the env, even
unchanged; a loop that rebinds the env at the top of each iteration and
changes it after the launch; and any object's ``.run([sys.executable],
env=...)``, which is taken for ``subprocess.run``.
"""
from __future__ import annotations

from typing import Any

import ast
import re
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


_REQUIRED = frozenset(GPU_FREE)
_LOOPS = (ast.For, ast.AsyncFor, ast.While)
# Constructs whose body runs zero, one or several times, or on one path only.
_CONDITIONAL = (ast.If, *_LOOPS, ast.Match, ast.match_case, ast.ExceptHandler, ast.IfExp,
                ast.BoolOp, ast.comprehension)
_READ_ONLY_METHODS = {"copy", "get", "items", "keys", "values", "__contains__", "__getitem__"}


class _Env:
    """One scratch env dict's state at a point: its constant keys and values;
    ``inherits`` when anything non-literal is mixed in (then it is left alone);
    ``maybe_changed`` -- required keys a write through a non-constant key may
    have overwritten since they were last set; ``unresolved`` -- why the state
    at the launch cannot be known (reported, never passed)."""

    def __init__(self) -> None:
        self.values: dict[str, object] = {}
        self.inherits = False
        self.maybe_changed: set[str] = set()
        self.unresolved: str | None = None

    def copy(self) -> _Env:
        out = _Env()
        out.values = dict(self.values)
        out.inherits, out.maybe_changed, out.unresolved = (
            self.inherits, set(self.maybe_changed), self.unresolved)
        return out

    def merge(self, other: _Env) -> None:
        self.values.update(other.values)
        self.inherits |= other.inherits
        self.maybe_changed = (self.maybe_changed - set(other.values)) | other.maybe_changed
        self.unresolved = self.unresolved or other.unresolved

    def set(self, key: ast.expr | str | None, value: ast.expr | None) -> None:
        if isinstance(key, ast.Constant) and isinstance(key.value, str):
            key = key.value
        if isinstance(key, str):
            self.values[key] = value.value if isinstance(value, ast.Constant) else ...
            self.maybe_changed.discard(key)
        else:  # a non-constant key may name a required variable
            self.maybe_changed |= _REQUIRED

    def remove(self, key: ast.expr | None) -> None:
        if isinstance(key, ast.Constant) and isinstance(key.value, str):
            self.values.pop(key.value, None)
            self.maybe_changed.discard(key.value)
        else:
            self.maybe_changed |= _REQUIRED


def _problems(env: _Env) -> list[str]:
    if env.unresolved:
        return [f"env at this launch cannot be resolved: {env.unresolved}"]
    out = []
    for key, want in GPU_FREE.items():
        if key not in env.values:
            out.append(f"no {key}")
        elif env.values[key] != want:
            out.append(f"{key} is not {want!r}")
        elif key in env.maybe_changed:
            out.append(f"{key} may be overwritten through a non-constant key")
    return out


def _binds(name: str, scope_nodes: list[ast.AST]) -> list[ast.Assign | ast.AnnAssign]:
    """Every ``name = ...`` / ``name: T = ...`` in the scope (a bare
    ``name: T`` declaration binds nothing)."""
    return [n for n in scope_nodes
            if (isinstance(n, ast.Assign) or (isinstance(n, ast.AnnAssign) and n.value is not None))
            and any(isinstance(t, ast.Name) and t.id == name
                    for t in (n.targets if isinstance(n, ast.Assign) else [n.target]))]


def _pos(node: ast.AST) -> tuple[int, int]:
    return (getattr(node, "lineno", 0), getattr(node, "col_offset", 0))


class _Scan:
    """One parsed file: parent links, so an env is resolved in SOURCE ORDER up
    to the launch, on the launch's own control-flow path."""

    def __init__(self, tree: ast.Module) -> None:
        self.parents: dict[ast.AST, tuple[ast.AST, str]] = {}
        for node in ast.walk(tree):
            for field, value in ast.iter_fields(node):
                for child in value if isinstance(value, list) else [value]:
                    if isinstance(child, ast.AST):
                        self.parents[child] = (node, field)

    def _path(self, node: ast.AST, scope: ast.AST) -> list[tuple[ast.AST, str]]:
        out = []
        while node is not scope and node in self.parents:
            parent, field = self.parents[node]
            out.append((parent, field))
            node = parent
        return out

    def contains(self, outer: ast.AST, node: ast.AST) -> bool:
        while node in self.parents:
            node = self.parents[node][0]
            if node is outer:
                return True
        return False

    def relation(self, event: ast.AST, anchor: ast.AST, scope: ast.AST) -> str:
        """``before``: runs before ``anchor`` on every path to it; ``never``:
        not before it on any path; ``maybe``: on some paths only (a branch the
        anchor is not in, a loop that may run zero times, a later statement of a
        loop body that also holds the anchor)."""
        p_event, p_anchor = self._path(event, scope), self._path(anchor, scope)
        shared = set(p_anchor)
        anchor_ancestors = {a for a, _ in p_anchor}
        if _pos(event) >= _pos(anchor):
            in_shared_loop = any(isinstance(a, _LOOPS) and f == "body" and (a, f) in shared
                                 for a, f in p_event)
            return "maybe" if in_shared_loop else "never"
        for anc, field in p_event:
            if (anc, field) in shared:
                return "before"  # the anchor is in this same block, later
            if anc in anchor_ancestors:
                return "never"  # another branch of a statement that holds the anchor
            if isinstance(anc, _CONDITIONAL) or (isinstance(anc, ast.Try)
                                                  and field in ("handlers", "orelse")):
                if isinstance(anc, ast.BoolOp) and field == "values" and anc.values[0] is event:
                    continue  # the first operand always runs
                return "maybe"
        return "before"


def _events(name: str, nodes: list[ast.AST]) -> list[tuple[tuple[int, int], str, ast.AST, Any]]:
    """``(position, kind, node, payload)`` for everything a scope does to ``name``."""
    out: list[tuple[tuple[int, int], str, ast.AST, Any]] = []

    def named(t: ast.AST) -> bool:
        return isinstance(t, ast.Name) and t.id == name

    for n in nodes:
        if isinstance(n, ast.Assign):
            for t in n.targets:
                if named(t):
                    out.append((_pos(n), "bind", n, n.value))
                elif isinstance(t, ast.Subscript) and named(t.value):
                    out.append((_pos(n), "set", n, (t.slice, n.value)))
                elif isinstance(t, (ast.Tuple, ast.List)) and any(named(e) for e in ast.walk(t)):
                    out.append((_pos(n), "opaque", n, "bound by an unpacking assignment"))
            if named(n.value) and not any(named(t) for t in n.targets):
                out.append((_pos(n), "opaque", n, f"aliased to another name ({name} is shared)"))
        elif isinstance(n, (ast.AnnAssign, ast.NamedExpr)) and n.value is not None and named(n.target):
            out.append((_pos(n), "bind", n, n.value))
        elif isinstance(n, ast.AugAssign):
            if named(n.target):
                if isinstance(n.op, ast.BitOr):
                    out.append((_pos(n), "update", n, ([n.value], [])))
                else:
                    out.append((_pos(n), "opaque", n, "an augmented assignment"))
            elif isinstance(n.target, ast.Subscript) and named(n.target.value):
                out.append((_pos(n), "set", n, (n.target.slice, None)))
        elif isinstance(n, ast.Delete):
            out.extend((_pos(n), "remove", n, t.slice) for t in n.targets
                       if isinstance(t, ast.Subscript) and named(t.value))
        elif isinstance(n, (ast.For, ast.AsyncFor, ast.With, ast.AsyncWith)):
            targets = [n.target] if isinstance(n, (ast.For, ast.AsyncFor)) else [
                i.optional_vars for i in n.items if i.optional_vars is not None]
            if any(named(e) for t in targets for e in ast.walk(t)):
                out.append((_pos(n), "opaque", n, "bound by a for / with target"))
        elif (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
              and named(n.func.value)):
            attr = n.func.attr
            if attr == "update":
                out.append((_pos(n), "update", n, (n.args, n.keywords)))
            elif attr == "pop" and n.args:
                out.append((_pos(n), "remove", n, n.args[0]))
            elif attr == "setdefault" and n.args:
                out.append((_pos(n), "setdefault", n, (n.args[0], n.args[1] if len(n.args) > 1 else None)))
            elif attr == "clear":
                out.append((_pos(n), "clear", n, None))
            elif attr not in _READ_ONLY_METHODS:
                out.append((_pos(n), "opaque", n, f"changed by .{attr}()"))
    out.sort(key=lambda e: e[0])
    return out


def _scratch_literal(value: ast.expr | None) -> bool:
    return isinstance(value, ast.Dict) or (
        isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == "dict")


class _Resolver:
    def __init__(self, scan: _Scan) -> None:
        self.scan = scan
        self._nodes: dict[int, list[ast.AST]] = {}

    def nodes(self, scope: ast.AST) -> list[ast.AST]:
        if id(scope) not in self._nodes:
            self._nodes[id(scope)] = _own_nodes(scope)
        return self._nodes[id(scope)]

    def value(self, node: ast.expr, scope: ast.AST, depth: int) -> _Env | None:
        """A bound value's env state, or None when it is not a scratch dict."""
        if isinstance(node, ast.Dict):
            env = _Env()
            for k, v in zip(node.keys, node.values):
                if k is None:
                    self.spread(env, v, node, scope, depth)
                else:
                    env.set(k, v)
            return env
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "dict":
            if len(node.args) > 1:
                return None
            env = _Env()
            for a in node.args:
                self.spread(env, a, node, scope, depth)
            self.keywords(env, node.keywords)
            return env
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "copy" and not node.args and isinstance(node.func.value, ast.Name)):
            src = self.state(node.func.value.id, node, scope, depth + 1)
            return src.copy() if src is not None else None
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            # `a | b` builds a new dict: a copy of `a`, not an alias of it.
            if isinstance(node.left, ast.Name):
                src = self.state(node.left.id, node, scope, depth + 1)
                left = src.copy() if src is not None else None
            else:
                left = self.value(node.left, scope, depth + 1)
            if left is None:
                return None
            self.spread(left, node.right, node, scope, depth)
            return left
        if isinstance(node, ast.Name):
            src = self.state(node.id, node, scope, depth + 1)
            if src is None:
                return None
            env = src.copy()
            env.unresolved = env.unresolved or f"an alias of {node.id}"
            return env
        return None

    def spread(self, env: _Env, node: ast.expr, anchor: ast.AST, scope: ast.AST, depth: int) -> None:
        """``env`` updated from ``node`` (``**node``, ``dict(node)``, ``.update(node)``)."""
        if isinstance(node, ast.Dict) or (
                isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "dict"):
            sub = self.value(node, scope, depth + 1)
        elif isinstance(node, ast.Name):
            sub = self.state(node.id, anchor, scope, depth + 1)
        else:
            sub = None
        if sub is None:
            env.inherits = True  # os.environ or anything else: not a scratch env
        else:
            env.merge(sub)

    @staticmethod
    def keywords(env: _Env, keywords: list[ast.keyword]) -> None:
        for kw in keywords:
            if kw.arg is None:
                env.inherits = True
            else:
                env.set(kw.arg, kw.value)

    def state(self, name: str, anchor: ast.AST, scope: ast.AST, depth: int = 0) -> _Env | None:
        """``name``'s env state just before ``anchor``, or None when it is not a
        scratch dict built in this scope (a parameter, a helper's return, ...)."""
        if depth > 8:
            env = _Env()
            env.unresolved = "too deeply derived"
            return env
        env: _Env | None = None
        seen_bind = False
        for _, kind, node, payload in _events(name, self.nodes(scope)):
            if self.scan.contains(node, anchor):
                continue  # `env = {**env, ...}`: a binding does not precede its own value
            rel = self.scan.relation(node, anchor, scope)
            if rel == "never":
                continue
            if kind == "bind":
                seen_bind = True
                new = self.value(payload, scope, depth)
                if rel == "maybe":
                    if new is not None or env is not None:
                        env = (env or new or _Env()).copy()
                        env.unresolved = env.unresolved or "rebound on a branch / in a loop"
                    continue
                env = new
                continue
            if env is None or env.inherits:
                continue
            if rel == "maybe":
                env.unresolved = env.unresolved or "changed on a branch / in a loop"
                continue
            if kind == "set":
                key, value = payload
                env.set(key, value)
            elif kind == "remove":
                env.remove(payload)
            elif kind == "setdefault":
                key, value = payload
                if not (isinstance(key, ast.Constant) and key.value in env.values):
                    env.set(key, value)
            elif kind == "clear":
                env.values.clear()
                env.maybe_changed.clear()
            elif kind == "update":
                args, kws = payload
                for a in args:
                    self.spread(env, a, node, scope, depth)
                self.keywords(env, kws)
            elif kind == "opaque":
                env.unresolved = env.unresolved or str(payload)
        if not seen_bind and not self._is_param(name, scope):
            return self._enclosing(name, scope)
        return env

    @staticmethod
    def _is_param(name: str, scope: ast.AST) -> bool:
        if not isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return False
        a = scope.args
        params = [*a.posonlyargs, *a.args, *a.kwonlyargs, *(x for x in (a.vararg, a.kwarg) if x)]
        return any(p.arg == name for p in params)

    def _enclosing(self, name: str, scope: ast.AST) -> _Env | None:
        """A name the launch's function closes over: if an enclosing function
        binds it to a scratch dict, when the launch runs relative to that
        function's changes is unknown -- reported."""
        node = scope
        while node in self.scan.parents:
            node = self.scan.parents[node][0]
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module)):
                if any(_scratch_literal(b.value) for b in _binds(name, self.nodes(node))):
                    env = _Env()
                    env.unresolved = f"{name} is built in an enclosing function"
                    return env
                if isinstance(node, ast.Module):
                    return None
        return None


def _resolve_env(node: ast.expr | None, launch: ast.Call, scope: ast.AST, resolver: _Resolver) -> _Env | None:
    """The env a launch passes, as it stands AT the launch, or None when it
    inherits / is not built from a literal here."""
    if node is None:
        return None
    if isinstance(node, ast.Name):
        env = resolver.state(node.id, launch, scope)
    else:
        env = resolver.value(node, scope, 0)
    return None if env is None or env.inherits else env


def _resolve_argv(node: ast.expr | None, scope_nodes: list[ast.AST]) -> list[ast.expr] | None:
    if isinstance(node, ast.Name):
        binds = _binds(node.id, scope_nodes)
        node = binds[0].value if len(binds) == 1 else None
    if isinstance(node, (ast.List, ast.Tuple)):
        return list(node.elts)
    return None


def _launch(node: ast.Call, bare: set[str]) -> tuple[ast.expr | None, ast.expr | None, bool] | None:
    """``(argv or program node, env node, is_subprocess_argv)`` if ``node``
    launches a process; the flag says a string first argument is a command line."""
    func = node.func
    name = func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else None
    env = next((k.value for k in node.keywords if k.arg == "env"), None)
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


def scratch_env_children(source: str) -> list[tuple[int, list[str]]]:
    """``(line, problems)`` for each python child given a scratch env whose
    GPU-free variables are missing or wrong."""
    tree = ast.parse(source)
    bare = _subprocess_names(tree)
    resolver = _Resolver(_Scan(tree))
    found: list[tuple[int, list[str]]] = []
    for scope in _scopes(tree):
        nodes = resolver.nodes(scope)
        for node in nodes:
            if not isinstance(node, ast.Call):
                continue
            launch = _launch(node, bare)
            if launch is None:
                continue
            argv_node, env_node, command_line = launch
            shell = False
            if isinstance(argv_node, (ast.Name, ast.List, ast.Tuple)):
                argv = _resolve_argv(argv_node, nodes)
                program = _program(argv) if argv is not None else None
            else:
                program, shell = argv_node, command_line
            if not _is_python(program, shell=shell):
                continue
            env = _resolve_env(env_node, node, scope, resolver)
            if env is None:
                continue
            problems = _problems(env)
            if problems:
                found.append((node.lineno, problems))
    return sorted(found)


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


_UNRESOLVED = "env at this launch cannot be resolved: "


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
        (32, ["no CUDA_VISIBLE_DEVICES"]),
        (36, ["no PYTORCH_NVML_BASED_CUDA_CHECK"]),
        (40, _BOTH),
        (45, [_UNRESOLVED + "aliased to another name (env is shared)"]),
        (54, _BOTH),
        (55, _BOTH),
        (61, [_UNRESOLVED + "changed on a branch / in a loop"]),
        (67, _BOTH),
        (71, [_UNRESOLVED + "changed on a branch / in a loop"]),
        (76, [_UNRESOLVED + "env is built in an enclosing function"]),
    ]
    # A rebinding from the env itself reads the state before that statement.
    assert scratch_env_children(
        "import subprocess, sys\ndef f():\n    env = {'PATH': '/bin'}\n"
        f"    env = {{**env, **{ok}}}\n    subprocess.run([sys.executable], env=env)\n"
    ) == []


def test_gpu_free_env_adds_both_variables_and_overrides_a_visible_gpu() -> None:
    assert GPU_FREE == {"CUDA_VISIBLE_DEVICES": "", "PYTORCH_NVML_BASED_CUDA_CHECK": "1"}
    env = gpu_free_env({"PATH": "/bin", "CUDA_VISIBLE_DEVICES": "0"})
    assert env == {"PATH": "/bin", "CUDA_VISIBLE_DEVICES": "", "PYTORCH_NVML_BASED_CUDA_CHECK": "1"}
