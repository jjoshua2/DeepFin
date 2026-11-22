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
literal, positional or ``args=``, or a name bound once in the same function to
one. A launch counts as starting python when its first program, after any
``timeout`` / ``nice`` / ``env`` / ``setsid`` / ``stdbuf`` wrapper and that
wrapper's options, is ``sys.executable`` or a ``python*`` literal.

The env may be a dict literal, ``dict(K=...)``, or a name bound once in the
same function to either; ``env[K] = v`` and ``env.update({...})`` /
``env.update(K=...)`` on that name in the same function count towards it. A
dict that spreads anything (``**os.environ``), or is updated from anything
but a literal, inherits from the suite and is left alone; so is any other
expression (``gpu_free_env(...)``, ``os.environ.copy()``).

Still missed: an env from a dict comprehension, or returned by a helper
function, or built in another function than the launch; a launch through a
helper of its own (e.g. a ``_run`` wrapper whose argv is a parameter); an
argv assembled with ``+`` / ``*`` unpacking from names. A guard against the
common shapes, not a proof.
"""
from __future__ import annotations

import ast
from pathlib import Path

from tests._child_env import GPU_FREE, gpu_free_env

TESTS = Path(__file__).resolve().parent
_SUBPROCESS_CALLS = {"run", "Popen", "check_output", "check_call", "call"}
_ASYNC_CALLS = {"create_subprocess_exec"}
# name -> (index of the program argument, index of a positional env or None)
_OS_EXEC_CALLS = {"execve": (0, 2), "execvpe": (0, 2), "posix_spawn": (0, 2), "posix_spawnp": (0, 2)}
_WRAPPERS = {"timeout", "nice", "env", "setsid", "stdbuf", "ionice", "taskset"}


def _is_python(node: ast.expr | None) -> bool:
    if isinstance(node, ast.Attribute) and node.attr == "executable":
        return True
    return (isinstance(node, ast.Constant) and isinstance(node.value, str)
            and node.value.rsplit("/", 1)[-1].startswith("python"))


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
            # the wrapper's own options / arguments: -n 10, 5s, -k 1, VAR=val, ...
            while i < len(elts):
                w = elts[i]
                if not (isinstance(w, ast.Constant) and isinstance(w.value, str)):
                    break
                t = w.value
                if t.startswith("-") or "=" in t or t.replace(".", "", 1).rstrip("smhd").isdigit():
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


class _Env:
    """What a scope does to one env dict: its constant keys and values, or
    ``inherits`` when anything non-literal is mixed in."""

    def __init__(self) -> None:
        self.values: dict[str, object] = {}
        self.inherits = False

    def take_dict(self, node: ast.expr) -> bool:
        if isinstance(node, ast.Dict):
            for k, v in zip(node.keys, node.values):
                if k is None:
                    self.inherits = True
                elif isinstance(k, ast.Constant) and isinstance(k.value, str):
                    self.values[k.value] = v.value if isinstance(v, ast.Constant) else ...
            return True
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "dict"
                and not node.args):
            self.take_keywords(node.keywords)
            return True
        return False

    def take_keywords(self, keywords: list[ast.keyword]) -> None:
        for kw in keywords:
            if kw.arg is None:
                self.inherits = True
            else:
                self.values[kw.arg] = kw.value.value if isinstance(kw.value, ast.Constant) else ...


def _problems(env: _Env) -> list[str]:
    out = []
    for key, want in GPU_FREE.items():
        if key not in env.values:
            out.append(f"no {key}")
        elif env.values[key] != want:
            out.append(f"{key} is not {want!r}")
    return out


def _resolve_env(node: ast.expr | None, scope_nodes: list[ast.AST]) -> _Env | None:
    """The env a launch passes, or None when it inherits / is not a literal."""
    if node is None:
        return None
    env = _Env()
    if isinstance(node, ast.Name):
        name = node.id
        binds = [n for n in scope_nodes if isinstance(n, (ast.Assign, ast.AnnAssign))
                 and any(isinstance(t, ast.Name) and t.id == name
                         for t in (n.targets if isinstance(n, ast.Assign) else [n.target]))]
        if len(binds) != 1 or binds[0].value is None or not env.take_dict(binds[0].value):
            return None
        for n in scope_nodes:
            if isinstance(n, ast.Assign):
                for t in n.targets:
                    if (isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name)
                            and t.value.id == name):
                        key = t.slice
                        if isinstance(key, ast.Constant) and isinstance(key.value, str):
                            env.values[key.value] = n.value.value if isinstance(n.value, ast.Constant) else ...
                        else:
                            env.inherits = True
            if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "update"
                    and isinstance(n.func.value, ast.Name) and n.func.value.id == name):
                for a in n.args:
                    if not env.take_dict(a):
                        env.inherits = True
                env.take_keywords(n.keywords)
    elif not env.take_dict(node):
        return None
    return None if env.inherits else env


def _resolve_argv(node: ast.expr | None, scope_nodes: list[ast.AST]) -> list[ast.expr] | None:
    if isinstance(node, ast.Name):
        binds = [n for n in scope_nodes if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == node.id for t in n.targets)]
        node = binds[0].value if len(binds) == 1 else None
    if isinstance(node, (ast.List, ast.Tuple)):
        return list(node.elts)
    return None


def _launch(node: ast.Call, bare: set[str]) -> tuple[ast.expr | None, ast.expr | None] | None:
    """``(argv or program node, env node)`` if ``node`` launches a process."""
    func = node.func
    name = func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else None
    env = next((k.value for k in node.keywords if k.arg == "env"), None)
    if ((isinstance(func, ast.Attribute) and name in _SUBPROCESS_CALLS)
            or (isinstance(func, ast.Name) and name in bare)):
        argv = node.args[0] if node.args else next((k.value for k in node.keywords if k.arg == "args"), None)
        return argv, env
    if name in _ASYNC_CALLS and isinstance(func, ast.Attribute):
        return (node.args[0] if node.args else None), env
    if name in _OS_EXEC_CALLS and isinstance(func, ast.Attribute):
        prog, env_at = _OS_EXEC_CALLS[name]
        if env is None and len(node.args) > env_at:
            env = node.args[env_at]
        return (node.args[prog] if len(node.args) > prog else None), env
    return None


def scratch_env_children(source: str) -> list[tuple[int, list[str]]]:
    """``(line, problems)`` for each python child given a scratch env whose
    GPU-free variables are missing or wrong."""
    tree = ast.parse(source)
    bare = _subprocess_names(tree)
    found: list[tuple[int, list[str]]] = []
    for scope in _scopes(tree):
        nodes = _own_nodes(scope)
        for node in nodes:
            if not isinstance(node, ast.Call):
                continue
            launch = _launch(node, bare)
            if launch is None:
                continue
            argv_node, env_node = launch
            if isinstance(argv_node, (ast.Name, ast.List, ast.Tuple)):
                argv = _resolve_argv(argv_node, nodes)
                program = _program(argv) if argv is not None else None
            else:
                program = argv_node
            if not _is_python(program):
                continue
            env = _resolve_env(env_node, nodes)
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
    ]


def test_gpu_free_env_adds_both_variables_and_overrides_a_visible_gpu() -> None:
    assert GPU_FREE == {"CUDA_VISIBLE_DEVICES": "", "PYTORCH_NVML_BASED_CUDA_CHECK": "1"}
    env = gpu_free_env({"PATH": "/bin", "CUDA_VISIBLE_DEVICES": "0"})
    assert env == {"PATH": "/bin", "CUDA_VISIBLE_DEVICES": "", "PYTORCH_NVML_BASED_CUDA_CHECK": "1"}
