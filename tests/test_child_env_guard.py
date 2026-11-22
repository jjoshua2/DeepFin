"""Guard: a test that starts a python child with an `env` dict built from
scratch must give it the two variables that keep it off the real GPU
(`tests._child_env.gpu_free_env`), because such a child does not inherit what
`tests/conftest.py` sets.

The scan is an AST pass over `tests/*.py`. It looks at calls to
``subprocess.run`` / ``Popen`` / ``check_output`` / ``check_call`` / ``call``
whose argv starts with ``sys.executable`` (or a ``python*`` literal) and whose
``env=`` is a dict literal, or a name that every assignment in the same
function binds to a dict literal. A dict that spreads anything (``**os.environ``)
inherits from the suite and is left alone; so is any other expression
(``gpu_free_env(...)``, a dict comprehension over ``os.environ``, ...).

It misses an env built some other way (a helper returning a literal, a dict
mutated after it is built): a guard against the common shape, not a proof.
"""
from __future__ import annotations

import ast
from pathlib import Path

from tests._child_env import GPU_FREE, gpu_free_env

TESTS = Path(__file__).resolve().parent
_SUBPROCESS_CALLS = {"run", "Popen", "check_output", "check_call", "call"}


def _starts_python(call: ast.Call) -> bool:
    if not call.args:
        return False
    argv = call.args[0]
    first = (argv.elts[0] if argv.elts else None) if isinstance(argv, (ast.List, ast.Tuple)) else argv
    if isinstance(first, ast.Attribute) and first.attr == "executable":
        return True
    return (isinstance(first, ast.Constant) and isinstance(first.value, str)
            and first.value.rsplit("/", 1)[-1].startswith("python"))


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


def scratch_env_children(source: str) -> list[tuple[int, list[str]]]:
    """``(line, missing variables)`` for each python child given a scratch
    ``env`` dict without both GPU-free variables."""
    found: list[tuple[int, list[str]]] = []
    for scope in _scopes(ast.parse(source)):
        nodes = _own_nodes(scope)
        bound: dict[str, list[ast.expr]] = {}
        for node in nodes:
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Name) and node.value is not None:
                        bound.setdefault(target.id, []).append(node.value)
        for node in nodes:
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr in _SUBPROCESS_CALLS and _starts_python(node)):
                continue
            env = next((k.value for k in node.keywords if k.arg == "env"), None)
            if isinstance(env, ast.Name):
                values = bound.get(env.id, [])
                env = values[0] if len(values) == 1 and isinstance(values[0], ast.Dict) else None
            if not isinstance(env, ast.Dict) or any(k is None for k in env.keys):
                continue
            keys = {k.value for k in env.keys if isinstance(k, ast.Constant)}
            missing = sorted(set(GPU_FREE) - keys)
            if missing:
                found.append((node.lineno, missing))
    return sorted(found)


def test_no_python_child_gets_a_scratch_env_that_can_see_the_gpu() -> None:
    offenders = [
        f"{path.name}:{line} lacks {', '.join(missing)}"
        for path in sorted(TESTS.glob("*.py"))
        for line, missing in scratch_env_children(path.read_text(encoding="utf-8"))
    ]
    assert offenders == [], "wrap the env in tests._child_env.gpu_free_env(...):\n" + "\n".join(offenders)


def test_the_guard_sees_the_shapes_it_claims() -> None:
    src = '''
import os, subprocess, sys
def literal():
    subprocess.run([sys.executable, "x.py"], env={"PATH": "/bin"})
def named():
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
'''
    assert scratch_env_children(src) == [
        (4, ["CUDA_VISIBLE_DEVICES", "PYTORCH_NVML_BASED_CUDA_CHECK"]),
        (7, ["CUDA_VISIBLE_DEVICES", "PYTORCH_NVML_BASED_CUDA_CHECK"]),
        (9, ["PYTORCH_NVML_BASED_CUDA_CHECK"]),
    ]


def test_gpu_free_env_adds_both_variables_and_overrides_a_visible_gpu() -> None:
    assert GPU_FREE == {"CUDA_VISIBLE_DEVICES": "", "PYTORCH_NVML_BASED_CUDA_CHECK": "1"}
    env = gpu_free_env({"PATH": "/bin", "CUDA_VISIBLE_DEVICES": "0"})
    assert env == {"PATH": "/bin", "CUDA_VISIBLE_DEVICES": "", "PYTORCH_NVML_BASED_CUDA_CHECK": "1"}
