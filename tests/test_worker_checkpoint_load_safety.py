"""The worker loads checkpoints that came off the network.

The manifest sha256 is verified before the load, but it travels the SAME
unauthenticated channel as the file itself, so it does not authenticate
anything an on-path attacker could not also rewrite. `weights_only=True` is
what stands between a swapped checkpoint and arbitrary code execution on a
volunteer's machine.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

# Resolved by path, NOT imported: `chess_anti_engine.worker` pulls in the
# compiled `encoding._lc0_ext` at import time, which would make this a test of
# whether the extension is built rather than of the load sites.
WORKER_PY = Path(__file__).resolve().parents[1] / "chess_anti_engine" / "worker.py"


def _torch_load_calls(path: Path) -> list[ast.Call]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: list[ast.Call] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if isinstance(fn, ast.Attribute) and fn.attr == "load":
            base = fn.value
            if isinstance(base, ast.Name) and base.id == "torch":
                out.append(node)
    return out


def test_every_worker_torch_load_is_weights_only() -> None:
    """Reachability pin, deliberately AST-based.

    A behavioural test can only cover the load sites it happens to reach; this
    fails on a NEW unguarded `torch.load` added anywhere in the worker, which
    is the regression that actually matters. It is a source-shape assertion and
    is not evidence that the flag takes effect -- that is the round-trip test
    below.
    """
    assert WORKER_PY.exists(), f"worker.py not found at {WORKER_PY}"
    calls = _torch_load_calls(WORKER_PY)
    assert calls, "no torch.load found in worker.py — did the module move?"

    unguarded = []
    for call in calls:
        kw = {k.arg: k.value for k in call.keywords}
        val = kw.get("weights_only")
        if not (isinstance(val, ast.Constant) and val.value is True):
            unguarded.append(call.lineno)

    assert not unguarded, (
        f"worker.py:{unguarded} calls torch.load without weights_only=True on a "
        "file fetched over the network"
    )


def test_ast_guard_can_actually_fail(tmp_path) -> None:
    """A gate that cannot fail is worse than no gate."""
    src = tmp_path / "sample.py"
    src.write_text("import torch\nx = torch.load('a.pt')\n", encoding="utf-8")
    calls = _torch_load_calls(src)
    assert len(calls) == 1
    assert "weights_only" not in {k.arg for k in calls[0].keywords}


def test_published_export_payload_loads_under_weights_only(tmp_path) -> None:
    """The flag must not break the payload the worker actually downloads.

    `Trainer.export_swa` writes `{"model": state_dict, "arch": {...primitives}}`,
    which is what `/v1/model` serves. If weights_only=True refused this shape,
    the change would take the whole fleet down rather than harden it.
    """
    export = {
        "model": {
            "block.0.weight": torch.zeros(2, 3),
            "block.0.bias": torch.zeros(3),
        },
        "arch": {
            "_schema_version": 3,
            "dim": 512,
            "n_layers": 16,
            "ffn_mult": [1.5, 1.9],
            "use_smolgen": True,
            "name": "pbt2_small",
        },
    }
    path = tmp_path / "model_abc.pt"
    torch.save(export, str(path))

    ckpt = torch.load(str(path), map_location="cpu", weights_only=True)
    sd = ckpt.get("model", ckpt)
    assert set(sd) == {"block.0.weight", "block.0.bias"}
    assert ckpt["arch"]["dim"] == 512
    assert ckpt["arch"]["ffn_mult"] == [1.5, 1.9]


def test_weights_only_refuses_a_non_tensor_object() -> None:
    """The property being bought: a checkpoint carrying an arbitrary object is
    refused rather than unpickled."""

    import io

    class Evil:
        def __reduce__(self):
            return (print, ("pwned",))

    # torch.save round-trips through pickle, so an object with __reduce__ is
    # exactly the payload weights_only is designed to refuse.
    buf = io.BytesIO()
    torch.save({"model": {"w": torch.zeros(1)}, "extra": Evil()}, buf)

    # `match` covers both wordings torch has used for this refusal; the point
    # is that it refuses, not which sentence it refuses with.
    buf.seek(0)
    with pytest.raises(Exception, match=r"(?is).*(weights_only|unsupported|not allowed).*"):
        torch.load(buf, map_location="cpu", weights_only=True)

    # Negative control: the SAME bytes load fine without the flag, so this test
    # measures the flag rather than a malformed file. `Evil.__reduce__` returns
    # `(print, ...)`, so a permissive load CALLS it and stores its None return
    # -- the "extra" value being None is the proof that the pickle executed.
    buf.seek(0)
    permissive = torch.load(buf, map_location="cpu", weights_only=False)
    assert permissive["extra"] is None, "negative control did not execute the payload"


def test_upload_digest_header_name_is_shared_not_duplicated() -> None:
    """The worker sends this header and the server matches it BY NAME.

    Two string literals across a process boundary are one typo away from a
    check that silently never fires, and the server cannot tell a misspelled
    header from an absent one -- absent is the backward-compatible accept path.
    So the name must come from one definition, and neither side may re-spell it.
    """
    from chess_anti_engine.version import UPLOAD_CONTENT_SHA256_HEADER

    assert UPLOAD_CONTENT_SHA256_HEADER == "X-CAE-Content-SHA256"

    app_py = Path(__file__).resolve().parents[1] / "chess_anti_engine" / "server" / "app.py"
    for path in (WORKER_PY, app_py):
        src = path.read_text(encoding="utf-8")
        assert "UPLOAD_CONTENT_SHA256_HEADER" in src, f"{path.name} does not use the shared constant"
        assert f'"{UPLOAD_CONTENT_SHA256_HEADER}"' not in src, (
            f"{path.name} re-spells the header as a literal instead of importing it"
        )
