"""Issue #411: an uploaded ``.zarr`` array whose dtype/codec runs an
attacker-chosen deserializer must be rejected BEFORE the training host decodes
any chunk.

⚑ These tests do NOT assert "the upload was rejected". The vulnerable code
already returns ``{"rejected": true}`` -- the deserializer executes on the
materialization pass and the rejection is reported AFTERWARDS. A rejection
assertion therefore passes against the unfixed code and proves nothing. Every
test here asserts the SIDE EFFECT (a marker file the malicious pickle would
create) did NOT occur. On unfixed code the marker appears; the guard must keep
it absent.

Channels covered: object dtype + Pickle object_codec (the audit's channel),
numeric dtype + Pickle as the COMPRESSOR, and numeric dtype + Pickle as a
FILTER. Fields covered: ``wdl_target`` (the probe's field), ``x`` and
``policy_target`` (every declared field decodes through its own codec). Sinks
covered: the live ``POST /v1/upload_shard`` route and the boot-time
``_scan_pending_dir`` recovery scan, which materializes eagerly with no lazy
pre-pass.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest
import zarr
from numcodecs import Pickle

from chess_anti_engine.replay.buffer import ReplaySample
from chess_anti_engine.replay.shard import (
    ShardMeta,
    pack_shard_for_upload,
    samples_to_arrays,
    save_local_shard_arrays,
)


def _sample(i: int = 0) -> ReplaySample:
    p = np.zeros(4672, dtype=np.float32)
    p[i % 4672] = 1.0
    return ReplaySample(
        x=np.zeros((146, 8, 8), dtype=np.float32),
        policy_target=p,
        wdl_target=1,
    )


def _seed_user(server_root: Path, username: str = "u", password: str = "p") -> None:
    from chess_anti_engine.server.auth import UserRecord, hash_password, save_users

    salt, hsh, iters = hash_password(password)
    users = {username: UserRecord(username=username, salt_b64=salt, hash_b64=hsh, iterations=iters)}
    save_users(server_root / "users.json", users)


def _build_client(server_root: Path, **kwargs):
    from fastapi.testclient import TestClient

    from chess_anti_engine.server.app import create_app

    app = create_app(server_root=str(server_root), users_db="users.json", **kwargs)
    return TestClient(app)


def _default_headers() -> dict[str, str]:
    return {"X-CAE-Worker-Version": "0.0.0", "X-CAE-Protocol-Version": "1"}


def _touch_marker(path: str) -> int:
    """Stand-in for the attacker's payload. Unpickling ``_Exploit`` invokes this
    module-global by reference -- exactly the arbitrary-callable execution the
    real exploit uses (``os.system`` in the scratchpad reproducer) -- and its
    only, observable effect is to create the marker file."""
    Path(path).write_text("pwned", encoding="utf-8")
    return 0


class _Exploit:
    """``pickle.loads`` on this calls ``_touch_marker(<marker>)`` -- the
    concrete, observable side effect that proves attacker-chosen code executed
    on the host during array materialization."""

    def __init__(self, marker: Path) -> None:
        self._marker = str(marker)

    def __reduce__(self) -> tuple:
        return (_touch_marker, (self._marker,))


def _build_valid_zarr(dirpath: Path, *, n: int = 3) -> Path:
    dirpath.mkdir(parents=True, exist_ok=True)
    zp = dirpath / "shard.zarr"
    save_local_shard_arrays(
        zp,
        arrs=samples_to_arrays([_sample(i) for i in range(n)]),
        meta=ShardMeta(username="u", games=1, positions=n, model_sha256="abc1234567", model_step=0),
    )
    return zp


def _poison(
    zp: Path,
    marker: Path,
    *,
    field: str = "wdl_target",
    channel: str = "object_dtype",
) -> None:
    """Replace ``field`` with an array whose codec runs a malicious pickle, then
    overwrite its single chunk with the weaponized payload."""
    g = zarr.open_group(str(zp), mode="a")
    shape = tuple(int(d) for d in g[field].shape)
    del g[field]
    if channel == "object_dtype":
        g.create_dataset(field, shape=shape, chunks=shape, dtype=object,
                         object_codec=Pickle(), compressor=None)
    elif channel == "compressor":
        # numeric dtype, but Pickle sits in the compressor slot and runs on decode
        g.create_dataset(field, shape=shape, chunks=shape, dtype="i1", compressor=Pickle())
    elif channel == "filter":
        g.create_dataset(field, shape=shape, chunks=shape, dtype="i1",
                         filters=[Pickle()], compressor=None)
    else:  # pragma: no cover - test wiring error
        raise ValueError(channel)
    # A single fully-covering chunk of an ndim-d array is keyed "0.0.…" with one
    # "0" per dimension (zarr v2 "." separator) -- NOT just "0" for d>1.
    chunk_key = ".".join("0" for _ in shape) if shape else "0"
    (zp / field / chunk_key).write_bytes(pickle.dumps(_Exploit(marker)))


def _poisoned_tar_bytes(
    tmp_path: Path,
    marker: Path,
    *,
    field: str = "wdl_target",
    channel: str = "object_dtype",
) -> bytes:
    zp = _build_valid_zarr(tmp_path / "src")
    _poison(zp, marker, field=field, channel=channel)
    _, buf = pack_shard_for_upload(zp)
    return buf.getvalue()


# --- The live upload route -------------------------------------------------

@pytest.mark.parametrize(
    ("field", "channel"),
    [
        ("wdl_target", "object_dtype"),
        ("wdl_target", "compressor"),
        ("wdl_target", "filter"),
        ("x", "object_dtype"),
        ("policy_target", "compressor"),
    ],
)
def test_upload_route_does_not_execute_uploader_codec(tmp_path, field, channel) -> None:
    server_root = tmp_path / "server"
    server_root.mkdir()
    _seed_user(server_root)
    client = _build_client(server_root)

    marker = tmp_path / f"PWNED_{field}_{channel}"
    marker.unlink(missing_ok=True)
    tar_bytes = _poisoned_tar_bytes(tmp_path / channel / field, marker, field=field, channel=channel)

    r = client.post(
        "/v1/upload_shard",
        auth=("u", "p"),
        files={"file": ("shard.zarr.tar", tar_bytes, "application/x-tar")},
        headers=_default_headers(),
    )

    # THE test: the deserializer must never have run.
    assert not marker.exists(), (
        f"uploaded {channel} codec on field {field!r} executed on the host "
        f"(marker {marker} was created) -- untrusted-deserialization guard breached"
    )
    # It is additionally rejected (defensive; NOT the security assertion).
    assert r.status_code == 200, r.text
    assert r.json().get("stored") is False
    assert r.json().get("rejected") is True
    # And nothing hostile was persisted into the inbox.
    inbox = server_root / "inbox"
    stored = list(inbox.rglob("*.zarr")) if inbox.exists() else []
    assert stored == [], f"a poisoned shard was stored: {stored}"


def test_upload_route_still_accepts_a_clean_production_shard(tmp_path) -> None:
    """The guard must admit exactly what production writes -- a shape-clean
    Blosc shard -- or it takes the fleet to zero."""
    server_root = tmp_path / "server"
    server_root.mkdir()
    _seed_user(server_root)
    client = _build_client(server_root)

    zp = _build_valid_zarr(tmp_path / "clean")
    _, buf = pack_shard_for_upload(zp)

    r = client.post(
        "/v1/upload_shard",
        auth=("u", "p"),
        files={"file": ("shard.zarr.tar", buf.getvalue(), "application/x-tar")},
        headers=_default_headers(),
    )
    assert r.status_code == 200, r.text
    assert r.json().get("stored") is True
    assert r.json().get("positions") == 3


# --- The guard must FAIL CLOSED, not default ------------------------------

def test_guard_rejects_field_slot_holding_a_subgroup(tmp_path) -> None:
    """⚑ A ``_SHARD_FIELDS`` slot holding a zarr sub-group instead of an array
    must be REJECTED BY THE GUARD, not defaulted through it.

    The natural spelling ``np.dtype(getattr(arr, "dtype", None))`` maps a
    MISSING dtype to ``float64`` -- kind ``f``, which the allowlist PERMITS --
    so the guard would accept precisely the member it could not classify. That
    is a security gate whose default is ACCEPT. This asserts the guard itself
    refuses, rather than leaning on the downstream ``validate_arrays``, which
    lives in a different function a later reordering could move or skip.
    """
    from chess_anti_engine.replay.shard import load_shard_arrays

    zp = _build_valid_zarr(tmp_path / "subgroup")
    g = zarr.open_group(str(zp), mode="a")
    del g["wdl_target"]
    g.create_group("wdl_target")  # a Group where an Array must be

    with pytest.raises(ValueError, match="not a zarr array"):
        load_shard_arrays(zp, lazy=True)
    # ...and on the eager path, which is what the upload/recovery sinks call.
    with pytest.raises(ValueError, match="not a zarr array"):
        load_shard_arrays(zp)


@pytest.mark.parametrize("bad_dtype", ["<U8", "<c8", "<M8[s]"])
def test_guard_rejects_non_numeric_dtype_carrying_no_codec(tmp_path, bad_dtype) -> None:
    """The dtype allowlist must earn its place independently of the filters check.

    ⚑ zarr refuses to OPEN an object dtype that carries no object codec
    (``MetadataError``), and an object codec always lands in ``filters`` -- so
    every object-dtype shard is already killed by the filters rule, and a
    mutation that admits kind ``O`` is an *equivalent mutant*. The dtype rule is
    not thereby redundant: these dtypes -- unicode, complex, datetime -- are
    accepted by zarr with NO filters and a normal blosc compressor, so the dtype
    check is the only rule standing between them and a materialization that
    downstream code never expects to type-check.
    """
    from numcodecs import Blosc

    from chess_anti_engine.replay.shard import load_shard_arrays

    zp = _build_valid_zarr(tmp_path / f"dt{abs(hash(bad_dtype))}")
    g = zarr.open_group(str(zp), mode="a")
    shape = tuple(int(d) for d in g["wdl_target"].shape)
    del g["wdl_target"]
    g.create_dataset(
        "wdl_target", shape=shape, chunks=shape, dtype=bad_dtype,
        filters=None, compressor=Blosc(cname="zstd", clevel=2),
    )
    # Precondition: this is the isolated channel -- no filters, allowlisted compressor.
    assert g["wdl_target"].filters in (None, [], ())
    assert getattr(g["wdl_target"].compressor, "codec_id", None) == "blosc"

    with pytest.raises(ValueError, match="non-numeric dtype"):
        load_shard_arrays(zp, lazy=True)


def test_guard_rejects_array_with_unreadable_dtype(tmp_path) -> None:
    """Fail-closed contract on a REAL ``zarr.core.Array`` whose dtype is absent.

    Pins the branch the fail-open spelling would have swallowed:
    ``np.dtype(None)`` is ``float64`` (kind ``f``), which the allowlist PERMITS.
    ``isinstance`` gates most of this, so without this test the branch is
    unkillable code -- and unkillable code is how a guard rots.
    """
    from chess_anti_engine.replay.shard import _reject_unsafe_shard_codecs

    zp = _build_valid_zarr(tmp_path / "nodtype")
    arr = zarr.open_group(str(zp), mode="r")["wdl_target"]
    assert isinstance(arr, zarr.Array)
    # A real Array that cannot report its dtype. `object.__setattr__` rather
    # than a plain assignment so this does not depend on zarr's slot layout.
    object.__setattr__(arr, "_dtype", None)

    with pytest.raises(ValueError, match="declares no dtype"):
        _reject_unsafe_shard_codecs({"wdl_target": arr})


def test_guard_rejects_non_array_member_directly() -> None:
    """The guard's own contract, independent of any store: a member it cannot
    positively identify as a zarr array is refused."""
    from chess_anti_engine.replay.shard import _reject_unsafe_shard_codecs

    class _NotAnArray:
        """Duck-types the attributes the guard reads, and is still not an array."""

        dtype = np.dtype("i1")
        filters = None
        compressor = None

    with pytest.raises(ValueError, match="not a zarr array"):
        _reject_unsafe_shard_codecs({"wdl_target": _NotAnArray()})


def test_guard_inspects_exactly_what_the_loader_materializes(tmp_path) -> None:
    """The guard receives the SAME proxy objects the lazy loader returns.

    Two independent walks over ``_SHARD_FIELDS`` -- one to guard, one to load --
    is how a guard and its loader drift apart later. Pinning identity keeps
    "the guard inspected what was decoded" true by construction.
    """
    from chess_anti_engine.replay import shard as shard_mod

    zp = _build_valid_zarr(tmp_path / "identity")
    seen: dict[str, object] = {}
    original = shard_mod._reject_unsafe_shard_codecs

    def _capture(proxies):
        seen.update(proxies)
        return original(proxies)

    shard_mod._reject_unsafe_shard_codecs = _capture  # type: ignore[assignment]
    try:
        arrs, _meta = shard_mod.load_shard_arrays(zp, lazy=True)
    finally:
        shard_mod._reject_unsafe_shard_codecs = original  # type: ignore[assignment]

    assert seen, "guard was not called on the lazy path"
    for name, proxy in seen.items():
        assert arrs[name] is proxy, (
            f"field {name!r} was guarded as one object and returned as another -- "
            f"the guard and the loader walked the group separately"
        )


# --- The sibling sink: boot-time _scan_pending_dir recovery ----------------

def test_pending_recovery_scan_does_not_execute_uploader_codec(tmp_path) -> None:
    """``_scan_pending_dir`` (added by #407) materializes a recovered shard with
    an EAGER ``load_shard_arrays`` and no lazy pre-pass, so it is a second
    materialization sink for the same uploaded bytes. It runs at boot, inside
    ``create_app``."""
    server_root = tmp_path / "server"
    server_root.mkdir()
    _seed_user(server_root)

    # Stage a poisoned shard where the default-inbox pending scan will find it.
    pending = server_root / "inbox" / "_pending"
    pending.mkdir(parents=True)
    zp = _build_valid_zarr(tmp_path / "pending_src")
    marker = tmp_path / "PWNED_recovery"
    marker.unlink(missing_ok=True)
    _poison(zp, marker, field="wdl_target", channel="object_dtype")
    # Move it into the pending dir under a plausible recovered-shard name.
    staged = pending / "recovered.zarr"
    zp.rename(staged)

    # Booting the app runs _recover_pending_uploads() -> _scan_pending_dir().
    _build_client(server_root)

    assert not marker.exists(), (
        "boot-time pending recovery decoded a poisoned shard and executed its "
        "codec -- the _scan_pending_dir sink bypassed the guard"
    )
