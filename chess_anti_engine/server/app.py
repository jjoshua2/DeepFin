from __future__ import annotations

import json
import logging
import os
import secrets
from pathlib import Path
from typing import Any

from chess_anti_engine.utils.versioning import version_lt


def create_app(
    *,
    server_root: str | Path = "server",
    publish_dir: str = "publish",
    inbox_dir: str = "inbox",
    quarantine_dir: str = "quarantine",
    users_db: str = "users.json",
    opening_book_path: str | None = None,
    max_upload_mb: int = 256,
):
    """Create the HTTP server.

    Layout under server_root:
    - publish/manifest.json
    - publish/latest_model.pt
    - inbox/<username>/<sha256>.npz
    - users.json
    """

    try:
        from fastapi import Body, Depends, FastAPI, File, Header, HTTPException, UploadFile
        from fastapi.responses import FileResponse, JSONResponse
        from fastapi.security import HTTPBasic, HTTPBasicCredentials

        # Important: this module uses `from __future__ import annotations`, so FastAPI/Pydantic
        # will resolve annotations (e.g. UploadFile) via the *module* globals, not function locals.
        # Export these types into globals() so file upload endpoints work under Pydantic v2.
        globals()["UploadFile"] = UploadFile
        globals()["HTTPBasicCredentials"] = HTTPBasicCredentials
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "FastAPI server requires optional dependencies. Install with: pip install -e '.[server]'"
        ) from e

    import hashlib

    from chess_anti_engine.replay.shard import load_npz

    from .auth import load_users, record_upload, save_users, verify_password

    root = Path(server_root)
    pub = root / publish_dir
    inbox = root / inbox_dir
    quarantine = root / quarantine_dir
    arena_inbox = root / "arena_inbox"
    users_path = root / users_db

    inbox.mkdir(parents=True, exist_ok=True)
    quarantine.mkdir(parents=True, exist_ok=True)
    arena_inbox.mkdir(parents=True, exist_ok=True)
    pub.mkdir(parents=True, exist_ok=True)

    log = logging.getLogger("chess_anti_engine.server")

    app = FastAPI(title="chess-anti-engine server", version="0.1")

    def _load_manifest() -> dict[str, Any] | None:
        mf = pub / "manifest.json"
        if not mf.exists():
            return None
        try:
            return dict(json.loads(mf.read_text(encoding="utf-8")))
        except Exception:
            return None

    def _check_worker_compat(
        *,
        worker_version: str | None,
        worker_protocol: str | None,
    ) -> tuple[bool, str]:
        """Check whether a worker is allowed to participate.

        This is intentionally driven by the learner-published manifest so the learner
        can upgrade protocol requirements without server CLI changes.
        """
        mf = _load_manifest()
        if mf is None:
            return True, ""

        min_v = mf.get("min_worker_version")
        req_proto = mf.get("protocol_version")

        # Backward-compat: if fields are missing, don't enforce.
        if min_v is None and req_proto is None:
            return True, ""

        wv = str(worker_version or "0.0.0")
        wp = str(worker_protocol or "0")

        if req_proto is not None:
            try:
                req_p = int(req_proto)
                got_p = int(wp)
            except Exception:
                return False, f"bad protocol version header (got {wp!r})"
            if got_p != req_p:
                return False, f"protocol mismatch: worker={got_p} required={req_p}"

        if min_v is not None:
            try:
                if version_lt(wv, str(min_v)):
                    return False, f"worker too old: worker={wv} min_required={min_v}"
            except Exception:
                return False, f"bad worker version header (got {wv!r})"

        return True, ""

    basic = HTTPBasic()

    def _sha256_file(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as f:
            while True:
                b = f.read(1024 * 1024)
                if not b:
                    break
                h.update(b)
        return h.hexdigest()

    def _auth_user(creds: HTTPBasicCredentials = Depends(basic)) -> str:
        users = load_users(users_path)
        rec = users.get(str(creds.username))
        if rec is None:
            raise HTTPException(status_code=401, detail="unknown user")
        if bool(rec.disabled):
            raise HTTPException(status_code=403, detail="user disabled")
        if not verify_password(str(creds.password), rec):
            raise HTTPException(status_code=401, detail="bad password")
        return str(creds.username)

    @app.get("/v1/manifest")
    def get_manifest(
        x_cae_worker_version: str | None = Header(None, alias="X-CAE-Worker-Version"),
        x_cae_protocol_version: str | None = Header(None, alias="X-CAE-Protocol-Version"),
    ) -> Any:
        mf = pub / "manifest.json"
        if not mf.exists():
            raise HTTPException(status_code=404, detail="manifest not published yet")

        ok, reason = _check_worker_compat(
            worker_version=x_cae_worker_version,
            worker_protocol=x_cae_protocol_version,
        )
        if not ok:
            # 426 (Upgrade Required) communicates "update your client".
            raise HTTPException(status_code=426, detail=reason)

        return JSONResponse(content=json.loads(mf.read_text(encoding="utf-8")))

    @app.get("/v1/model")
    def get_model() -> Any:
        mp = pub / "latest_model.pt"
        if not mp.exists():
            raise HTTPException(status_code=404, detail="model not published yet")
        return FileResponse(str(mp), media_type="application/octet-stream", filename="latest_model.pt")

    @app.get("/v1/best_model")
    def get_best_model() -> Any:
        mp = pub / "best_model.pt"
        if not mp.exists():
            raise HTTPException(status_code=404, detail="best model not published yet")
        return FileResponse(str(mp), media_type="application/octet-stream", filename="best_model.pt")

    @app.get("/v1/opening_book")
    def get_opening_book() -> Any:
        if opening_book_path is None:
            raise HTTPException(status_code=404, detail="no opening book configured")
        p = Path(opening_book_path)
        if not p.exists():
            raise HTTPException(status_code=404, detail="opening book not found")
        return FileResponse(str(p), media_type="application/octet-stream", filename=p.name)

    def _artifact_from_publish(key: str, *, default_name: str) -> Path | None:
        mf = _load_manifest() or {}
        rec = mf.get(key)
        if isinstance(rec, dict) and rec.get("filename"):
            name = str(rec.get("filename"))
        else:
            name = str(default_name)
        p = pub / name
        if p.exists() and p.is_file():
            return p
        return None

    @app.get("/v1/stockfish")
    def get_stockfish() -> Any:
        p = _artifact_from_publish("stockfish", default_name="stockfish")
        if p is None:
            raise HTTPException(status_code=404, detail="stockfish not published")
        return FileResponse(str(p), media_type="application/octet-stream", filename=p.name)

    @app.get("/v1/update_info")
    def get_update_info() -> Any:
        """Minimal compatibility/update metadata.

        This endpoint intentionally does NOT enforce worker compatibility so an out-of-date
        worker can still learn how to update itself.
        """
        mf = _load_manifest() or {}
        out: dict[str, Any] = {
            "server_version": mf.get("server_version"),
            "protocol_version": mf.get("protocol_version"),
            "min_worker_version": mf.get("min_worker_version"),
        }
        if isinstance(mf.get("worker_wheel"), dict):
            out["worker_wheel"] = mf.get("worker_wheel")
        return JSONResponse(content=out)

    @app.get("/v1/worker_wheel")
    def get_worker_wheel() -> Any:
        p = _artifact_from_publish("worker_wheel", default_name="worker.whl")
        if p is None:
            raise HTTPException(status_code=404, detail="worker wheel not published")
        return FileResponse(str(p), media_type="application/octet-stream", filename=p.name)

    @app.post("/v1/upload_shard")
    async def upload_shard(
        file: UploadFile = File(...),
        username: str = Depends(_auth_user),
        x_cae_worker_version: str | None = Header(None, alias="X-CAE-Worker-Version"),
        x_cae_protocol_version: str | None = Header(None, alias="X-CAE-Protocol-Version"),
    ) -> Any:
        ok, reason = _check_worker_compat(
            worker_version=x_cae_worker_version,
            worker_protocol=x_cae_protocol_version,
        )
        if not ok:
            log.warning("rejecting shard upload from user=%s: %s", username, reason)
            return {"stored": False, "rejected": True, "reason": reason}
        # Size guard: FastAPI doesn't enforce this automatically.
        max_bytes = int(max_upload_mb) * 1024 * 1024

        # Write to a temp file first.
        tmp = inbox / f"tmp_{os.getpid()}_{secrets.token_hex(8)}.npz"
        n = 0
        with tmp.open("wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                n += len(chunk)
                if n > max_bytes:
                    tmp.unlink(missing_ok=True)
                    raise HTTPException(status_code=413, detail="upload too large")
                f.write(chunk)

        # Validate shard by parsing it.
        try:
            samples, meta = load_npz(tmp)
        except Exception as e:
            # Quarantine so we can inspect bad uploads without causing worker retry storms.
            qdir = quarantine / "invalid"
            qdir.mkdir(parents=True, exist_ok=True)
            qpath = qdir / tmp.name
            try:
                tmp.replace(qpath)
                (qpath.with_suffix(qpath.suffix + ".reason.txt")).write_text(
                    f"{type(e).__name__}: {e}", encoding="utf-8"
                )
            except Exception:
                tmp.unlink(missing_ok=True)

            # Return 200 so well-behaved workers can move on; mark as rejected.
            return {
                "stored": False,
                "rejected": True,
                "reason": f"invalid shard: {type(e).__name__}: {e}",
            }

        sha = _sha256_file(tmp)
        user_dir = inbox / username
        user_dir.mkdir(parents=True, exist_ok=True)
        final = user_dir / f"{sha}.npz"

        if final.exists():
            tmp.unlink(missing_ok=True)
            stored = False
        else:
            tmp.replace(final)
            stored = True

        # Update user stats.
        try:
            users = load_users(users_path)
            record_upload(users, username=username, bytes_uploaded=int(n), positions=len(samples))
            save_users(users_path, users)
        except Exception:
            # Stats failure should not fail the upload.
            pass

        return {
            "stored": bool(stored),
            "sha256": sha,
            "bytes": int(n),
            "positions": int(len(samples)),
            "meta": meta,
        }

    @app.post("/v1/upload_arena_result")
    async def upload_arena_result(
        payload: dict[str, Any] = Body(...),
        username: str = Depends(_auth_user),
        x_cae_worker_version: str | None = Header(None, alias="X-CAE-Worker-Version"),
        x_cae_protocol_version: str | None = Header(None, alias="X-CAE-Protocol-Version"),
    ) -> Any:
        ok, reason = _check_worker_compat(
            worker_version=x_cae_worker_version,
            worker_protocol=x_cae_protocol_version,
        )
        if not ok:
            log.warning("rejecting arena upload from user=%s: %s", username, reason)
            return {"stored": False, "rejected": True, "reason": reason}
        # Basic schema validation
        def _req_int(k: str) -> int:
            if k not in payload:
                raise HTTPException(status_code=400, detail=f"missing field {k}")
            try:
                return int(payload[k])
            except Exception:
                raise HTTPException(status_code=400, detail=f"bad int field {k}")

        def _req_str(k: str) -> str:
            if k not in payload:
                raise HTTPException(status_code=400, detail=f"missing field {k}")
            v = str(payload[k])
            if not v:
                raise HTTPException(status_code=400, detail=f"empty field {k}")
            return v

        games = _req_int("games")
        a_win = _req_int("a_win")
        a_draw = _req_int("a_draw")
        a_loss = _req_int("a_loss")
        if a_win + a_draw + a_loss != games:
            raise HTTPException(status_code=400, detail="W/D/L must sum to games")

        a_sha = _req_str("a_sha256")
        b_sha = _req_str("b_sha256")
        ts = int(payload.get("generated_at_unix") or 0)

        # Store under arena_inbox/<username>/
        user_dir = arena_inbox / username
        user_dir.mkdir(parents=True, exist_ok=True)

        body = json.dumps(payload, sort_keys=True).encode("utf-8")
        import hashlib

        sha = hashlib.sha256(body).hexdigest()
        out = user_dir / f"{sha}.json"
        if not out.exists():
            out.write_bytes(body)

        return {
            "stored": True,
            "sha256": sha,
            "username": username,
            "games": int(games),
            "a_sha256": a_sha,
            "b_sha256": b_sha,
            "generated_at_unix": int(ts),
        }

    return app
