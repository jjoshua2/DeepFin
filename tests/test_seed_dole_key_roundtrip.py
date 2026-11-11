"""Seed-dole key contract: the server's WRITER and the trainer's READER agree.

PR #1192 made ``_seed_dole_generation_for_iteration`` match the trial id
exactly, and its tests hand-write the gate/winner JSON. A hand-written file
proves the reader parses what the test author THINKS the server writes; it
cannot catch the server keying the files differently. This test lets the real
writers produce the files and the real reader consume them.

The trial id travels, end to end:

* trainer: ``trainable.py`` ``trial_id = str(_ctx.get_trial_id() or "trial")``
  -> ``_publish_distributed_trial_state(trial_id=...)`` publishes under
  ``trials/<trial_id>/publish`` with ``api_prefix=/v1/trials/{trial_id}``;
* worker: polls that manifest and POSTs the advertised
  ``seed_dole_claim_endpoint`` (``/v1/trials/{trial_id}/seed_dole_claim``);
* server: ``_serve_seed_dole_claim`` keys the gate by
  ``str(_normalize_trial_id(trial_id) or "")`` and ``_SeedDoleGate`` writes
  ``seed_dole_gate.json`` + ``seed_dole_gate.json.winners.json``;
* trainer again: ``_seed_dole_generation_for_iteration(trial_id=<same id>)``.

The default lane (``/v1/seed_dole_claim``, gate key ``""``) has no trainer
publisher -- the trainable always publishes under ``trials/<id>`` -- so it is
driven through the real route by placing a manifest at the server-root
publish dir, and additionally through ``_SeedDoleGate.claim_token`` directly.
"""
from __future__ import annotations

import asyncio
import shutil
import uuid
from pathlib import Path

import httpx

from chess_anti_engine.server.app import _SeedDoleGate, create_app
from chess_anti_engine.tune.distributed_runtime import (
    _publish_distributed_trial_state,
    _seed_dole_generation_for_iteration,
)
from chess_anti_engine.worker import _manifest_poll_headers
from tests.test_distributed_selfplay_backpressure import (
    _DOLE_SEED_FEN,
    _FakeTrainer,
    _model_cfg,
    _seed_dole_user,
)

# A Ray trial id as `get_trial_id()` returns it, and two ids that must NOT
# alias it: a sibling trial and the trial NAME form that embeds the id.
TRIAL_ID = "4c17c_00000"
SIBLING_ID = "4c17c_00001"
NAME_FORM = "train_x_4c17c_00000"
ITERATION = 12
# Distinct from ITERATION on purpose: with the two equal, a server that keyed
# the gate by the manifest's trainer_step instead of training_iteration
# would round-trip just as well and the test could not tell them apart.
TRAINER_STEP = ITERATION * 88 + 7


def _publish(server_root: Path, trial_id: str) -> None:
    fen_path = server_root / "blindspot.txt"
    fen_path.write_text(_DOLE_SEED_FEN + "\n", encoding="utf-8")
    _publish_distributed_trial_state(
        trainer=_FakeTrainer(),
        config={"opening_fen_dole_per_iter": 1, "opening_fen_list_path": str(fen_path)},
        model_cfg=_model_cfg(),
        server_root=server_root,
        trial_id=trial_id,
        training_iteration=ITERATION,
        trainer_step=TRAINER_STEP,
        sf_nodes=1000,
        mcts_simulations=64,
    )


def _claim_via_http(app, manifest_path: str) -> dict:
    """The worker's two-step protocol: GET the manifest, POST its endpoint."""
    async def _run() -> dict:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            headers = _manifest_poll_headers(worker_id="w")
            manifest = (await client.get(manifest_path, headers=headers)).json()
            endpoint = manifest["seed_dole_claim_endpoint"]
            r = await client.post(
                endpoint,
                json={
                    "claim_id": uuid.uuid4().hex,
                    "manifest_revision": manifest["manifest_revision"],
                },
                auth=("u", "p"),
                headers=headers,
            )
            r.raise_for_status()
            return {"endpoint": endpoint, **r.json()}

    return asyncio.run(_run())


def _read(server_root: Path, trial_id: str) -> str | None:
    return _seed_dole_generation_for_iteration(
        server_root=server_root, trial_id=trial_id, training_iteration=ITERATION,
    )


def test_named_trial_claim_round_trips_through_the_real_route(tmp_path: Path) -> None:
    _publish(tmp_path, TRIAL_ID)
    _seed_dole_user(tmp_path)
    app = create_app(server_root=tmp_path, users_db="users.json")

    res = _claim_via_http(app, f"/v1/trials/{TRIAL_ID}/manifest")
    assert res["endpoint"] == f"/v1/trials/{TRIAL_ID}/seed_dole_claim"
    assert res["granted"] is True
    token = res["grant_token"]
    assert token

    # The id the trainer passes reads back exactly the token the worker got.
    assert _read(tmp_path, TRIAL_ID) == token
    # Whitespace is normalised the same way on both sides.
    assert _read(tmp_path, f" {TRIAL_ID} ") == token
    # No aliasing: a sibling trial, the trial-name form, and the default lane
    # must not see this trial's generation.
    assert _read(tmp_path, SIBLING_ID) is None
    assert _read(tmp_path, NAME_FORM) is None
    assert _read(tmp_path, "") is None


def test_default_lane_claim_round_trips_through_the_real_route(tmp_path: Path) -> None:
    # The trainable never publishes the default lane, so stage a valid
    # manifest (and its assets) at the server-root publish dir.
    _publish(tmp_path, TRIAL_ID)
    shutil.copytree(tmp_path / "trials" / TRIAL_ID / "publish", tmp_path / "publish")
    _seed_dole_user(tmp_path)
    app = create_app(server_root=tmp_path, users_db="users.json")

    res = _claim_via_http(app, "/v1/manifest")
    assert res["endpoint"] == "/v1/seed_dole_claim"
    assert res["granted"] is True
    token = res["grant_token"]
    assert token

    assert _read(tmp_path, "") == token
    assert _read(tmp_path, TRIAL_ID) is None


def test_gate_writer_keys_named_and_default_lanes_independently(tmp_path: Path) -> None:
    """Directly through `_SeedDoleGate.claim_token` (and its winner sidecar),
    both lanes in ONE pair of files: each reads back its own token only."""
    gate = _SeedDoleGate(tmp_path / "seed_dole_gate.json", lease_seconds=10)

    async def _run() -> tuple[str, str]:
        named = await gate.claim_token(TRIAL_ID, ITERATION, claim_id="A", manifest_revision="r")
        default = await gate.claim_token("", ITERATION, claim_id="B", manifest_revision="r")
        return named, default

    named, default = asyncio.run(_run())
    assert named
    assert default
    assert named != default
    assert (tmp_path / "seed_dole_gate.json.winners.json").exists()
    assert _read(tmp_path, TRIAL_ID) == named
    assert _read(tmp_path, "") == default
    assert _read(tmp_path, SIBLING_ID) is None
    assert _read(tmp_path, NAME_FORM) is None
