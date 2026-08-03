"""The worker's inference-slot width must be checked on the path selfplay takes.

PR #321 review F2. ``--inference-slot-input-planes`` defaults to the v1 146
while production is 175. A narrower client fits inside the broker's larger
shared-memory segment, the request/response protocol completes, and the client
reads policy and WDL out of the middle of the broker's INPUT region — all-zero,
with no error (``INFERENCE_AUDIT.md`` I2).

The first version of this guard sat in ``_load_and_compile_model``, which a
production selfplay worker never reaches: with a broker client ``_sync_model``
returns before the load and ``_swap_model_from_manifest`` returns after only
re-tagging. So it fired on arena tasks and nowhere else. These tests drive the
REAL methods, on the real selfplay+client shape, and would go green again only
if the check moved back somewhere unreachable.
"""
from __future__ import annotations

import logging
from typing import Any, cast
from unittest.mock import Mock

import pytest

from chess_anti_engine.model import ModelConfig
from chess_anti_engine.worker import WorkerSession

PROD_PLANES = 175
V1_PLANES = 146


class _Client:
    """Stands in for SlotInferenceClient/MultiSlotInferenceClient."""

    def __init__(self, planes: int) -> None:
        self.input_planes = int(planes)


def _session(*, client: _Client | None) -> Any:
    session = object.__new__(WorkerSession)
    session.log = logging.getLogger("test.worker_slot_planes")
    session.inference_client = cast("Any", client)
    session._slot_planes_unknown_warned = False
    session.model_sha = ""
    session.model_step = 0
    session.last_model_sha = None
    session._flush_pre_swap_buffer_if_stale = Mock()
    # If the guard ever fails to fire, the flow must not silently reach a model
    # download — these blow up loudly instead of quietly passing.
    session._ensure_local_model_at_sha = Mock(
        side_effect=AssertionError("reached the model download past a plane skew")
    )
    session._load_and_compile_model = Mock(
        side_effect=AssertionError("built a local model past a plane skew")
    )
    return session


def _manifest(*, task: str, extra: str | None, sha: str = "a" * 8) -> dict:
    model_config: dict[str, Any] = {"num_layers": 2}
    if extra is not None:
        model_config["input_extra_features"] = extra
    return {
        "task": {"type": task},
        "model": {"sha256": sha},
        "trainer_step": 7,
        "model_config": model_config,
    }


def test_sync_model_selfplay_with_client_rejects_the_146_default() -> None:
    """The production shape: selfplay worker, broker client left at v1 146,
    manifest declaring the v2_threats model. This is the case the old placement
    could not observe."""
    session = _session(client=_Client(V1_PLANES))
    with pytest.raises(ValueError, match=r"146 input planes.*v2_threats.*175"):
        WorkerSession._sync_model(session, _manifest(task="selfplay", extra="v2_threats"))


def test_sync_model_selfplay_with_client_returns_before_any_model_load() -> None:
    """Reachability, stated as an assertion rather than as prose.

    On a matched width the selfplay+client path proceeds past the guard and then
    returns WITHOUT loading a model — which is exactly why a guard placed in
    ``_load_and_compile_model`` was dead here. If someone moves the check back
    there, the test above goes green-but-meaningless; this one records the reason.
    """
    session = _session(client=_Client(PROD_PLANES))
    WorkerSession._sync_model(session, _manifest(task="selfplay", extra="v2_threats"))
    session._ensure_local_model_at_sha.assert_not_called()
    session._load_and_compile_model.assert_not_called()
    assert session.model_sha == "a" * 8  # it did run, it just never loads


def test_sync_model_arena_with_client_also_checks() -> None:
    """Arena DOES build a local model; the guard must fire before the download."""
    session = _session(client=_Client(V1_PLANES))
    with pytest.raises(ValueError, match="inference slot is 146 input planes"):
        WorkerSession._sync_model(session, _manifest(task="arena", extra="v2_threats"))
    session._ensure_local_model_at_sha.assert_not_called()


def test_swap_model_from_manifest_with_client_checks_before_retagging() -> None:
    """The mid-batch tier-2 path re-tags shards and returns; it must not adopt a
    new model sha whose declared width the slot cannot serve."""
    session = _session(client=_Client(V1_PLANES))
    with pytest.raises(ValueError, match="INFERENCE_AUDIT I2"):
        WorkerSession._swap_model_from_manifest(
            session, _manifest(task="selfplay", extra="v2_threats", sha="b" * 8),
        )
    assert session.model_sha == ""  # not re-tagged behind a broken slot


def test_matching_width_passes_on_every_entry_point() -> None:
    for extra, planes in (("v2_threats", PROD_PLANES), ("v1", V1_PLANES)):
        session = _session(client=_Client(planes))
        WorkerSession._sync_model(session, _manifest(task="selfplay", extra=extra))
        session = _session(client=_Client(planes))
        WorkerSession._swap_model_from_manifest(
            session, _manifest(task="selfplay", extra=extra, sha="c" * 8),
        )


def test_no_client_is_not_an_error() -> None:
    """A worker with a local model has no slot to disagree with."""
    session = _session(client=None)
    session._ensure_local_model_at_sha = Mock(return_value=None)
    WorkerSession._sync_model(session, _manifest(task="selfplay", extra="v2_threats"))


def test_manifest_without_an_encoding_warns_once_instead_of_skipping_silently(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The one case the check cannot make must be visible, not silent — and must
    not spam a poll loop that runs every few seconds."""
    session = _session(client=_Client(V1_PLANES))
    with caplog.at_level(logging.WARNING):
        for _ in range(3):
            WorkerSession._sync_model(session, _manifest(task="selfplay", extra=None))
    hits = [r for r in caplog.records if "cannot verify inference slot width" in r.message]
    assert len(hits) == 1, [r.message for r in caplog.records]


def test_manifest_carrying_a_modelconfig_object_is_checked_too() -> None:
    """``_swap_model_from_manifest`` accepts ``model_config`` as a ModelConfig
    instance, not only as a dict. Reading only the dict form would downgrade the
    object form to the "cannot check" branch — a guard that quietly stops
    guarding on one of its two real inputs."""
    session = _session(client=_Client(V1_PLANES))
    manifest = _manifest(task="selfplay", extra="v2_threats", sha="d" * 8)
    manifest["model_config"] = ModelConfig(input_extra_features="v2_threats")
    with pytest.raises(ValueError, match="inference slot is 146 input planes"):
        WorkerSession._swap_model_from_manifest(session, manifest)

    session = _session(client=_Client(PROD_PLANES))
    WorkerSession._swap_model_from_manifest(session, manifest)
