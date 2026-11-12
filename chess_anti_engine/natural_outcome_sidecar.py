"""Opt-in durable finite-bank crosswalk. Not a distributed shard schema change.

Install NaturalOutcomeSidecar on SelfplayState.natural_outcome_sink through
play_batch's on_state_ready callback. No existing worker installs it by default.
Capture is pre-rescore; post-rescore mismatch aborts before sample emission.
Stored input bytes are bound, not independently re-encoded or authenticated.
Every file is a CAPTURE_CANDIDATE, never a bank admission or emission receipt.
A directory-fsync failure may leave a visible candidate; consumers MUST require
an independently durable bank manifest binding actual emitted rows and this
candidate hash, plus input re-encoding/target-source gates. No bank commit writer
or qualified bank loader is provided here. Even a successful sink is not proof
that the later upload or bank commit succeeded.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any
from collections.abc import Sequence

import numpy as np

from chess_anti_engine.natural_outcome import NaturalOutcome, admit_rows


class NaturalOutcomeSidecar:
    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def __call__(
        self,
        receipt: NaturalOutcome,
        samples: Sequence[Any],
        *,
        post_rescore_result: str,
        tb_rescore_enabled: bool,
    ) -> None:
        if post_rescore_result != receipt.result:
            raise ValueError("TB/post-rescore outcome differs from natural result")
        if type(tb_rescore_enabled) is not bool or not 1 <= len(samples) <= 1600:
            raise ValueError("invalid bounded game crosswalk")
        rows = []
        for sample in samples:
            x = np.array(sample.x, copy=True)
            if (
                x.shape != (175, 8, 8)
                or x.dtype not in (np.dtype(np.float16), np.dtype(np.float32))
                or not np.isfinite(x).all()
            ):
                raise ValueError("invalid stored175 input")
            black = x[108, 0, 0]
            if black not in (0, 1) or not np.all(x[108] == black):
                raise ValueError("invalid STM plane")
            gid, ply, label = sample.game_id, sample.ply_index, sample.wdl_target
            if (
                type(gid) is not int
                or not 0 <= gid < 2**63
                or type(ply) is not int
                or type(label) is not int
            ):
                raise ValueError("invalid row identity")
            stored = x.astype("<f2")
            if not np.isfinite(stored).all():
                raise ValueError("stored input overflow")
            rows.append(
                {
                    "game_id": gid,
                    "ply": ply,
                    "stm_white": not bool(black),
                    "natural_wdl_class": label,
                    "stored_float16_x_sha256": hashlib.sha256(
                        stored.tobytes()
                    ).hexdigest(),
                }
            )
        if len({row["game_id"] for row in rows}) != 1 or len(
            {row["ply"] for row in rows}
        ) != len(rows):
            raise ValueError("mixed game or duplicate ply")
        for start in range(0, len(rows), 256):
            part = rows[start : start + 256]
            # Check this independent natural channel only, NOT stored-bank admission.
            # Even unchanged labels with TB enabled need a later bank gate; the
            # false flag here does not assert that TB rescoring never ran.
            admit_rows(
                receipt,
                expected_trace_sha256=receipt.trace_sha256,
                plies=[v["ply"] for v in part],
                stm_white=[v["stm_white"] for v in part],
                outcomes=[v["natural_wdl_class"] for v in part],
                tb_rescored=False,
            )
        payload = {
            "schema": 1,
            "status": "CAPTURE_CANDIDATE_NOT_BANK_COMMITTED",
            "stored_bank_labels_admitted": False,
            "input_planes_reencoded": False,
            "bank_gate": "independent durable bank manifest, actual rows and target-source admission required",
            "natural": asdict(receipt),
            "rows": rows,
            "post_rescore_result": post_rescore_result,
            "tb_rescore_enabled": tb_rescore_enabled,
            "source": "automatic natural terminal; never TB substituted",
            "history_authentication": "replay consistency only; external bank pin required",
        }
        data = (
            json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        ).encode()
        name = receipt.trace_sha256 + ".candidate.json"
        fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=self.directory)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            # Exclusive link, never replace any earlier receipt, including retries.
            os.link(temporary, self.directory / name)
            directory_fd = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            os.unlink(temporary)
