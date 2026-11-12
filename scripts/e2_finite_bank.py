"""Owner-local finite bank sink; no generator/trainer launch or upload.

Compose on_state_ready/on_game_complete/on_game_provenance from join with one
finite producer call. finish selects by start ordinal, never result or timing.
GAME readback is qualified; experiment/TB/optimizer/batchtrace gates remain.
"""

from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any
from chess_anti_engine.train.e2_game_join import BufferedGameJoin, JoinedAttempt
from chess_anti_engine.replay.shard import save_local_shard_arrays, samples_to_arrays
from scripts.e2_bank_gate import qualify_game, atomic_exclusive


class FiniteBank:
    def __init__(self, root: Path, *, run_id: str, attempts: int, eligible_games: int):
        if (
            type(eligible_games) is not int
            or not 1 <= eligible_games <= attempts <= 160
        ):
            raise ValueError("bounded eligible/attempt counts required")
        self.root = Path(root)
        self.eligible_games = eligible_games
        self.records: dict[int, dict[str, Any]] = {}
        self.join = BufferedGameJoin(
            root, run_id=run_id, attempts=attempts, on_joined=self._accept
        )

    def _accept(self, attempt: JoinedAttempt, samples: tuple[Any, ...]):
        row: dict[str, Any] = {
            "ordinal": attempt.ordinal,
            "game_id": attempt.game_id,
            "exclusion": attempt.exclusion,
            "observation_sha256": attempt.observation_sha256,
        }
        if attempt.exclusion is None:
            assert attempt.natural is not None
            assert attempt.candidate_sha256 is not None
            directory = self.join.directory / str(attempt.ordinal)
            candidate = directory / (attempt.natural.trace_sha256 + ".candidate.json")
            if (
                hashlib.sha256(candidate.read_bytes()).hexdigest()
                != attempt.candidate_sha256
            ):
                raise ValueError("candidate changed before admission")
            shard = directory / "emitted.zarr"
            if shard.exists():
                raise FileExistsError(shard)
            save_local_shard_arrays(
                shard,
                arrs=samples_to_arrays(list(samples)),
                meta={
                    "history_rep_fix": True,
                    "input_history_encoding": "lc0_root_legacy_meta",
                    "policy_encoding": "lc0_1858",
                },
            )
            qualified = qualify_game(
                shard, candidate, attempt.candidate_sha256, directory
            )
            marker = qualified / "GAME-READBACK-COMMITTED.json"
            row.update(
                shard=str(shard.relative_to(self.join.directory)),
                candidate=str(candidate.relative_to(self.join.directory)),
                candidate_sha256=attempt.candidate_sha256,
                qualified=str(qualified.relative_to(self.join.directory)),
                readback_sha256=hashlib.sha256(marker.read_bytes()).hexdigest(),
            )
        self.records[attempt.ordinal] = row

    def finish(self):
        self.join.finish()
        ordered = [self.records[i] for i in range(self.join.attempts)]
        eligible = [r for r in ordered if r["exclusion"] is None]
        if len(eligible) < self.eligible_games:
            raise ValueError("insufficient eligible natural games; no bank committed")
        # Revalidate externally recorded artifact hashes before final publication.
        for row in ordered:
            observation = self.join.directory / f"{row['ordinal']}.observation.json"
            if (
                hashlib.sha256(observation.read_bytes()).hexdigest()
                != row["observation_sha256"]
            ):
                raise ValueError("observation changed")
            if row["exclusion"] is None:
                for path, key in (
                    (self.join.directory / row["candidate"], "candidate_sha256"),
                    (
                        self.join.directory
                        / row["qualified"]
                        / "GAME-READBACK-COMMITTED.json",
                        "readback_sha256",
                    ),
                ):
                    if hashlib.sha256(path.read_bytes()).hexdigest() != row[key]:
                        raise ValueError("readback/candidate changed")
        payload = {
            "status": "FINITE_READBACK_BANK_NOT_EXPERIMENT_READY",
            "run_id": self.join.run_id,
            "attempts": ordered,
            "selected_start_ordinals": [
                r["ordinal"] for r in eligible[: self.eligible_games]
            ],
            "selection": "first eligible start ordinals, independent of completion order",
            "remaining_gates": [
                "load-time shard/target revalidation",
                "TB execution controls",
                "shared 64x256 batch trace",
                "optimizer/model/LR runtime qualification",
            ],
        }
        data = (
            json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        ).encode()
        path = self.join.directory / "BANK-READBACK-COMMITTED.json"
        atomic_exclusive(path, data)
        return path
