"""Rolling full trainer state; recovery restarts sampling rather than replaying it."""
from __future__ import annotations

import hashlib
import json
import math
import random
import shutil
import time
import uuid
from pathlib import Path
from typing import Any
from collections.abc import Callable

import numpy as np
import torch

from chess_anti_engine.utils.atomic import _fsync_dir, atomic_write


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


class RollingRecoveryCheckpoints:
    """Commit a complete checkpoint bundle before pruning older committed bundles.

    Saves after the first completed window and then the first completed window
    after each interval. No signal handler interrupts an optimizer update.
    """

    def __init__(self, root: Path, *, interval_seconds: float = 3600,
                 keep: int = 2, clock: Callable[[], float] = time.monotonic) -> None:
        if not math.isfinite(interval_seconds) or interval_seconds < 0 or keep < 1:
            raise ValueError('recovery interval must be finite/nonnegative and keep positive')
        self.root = Path(root)
        self.interval = interval_seconds
        self.keep = keep
        self.clock = clock
        self.last_save: float | None = None

    def maybe_save(self, trainer: Any, *, progress: dict[str, Any],
                   metrics: dict[str, Any], sampler_rng: Any = None) -> Path | None:
        now = self.clock()
        if self.interval == 0 or (self.last_save is not None and now - self.last_save < self.interval):
            return None
        # A recovery snapshot is not permission to bank an invalid optimizer window.
        if (metrics.get('train_steps_done') != progress['window_steps']
                or metrics.get('transient_cuda_retry_batches', 0) != 0
                or metrics.get('grad_nonfinite_skip_rate', 0) != 0
                or any(isinstance(value, float) and not math.isfinite(value)
                       for value in metrics.values())):
            return None
        self.root.mkdir(parents=True, exist_ok=True)
        _fsync_dir(self.root.parent)  # Commit the new recovery directory entry.
        index = self.root / 'latest.json'
        previous = json.loads(index.read_text())['snapshots'] if index.exists() else []
        name = f'step-{int(trainer.step):012d}'
        final = self.root / name
        if final.exists():
            raise FileExistsError(f'recovery step already exists: {final}')
        staging = self.root / ('.writing-' + uuid.uuid4().hex)
        staging.mkdir()
        try:
            checkpoint = staging / 'checkpoint.pt'
            trainer.save(checkpoint)  # Trainer.save atomically persists all optimizer state.
            rng = {'python': random.getstate(), 'numpy': np.random.get_state(),
                   'torch_cpu': torch.get_rng_state(),
                   'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
                   'sampler': getattr(getattr(sampler_rng, 'bit_generator', None), 'state', None)}
            atomic_write(staging / 'rng.pt', lambda path: torch.save(rng, path))
            manifest = {'schema': 1, 'status': 'FULL_STATE_RECOVERY_WINDOW_NOT_FINAL_QUALIFICATION',
                        'global_optimizer_step': int(trainer.step), 'saved_unix': time.time(),
                        'progress': progress, 'checkpoint_sha256': _sha(checkpoint),
                        'rng_sha256': _sha(staging / 'rng.pt'),
                        'resume_semantics': 'Restore trainer state, then start a fresh explicitly seeded sampling pass. Sampler cursor/prefetch state is not persisted; exact interrupted-epoch replay is not supported.',
                        'qualification': 'Completed finite optimizer window; final corpus/recipe loss guards may not have run.'}
            atomic_write(staging / 'manifest.json', lambda path: path.write_text(json.dumps(manifest, indent=2) + '\n'))
            staging.rename(final)
            retained = [name, *previous][:self.keep]
            # Durable index publication fsyncs this directory, also committing the
            # bundle rename. Readers use only entries in this index.
            atomic_write(index, lambda path: path.write_text(json.dumps({'schema': 1, 'snapshots': retained}, indent=2) + '\n'))
            self.last_save = self.clock()
        finally:
            if staging.exists():
                shutil.rmtree(staging)
        for old in previous:
            if old not in retained:
                target = self.root / old
                if target.parent != self.root or not old.startswith('step-') or target.is_symlink():
                    raise ValueError('invalid recovery retention entry')
                shutil.rmtree(target)
        print(f'[recovery] saved step {trainer.step}: {final / "checkpoint.pt"}', flush=True)
        return final
