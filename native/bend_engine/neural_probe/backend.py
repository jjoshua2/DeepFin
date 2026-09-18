"""Persistent native CPU AOTI evaluator with bounded pipe IO and a pinned manifest."""
from __future__ import annotations

from contextlib import ExitStack
import hashlib
import json
import math
import os
from pathlib import Path
import select
import struct
import subprocess
import tempfile
import time
from typing import TYPE_CHECKING

import numpy as np

from .adapter import Encoding

if TYPE_CHECKING:
    from typing import BinaryIO

MAGIC = 0x44464E31
FORMAT = 'deepfin-tuple-policy-wdl-cpu-f32-v1'
HERE = Path(__file__).resolve().parent


def package_manifest(package: Path) -> tuple[dict[str, object], Encoding]:
    data = json.loads(package.with_suffix('.json').read_text())
    if not isinstance(data, dict) or data.get('format') != FORMAT:
        raise ValueError('unsupported evaluator manifest format')
    import torch
    if data.get('torch_version') != str(torch.__version__):
        raise ValueError('AOTI package/runtime Torch version mismatch')
    if data.get('sha256') != hashlib.sha256(package.read_bytes()).hexdigest():
        raise ValueError('evaluator package fingerprint mismatch')
    if (type(data.get('policy_width')) is not int or data.get('policy_width') != 1858
            or type(data.get('batch')) is not int or data.get('batch') != 1):
        raise ValueError('unsupported evaluator policy width or batch')
    history, extra, fix = (data.get(k) for k in ('input_history_encoding', 'input_extra_features', 'history_rep_fix'))
    if not isinstance(history, str) or not isinstance(extra, str) or type(fix) is not bool:
        raise ValueError('missing or invalid model encoding in evaluator manifest')
    encoding = Encoding(history, extra, fix)
    if type(data.get('channels')) is not int or data.get('channels') != encoding.channels:
        raise ValueError('manifest channels disagree with encoding')
    return data, encoding


def build_worker(directory: Path, cxx: str) -> Path:
    import torch
    directory.mkdir(parents=True, exist_ok=True)
    for command in ([
        'cmake', '-S', str(HERE), '-B', str(directory), '-DCMAKE_BUILD_TYPE=Release',
        '-DCMAKE_CXX_COMPILER=' + cxx, '-DCMAKE_PREFIX_PATH=' + torch.utils.cmake_prefix_path,
    ], ['cmake', '--build', str(directory), '--parallel', '1']):
        result = subprocess.run(command, capture_output=True, text=True, timeout=180, check=False)
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
    binary = directory / 'aoti_worker'
    needed = subprocess.run(['ldd', str(binary)], capture_output=True, text=True, timeout=10, check=True)
    if 'libpython' in needed.stdout.lower():
        raise RuntimeError('native evaluator unexpectedly links libpython')
    return binary


def read_exact(stream: BinaryIO, size: int, deadline: float) -> bytes:
    data = bytearray()
    while len(data) < size:
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not select.select([stream], [], [], remaining)[0]:
            raise TimeoutError('native evaluator reply deadline')
        part = os.read(stream.fileno(), size - len(data))
        if not part:
            raise RuntimeError('native evaluator closed its output')
        data.extend(part)
    return bytes(data)


def write_all(stream: BinaryIO, data: bytes, deadline: float) -> None:
    view = memoryview(data)
    while view:
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not select.select([], [stream], [], remaining)[1]:
            raise TimeoutError('native evaluator input deadline')
        try:
            written = os.write(stream.fileno(), view)
        except BlockingIOError:
            continue
        view = view[written:]


class NativeEvaluator:
    """One loaded package, many calls. No mutation/rebinding of package weights."""
    def __init__(self, binary: Path, package: Path, *, timeout: float = 30):
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('evaluator timeout must be finite and positive')
        self.manifest, self.encoding = package_manifest(package)
        self.timeout, self.sequence, self.failed = timeout, 1, False
        with ExitStack() as resources:
            self.errors = resources.enter_context(tempfile.TemporaryFile(mode='w+'))
            self.proc = subprocess.Popen([str(binary), str(package), str(self.encoding.channels)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.errors, bufsize=0)
            self.resources = resources.pop_all()
        assert self.proc.stdin is not None
        assert self.proc.stdout is not None
        self.input, self.output = self.proc.stdin, self.proc.stdout
        os.set_blocking(self.input.fileno(), False)
        try:
            if struct.unpack('<2I', read_exact(self.output, 8, time.monotonic() + timeout)) != (MAGIC, 0):
                raise ValueError('invalid evaluator handshake')
        except BaseException:
            self.close()
            raise

    def evaluate(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.failed:
            raise RuntimeError('failed evaluator cannot be reused')
        if x.shape != (1, self.encoding.channels, 8, 8) or not np.isfinite(x).all():
            raise ValueError('invalid evaluator input tensor')
        data = np.asarray(x, dtype='<f4').tobytes()
        deadline = time.monotonic() + self.timeout
        try:
            write_all(self.input, struct.pack('<3I', MAGIC, self.sequence, x.size) + data, deadline)
            header = struct.unpack('<4I', read_exact(self.output, 16, deadline))
            if header != (MAGIC, self.sequence, 1858, 3):
                raise ValueError('invalid evaluator sequence or output header')
            result = np.frombuffer(read_exact(self.output, (1858 + 3) * 4, deadline), dtype='<f4').copy()
            if not np.isfinite(result).all():
                raise ValueError('nonfinite native evaluator response')
            self.sequence += 1
            return result[:1858][None], result[1858:][None]
        except BaseException:
            self.failed = True
            raise

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()
        self.proc.wait(timeout=5)
        self.input.close()
        self.output.close()
        self.resources.close()

    def finish(self) -> None:
        write_all(self.input, struct.pack('<3I', MAGIC, 0, 0), time.monotonic() + self.timeout)
        if self.proc.wait(timeout=5) != 0:
            self.errors.seek(0)
            raise RuntimeError('evaluator shutdown: ' + self.errors.read())
