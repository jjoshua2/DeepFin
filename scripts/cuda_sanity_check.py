from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from dataclasses import dataclass


@dataclass
class Result:
    worker: int
    ok: bool
    detail: str
    elapsed_s: float


def _run_once(worker_idx: int, matrix_size: int) -> Result:
    t0 = time.time()
    try:
        import torch

        if not torch.cuda.is_available():
            return Result(worker=worker_idx, ok=False, detail="torch.cuda.is_available() == False", elapsed_s=time.time() - t0)

        x = torch.randn(matrix_size, matrix_size, device="cuda")
        y = x @ x
        torch.cuda.synchronize()
        dev = torch.cuda.get_device_name(0)
        return Result(worker=worker_idx, ok=True, detail=f"ok device={dev} sum={float(y.sum().item()):.3f}", elapsed_s=time.time() - t0)
    except Exception as exc:
        return Result(worker=worker_idx, ok=False, detail=f"{type(exc).__name__}: {exc}", elapsed_s=time.time() - t0)


def _child_main(worker_idx: int, matrix_size: int, queue: mp.Queue) -> None:
    res = _run_once(worker_idx=worker_idx, matrix_size=matrix_size)
    queue.put(res)


def _run_multi(workers: int, matrix_size: int, stagger_s: float) -> int:
    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue()
    procs: list[mp.Process] = []

    for i in range(workers):
        p = ctx.Process(target=_child_main, args=(i, matrix_size, queue))
        p.start()
        procs.append(p)
        if stagger_s > 0 and i + 1 < workers:
            time.sleep(stagger_s)

    results: list[Result] = []
    for _ in range(workers):
        results.append(queue.get())

    exit_codes: list[int] = []
    for p in procs:
        p.join(timeout=30.0)
        exit_codes.append(int(p.exitcode or 0))

    results.sort(key=lambda r: r.worker)
    for res in results:
        status = "OK" if res.ok else "FAIL"
        print(f"[worker {res.worker}] {status} elapsed={res.elapsed_s:.3f}s {res.detail}")

    print(f"exit_codes={exit_codes}")
    return 0 if all(code == 0 for code in exit_codes) and all(r.ok for r in results) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Minimal PyTorch CUDA sanity/repro check.")
    parser.add_argument("--workers", type=int, default=1, help="Number of spawned processes to test.")
    parser.add_argument("--matrix-size", type=int, default=1024, help="Square matrix size for a tiny GEMM.")
    parser.add_argument("--stagger-seconds", type=float, default=0.0, help="Delay between worker launches.")
    args = parser.parse_args()

    print(f"python={sys.version.split()[0]} pid={os.getpid()}")
    if args.workers <= 1:
        res = _run_once(worker_idx=0, matrix_size=args.matrix_size)
        status = "OK" if res.ok else "FAIL"
        print(f"[worker 0] {status} elapsed={res.elapsed_s:.3f}s {res.detail}")
        return 0 if res.ok else 1

    return _run_multi(
        workers=int(args.workers),
        matrix_size=int(args.matrix_size),
        stagger_s=float(args.stagger_seconds),
    )


if __name__ == "__main__":
    raise SystemExit(main())
