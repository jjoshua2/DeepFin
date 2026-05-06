from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import queue
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


def _run_multi(workers: int, matrix_size: int, stagger_s: float, timeout_s: float) -> int:
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
        try:
            results.append(queue.get(timeout=timeout_s))
        except queue.Empty:
            break

    exit_codes: list[int] = []
    for p in procs:
        p.join(timeout=timeout_s)
        if p.is_alive():
            p.terminate()
            p.join(timeout=5.0)
        exit_codes.append(-1 if p.exitcode is None else int(p.exitcode))

    results.sort(key=lambda r: r.worker)
    for res in results:
        status = "OK" if res.ok else "FAIL"
        print(f"[worker {res.worker}] {status} elapsed={res.elapsed_s:.3f}s {res.detail}")
    missing = workers - len(results)
    if missing:
        print(f"missing_results={missing}")

    print(f"exit_codes={exit_codes}")
    return 0 if len(results) == workers and all(code == 0 for code in exit_codes) and all(r.ok for r in results) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Minimal PyTorch CUDA sanity/repro check.")
    parser.add_argument("--workers", type=int, default=1, help="Number of spawned processes to test.")
    parser.add_argument("--matrix-size", type=int, default=1024, help="Square matrix size for a tiny GEMM.")
    parser.add_argument("--stagger-seconds", type=float, default=0.0, help="Delay between worker launches.")
    parser.add_argument("--timeout-seconds", type=float, default=60.0, help="Per-worker result/join timeout.")
    args = parser.parse_args()

    if args.workers <= 0:
        raise SystemExit("--workers must be > 0")
    if args.matrix_size <= 0:
        raise SystemExit("--matrix-size must be > 0")
    if args.stagger_seconds < 0:
        raise SystemExit("--stagger-seconds must be >= 0")
    if args.timeout_seconds <= 0:
        raise SystemExit("--timeout-seconds must be > 0")

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
        timeout_s=float(args.timeout_seconds),
    )


if __name__ == "__main__":
    raise SystemExit(main())
