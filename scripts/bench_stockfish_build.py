#!/usr/bin/env python3
"""Paired benchmark for two same-source Stockfish binaries."""
from __future__ import annotations

import argparse
import hashlib
import re
import statistics
import subprocess
from dataclasses import dataclass
from pathlib import Path


_TOTAL_MS_RE = re.compile(r"^Total time \(ms\)\s*:\s*(\d+)$", re.MULTILINE)
_NODES_RE = re.compile(r"^Nodes searched\s*:\s*(\d+)$", re.MULTILINE)
_NPS_RE = re.compile(r"^Nodes/second\s*:\s*(\d+)$", re.MULTILINE)
_BESTMOVE_RE = re.compile(r"^bestmove\s+(.+)$", re.MULTILINE)


@dataclass(frozen=True)
class _Result:
    total_ms: int
    nodes: int
    nps: int
    semantic_hash: str


def _required_int(pattern: re.Pattern[str], output: str, label: str) -> int:
    match = pattern.search(output)
    if match is None:
        raise RuntimeError(f"Stockfish bench output omitted {label}")
    return int(match.group(1))


def _run(binary: Path, cpu: int, hash_mb: int, threads: int, depth: int) -> _Result:
    command = [
        "taskset",
        "-c",
        str(cpu),
        str(binary),
        "bench",
        str(hash_mb),
        str(threads),
        str(depth),
        "default",
        "depth",
    ]
    completed = subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    output = completed.stdout
    nodes = _required_int(_NODES_RE, output, "node count")
    bestmoves = _BESTMOVE_RE.findall(output)
    if not bestmoves:
        raise RuntimeError("Stockfish bench output omitted best moves")
    semantic = "\n".join((*bestmoves, f"nodes={nodes}"))
    return _Result(
        total_ms=_required_int(_TOTAL_MS_RE, output, "total time"),
        nodes=nodes,
        nps=_required_int(_NPS_RE, output, "nodes/second"),
        semantic_hash=hashlib.sha256(semantic.encode()).hexdigest()[:16],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--cpu", type=int, default=15)
    parser.add_argument("--hash-mb", type=int, default=16)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--depth", type=int, default=13)
    args = parser.parse_args()
    if args.rounds <= 0:
        raise SystemExit("--rounds must be positive")
    for binary in (args.baseline, args.candidate):
        if not binary.is_file():
            raise SystemExit(f"binary does not exist: {binary}")

    # Pay binary/page-cache initialization before the deciding alternation.
    _run(args.baseline, args.cpu, args.hash_mb, args.threads, args.depth)
    _run(args.candidate, args.cpu, args.hash_mb, args.threads, args.depth)

    baseline: list[_Result] = []
    candidate: list[_Result] = []
    for round_idx in range(args.rounds):
        order = (
            ((args.baseline, baseline), (args.candidate, candidate))
            if round_idx % 2 == 0
            else ((args.candidate, candidate), (args.baseline, baseline))
        )
        for binary, destination in order:
            destination.append(
                _run(binary, args.cpu, args.hash_mb, args.threads, args.depth),
            )

    baseline_nps = statistics.median(result.nps for result in baseline)
    candidate_nps = statistics.median(result.nps for result in candidate)
    hashes = {result.semantic_hash for result in (*baseline, *candidate)}
    nodes = {result.nodes for result in (*baseline, *candidate)}
    print(f"baseline median:  {baseline_nps:.0f} nodes/s")
    print(f"candidate median: {candidate_nps:.0f} nodes/s")
    print(f"candidate speedup: {candidate_nps / baseline_nps:.6f}x")
    print(f"semantic hashes equal: {len(hashes) == 1} ({', '.join(sorted(hashes))})")
    print(f"node counts equal: {len(nodes) == 1} ({', '.join(str(n) for n in sorted(nodes))})")


if __name__ == "__main__":
    main()
