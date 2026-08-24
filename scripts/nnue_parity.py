"""EXACT-integer parity gate: our native NNUE evaluator against Stockfish itself.

The oracle is Stockfish, not a second implementation of ours. Internal
Python-vs-C parity cannot find a rule that is wrong in BOTH, and a value-label
defect would poison a hundred-million-row corpus without ever raising — so the
reference this gate matches is the engine's own printed output:

    (Big net) NNUE evaluation          -71 (side to move, internal units)

which is ``psqt / 16 + positional / 16``, side-to-move POV, straight off the big
net with no post-processing. We deliberately do NOT match Stockfish's *final*
evaluation: that blends optimism, complexity, material and rule50 damping on top,
and would hide an eval defect behind four layers of scaling. The small-net
selection rule is likewise out of scope — this project always uses the big net,
so labels differ from SF's ``eval`` exactly where SF would pick the small one,
and matching the big-net line sidesteps that by construction.

**ANY mismatch on a non-check FEN is a bug. There is no tolerance band.**

⚑ In-check FENs are excluded and the excluded fraction is REPORTED. Stockfish's
``eval`` refuses them outright (``Final evaluation: none (in check)``) because
the network is undefined there; our evaluator refuses them at the seam for the
same reason. Callers must resolve check nodes RECURSIVELY (an evasion can itself
give check) before asking for a static value; the refusal is the enforcement
backstop for that invariant, not a substitute for it.

Three-layer localisation on failure — per-bucket PSQT, per-bucket positional,
then the total — is done against ``scripts/nnue_reference.py``'s numpy
implementation, which exposes the split that Stockfish's exact line does not.
That reference is a BISECTOR, never the gate: this harness always also reports
reference-vs-Stockfish on the total, so "wrong in both" shows up as the reference
failing too rather than as a green run.

Usage::

    PYTHONPATH=. python3 scripts/nnue_parity.py --pack big.pack --n 5000
    PYTHONPATH=. nice -n 19 python3 scripts/nnue_parity.py --pack big.pack --n 50000
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import chess

from scripts.nnue_fens import sample_fens, sample_fens_pooled
from scripts.nnue_parse import PSQT_BUCKETS

_BIG_EVAL_RE = re.compile(
    r"^\(Big net\) NNUE evaluation\s+([+-]?\d+)\s+\(side to move, internal units\)"
)
_EVAL_FILE_RE = re.compile(r"^option name EvalFile type string default (\S+)")


class InCheckRefused(Exception):
    """The position is in check, so no evaluator may return a number for it."""


class Backend(Protocol):
    """What the parity harness needs from an evaluator."""

    name: str

    def evaluate(self, fen: str) -> int:
        """Internal units, side-to-move POV. Raises for in-check positions."""
        ...


@dataclass
class Mismatch:
    fen: str
    stockfish: int
    ours: int
    detail: str


class StockfishDriver:
    """One long-lived Stockfish process driven with ``position fen`` + ``eval``."""

    def __init__(self, binary: str | Path) -> None:
        self.binary = str(binary)
        self.proc = subprocess.Popen(
            [self.binary],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        assert self.proc.stdin is not None
        assert self.proc.stdout is not None
        self.eval_file = ""
        self._send("uci")
        for line in self._read_until("uciok"):
            m = _EVAL_FILE_RE.match(line)
            if m:
                self.eval_file = m.group(1)
        self._sync()

    def _send(self, cmd: str) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()

    def _read_until(self, sentinel: str) -> list[str]:
        assert self.proc.stdout is not None
        lines: list[str] = []
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError(f"stockfish exited while waiting for {sentinel!r}")
            line = line.rstrip("\n")
            if line.strip() == sentinel:
                return lines
            lines.append(line)

    def _sync(self) -> list[str]:
        self._send("isready")
        return self._read_until("readyok")

    def eval_lines(self, fen: str) -> list[str]:
        self._send(f"position fen {fen}")
        self._send("eval")
        return self._sync()

    def evaluate(self, fen: str) -> int:
        """The big net's internal-units line. Raises if the engine refused."""
        lines = self.eval_lines(fen)
        for line in lines:
            m = _BIG_EVAL_RE.match(line.strip())
            if m:
                return int(m.group(1))
        if any("none (in check)" in line for line in lines):
            raise InCheckRefused(fen)
        raise RuntimeError(f"no big-net eval line for {fen!r}; got {lines!r}")

    def close(self) -> None:
        try:
            self._send("quit")
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()

    def __enter__(self) -> StockfishDriver:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


class ReferenceBackend:
    """The numpy reference forward pass (``scripts/nnue_reference.py``)."""

    def __init__(self, nnue_path: Path) -> None:
        self.name = "reference"
        from scripts.nnue_parse import parse
        from scripts.nnue_reference import ReferenceEvaluator

        self.net = parse(nnue_path)
        self.evaluator = ReferenceEvaluator(self.net)
        self.sha256 = self.net.source_sha256

    def evaluate(self, fen: str) -> int:
        return self.evaluator.evaluate(chess.Board(fen))


class NativeBackend:
    """The C evaluator, reached the way production will reach it.

    Positions go through ``CBoard.from_board`` — the same adapter the MCTS tree
    feeds the seam — so a defect in the CBoard→feature mapping fails this gate
    rather than hiding behind a parallel FEN path written just for the harness.
    """

    def __init__(self, pack_path: Path, simd: bool | None = None) -> None:
        from chess_anti_engine.encoding._lc0_ext import CBoard
        from chess_anti_engine.nnue import _nnue_ext

        self.ext = _nnue_ext
        self.cboard = CBoard
        if simd is not None:
            _nnue_ext.set_simd(simd)
        self.handle = _nnue_ext.load(str(pack_path))
        self.sha256 = _nnue_ext.source_sha256(self.handle)
        self.name = f"native-c ({'avx2' if _nnue_ext.simd_active() else 'scalar'})"

    def evaluate(self, fen: str) -> int:
        return self.ext.evaluate(self.handle, self.cboard.from_board(chess.Board(fen)))


def _localise(fen: str, ours: int, theirs: int, nnue_path: Path | None) -> str:
    """Three-layer localisation: per-bucket PSQT, per-bucket positional, total."""
    if nnue_path is None:
        return "(no --nnue given, cannot localise: pass the .nnue the pack was built from)"
    from scripts.nnue_parse import parse
    from scripts.nnue_reference import ReferenceEvaluator

    evaluator = ReferenceEvaluator(parse(nnue_path))
    trace = evaluator.trace(chess.Board(fen))
    rows = [
        f"    reference total  = {trace.total}  (bucket {trace.bucket})",
        f"    stockfish total  = {theirs}",
        f"    our total        = {ours}",
        "    per-bucket reference psqt/positional:",
    ]
    for b in range(PSQT_BUCKETS):
        marker = " <-- used" if b == trace.bucket else ""
        rows.append(f"      b{b}: psqt={trace.psqt[b]:>7} positional={trace.positional[b]:>7}{marker}")
    if trace.total == theirs:
        rows.append("    ⇒ LAYER: reference agrees with Stockfish, so the defect is in the C path.")
    else:
        rows.append(
            "    ⇒ LAYER: reference ALSO disagrees with Stockfish — the shared understanding of "
            "the format/feature rules is wrong, not just the C."
        )
    return "\n".join(rows)


def run_parity(
    backend: Backend,
    fens: list[str],
    stockfish: StockfishDriver,
    nnue_path: Path | None,
    max_report: int = 10,
) -> tuple[int, list[Mismatch], int]:
    """Compare every FEN. Returns ``(checked, mismatches, refused_in_check)``."""
    mismatches: list[Mismatch] = []
    checked = 0
    refused = 0
    for fen in fens:
        try:
            theirs = stockfish.evaluate(fen)
        except InCheckRefused:
            refused += 1
            continue
        ours = backend.evaluate(fen)
        checked += 1
        if ours != theirs:
            detail = (
                _localise(fen, ours, theirs, nnue_path) if len(mismatches) < max_report else ""
            )
            mismatches.append(Mismatch(fen=fen, stockfish=theirs, ours=ours, detail=detail))
    return checked, mismatches, refused


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", type=Path, help="weight pack for the native backend")
    ap.add_argument("--nnue", type=Path, help=".nnue file (reference backend / localisation)")
    ap.add_argument(
        "--backend",
        choices=("native", "reference"),
        default="native",
        help="which evaluator to gate (default: native)",
    )
    ap.add_argument("--n", type=int, default=5000, help="number of FENs (5k dev, 50k final gate)")
    ap.add_argument("--seed", type=int, default=20260824)
    ap.add_argument(
        "--seeds",
        type=int,
        default=1,
        help="pool this many independent stratified draws (see sample_fens_pooled: "
        "one seed does not scale past ~5k because some cells are structurally thin)",
    )
    ap.add_argument("--stockfish", help="path to a Stockfish binary (default: repo discovery)")
    ap.add_argument("--fens-in", type=Path, help="read FENs from this file instead of sampling")
    ap.add_argument("--fens-out", type=Path, help="write the sampled FENs here")
    ap.add_argument("--max-report", type=int, default=10)
    ap.add_argument(
        "--simd",
        choices=("avx2", "scalar"),
        help="native backend only: which kernels to gate. BOTH must pass — a SIMD "
        "path is only checked once it has been run against the engine itself.",
    )
    args = ap.parse_args(argv)

    if args.backend == "native" and args.pack is None:
        ap.error("--pack is required for the native backend")
    if args.backend == "reference" and args.nnue is None:
        ap.error("--nnue is required for the reference backend")

    sf_path = args.stockfish
    if sf_path is None:
        from chess_anti_engine.utils.engine_discovery import find_stockfish

        sf_path = find_stockfish()
    if not sf_path:
        print("no Stockfish binary found; pass --stockfish", file=sys.stderr)
        return 2

    simd = None if args.simd is None else (args.simd == "avx2")
    backend: Backend = (
        NativeBackend(args.pack, simd=simd)
        if args.backend == "native"
        else ReferenceBackend(args.nnue)
    )

    if args.fens_in:
        fens = [ln.strip() for ln in args.fens_in.read_text().splitlines() if ln.strip()]
        stats = None
    else:
        t0 = time.perf_counter()
        if args.seeds > 1:
            fens, stats = sample_fens_pooled(args.n, args.seeds, base_seed=args.seed)
        else:
            fens, stats = sample_fens(args.n, seed=args.seed)
        print(
            f"sampled {len(fens):,} FENs from {args.seeds} seed(s) base {args.seed} "
            f"in {time.perf_counter() - t0:.1f}s"
        )
        print(stats.coverage_report())
    if args.fens_out:
        args.fens_out.write_text("\n".join(fens) + "\n")

    with StockfishDriver(sf_path) as sf:
        print(f"stockfish        : {sf_path}")
        print(f"stockfish EvalFile: {sf.eval_file}")
        our_sha = getattr(backend, "sha256", "")
        if our_sha:
            print(f"our net sha256   : {our_sha}")
            # Stockfish names its nets nn-<first 12 hex of sha256>.nnue, so this
            # is a real provenance check, not a label comparison.
            expected = f"nn-{our_sha[:12]}.nnue"
            if sf.eval_file and sf.eval_file != expected:
                print(
                    f"⚑ NET MISMATCH: stockfish is running {sf.eval_file}, our weights are "
                    f"{expected}. The gate would be comparing two different networks.",
                    file=sys.stderr,
                )
                return 2

        t0 = time.perf_counter()
        checked, mismatches, refused = run_parity(
            backend, fens, sf, args.nnue, max_report=args.max_report
        )
        elapsed = time.perf_counter() - t0

    print()
    print(f"backend          : {backend.name}")
    print(f"FENs checked     : {checked:,}  in {elapsed:.1f}s")
    print(f"refused in check : {refused:,}  (should be 0 — the sampler excludes them)")
    if stats is not None:
        print(
            f"in-check excluded during sampling: {stats.in_check_excluded:,} / "
            f"{stats.considered:,} = {100.0 * stats.in_check_fraction:.2f}%"
        )
    print(f"MISMATCHES       : {len(mismatches):,}")
    for mm in mismatches[: args.max_report]:
        print(f"\n  FEN {mm.fen}\n    stockfish={mm.stockfish} ours={mm.ours}")
        if mm.detail:
            print(mm.detail)
    if mismatches:
        print("\nPARITY FAILED — any mismatch on a non-check FEN is a bug.")
        return 1
    print("\nPARITY PASSED — exact integer equality on every FEN.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
