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
import gzip
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TextIO

import chess

from scripts.nnue_fens import (
    SampledPosition,
    read_sample,
    state_key_of_fen,
    sample_fens,
    sample_fens_pooled,
    write_sample,
)
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


@dataclass
class ParityResult:
    checked: int
    refused: int
    mismatches: list[Mismatch]
    observations: int


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
    fens: list[SampledPosition],
    stockfish: StockfishDriver,
    nnue_path: Path | None,
    max_report: int = 10,
    observations: TextIO | None = None,
) -> ParityResult:
    """Compare every FEN, banking every observation.

    ⚑ EVERY pair is written to ``observations``, not only the disagreements. A
    summary plus the exceptional rows is exactly enough to answer the question we
    already asked and nothing else: re-stratifying the sample, re-estimating on a
    subset, or checking a later engine build against this run all need the equal
    rows too, and re-running the engine to recover them re-rolls the sampling and
    the engine version at the same time, confounding the method change with them.
    The dump is the artifact; the count is a reading off it.
    """
    mismatches: list[Mismatch] = []
    checked = 0
    refused = 0
    written = 0
    for item in fens:
        fen = item.fen
        # ⚑ The resampling unit goes into every row. The sampler walks whole
        # random games, so positions from one playout are correlated; a banked
        # row without its cluster key cannot be re-analysed as anything but an
        # independent draw, which it is not.
        cluster = {"playout": item.playout, "ply": item.ply}
        try:
            theirs = stockfish.evaluate(fen)
        except InCheckRefused:
            refused += 1
            if observations is not None:
                observations.write(
                    json.dumps({"fen": fen, "refused": "in_check", **cluster}) + "\n"
                )
                written += 1
            continue
        ours = backend.evaluate(fen)
        checked += 1
        if observations is not None:
            observations.write(
                json.dumps({"fen": fen, "sf": theirs, "ours": ours, **cluster}) + "\n"
            )
            written += 1
        if ours != theirs:
            detail = (
                _localise(fen, ours, theirs, nnue_path) if len(mismatches) < max_report else ""
            )
            mismatches.append(Mismatch(fen=fen, stockfish=theirs, ours=ours, detail=detail))
    return ParityResult(
        checked=checked, refused=refused, mismatches=mismatches, observations=written
    )


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
        "--observations",
        type=Path,
        default=Path("nnue_parity_observations.jsonl.gz"),
        help="bank EVERY (fen, ours, stockfish) triple here, gzipped JSONL "
        "(default: %(default)s). Banking only the mismatches makes any later "
        "re-analysis need a fresh engine run, which re-rolls the sample and the "
        "engine build along with whatever was being changed.",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="allow the observation bank to replace an existing file at that path",
    )
    ap.add_argument(
        "--no-observations",
        action="store_true",
        help="skip the observation dump (for a quick local check, not for a gate run)",
    )
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
        fens = read_sample(args.fens_in)
        stats = None
    else:
        t0 = time.perf_counter()
        if args.seeds > 1:
            sampled, stats = sample_fens_pooled(args.n, args.seeds, base_seed=args.seed)
        else:
            sampled, stats = sample_fens(args.n, seed=args.seed)
        fens = [
            SampledPosition(f, *stats.origin.get(f, (None, None))) for f in sampled
        ]
        print(
            f"sampled {len(fens):,} FENs from {args.seeds} seed(s) base {args.seed} "
            f"in {time.perf_counter() - t0:.1f}s"
        )
        print(stats.coverage_report())
    if args.fens_out:
        if stats is not None:
            write_sample(args.fens_out, [f.fen for f in fens], stats)
        else:
            args.fens_out.write_text("\n".join(f.fen for f in fens) + "\n")

    # ⚑ A gate with nothing to compare is not a gate that passed. An empty
    # --fens-in, --n 0, or a sampler regression returning [] each used to print
    # PARITY PASSED and exit 0 — a green result whose meaning was "we checked
    # nothing", reported in the same words as "we checked fifty thousand".
    if not fens:
        print(
            "no FENs to check — a parity run with an empty sample proves nothing "
            "and does not pass",
            file=sys.stderr,
        )
        return 2
    # ⚑ With --fens-in there is no request to compare against, and printing the
    # unused --n default beside the delivered count invents a shortfall that
    # never happened.
    if args.fens_in:
        print(f"FENs read from   : {args.fens_in}   delivered: {len(fens):,}")
    else:
        print(f"FENs requested   : {args.n:,}   delivered: {len(fens):,}")
    # ⚑ Distinct EVALUATOR INPUTS and distinct RESAMPLING UNITS, both reported.
    # "50,000 FENs" is neither of those numbers, and it is the one that flatters.
    n_states = len({state_key_of_fen(f.fen) for f in fens})
    n_clusters = len({f.playout for f in fens if f.playout})
    print(
        f"distinct states  : {n_states:,} (placement+STM, what the net sees)   "
        f"playouts: {n_clusters:,}"
    )

    obs_handle: TextIO | None = None
    obs_path: Path | None = None
    obs_tmp: Path | None = None
    if not args.no_observations:
        obs_path = Path(args.observations)
        # ⚑⚑ NEVER DESTROY A PRIOR BANK. Opening the destination "wt" truncates
        # it at once — so a run that then bails out at the provenance check, or
        # dies on a missing engine, had already wiped an expensive previous
        # artifact before it discovered it had nothing to write. Two rules:
        # refuse an existing bank outright unless --overwrite, and stage into a
        # temp file that is published only once the comparison has actually run.
        if obs_path.exists() and not args.overwrite:
            print(
                f"observation bank {obs_path} already exists. Refusing to overwrite a "
                "previous run's dump — pass --overwrite, or give --observations a new "
                "path.",
                file=sys.stderr,
            )
            return 2
        obs_path.parent.mkdir(parents=True, exist_ok=True)
        obs_tmp = obs_path.with_name(obs_path.name + f".partial-{os.getpid()}")
        # Held open across the whole run and closed in the finally below; a
        # context manager here would need to wrap the entire engine session.
        obs_handle = gzip.open(obs_tmp, "wt", encoding="utf-8")  # noqa: SIM115

    published = False
    try:
        with StockfishDriver(sf_path) as sf:
            print(f"stockfish        : {sf_path}")
            print(f"stockfish EvalFile: {sf.eval_file or '(unreadable)'}")
            our_sha = getattr(backend, "sha256", "")
            if our_sha:
                print(f"our net sha256   : {our_sha}")
                # Stockfish names its nets nn-<first 12 hex of sha256>.nnue, so
                # this is a real provenance check, not a label comparison.
                expected = f"nn-{our_sha[:12]}.nnue"
                # ⚑ AN UNREADABLE EvalFile IS A FAILURE, NOT A PASS. This used to
                # read `if sf.eval_file and ...`, so an engine whose option line
                # the regex could not match — a build change, a renamed option —
                # skipped the comparison entirely and the gate ran against an
                # unverified oracle. That is the exact drift the check exists to
                # catch, and it was the one case the check could not fire on.
                if not sf.eval_file:
                    print(
                        "⚑ could not read the engine's EvalFile option, so we cannot "
                        "prove it is running the network we packed. Refusing to "
                        "report parity against an unverified oracle.",
                        file=sys.stderr,
                    )
                    return 2
                if sf.eval_file != expected:
                    print(
                        f"⚑ NET MISMATCH: stockfish is running {sf.eval_file}, our weights "
                        f"are {expected}. The gate would be comparing two different "
                        "networks.",
                        file=sys.stderr,
                    )
                    return 2

            if obs_handle is not None:
                obs_handle.write(
                    json.dumps(
                        {
                            "record": "run",
                            "backend": backend.name,
                            "our_sha256": our_sha,
                            "stockfish_eval_file": sf.eval_file,
                            "fens_requested": args.n,
                            "fens_delivered": len(fens),
                            "distinct_states": n_states,
                            "distinct_playouts": n_clusters,
                            "seed": args.seed,
                            "seeds": args.seeds,
                            "simd": args.simd,
                        }
                    )
                    + "\n"
                )

            t0 = time.perf_counter()
            result = run_parity(
                backend,
                fens,
                sf,
                args.nnue,
                max_report=args.max_report,
                observations=obs_handle,
            )
            elapsed = time.perf_counter() - t0
        published = True
    finally:
        if obs_handle is not None:
            obs_handle.close()
        if obs_tmp is not None and obs_path is not None:
            if published:
                # Atomic within the directory: a reader sees the old bank or the
                # new one, never a half-written file.
                os.replace(obs_tmp, obs_path)
            else:
                obs_tmp.unlink(missing_ok=True)

    print()
    print(f"backend          : {backend.name}")
    print(f"FENs checked     : {result.checked:,}  in {elapsed:.1f}s")
    print(f"refused in check : {result.refused:,}  (should be 0 — the sampler excludes them)")
    if stats is not None:
        print(
            f"in-check excluded during sampling: {stats.in_check_excluded:,} / "
            f"{stats.considered:,} = {100.0 * stats.in_check_fraction:.2f}%"
        )
    if obs_path is not None:
        print(f"observations     : {result.observations:,} rows banked to {obs_path}")
    print(f"MISMATCHES       : {len(result.mismatches):,}")
    for mm in result.mismatches[: args.max_report]:
        print(f"\n  FEN {mm.fen}\n    stockfish={mm.stockfish} ours={mm.ours}")
        if mm.detail:
            print(mm.detail)

    # Every FEN is either compared or refused; anything else means the loop
    # silently dropped positions and the count above overstates the coverage.
    if result.checked + result.refused != len(fens):
        print(
            f"\nPARITY INCONCLUSIVE — {len(fens):,} FENs in, but only "
            f"{result.checked:,} checked + {result.refused:,} refused accounted for.",
            file=sys.stderr,
        )
        return 2
    if result.checked == 0:
        print(
            f"\nPARITY INCONCLUSIVE — 0 of {len(fens):,} FENs were comparable "
            f"({result.refused:,} refused in check). Nothing was gated.",
            file=sys.stderr,
        )
        return 2
    if result.mismatches:
        print("\nPARITY FAILED — any mismatch on a non-check FEN is a bug.")
        return 1
    print(f"\nPARITY PASSED — exact integer equality on all {result.checked:,} FENs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
