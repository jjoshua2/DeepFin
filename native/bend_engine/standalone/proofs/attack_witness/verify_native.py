"""Bounded actual attack/check observations against independent coordinate geometry.

This does not execute the proof-only Word scan. It tests the current real table
builder, five mask queries, attacked reducer and bounded in_check calls.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import tempfile

SUITE = Path(__file__).resolve().parent
ENGINE = SUITE.parents[2]
MODES = {
    "generic": [],
    "portable": ["-DBEND_U64_PORTABLE"],
    "native": ["-march=native"],
    "ubsan": ["-fsanitize=undefined", "-fno-sanitize-recover=all"],
}
Board = list[tuple[set[int], set[int]]]


def run(args: list[str], timeout: int = 180) -> subprocess.CompletedProcess[str]:
    p = subprocess.run(args, capture_output=True, text=True, timeout=timeout,
                       env={**os.environ, "BEND_NO_TELEMETRY": "1", "TERM": "dumb"}, check=False)
    if p.returncode != 0 or p.stderr:
        raise AssertionError((args[:3], p.returncode, p.stderr[-2000:]))
    return p


def empty() -> Board:
    return [(set(), set()) for _ in range(64)]


def put(board: Board, square: int, kind: int, color: int) -> None:
    board[square][0].add(kind)
    board[square][1].add(color)


def encode(board: Board, metadata: tuple[int, int, int]) -> list[int]:
    words = [0] * 8
    for sq, (kinds, colors) in enumerate(board):
        for kind in kinds:
            words[kind] |= 1 << sq
        for color in colors:
            words[6 if color == 1 else 7] |= 1 << sq
    return [z for w in words for z in (w >> 32, w & 0xFFFFFFFF)] + list(metadata)


def geometry(board: Board, origin: int, kind: int, color: int) -> set[int]:
    """Forward geometric attacks. Sliders include and stop at first occupancy."""
    x, y = origin % 8, origin // 8
    if kind == 0:
        offsets = [(dx, 1 if color == 1 else -1) for dx in (-1, 1)]
    elif kind == 1:
        offsets = [(dx, dy) for dx in (-2, -1, 1, 2)
                   for dy in (-2, -1, 1, 2) if abs(dx * dy) == 2]
    elif kind == 5:
        offsets = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy]
    else:
        directions = ([(dx, dy) for dx in (-1, 1) for dy in (-1, 1)] if kind in (2, 4) else [])
        directions += ([(0, 1), (0, -1), (1, 0), (-1, 0)] if kind in (3, 4) else [])
        result = set()
        for dx, dy in directions:
            xx, yy = x + dx, y + dy
            while 0 <= xx < 8 and 0 <= yy < 8:
                sq = yy * 8 + xx
                result.add(sq)
                if board[sq][1]:
                    break
                xx, yy = xx + dx, yy + dy
        return result
    return {(y + dy) * 8 + x + dx for dx, dy in offsets
            if 0 <= x + dx < 8 and 0 <= y + dy < 8}


def attacked(board: Board, target: int, by: int) -> bool:
    # Forward from candidate attackers, rather than reverse-query mask reduction.
    return any(by in colors and any(target in geometry(board, sq, k, by) for k in kinds)
               for sq, (kinds, colors) in enumerate(board))


def expected(board: Board, target: int, by: int) -> list[int]:
    masks = [geometry(board, target, k, 1 - by if k == 0 else by) for k in (0, 1, 5, 3, 2)]
    words = [sum(1 << sq for sq in m) for m in masks]
    kings = [sq for sq, (ks, cs) in enumerate(board) if 5 in ks and 1 - by in cs]
    selected = min(kings) if kings else 64
    check = int(attacked(board, selected, by)) if kings else 2
    return [z for w in words for z in (w >> 32, w & 0xFFFFFFFF)] + [int(attacked(board, target, by)), selected, check]


def fixtures() -> list[dict]:
    cases: list[dict] = []
    seen: set[tuple[int, ...]] = set()

    def add(label: str, b: Board, target: int, by: int,
            metadata: tuple[int, int, int] | None = None) -> None:
        inp = [target, by, *encode(b, metadata or (1 - by, 15, 64))]
        key = tuple(inp)
        if key not in seen:
            seen.add(key)
            kings = [sq for sq, (ks, cs) in enumerate(b) if 5 in ks and 1 - by in cs]
            cases.append({"label": label, "input": inp, "expected": expected(b, target, by),
                          "singleton_at_target": kings == [target], "king_count": len(kings)})

    fixed = [0, 7, 8, 15, 24, 31, 32, 39, 48, 55, 56, 63]
    for target in range(64):
        for by in (0, 1):
            for kind in range(6):
                for source in sorted(set(fixed + [target ^ x for x in (1, 8, 9, 17)])):
                    b = empty()
                    put(b, target, 5, 1 - by)
                    put(b, source, kind, by)
                    add("all-query-kind-color-sampled-origins", b, target, by)
                wrong = empty()
                put(wrong, target, 5, 1 - by)
                put(wrong, target ^ 17, kind, 1 - by)
                add("wrong-color", wrong, target, by)
            for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                path = []
                x, y = target % 8 + dx, target // 8 + dy
                while 0 <= x < 8 and 0 <= y < 8:
                    path.append(y * 8 + x)
                    x, y = x + dx, y + dy
                if len(path) > 1:
                    for blocker_color in (0, 1):
                        b = empty()
                        put(b, target, 5, 1 - by)
                        put(b, path[-1], 4, by)
                        put(b, path[0], 0, blocker_color)
                        add("first-blocker-stops-slider", b, target, by)
    rng = random.Random(20260926)
    for i in range(256):
        b = empty()
        target, by = rng.randrange(64), rng.randrange(2)
        for sq in rng.sample(range(64), rng.randrange(1, 33)):
            put(b, sq, rng.randrange(6), rng.randrange(2))
        add("mixed-metadata-and-king-count", b, target, by, (rng.getrandbits(32), rng.getrandbits(32), rng.getrandbits(32)))
    for target in (0, 31, 32, 63):
        for by in (0, 1):
            add("missing-king", empty(), target, by)
    b = empty()
    put(b, 36, 5, 1)  # white e5 king
    put(b, 58, 5, 0)  # black c8 king
    put(b, 42, 1, 0)  # black c6 knight, pinned along c-file
    put(b, 2, 3, 1)   # white c1 rook
    add("pinned-knight-still-attacks", b, 36, 0)
    b = empty()
    put(b, 0, 5, 1)
    put(b, 63, 5, 1)
    put(b, 53, 1, 0)
    add("two-kings-lowest-only", b, 63, 0)
    assert cases[-1]["expected"][-3:] == [1, 0, 0]
    assert any(c["label"] == "pinned-knight-still-attacks" and c["expected"][-3] == 1 for c in cases)
    return cases


def observe(binary: Path, cases: list[dict]) -> list[list[int]]:
    outputs = []
    for start in range(0, len(cases), 512):
        args = [str(x) for c in cases[start:start + 512] for x in c["input"]]
        raw = run([str(binary), *args], 90).stdout
        lines = raw.splitlines()
        if len(lines) != len(cases[start:start + 512]):
            raise AssertionError(("record count", start, len(lines)))
        for line in lines:
            fields = line.split()
            assert fields[0] == "case" and len(fields) == 14, line
            outputs.append([int(x) for x in fields[1:]])
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("compiler", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    compiler = args.compiler.resolve()
    bun = os.environ.get("BUN", "bun")
    cc = os.environ.get("CC", "clang")
    identity = run([bun, str(ENGINE / "standalone/verify_compiler.js"), str(compiler)]).stdout
    originals = {str(p.relative_to(ENGINE)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in [*SUITE.glob("*.bend"), SUITE / "focused.js", Path(__file__),
                           ENGINE / "legal_probe/Chess.bend", ENGINE / "standalone/Tables.bend"]}
    cases = fixtures()
    exp = [c["expected"] for c in cases]
    modes, mutations = [], []
    with tempfile.TemporaryDirectory(prefix="deepfin-attack-witness-") as td:
        tmp = Path(td)

        def generate(probe: Path, name: str) -> Path:
            out = tmp / (name + ".c")
            run([bun, str(compiler / "bend2/main.ts"), str(probe), "-o", str(out)])
            assert out.is_file()
            return out

        def build(c: Path, name: str, flags: list[str]) -> Path:
            binary = tmp / name
            run([cc, "-std=c11", "-O2", *flags, str(c), "-pthread", "-lm", "-o", str(binary)])
            return binary

        c = generate(SUITE / "probe.bend", "probe")
        invalid = [["x"], ["-1"], ["4294967296"], ["0"],
                   ["64", *map(str, cases[0]["input"][1:])],
                   [str(cases[0]["input"][0]), "2", *map(str, cases[0]["input"][2:])],
                   [*map(str, cases[0]["input"]), "0"]]
        for mode, flags in MODES.items():
            binary = build(c, mode, flags)
            actual = observe(binary, cases)
            assert actual == exp, next((i, a, b) for i, (a, b) in enumerate(zip(actual, exp)) if a != b)
            for batch in invalid:
                p = subprocess.run([str(binary), *batch], capture_output=True, text=True, timeout=30, check=False)
                assert p.returncode == 2 and "invalid" in p.stderr + p.stdout and "runtime error" not in p.stderr
            digest = hashlib.sha256(json.dumps(actual, separators=(",", ":")).encode()).hexdigest()
            modes.append({"mode": mode, "rows": len(actual), "output_sha256": digest, "invalid_rejections": len(invalid)})
            print(f"PASS {mode}: {len(actual)} actual attack/check observations", flush=True)
        for name, old, new in [
            ("wrong-attacker-color", "U64.and(hits, color(b, by))", "U64.and(hits, color(b, U32.xor(by, 1)))"),
            ("omitted-knight", "U64.and(na, get_knights(b))", "U64.zero()"),
            ("wrong-pawn-direction", "attacked_pawn(b, sq, by, attack(0, table, sq, U32.xor(by, 1), occupied(b)))",
             "attacked_pawn(b, sq, by, attack(0, table, sq, by, occupied(b)))"),
        ]:
            copy = tmp / name
            shutil.copytree(ENGINE, copy)
            chess = copy / "legal_probe/Chess.bend"
            text = chess.read_text()
            assert text.count(old) == 1
            chess.write_text(text.replace(old, new))
            source = generate(copy / "standalone/proofs/attack_witness/probe.bend", name)
            binary = build(source, name + "-exe", [])
            observed = observe(binary, cases[:512])
            mismatch = next((i for i, (a, b) in enumerate(zip(observed, exp[:512])) if a != b), None)
            assert mismatch is not None, name
            mutations.append({"name": name, "compiled_and_executed": True, "rejected": True,
                              "row": mismatch, "observed": observed[mismatch], "expected": exp[mismatch]})
    for f, digest in originals.items():
        assert hashlib.sha256((ENGINE / f).read_bytes()).hexdigest() == digest, f
    assert run([bun, str(ENGINE / "standalone/verify_compiler.js"), str(compiler)]).stdout == identity
    report = {"native_gate": "PASS", "compiler_identity": identity, "cc": run([cc, "--version"]).stdout,
              "rows_per_mode": len(cases), "distinct_inputs": len({tuple(c["input"]) for c in cases}),
              "fields_per_mode": len(cases) * 13, "singleton_at_target": sum(c["singleton_at_target"] for c in cases),
              "missing_king_diagnostics": sum(c["king_count"] == 0 for c in cases),
              "multiple_king_diagnostics": sum(c["king_count"] > 1 for c in cases),
              "fixture_sha256": hashlib.sha256(json.dumps(cases, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
              "modes": modes, "mutations": mutations, "source_sha256s": originals,
              "scope": "Actual Tables.build, reverse mask queries, attacked and bounded in_check versus independent forward coordinate attacks; no native proof-scan execution or exhaustive game coverage."}
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
