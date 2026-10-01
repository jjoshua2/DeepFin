"""Emit finite coordinate-prefix certificates, then require the Bend checker.

The generator only chooses tuple lengths: it emits equality constructors, never
assumed occupancy facts or bitboard answers. --write is explicit; default verifies.
"""
from pathlib import Path
import argparse

DIRECTIONS = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))


def path_length(square: int, direction: int) -> int:
    x, y = square % 8, square // 8
    dx, dy = DIRECTIONS[direction]
    count = 0
    while 0 <= x + dx < 8 and 0 <= y + dy < 8:
        x, y = x + dx, y + dy
        count += 1
    return count


def outputs() -> dict[str, str]:
    result = {}
    header = '# Generated coordinate-only prefix certificates; every equality is checked by Bend.\nimport Base\nimport ./Spec.bend as S\nimport ../layout/Domain.bend as D\n'
    for rank in range(8):
        text = header
        for square in range(rank * 8, rank * 8 + 8):
            text += f'\ndef square{square}(+dir: Nat,bound: {{Nat.is_lt(dir,8n) == True{{}} : Bool}}) -> S.certificate({square}n,dir):\n  match dir:\n'
            for direction in range(8):
                proof = 'Unit{}'
                for _ in range(path_length(square, direction)):
                    proof = '({==},' + proof + ')'
                text += f'    case {direction}n: {proof}\n'
            text += f'    case 8n+p: D.impossible(S.certificate({square}n,8n+p),p,bound)\n'
        result[f'Rank{rank}.bend'] = text
    text = 'import Base\nimport ./Spec.bend as S\nimport ../layout/Domain.bend as D\n'
    text += ''.join(f'import ./Rank{r}.bend as R{r}\n' for r in range(8))
    text += '\ndef all(+src: Nat,+dir: Nat,source_bound: {Nat.is_lt(src,64n) == True{} : Bool},direction_bound: {Nat.is_lt(dir,8n) == True{} : Bool}) -> S.certificate(src,dir):\n  match src:\n'
    for square in range(64):
        text += f'    case {square}n: R{square // 8}.square{square}(dir,direction_bound)\n'
    text += '    case 64n+p: D.impossible(S.certificate(64n+p,dir),p,source_bound)\n'
    result['Certificate.bend'] = text
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    for name, text in outputs().items():
        target = root / name
        if args.write:
            target.write_text(text)
        elif target.read_text() != text:
            raise RuntimeError(f'Certificate differs: {name}')
    print('9 certificate files match')


if __name__ == '__main__':
    main()
