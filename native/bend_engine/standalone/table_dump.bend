# Test-only table dump; not imported or linked into the engine.
import Base
import ./Tables.bend as Tables

def emit(r: Array<U64> & U64) -> IO(Array<U64>):
  (a, +value) = r
  do IO<Array<U64>>:
    IO.print(U32.show(U64.high(value)) ++ " " ++ U32.show(U64.low(value)))
    return a

def dump(n: Nat, +i: U32, a: Array<U64>) -> IO(Unit):
  match n:
    case 0n: IO.pure(Unit, Unit{})
    case 1n+p:
      do IO<Unit>:
        a : Array<U64> <- emit(Array.get(U64, a, i))
        dump(p, U32.inc(i), a)

def main() -> IO(Unit):
  dump(108160n, 0, Tables.build())
