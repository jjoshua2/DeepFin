# Opt-in native test only. Never imported by the application or proof module.
# Chess masks are constructed by the actual Tables.slider; external arguments
# are bounded synthetic mask/start/count fixtures, not engine decisions.
import Base
import ../Subsets.bend as Subsets
import ../Tables.bend as Tables
import ../Text.bend as Text

def show(+x: U64) -> String:
  U32.show(U64.high(x)) ++ " " ++ U32.show(U64.low(x))

def walk(n: Nat, +i: U32, +mask: U64, +subset: U64) -> IO(Unit):
  match n:
    case 0n: IO.print("end " ++ show(subset))
    case 1n+p:
      do IO<Unit>:
        IO.print("v " ++ U32.show(i) ++ " " ++ show(subset) ++ " " ++ show(U64.pext(subset, mask)))
        walk(p, U32.inc(i), mask, Subsets.next(subset, mask))

def run(+id: U32, +mask: U64, +start: U64, +count: U32) -> IO(Unit):
  do IO<Unit>:
    IO.print("begin " ++ U32.show(id) ++ " " ++ show(mask) ++ " " ++ show(start)
      ++ " " ++ U32.show(count) ++ " " ++ U32.show(U64.popcount(mask)))
    walk(U32.to_nat(count), 0, mask, start)

def chess_checked(ok: Bool, +id: U32, +mask: U64, +population: U32) -> IO(Unit):
  # Fail instead of overflowing a machine shift or expanding a corrupted mask.
  match ok:
    case True{}: run(id, mask, U64.zero(), U32.shln(1, U32.to_nat(population)))
    case False{}: IO.die(Unit, 2, "chess mask exceeds the test budget")

def chess(n: Nat, +id: U32) -> IO(Unit):
  match n:
    case 0n: IO.pure(Unit, Unit{})
    case 1n+p:
      +mask = Tables.slider(U32.mod(id, 64), U32.is_ge(id, 64), U64.zero(), True{})
      do IO<Unit>:
        chess_checked(U32.is_le(U64.popcount(mask), 12), id, mask, U64.popcount(mask))
        chess(p, U32.inc(id))

def custom_checked(ok: Bool, id: U32, mask: U64, start: U64, count: U32) -> IO(Unit):
  match ok:
    case True{}: run(id, mask, start, count)
    case False{}: IO.die(Unit, 2, "synthetic subset case exceeds the test budget")

def custom(+id: U32, mh: Maybe<&2, U32>, ml: Maybe<&2, U32>, sh: Maybe<&2, U32>, sl: Maybe<&2, U32>, count: Maybe<&2, U32>) -> IO(Unit):
  match mh ml sh sl count:
    case Some{hi} Some{lo} Some{shi} Some{slo} Some{+n}:
      custom_checked(Bool.and(U32.is_lt(id, 384), Bool.and(U32.is_gt(n, 0), U32.is_le(n, 4096))),
        id, U64.from_parts(hi, lo), U64.from_parts(shi, slo), n)
    case _ _ _ _ _: IO.die(Unit, 2, "invalid subset integer")

def cases(args: List<String>, +id: U32) -> IO(Unit):
  match args:
    case Nil{}: IO.pure(Unit, Unit{})
    case mh <> ml <> sh <> sl <> n <> rest:
      do IO<Unit>:
        custom(id, Text.number(mh), Text.number(ml), Text.number(sh), Text.number(sl), Text.number(n))
        cases(rest, U32.inc(id))
    case _: IO.die(Unit, 2, "expected mask_hi mask_lo start_hi start_lo count")

def main() -> IO(Unit):
  do IO<Unit>:
    args : List<String> <- IO.args()
    chess(128n, 0)
    cases(args, 128)
