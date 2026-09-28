# Opt-in raw native observation only; never imported by proofs or the engine.
# Inputs are bounded test operands, not reference answers or chess decisions.
import Base
import ../../Subsets.bend as Subsets
import ../../Text.bend as Text

def show(+x: U64) -> String:
  U32.show(U64.high(x)) ++ " " ++ U32.show(U64.low(x))

def observe(id: U32, +mask: U64, +x: U64) -> IO(Unit):
  +subset = U64.and(x,mask)
  +next = Subsets.next(subset,mask)
  IO.print("s " ++ U32.show(id) ++ " " ++ U32.show(U64.popcount(mask))
    ++ " " ++ show(U64.sub(x,mask)) ++ " " ++ show(next)
    ++ " " ++ show(U64.pext(subset,mask)) ++ " " ++ show(U64.pext(next,mask)))

def checked(ok: Bool, id: U32, mask: U64, x: U64) -> IO(Unit):
  match ok:
    case True{}: observe(id, mask, x)
    case False{}: IO.die(Unit, 2, "successor test budget exceeded")

def parse(+id: U32, mh: Maybe<&2, U32>, ml: Maybe<&2, U32>, xh: Maybe<&2, U32>, xl: Maybe<&2, U32>) -> IO(Unit):
  match mh ml xh xl:
    case Some{hi} Some{lo} Some{xh} Some{xl}:
      checked(U32.is_lt(id, 2048), id, U64.from_parts(hi, lo), U64.from_parts(xh, xl))
    case _ _ _ _: IO.die(Unit, 2, "invalid successor test integer")

def cases(args: List<String>, +id: U32) -> IO(Unit):
  match args:
    case Nil{}: IO.pure(Unit, Unit{})
    case mh <> ml <> xh <> xl <> rest:
      do IO<Unit>:
        parse(id, Text.number(mh), Text.number(ml), Text.number(xh), Text.number(xl))
        cases(rest, U32.inc(id))
    case _: IO.die(Unit, 2, "expected mask_hi mask_lo value_hi value_lo")

def main() -> IO(Unit):
  do IO<Unit>:
    args : List<String> <- IO.args()
    cases(args, 0)
