# Opt-in observation of actual recurrence and runtime scalar lookup; no oracle.
import Base
import ../../Text.bend as Text
import ../../Subsets.bend as Subsets
import ../layout/Spec.bend as Spec
import ../../../bitboard_probe/Sliders.bend as Sliders

def show(+x: U64) -> String:
  U32.show(U64.high(x)) ++ " " ++ U32.show(U64.low(x))

def rows(n: Nat, +key: U32, +mask: U64, +state: U64, +index: U32) -> IO(Unit):
  match n:
    case 0n: IO.print("e " ++ U32.show(key) ++ " " ++ U32.show(index) ++ " " ++ show(state))
    case 1n+p:
      do IO<Unit>:
        IO.print("r " ++ U32.show(key) ++ " " ++ U32.show(index) ++ " " ++ show(state)
          ++ " " ++ U32.show(Sliders.pext_index(state,mask))
          ++ " " ++ U32.show(Sliders.pext_index(U64.or(state,U64.not(mask)),mask))
          ++ " " ++ show(U64.pext(state,mask)))
        rows(p,key,mask,Subsets.next(state,mask),U32.inc(index))

def checked(ok: Bool, +key: U32) -> IO(Unit):
  match ok:
    case True{}: rows(U32.to_nat(Spec.size(key)),key,Spec.mask(key),U64.zero(),0)
    case False{}: IO.die(Unit,2,"invalid key or exact-index budget")

def parse(+id: U32, value: Maybe<&2,U32>) -> IO(Unit):
  match value:
    case None{}: IO.die(Unit,2,"invalid exact-index integer")
    case Some{+key}: checked(Bool.and(U32.is_lt(key,128),U32.is_lt(id,128)),key)

def cases(args: List<String>, +id: U32) -> IO(Unit):
  match args:
    case Nil{}: IO.pure(Unit,Unit{})
    case word <> rest:
      do IO<Unit>:
        parse(id,Text.number(word))
        cases(rest,U32.inc(id))

def main() -> IO(Unit):
  do IO<Unit>:
    args : List<String> <- IO.args()
    cases(args,0)
