# Opt-in native observation: operands enter; candidate mask/index leave.
# No proof, geometry oracle, engine controller or precomputed answer is imported.
import Base
import ../../Text.bend as Text
import ./Spec.bend as Spec
import ../../../bitboard_probe/Sliders.bend as Sliders

def show(+x: U64) -> String:
  U32.show(U64.high(x)) ++ " " ++ U32.show(U64.low(x))

def observe(id: U32, +key: U32, +occ: U64, +mask: U64) -> IO(Unit):
  IO.print("v " ++ U32.show(id) ++ " " ++ U32.show(key) ++ " " ++ show(mask)
    ++ " " ++ U32.show(U64.popcount(mask)) ++ " " ++ U32.show(Spec.size(key))
    ++ " " ++ show(U64.pext(occ,mask)) ++ " " ++ U32.show(Sliders.pext_index(occ,mask)))

def checked(ok: Bool, id: U32, +key: U32, occ: U64) -> IO(Unit):
  match ok:
    case True{}: observe(id,key,occ,Spec.mask(key))
    case False{}: IO.die(Unit,2,"invalid layout key or test budget")

def parse(+id: U32, key: Maybe<&2,U32>, hi: Maybe<&2,U32>, lo: Maybe<&2,U32>) -> IO(Unit):
  match key hi lo:
    case Some{+k} Some{h} Some{l}:
      checked(Bool.and(U32.is_lt(k,128),U32.is_lt(id,4096)),id,k,U64.from_parts(h,l))
    case _ _ _: IO.die(Unit,2,"invalid layout integer")

def cases(args: List<String>, +id: U32) -> IO(Unit):
  match args:
    case Nil{}: IO.pure(Unit,Unit{})
    case k <> hi <> lo <> rest:
      do IO<Unit>:
        parse(id,Text.number(k),Text.number(hi),Text.number(lo))
        cases(rest,U32.inc(id))
    case _: IO.die(Unit,2,"expected key occupancy_hi occupancy_lo")

def main() -> IO(Unit):
  do IO<Unit>:
    args : List<String> <- IO.args()
    cases(args,0)
