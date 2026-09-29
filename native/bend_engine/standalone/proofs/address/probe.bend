# Runtime operands only. No proof oracle or host-computed candidate decisions.
# key 0..127 uses the actual chess mask; key 128 selects an explicit raw mask.
import Base
import ../../Text.bend as Text
import ../layout/Spec.bend as Layout
import ../LAWS.bend as Sequence
import ../PROOF.bend as SequenceProof
import ../../../bitboard_probe/Sliders.bend as Sliders

def show(+x: U64) -> String:
  U32.show(U64.high(x)) ++ " " ++ U32.show(U64.low(x))

def choose_mask(chess: Bool, key: U32, raw: U64) -> U64:
  match chess:
    case True{}: Layout.mask(key)
    case False{}: raw

def observe(+id: U32, +key: U32, +occ: U64, +m: U64, +n: U32) -> IO(Unit):
  +state = Sequence.at(U32.to_nat(n),m)
  +index = Sliders.pext_index(occ,m)
  IO.print("v " ++ U32.show(id) ++ " " ++ U32.show(key) ++ " " ++ show(m)
    ++ " " ++ U32.show(U64.popcount(m)) ++ " " ++ U32.show(n)
    ++ " " ++ U32.show(index) ++ " " ++ show(U64.pext(occ,m))
    ++ " " ++ show(state) ++ " " ++ U32.show(Sliders.pext_index(state,m))
    ++ " " ++ show(U64.pdep(U64.from_u32(index),m)))

def checked(ok: Bool, id: U32, +key: U32, raw: U64, occ: U64, n: U32) -> IO(Unit):
  match ok:
    case True{}: observe(id,key,occ,choose_mask(U32.is_lt(key,128),key,raw),n)
    case False{}: IO.die(Unit,2,"invalid exact-index key or test budget")

def parse(+id: U32, key: Maybe<&2,U32>, mh: Maybe<&2,U32>, ml: Maybe<&2,U32>,
  oh: Maybe<&2,U32>, ol: Maybe<&2,U32>, n: Maybe<&2,U32>) -> IO(Unit):
  match key mh ml oh ol n:
    case Some{+k} Some{h} Some{l} Some{x} Some{y} Some{+i}:
      checked(Bool.and(U32.is_le(k,128),Bool.and(U32.is_lt(id,4096),U32.is_le(i,4096))),
        id,k,U64.from_parts(h,l),U64.from_parts(x,y),i)
    case _ _ _ _ _ _: IO.die(Unit,2,"invalid exact-index integer")

def cases(args: List<String>, +id: U32) -> IO(Unit):
  match args:
    case Nil{}: IO.pure(Unit,Unit{})
    case k <> mh <> ml <> oh <> ol <> n <> rest:
      do IO<Unit>:
        parse(id,Text.number(k),Text.number(mh),Text.number(ml),Text.number(oh),Text.number(ol),Text.number(n))
        cases(rest,U32.inc(id))
    case _: IO.die(Unit,2,"expected key mask_hi mask_lo occupancy_hi occupancy_lo ordinal")

def main() -> IO(Unit):
  do IO<Unit>:
    args : List<String> <- IO.args()
    cases(args,0)
