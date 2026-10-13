import Base
import ../storage/Observe.bend as O
import ../child_metadata/Range.bend as R
import ../initialized_pawn_double/Spec.bend as D
import ../../../legal_probe/Chess.bend as C

def dst(m: C.Ply) -> U32:
  C.Ply{s,d,p,f}=m
  d

def pawn(+b: C.Board,s: U32) -> Bool:
  U64.test_bit(C.get_pawns(b),U32.to_nat(s))

def trigger(+s: U32,d: U32) -> Bool:
  U32.is_eq(U32.xor(s,d),16)

def gate(flag: Bool,-table: Array<U64>,+b: C.Board,+s: U32,+d: U32) -> Data:
  match flag:
    case False{}: Unit
    case True{}: {U64.test_bit(D.mask(C.pawn_targets(table,b,s)),U32.to_nat(d)) == True{} : Bool}

def target(-table: Array<U64>,+b: C.Board,+s: U32,+d: U32) -> Data:
  gate(Bool.and(pawn(b,s),trigger(s,d)),table,b,s,d)

def occurrence(-table: Array<U64>,+b: C.Board,m: C.Ply) -> Type:
  match m:
    case C.Ply{s,d,p,f}: {Nat.is_lt(U32.to_nat(s),64n) == True{} : Bool} & target(table,b,s,d)

def cells(+c: O.Cells,+b: C.Board,m: C.Ply) -> Type:
  occurrence(O.pack(c),b,m)

def dispatch(+c: O.Cells,+b: C.Board,+s: U32,+bb: U64,+pawn: Bool) -> Data:
  {C.select_u64(pawn,bb,U64.zero()) == C.select_u64(pawn,D.mask(C.pawn_targets(O.pack(c),b,s)),U64.zero()) : U64}
