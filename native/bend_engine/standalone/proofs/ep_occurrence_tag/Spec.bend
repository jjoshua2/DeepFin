# The flag1 producer contract is independent of table geometry and root validity.
import Base
import ../child_metadata/Range.bend as R
import ../pawn_ep_occurrence/Spec.bend as Old
import ../bitboard_membership/IndexEquality.bend as Index
import ../../../legal_probe/Chess.bend as C

def src(m: C.Ply) -> U32: R.src(m)
def dst(m: C.Ply) -> U32: Old.dst(m)
def promotion(m: C.Ply) -> U32:
  C.Ply{s,d,p,f}=m
  p
def flag(m: C.Ply) -> U32:
  C.Ply{s,d,p,f}=m
  f
def pawn(+b: C.Board,s: U32) -> Bool:
  U64.test_bit(C.get_pawns(b),U32.to_nat(s))
def facts(-b: C.Board,-m: C.Ply) -> Type:
  {promotion(m) == 0 : U32} &
  ({pawn(b,src(m)) == True{} : Bool} &
  ({dst(m) == C.get_ep(b) : U32} &
  {Nat.is_lt(U32.to_nat(dst(m)),64n) == True{} : Bool}))
def gate(yes: Bool,-b: C.Board,-m: C.Ply) -> Type:
  match yes:
    case False{}: Unit
    case True{}: facts(b,m)
def property(+b: C.Board,+m: C.Ply) -> Type:
  gate(U32.is_eq(flag(m),1),b,m)
def bits(yes: Bool,-a: Bool,-z: Bool) -> Type:
  match yes:
    case False{}: Unit
    case True{}: {a == True{} : Bool} & {z == True{} : Bool}
def correlate(+a: Bool,+z: Bool) -> bits(Bool.and(a,z),a,z):
  match a z:
    case False{} _: Unit{}
    case True{} False{}: Unit{}
    case True{} True{}: ({==},{==})
def selected(yes: Bool,+b: C.Board,+s: U32,+d: U32,+p: Bool,+ep: U32,
  bound: {Nat.is_lt(U32.to_nat(d),64n) == True{} : Bool},
  same_pawn: {p == pawn(b,s) : Bool},same_ep: {ep == C.get_ep(b) : U32},
  evidence: bits(yes,p,U32.is_eq(d,ep))) -> property(b,C.Ply{s,d,0,Bool.to_u32(yes)}):
  match yes:
    case False{}: Unit{}
    case True{}:
      (head,equal)=evidence
      ({==},(Equal.trans(Bool,pawn(b,s),p,True{},Equal.sym(Bool,p,pawn(b,s),same_pawn),head),
        (Equal.trans(U32,d,ep,C.get_ep(b),Index.u32_eq_true_same(d,ep,equal),same_ep),bound)))
