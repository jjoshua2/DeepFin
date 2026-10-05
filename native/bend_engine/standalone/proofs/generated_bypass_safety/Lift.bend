import Base
import ./Spec.bend as S
import ./Safety.bend as Safety
import ../storage/Observe.bend as O
import ../board/Spec.bend as B
import ../prepared_king_selection/Selection.bend as K
import ../castle_chain/Spec.bend as C
import ../castle_chain/Lists.bend as L
import ../castle_emission/Spec.bend as E
import ../fast_full_equivalence/Equivalence.bend as F
import ../../../legal_probe/Chess.bend as Chess

# Each supplied occurrence keeps its own actual full-Ply membership witness.
# The input list may reorder or repeat generated values; no set conversion.
def occurrences(+c: O.Cells,+b: Chess.Board,+white: Bool,+xs: List<&2,Chess.Ply>,
  origin: C.every(m => E.member(m,S.candidates(c,b)),xs),
  +good: {B.valid(b) == True{} : Bool},
  +nonempty: {U64.is_zero(K.plane(b)) == False{} : Bool},
  +canonical: {Chess.get_turn(b) == Bool.to_u32(white) : U32},
  +old: S.old(c,b),+initial: S.initial(c,b),lookups: S.lookups(c,b,xs)) ->
  C.every(m => F.bypass_safe(c,b,F.prepared_sensitive(c,b),m),xs):
  match xs:
    case Nil{}: Unit{}
    case Con{+m,rest}:
      (here,later)=origin
      (lookup,remaining)=lookups
      (Safety.candidate(c,b,m,white,here,good,nonempty,canonical,old,initial,lookup),
        occurrences(c,b,white,rest,later,good,nonempty,canonical,old,initial,remaining))

def all(+c: O.Cells,+b: Chess.Board,+white: Bool,
  +good: {B.valid(b) == True{} : Bool},
  +nonempty: {U64.is_zero(K.plane(b)) == False{} : Bool},
  +canonical: {Chess.get_turn(b) == Bool.to_u32(white) : U32},
  +old: S.old(c,b),+initial: S.initial(c,b),lookups: S.lookups(c,b,S.candidates(c,b))) ->
  C.every(m => F.bypass_safe(c,b,F.prepared_sensitive(c,b),m),S.candidates(c,b)):
  +xs=S.candidates(c,b)
  occurrences(c,b,white,xs,L.self(xs),good,nonempty,canonical,old,initial,lookups)
