import Base
import ./Spec.bend as S
import ./Safety.bend as Safety
import ./Lift.bend as Lift
import ../storage/Observe.bend as O
import ../board/Spec.bend as B
import ../prepared_king_selection/Selection.bend as K
import ../castle_chain/Spec.bend as C
import ../castle_emission/Spec.bend as E
import ../accepted_castle/Spec.bend as Prepared
import ../accepted_castle/Generated.bend as Actual
import ../accepted_castle/Pair.bend as Pair
import ../fast_full_equivalence/Equivalence.bend as F
import ../../../legal_probe/Chess.bend as Chess

def actual_candidate_safe(+c: O.Cells,+b: Chess.Board,+m: Chess.Ply,+white: Bool,
  here: E.member(m,S.candidates(c,b)),
  +bypass: {Chess.filter_requires(F.prepared_sensitive(c,b),m) == False{} : Bool},
  +good: {B.valid(b) == True{} : Bool},
  +nonempty: {U64.is_zero(K.plane(b)) == False{} : Bool},
  +canonical: {Chess.get_turn(b) == Bool.to_u32(white) : U32},
  +old: S.old(c,b),+initial: S.initial(c,b),lookup: S.lookup(c,b,m)) ->
  {Chess.in_check(O.pack(c),Chess.make_move(b,m),Chess.get_turn(b)) ==
    (O.pack(c),False{}) : Array<U64> & Bool}:
  Pair.false_check(c,b,m,
    Safety.use_bypass(c,b,m,white,here,bypass,good,nonempty,canonical,old,initial,lookup))

# This bridge derives the list theorem's premise from actual producer membership;
# it does not ask callers to assume bypass safety. Tail is arbitrary and identical.
def member_list_fast_full(+c: O.Cells,+b: Chess.Board,+white: Bool,
  +xs: List<&2,Chess.Ply>,+tail: List<&2,Chess.Ply>,
  origins: C.every(m => E.member(m,S.candidates(c,b)),xs),
  +good: {B.valid(b) == True{} : Bool},
  +nonempty: {U64.is_zero(K.plane(b)) == False{} : Bool},
  +canonical: {Chess.get_turn(b) == Bool.to_u32(white) : U32},
  +old: S.old(c,b),+initial: S.initial(c,b),lookups: S.lookups(c,b,xs)) ->
  {Chess.filter_fast(xs,b,F.prepared_sensitive(c,b),(O.pack(c),tail)) ==
    Chess.filter_legal(xs,b,(O.pack(c),tail)) : Array<U64> & List<&2,Chess.Ply>}:
  F.actual_fast_full(xs,c,b,F.prepared_sensitive(c,b),tail,
    Lift.occurrences(c,b,white,xs,origins,good,nonempty,canonical,old,initial,lookups))

def generated_fast_full(+c: O.Cells,+b: Chess.Board,+white: Bool,+tail: List<&2,Chess.Ply>,
  +good: {B.valid(b) == True{} : Bool},
  +nonempty: {U64.is_zero(K.plane(b)) == False{} : Bool},
  +canonical: {Chess.get_turn(b) == Bool.to_u32(white) : U32},
  +old: S.old(c,b),+initial: S.initial(c,b),lookups: S.lookups(c,b,S.candidates(c,b))) ->
  {Chess.filter_fast(S.candidates(c,b),b,F.prepared_sensitive(c,b),(O.pack(c),tail)) ==
    Chess.filter_legal(S.candidates(c,b),b,(O.pack(c),tail)) : Array<U64> & List<&2,Chess.Ply>}:
  F.actual_fast_full(S.candidates(c,b),c,b,F.prepared_sensitive(c,b),tail,
    Lift.all(c,b,white,good,nonempty,canonical,old,initial,lookups))

def actual_legal_full(+c: O.Cells,+b: Chess.Board,+white: Bool,
  +good: {B.valid(b) == True{} : Bool},
  +nonempty: {U64.is_zero(K.plane(b)) == False{} : Bool},
  +canonical: {Chess.get_turn(b) == Bool.to_u32(white) : U32},
  +old: S.old(c,b),+initial: S.initial(c,b),lookups: S.lookups(c,b,S.candidates(c,b))) ->
  {Chess.legal_moves(O.pack(c),b) ==
    Chess.filter_legal(S.candidates(c,b),b,(O.pack(c),Nil{})) : Array<U64> & List<&2,Chess.Ply>}:
  Equal.trans(Array<U64> & List<&2,Chess.Ply>,Chess.legal_moves(O.pack(c),b),
    Chess.filter_prepare(b,(O.pack(c),S.candidates(c,b))),
    Chess.filter_legal(S.candidates(c,b),b,(O.pack(c),Nil{})),Actual.prepared_equal(c,b),
    F.actual_prepare_full(c,b,S.candidates(c,b),
      Lift.all(c,b,white,good,nonempty,canonical,old,initial,lookups)))
