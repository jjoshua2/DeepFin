# The actual initial-True branch needs no bypass-safety premise.
import Base
import ./Queries.bend as Queries
import ./Spec.bend as S
import ../storage/Observe.bend as O
import ../initialized_attacks/Build.bend as Init
import ../prepared_king_selection/Selection.bend as K
import ../board/Spec.bend as B
import ../generated_bypass_safety/Spec.bend as Contracts
import ../generated_bypass_safety/consumer.bend as Prior
import ../accepted_castle/Spec.bend as Prepared
import ../accepted_castle/Pair.bend as Pair
import ../accepted_castle/Filter.bend as Filter
import ../accepted_castle/Generated.bend as Actual
import ../castle_emission/Spec.bend as E
import ../fast_full_equivalence/Equivalence.bend as F
import ../../../legal_probe/Chess.bend as Chess

def initial_false(+c: O.Cells,+b: Chess.Board,
  no: {Prepared.initial(c,b) == False{} : Bool}) -> Contracts.initial(c,b):
  Equal.trans(Array<U64> & Bool,Chess.in_check(O.pack(c),b,Chess.get_turn(b)),
    (O.pack(c),Prepared.initial(c,b)),(O.pack(c),False{}),
    Pair.check(c,b,Chess.get_turn(b)),
    Equal.cong(Bool,Array<U64> & Bool,answer => (O.pack(c),answer),Prepared.initial(c,b),False{},no))

def prepare_true(+c: O.Cells,+b: Chess.Board,+xs: List<&2,Chess.Ply>,
  yes: {Prepared.initial(c,b) == True{} : Bool}) ->
  {Chess.filter_prepare(b,(O.pack(c),xs)) ==
    Chess.filter_legal(xs,b,(O.pack(c),Nil{})) : Array<U64> & List<&2,Chess.Ply>}:
  Equal.trans(Array<U64> & List<&2,Chess.Ply>,
    Chess.filter_prepare(b,(O.pack(c),xs)),(O.pack(c),Prepared.prepared(c,b,xs)),
    Chess.filter_legal(xs,b,(O.pack(c),Nil{})),Filter.prepare(c,b,xs),
    Equal.trans(Array<U64> & List<&2,Chess.Ply>,(O.pack(c),Prepared.prepared(c,b,xs)),
      (O.pack(c),Prepared.full(xs,c,b,Nil{})),Chess.filter_legal(xs,b,(O.pack(c),Nil{})),
      F.pair_lift(c,Prepared.prepared(c,b,xs),Prepared.full(xs,c,b,Nil{}),
        Equal.cong(Bool,List<&2,Chess.Ply>,
          check => Prepared.choose(check,c,b,xs,Prepared.rays(c,b)),Prepared.initial(c,b),True{},yes)),
      Equal.sym(Array<U64> & List<&2,Chess.Ply>,Chess.filter_legal(xs,b,(O.pack(c),Nil{})),
        (O.pack(c),Prepared.full(xs,c,b,Nil{})),Filter.full(xs,c,b,Nil{}))))

def branch(check: Bool,+c: O.Cells,+d: Nat,+seed: U64,+n: Nat,+extra: Nat,+b: Chess.Board,+white: Bool,
  +same: {O.pack(c) == Init.run(d,seed,n,extra) : Array<U64>},
  +depth: {d == 17n : Nat},+tables: {n == 128n : Nat},+full: {extra == 64n : Nat},
  good: {B.valid(b) == True{} : Bool},
  +nonempty: {U64.is_zero(K.plane(b)) == False{} : Bool},
  canonical: {Chess.get_turn(b) == Bool.to_u32(white) : U32},
  decision: {Prepared.initial(c,b) == check : Bool}) -> S.legal(O.pack(c),b):
  match check:
    case True{}:
      Equal.trans(Array<U64> & List<&2,Chess.Ply>,Chess.legal_moves(O.pack(c),b),
        Chess.filter_prepare(b,(O.pack(c),Contracts.candidates(c,b))),
        Chess.filter_legal(Contracts.candidates(c,b),b,(O.pack(c),Nil{})),
        Actual.prepared_equal(c,b),prepare_true(c,b,Contracts.candidates(c,b),decision))
    case False{}:
      Prior.actual_legal_full(c,b,white,good,nonempty,canonical,
        Queries.old(c,d,seed,n,extra,b,same,depth,tables,full,nonempty),
        initial_false(c,b,decision),
        Queries.occurrences(c,d,seed,n,extra,b,Contracts.candidates(c,b),same,depth,tables,full,nonempty))

def known(+c: O.Cells,+d: Nat,+seed: U64,+n: Nat,+extra: Nat,+b: Chess.Board,+white: Bool,
  same: {O.pack(c) == Init.run(d,seed,n,extra) : Array<U64>},
  depth: {d == 17n : Nat},tables: {n == 128n : Nat},full: {extra == 64n : Nat},
  good: {B.valid(b) == True{} : Bool},
  nonempty: {U64.is_zero(K.plane(b)) == False{} : Bool},
  canonical: {Chess.get_turn(b) == Bool.to_u32(white) : U32}) -> S.legal(O.pack(c),b):
  branch(Prepared.initial(c,b),c,d,seed,n,extra,b,white,same,depth,tables,full,good,nonempty,canonical,{==})
