# Symbolic consumer witnesses keep actual initialization and full-Ply membership.
import Base
import ./Spec.bend as S
import ./consumer.bend as Consumer
import ../storage/Observe.bend as O
import ../storage/Representation.bend as R
import ../initialized_attacks/Build.bend as Init
import ../board/Spec.bend as B
import ../prepared_king_selection/Selection.bend as K
import ../accepted_castle/Generated.bend as Actual
import ../castle_emission/Spec.bend as E
import ../../../legal_probe/Chess.bend as Chess

def twice(+m: Chess.Ply) -> List<&2,Chess.Ply>:
  Con{m,Con{m,Nil{}}}

def duplicate_lift(-a: Array<U64>,+d: Nat,+seed: U64,+n: Nat,+extra: Nat,+b: Chess.Board,+white: Bool,
  +m: Chess.Ply,+tail: List<&2,Chess.Ply>,
  initialized: {a == Init.run(d,seed,n,extra) : Array<U64>},
  depth: {d == 17n : Nat},tables: {n == 128n : Nat},full: {extra == 64n : Nat},
  good: {B.valid(b) == True{} : Bool},
  nonempty: {U64.is_zero(K.plane(b)) == False{} : Bool},
  canonical: {Chess.get_turn(b) == Bool.to_u32(white) : U32},
  cert: Exists(O.Cells,c => {O.pack(c) == a : Array<U64>})) ->
  E.member(m,E.moves(Actual.candidates(a,b))) -> S.initial(a,b) -> S.list(a,b,twice(m),tail):
  (+c,e)=cert
  +same: {O.pack(c) == a : Array<U64>}=e
  +built: {O.pack(c) == Init.run(d,seed,n,extra) : Array<U64>}=
    Equal.trans(Array<U64>,O.pack(c),a,Init.run(d,seed,n,extra),same,initialized)
  %same : E.member(m,E.moves(Actual.candidates(_,b))) -> S.initial(_,b) -> S.list(_,b,twice(m),tail)
  here => initial =>
    Actual.split_member(S.list(O.pack(c),b,twice(m),tail),E.moves(Actual.candidates(O.pack(c),b)),m,here,
      first => second => Consumer.use_list(O.pack(c),d,seed,n,extra,b,white,twice(m),tail,
        built,depth,tables,full,good,nonempty,canonical,(first,(second,Unit{})),initial))

def duplicate_generated(a: Array<U64>,+d: Nat,+seed: U64,+n: Nat,+extra: Nat,+b: Chess.Board,+white: Bool,
  +m: Chess.Ply,+tail: List<&2,Chess.Ply>,
  initialized: {a == Init.run(d,seed,n,extra) : Array<U64>},
  depth: {d == 17n : Nat},tables: {n == 128n : Nat},full: {extra == 64n : Nat},
  good: {B.valid(b) == True{} : Bool},
  nonempty: {U64.is_zero(K.plane(b)) == False{} : Bool},
  canonical: {Chess.get_turn(b) == Bool.to_u32(white) : U32},
  here: E.member(m,E.moves(Actual.candidates(a,b))),initial: S.initial(a,b)) ->
  S.list(a,b,twice(m),tail):
  duplicate_lift(a,d,seed,n,extra,b,white,m,tail,initialized,depth,tables,full,good,nonempty,canonical,R.reify(a))(here)(initial)

def actual_generated_full(a: Array<U64>,+d: Nat,+seed: U64,+n: Nat,+extra: Nat,+b: Chess.Board,+white: Bool,
  initialized: {a == Init.run(d,seed,n,extra) : Array<U64>},
  depth: {d == 17n : Nat},tables: {n == 128n : Nat},full: {extra == 64n : Nat},
  good: {B.valid(b) == True{} : Bool},
  nonempty: {U64.is_zero(K.plane(b)) == False{} : Bool},
  canonical: {Chess.get_turn(b) == Bool.to_u32(white) : U32}) -> S.legal(a,b):
  Consumer.use_legal_full(a,d,seed,n,extra,b,white,initialized,depth,tables,full,good,nonempty,canonical)

# Fixed full-Ply special records remain tied to actual producer membership.
def promotion_generated(a: Array<U64>,+d: Nat,+seed: U64,+n: Nat,+extra: Nat,+b: Chess.Board,+white: Bool,
  +tail: List<&2,Chess.Ply>,
  initialized: {a == Init.run(d,seed,n,extra) : Array<U64>},
  depth: {d == 17n : Nat},tables: {n == 128n : Nat},full: {extra == 64n : Nat},
  good: {B.valid(b) == True{} : Bool},
  nonempty: {U64.is_zero(K.plane(b)) == False{} : Bool},
  canonical: {Chess.get_turn(b) == Bool.to_u32(white) : U32},
  here: E.member(Chess.Ply{48,56,4,0},E.moves(Actual.candidates(a,b))),initial: S.initial(a,b)) ->
  S.list(a,b,twice(Chess.Ply{48,56,4,0}),tail):
  duplicate_generated(a,d,seed,n,extra,b,white,Chess.Ply{48,56,4,0},tail,
    initialized,depth,tables,full,good,nonempty,canonical,here,initial)

def ep_generated(a: Array<U64>,+d: Nat,+seed: U64,+n: Nat,+extra: Nat,+b: Chess.Board,+white: Bool,
  +tail: List<&2,Chess.Ply>,
  initialized: {a == Init.run(d,seed,n,extra) : Array<U64>},
  depth: {d == 17n : Nat},tables: {n == 128n : Nat},full: {extra == 64n : Nat},
  good: {B.valid(b) == True{} : Bool},
  nonempty: {U64.is_zero(K.plane(b)) == False{} : Bool},
  canonical: {Chess.get_turn(b) == Bool.to_u32(white) : U32},
  here: E.member(Chess.Ply{36,43,0,1},E.moves(Actual.candidates(a,b))),initial: S.initial(a,b)) ->
  S.list(a,b,twice(Chess.Ply{36,43,0,1}),tail):
  duplicate_generated(a,d,seed,n,extra,b,white,Chess.Ply{36,43,0,1},tail,
    initialized,depth,tables,full,good,nonempty,canonical,here,initial)
