# One affine input array; all new lookup certificates are constructed internally.
import Base
import ./Spec.bend as S
import ./Queries.bend as Queries
import ./Full.bend as Full
import ../storage/Observe.bend as O
import ../storage/Representation.bend as R
import ../initialized_attacks/Build.bend as Init
import ../board/Spec.bend as B
import ../prepared_king_selection/Selection.bend as K
import ../generated_bypass_safety/consumer.bend as Prior
import ../../../legal_probe/Chess.bend as Chess

def legal_lift(-a: Array<U64>,+d: Nat,+seed: U64,+n: Nat,+extra: Nat,+b: Chess.Board,+white: Bool,
  initialized: {a == Init.run(d,seed,n,extra) : Array<U64>},
  depth: {d == 17n : Nat},tables: {n == 128n : Nat},full: {extra == 64n : Nat},
  good: {B.valid(b) == True{} : Bool},
  nonempty: {U64.is_zero(K.plane(b)) == False{} : Bool},
  canonical: {Chess.get_turn(b) == Bool.to_u32(white) : U32},
  cert: Exists(O.Cells,c => {O.pack(c) == a : Array<U64>})) -> S.legal(a,b):
  (+c,e)=cert
  +same: {O.pack(c) == a : Array<U64>}=e
  +built: {O.pack(c) == Init.run(d,seed,n,extra) : Array<U64>}=
    Equal.trans(Array<U64>,O.pack(c),a,Init.run(d,seed,n,extra),same,initialized)
  %same : S.legal(_,b)
  Full.known(c,d,seed,n,extra,b,white,built,depth,tables,full,good,nonempty,canonical)

def use_legal_full(a: Array<U64>,+d: Nat,+seed: U64,+n: Nat,+extra: Nat,+b: Chess.Board,+white: Bool,
  initialized: {a == Init.run(d,seed,n,extra) : Array<U64>},
  depth: {d == 17n : Nat},tables: {n == 128n : Nat},full: {extra == 64n : Nat},
  good: {B.valid(b) == True{} : Bool},
  nonempty: {U64.is_zero(K.plane(b)) == False{} : Bool},
  canonical: {Chess.get_turn(b) == Bool.to_u32(white) : U32}) -> S.legal(a,b):
  legal_lift(a,d,seed,n,extra,b,white,initialized,depth,tables,full,good,nonempty,canonical,R.reify(a))

def list_lift(-a: Array<U64>,+d: Nat,+seed: U64,+n: Nat,+extra: Nat,+b: Chess.Board,+white: Bool,
  +xs: List<&2,Chess.Ply>,+tail: List<&2,Chess.Ply>,
  initialized: {a == Init.run(d,seed,n,extra) : Array<U64>},
  +depth: {d == 17n : Nat},+tables: {n == 128n : Nat},+full: {extra == 64n : Nat},
  good: {B.valid(b) == True{} : Bool},
  +nonempty: {U64.is_zero(K.plane(b)) == False{} : Bool},
  canonical: {Chess.get_turn(b) == Bool.to_u32(white) : U32},
  cert: Exists(O.Cells,c => {O.pack(c) == a : Array<U64>})) ->
  S.origins(a,b,xs) -> S.initial(a,b) -> S.list(a,b,xs,tail):
  (+c,e)=cert
  +same: {O.pack(c) == a : Array<U64>}=e
  +built: {O.pack(c) == Init.run(d,seed,n,extra) : Array<U64>}=
    Equal.trans(Array<U64>,O.pack(c),a,Init.run(d,seed,n,extra),same,initialized)
  %same : S.origins(_,b,xs) -> S.initial(_,b) -> S.list(_,b,xs,tail)
  origin => initial =>
    Prior.member_list_fast_full(c,b,white,xs,tail,origin,good,nonempty,canonical,
      Queries.old(c,d,seed,n,extra,b,built,depth,tables,full,nonempty),initial,
      Queries.occurrences(c,d,seed,n,extra,b,xs,built,depth,tables,full,nonempty))

def use_list(a: Array<U64>,+d: Nat,+seed: U64,+n: Nat,+extra: Nat,+b: Chess.Board,+white: Bool,
  +xs: List<&2,Chess.Ply>,+tail: List<&2,Chess.Ply>,
  initialized: {a == Init.run(d,seed,n,extra) : Array<U64>},
  depth: {d == 17n : Nat},tables: {n == 128n : Nat},full: {extra == 64n : Nat},
  good: {B.valid(b) == True{} : Bool},
  nonempty: {U64.is_zero(K.plane(b)) == False{} : Bool},
  canonical: {Chess.get_turn(b) == Bool.to_u32(white) : U32},
  origins: S.origins(a,b,xs),initial: S.initial(a,b)) -> S.list(a,b,xs,tail):
  list_lift(a,d,seed,n,extra,b,white,xs,tail,initialized,depth,tables,full,good,nonempty,canonical,R.reify(a))(origins)(initial)
