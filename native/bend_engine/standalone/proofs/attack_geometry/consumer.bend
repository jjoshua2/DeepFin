import Base
import ./PROOF.bend as Proof
import ./LAWS.bend as Laws
import ./Geometry.bend as G
import ./Slots.bend as S
import ./Query.bend as Q
import ./Initialized.bend as I
import ../../../legal_probe/Chess.bend as Chess
import ../../Tables.bend as Tables
import ../storage/Observe.bend as O
import ../complete/Spec.bend as C

def computed_all(+kind: G.Leaper,+sq: Nat,bound: {Nat.is_lt(sq,64n) == True{} : Bool}) ->
  {S.value(kind,sq) == G.mask(kind,sq) : U64}:
  Laws.computed_leaper_matches_coordinates(kind,sq,bound)

def stored_all(+extra: Nat,a: Array<U64>,+kind: G.Leaper,+sq: Nat,
  sh: {O.shape(a) == C.full(17n) : O.Shape},bound: {Nat.is_lt(sq,64n) == True{} : Bool},full: {extra == 64n : Nat}) ->
  {Array.get(U64,Tables.extras(extra,0,a),S.index(kind,sq)) == (Tables.extras(extra,0,a),G.mask(kind,sq)) : Array<U64> & U64}:
  Laws.extras_store_coordinate_mask(extra,a,kind,sq,sh,bound,full)

def initialized_all(+d: Nat,+seed: U64,+n: Nat,+key: U32,+at: U32,+extra: Nat,+kind: G.Leaper,+sq: Nat,+occ: U64,
  depth: {d == 17n : Nat},full: {extra == 64n : Nat},bound: {Nat.is_lt(sq,64n) == True{} : Bool}) ->
  {Chess.attack(Q.tag(kind),I.run(d,seed,n,key,at,extra),U32.from_nat(sq),Q.side(kind),occ) ==
    (I.run(d,seed,n,key,at,extra),G.mask(kind,sq)) : Array<U64> & U64}:
  Laws.initialized_leaper_query_matches_coordinates(d,seed,n,key,at,extra,kind,sq,occ,depth,full,bound)

# The production reverse pawn query swaps direction. This checks its bounded side
# convention, not a new forward/reverse attack-membership theorem.
def reverse_pawn(+d: Nat,+seed: U64,+n: Nat,+key: U32,+at: U32,+extra: Nat,+sq: Nat,+occ: U64,+by: Bool,
  depth: {d == 17n : Nat},full: {extra == 64n : Nat},bound: {Nat.is_lt(sq,64n) == True{} : Bool}) ->
  {Chess.attack(0,I.run(d,seed,n,key,at,extra),U32.from_nat(sq),U32.xor(Bool.to_u32(by),1),occ) ==
    (I.run(d,seed,n,key,at,extra),G.mask(Bool.pick(G.Leaper,by,G.BlackPawn{},G.WhitePawn{}),sq)) : Array<U64> & U64}:
  match by:
    case True{}: Laws.initialized_leaper_query_matches_coordinates(d,seed,n,key,at,extra,G.BlackPawn{},sq,occ,depth,full,bound)
    case False{}: Laws.initialized_leaper_query_matches_coordinates(d,seed,n,key,at,extra,G.WhitePawn{},sq,occ,depth,full,bound)

def corner_knight() -> {S.value(G.Knight{},0n) == G.mask(G.Knight{},0n) : U64}:
  Laws.computed_leaper_matches_coordinates(G.Knight{},0n,{==})

def last_black_pawn() -> {S.value(G.BlackPawn{},63n) == G.mask(G.BlackPawn{},63n) : U64}:
  Laws.computed_leaper_matches_coordinates(G.BlackPawn{},63n,{==})
