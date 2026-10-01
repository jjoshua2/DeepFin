import ./PROOF.bend as Proof
import ./LAWS.bend as Laws
import Base
# Retain nonempty satisfying inputs for all four side/route combinations.
import ../castle_kings/consumer.bend as Examples
import ../../../legal_probe/Chess.bend as Chess
import ../castle_kings/Spec.bend as K
import ../board/Spec.bend as B
import ../castle_emission/Spec.bend as E
import ../initialized_attacks/Build.bend as Init
import ./Spec.bend as S
import ./Runtime.bend as Runtime

def use_producer(+d: Nat,+seed: U64,+n: Nat,+extra: Nat,+b: Chess.Board,+white: Bool,+ks: Bool,
  +depth: {d == 17n : Nat},+tables: {n == 128n : Nat},+full: {extra == 64n : Nat},
  +good: {B.valid(b) == True{} : Bool},+turn: {Chess.get_turn(b) == Bool.to_u32(white) : U32},
  +one: {K.kings(b,white) == U64.bit(U32.to_nat(K.src(white))) : U64},+guard: {E.guard(b,ks) == True{} : Bool},+tail: List<&2,Chess.Ply>) ->
  {Chess.castle_side(b,ks,(Init.run(d,seed,n,extra),tail)) == (Init.run(d,seed,n,extra),S.before(b,white,ks,tail)) : Array<U64> & List<&2,Chess.Ply>}:
  Laws.initialized_castle_producer_matches_geometry(d,seed,n,extra,b,white,ks,depth,tables,full,good,turn,one,guard,tail)

def use_filtered(+d: Nat,+seed: U64,+n: Nat,+extra: Nat,+b: Chess.Board,+white: Bool,+ks: Bool,
  +depth: {d == 17n : Nat},+tables: {n == 128n : Nat},+full: {extra == 64n : Nat},
  +good: {B.valid(b) == True{} : Bool},+turn: {Chess.get_turn(b) == Bool.to_u32(white) : U32},
  +one: {K.kings(b,white) == U64.bit(U32.to_nat(K.src(white))) : U64},+guard: {E.guard(b,ks) == True{} : Bool}) ->
  {Runtime.run(Init.run(d,seed,n,extra),b,ks) == (Init.run(d,seed,n,extra),S.after(b,white,ks)) : Array<U64> & List<&2,Chess.Ply>}:
  Laws.initialized_filtered_castle_matches_three_stage_geometry(d,seed,n,extra,b,white,ks,depth,tables,full,good,turn,one,guard)
