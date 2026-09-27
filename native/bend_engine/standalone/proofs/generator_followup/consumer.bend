import Base
import ./PROOF.bend as Proof
import ./LAWS.bend as Laws
import ./Runtime.bend as Run
import ../initialized_attacks/Build.bend as Init
import ../initialized_attacks/Spec.bend as Geometry
import ../../../legal_probe/Chess.bend as Chess

def use_raw(a: Array<U64>,+b: Chess.Board,+q: Chess.Board,+side: U32) ->
  {Run.run(a,b,q,side) == Run.attach(Run.moves(Chess.legal_moves(a,b)),Chess.in_check(a,q,side)) : Array<U64> & (List<&2,Chess.Ply> & Bool)}:
  Laws.generation_then_check_matches_direct_check(a,b,q,side)

def use_initialized(+d: Nat,+seed: U64,+n: Nat,+extra: Nat,+b: Chess.Board,+q: Chess.Board,+sq: Nat,+side: Bool,
  depth: {d == 17n : Nat},tables: {n == 128n : Nat},full: {extra == 64n : Nat},
  bound: {Nat.is_lt(sq,64n) == True{} : Bool},
  one: {U64.and(Chess.get_kings(q),Chess.color(q,Bool.to_u32(side))) == U64.bit(sq) : U64}) ->
  {Run.run(Init.run(d,seed,n,extra),b,q,Bool.to_u32(side)) ==
    (Init.run(d,seed,n,extra),(Run.moves(Chess.legal_moves(Init.run(d,seed,n,extra),b)),Geometry.attacked(q,sq,Bool.not(side)))) : Array<U64> & (List<&2,Chess.Ply> & Bool)}:
  Laws.initialized_generation_then_check_matches_geometry(d,seed,n,extra,b,q,sq,side,depth,tables,full,bound,one)
