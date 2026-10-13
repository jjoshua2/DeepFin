import Base
import ./PawnScan.bend as Pawn
import ../destination_factorization/AfterCount.bend as AfterCount
import ../destination_factorization/QueryFactor.bend as Factor
import ../destination_factorization/Spec.bend as Spec
import ../destination_factorization/QueryCount.bend as Q
import ../../../legal_probe/Chess.bend as Chess

def use_actual_pawn_targets(+d: Nat,+seed: U64,+n: Nat,+key: U32,+at: U32,+extra: Nat,
  +b: Chess.Board,+sq: Nat,depth: {d == 17n : Nat},full: {extra == 64n : Nat},
  bound: {Nat.is_lt(sq,64n) == True{} : Bool}) ->
  {Chess.piece_targets(True{},Pawn.initialized_table(d,seed,n,key,at,extra),b,U32.from_nat(sq)) ==
    Pawn.pawn_targets_formula(Pawn.initialized_table(d,seed,n,key,at,extra),b,U32.from_nat(sq)) :
      Array<U64> & U64}:
  Pawn.piece_targets_pawn(d,seed,n,key,at,extra,b,sq,depth,full,bound)

def use_actual_pawn_scan_count(+d: Nat,+seed: U64,+n: Nat,+key: U32,+at: U32,+extra: Nat,
  +b: Chess.Board,+sq: Nat,+query: Chess.Ply,+tail: List<&2,Chess.Ply>,+targets: U64,
  depth: {d == 17n : Nat},full: {extra == 64n : Nat},
  bound: {Nat.is_lt(sq,64n) == True{} : Bool},
  pawn_present: {U64.test_bit(Chess.get_pawns(b),U32.to_nat(U32.from_nat(sq))) == True{} : Bool},
  target_exact: {Pawn.pawn_targets_formula(Pawn.initialized_table(d,seed,n,key,at,extra),b,
    U32.from_nat(sq)) == (Pawn.initialized_table(d,seed,n,key,at,extra),targets) : Array<U64> & U64}) ->
  {AfterCount.pair_count(Chess.scan_step(b,U32.from_nat(sq),
      (Pawn.initialized_table(d,seed,n,key,at,extra),tail)),query) ==
    Nat.add(Bool.pick(Nat,Bool.and(Nat.is_lt(U32.to_nat(Q.ply_destination(query)),64n),
      U64.test_bit(targets,U32.to_nat(Q.ply_destination(query)))),
      Factor.query_emission_count(U32.from_nat(sq),True{},Chess.get_ep(b),query),0n),
      Spec.count(tail,query)) : Nat}:
  Pawn.scan_step_pawn_closed_count(d,seed,n,key,at,extra,b,sq,query,tail,targets,
    depth,full,bound,pawn_present,target_exact)
