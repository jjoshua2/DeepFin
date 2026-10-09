# Public caller equations. The table is the actual tables-loop plus full extras
# image used by the independently qualified initialized-knight query proof.
import Base
import ./KnightScan.bend as N
import ../destination_factorization/AfterCount.bend as AfterCount
import ../destination_factorization/QueryFactor.bend as Factor
import ../../../legal_probe/Chess.bend as Chess
import ../destination_factorization/Spec.bend as Spec
import ../destination_factorization/QueryCount.bend as Q

def use_knight_piece_targets(+d: Nat,+seed: U64,+n: Nat,+key: U32,+at: U32,+extra: Nat,
  +b: Chess.Board,+sq: Nat,
  depth: {d == 17n : Nat},full: {extra == 64n : Nat},
  bound: {Nat.is_lt(sq,64n) == True{} : Bool},
  piece: {Chess.piece(b,U32.from_nat(sq)) == 1 : U32}) ->
  {Chess.piece_targets(False{},N.initialized_table(d,seed,n,key,at,extra),b,U32.from_nat(sq)) ==
    (N.initialized_table(d,seed,n,key,at,extra),N.own_excluded_knight_targets(b,sq)) : Array<U64> & U64}:
  N.piece_targets_knight(d,seed,n,key,at,extra,b,sq,depth,full,bound,piece)

def use_actual_knight_scan_count(+d: Nat,+seed: U64,+n: Nat,+key: U32,+at: U32,+extra: Nat,
  +b: Chess.Board,+sq: Nat,+query: Chess.Ply,+tail: List<&2,Chess.Ply>,
  depth: {d == 17n : Nat},full: {extra == 64n : Nat},
  bound: {Nat.is_lt(sq,64n) == True{} : Bool},
  piece: {Chess.piece(b,U32.from_nat(sq)) == 1 : U32},
  pawn_clear: {U64.test_bit(Chess.get_pawns(b),U32.to_nat(U32.from_nat(sq))) == False{} : Bool}) ->
  {AfterCount.pair_count(Chess.scan_step(b,U32.from_nat(sq),
    (N.initialized_table(d,seed,n,key,at,extra),tail)),query) ==
    Nat.add(Bool.pick(Nat,Bool.and(Nat.is_lt(U32.to_nat(Q.ply_destination(query)),64n),
      U64.test_bit(N.own_excluded_knight_targets(b,sq),U32.to_nat(Q.ply_destination(query)))),
      Factor.query_emission_count(U32.from_nat(sq),False{},Chess.get_ep(b),query),0n),
      Spec.count(tail,query)) : Nat}:
  N.scan_step_knight_closed_count(d,seed,n,key,at,extra,b,sq,query,tail,depth,full,bound,piece,pawn_clear)
