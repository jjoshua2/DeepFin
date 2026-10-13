# Public caller equations. The table is the actual tables-loop plus full extras
# image used by the independently qualified initialized-king query proof.
import Base
import ./KingScan.bend as N
import ../destination_factorization/AfterCount.bend as AfterCount
import ../destination_factorization/QueryFactor.bend as Factor
import ../../../legal_probe/Chess.bend as Chess
import ../destination_factorization/Spec.bend as Spec
import ../destination_factorization/QueryCount.bend as Q

def use_king_piece_targets(+d: Nat,+seed: U64,+n: Nat,+key: U32,+at: U32,+extra: Nat,
  +b: Chess.Board,+sq: Nat,
  depth: {d == 17n : Nat},full: {extra == 64n : Nat},
  bound: {Nat.is_lt(sq,64n) == True{} : Bool},
  piece: {Chess.piece(b,U32.from_nat(sq)) == 5 : U32},
  king_present: {U64.test_bit(Chess.get_kings(b),U32.to_nat(U32.from_nat(sq))) == True{} : Bool}) ->
  {Chess.piece_targets(False{},N.initialized_table(d,seed,n,key,at,extra),b,U32.from_nat(sq)) ==
    (N.initialized_table(d,seed,n,key,at,extra),N.own_excluded_king_targets(b,sq)) : Array<U64> & U64}:
  N.piece_targets_king(d,seed,n,key,at,extra,b,sq,depth,full,bound,piece,king_present)

def use_actual_king_scan_count(+d: Nat,+seed: U64,+n: Nat,+key: U32,+at: U32,+extra: Nat,
  +b: Chess.Board,+sq: Nat,+query: Chess.Ply,+tail: List<&2,Chess.Ply>,
  depth: {d == 17n : Nat},full: {extra == 64n : Nat},
  bound: {Nat.is_lt(sq,64n) == True{} : Bool},
  piece: {Chess.piece(b,U32.from_nat(sq)) == 5 : U32},
  king_present: {U64.test_bit(Chess.get_kings(b),U32.to_nat(U32.from_nat(sq))) == True{} : Bool},
  pawn_clear: {U64.test_bit(Chess.get_pawns(b),U32.to_nat(U32.from_nat(sq))) == False{} : Bool}) ->
  {AfterCount.pair_count(Chess.scan_step(b,U32.from_nat(sq),
    (N.initialized_table(d,seed,n,key,at,extra),tail)),query) ==
    Nat.add(Bool.pick(Nat,Bool.and(Nat.is_lt(U32.to_nat(Q.ply_destination(query)),64n),
      U64.test_bit(N.own_excluded_king_targets(b,sq),U32.to_nat(Q.ply_destination(query)))),
      Factor.query_emission_count(U32.from_nat(sq),False{},Chess.get_ep(b),query),0n),
      Spec.count(tail,query)) : Nat}:
  N.scan_step_king_closed_count(d,seed,n,key,at,extra,b,sq,query,tail,depth,full,bound,piece,king_present,pawn_clear)
