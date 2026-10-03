import Base
import ../../../legal_probe/Chess.bend as Chess
import ../bitboard_membership/PrefixDecomposition.bend as Prefix
import ../bitboard_membership/AllEmittedSound.bend as Squares
import ./Spec.bend as S
import ./Factor.bend as F
import ./Emission.bend as E
import ./Count.bend as C
import ./Scan.bend as Scan
import ./AfterCount.bend as AfterCount

def use_structural(n: Nat,empty: Bool,+bb: U64,+src: U32,+pawn: Bool,+ep: U32,
  +keys: List<&2,U32>,+tail: List<&2,Chess.Ply>) ->
  {Chess.destinations(n,empty,bb,src,pawn,ep,F.emit(keys,src,pawn,ep,tail)) ==
    F.emit(Chess.bit_squares(n,empty,bb,keys),src,pawn,ep,tail) : List<&2,Chess.Ply>}:
  F.structural(n,empty,bb,src,pawn,ep,keys,tail)

def use_destinations(n: Nat,empty: Bool,+bb: U64,+src: U32,+pawn: Bool,+ep: U32,+tail: List<&2,Chess.Ply>) ->
  {Chess.destinations(n,empty,bb,src,pawn,ep,tail) ==
    S.expand(Chess.bit_squares(n,empty,bb,Nil{}),src,pawn,ep,tail) : List<&2,Chess.Ply>}:
  E.destinations(n,empty,bb,src,pawn,ep,tail)

def use_full_ply_count(n: Nat,empty: Bool,+bb: U64,+src: U32,+pawn: Bool,+ep: U32,+query: Chess.Ply,+tail: List<&2,Chess.Ply>) ->
  {S.count(Chess.destinations(n,empty,bb,src,pawn,ep,tail),query) ==
    S.tally(Chess.bit_squares(n,empty,bb,Nil{}),src,pawn,ep,query,S.count(tail,query)) : Nat}:
  C.destinations(n,empty,bb,src,pawn,ep,query,tail)

def use_scan_after(table: Array<U64>,+targets: U64,+src: U32,+pawn: Bool,+ep: U32,+tail: List<&2,Chess.Ply>) ->
  {Chess.scan_after(src,pawn,ep,tail,(table,targets)) ==
    (table,S.expand(Chess.bit_squares(64n,U64.is_zero(targets),targets,Nil{}),src,pawn,ep,tail)) : Array<U64> & List<&2,Chess.Ply>}:
  Scan.exact(table,targets,src,pawn,ep,tail)

def zero_budget_stops() ->
  {Chess.destinations(0n,False{},U64.bit(12n),4,False{},64,Nil{}) == Nil{} : List<&2,Chess.Ply>}:
  {==}

def promotion_zero_count() ->
  {S.count(S.expand(56 <> Nil{},48,True{},64,Nil{}),Chess.Ply{48,56,0,0}) == 0n : Nat}:
  {==}

def use_actual_key_frequency(+targets: U64,+query: U32) ->
  {Prefix.occurrences(Chess.bit_squares(64n,U64.is_zero(targets),targets,Nil{}),query) ==
    Bool.pick(Nat,U64.test_bit(targets,U32.to_nat(query)),1n,0n) : Nat}:
  AfterCount.actual_key_frequency(targets,query)

def use_actual_keys_below64(+targets: U64) ->
  {Squares.all_below64(Chess.bit_squares(64n,U64.is_zero(targets),targets,Nil{})) == True{} : Bool}:
  AfterCount.actual_keys_below64(targets)

def use_scan_after_count(table: Array<U64>,+targets: U64,+src: U32,+pawn: Bool,+ep: U32,
  +query: Chess.Ply,+tail: List<&2,Chess.Ply>) ->
  {AfterCount.pair_count(Chess.scan_after(src,pawn,ep,tail,(table,targets)),query) ==
    S.tally(Chess.bit_squares(64n,U64.is_zero(targets),targets,Nil{}),
      src,pawn,ep,query,S.count(tail,query)) : Nat}:
  AfterCount.scan_after_count(table,targets,src,pawn,ep,query,tail)

def use_duplicate_tail_multiplicity(table: Array<U64>) ->
  {AfterCount.pair_count(Chess.scan_after(12,False{},64,
    Con{Chess.Ply{12,28,0,0},Con{Chess.Ply{12,28,0,0},Nil{}}},
    (table,U64.zero())),Chess.Ply{12,28,0,0}) == 2n : Nat}:
  AfterCount.duplicate_tail_multiplicity(table)
