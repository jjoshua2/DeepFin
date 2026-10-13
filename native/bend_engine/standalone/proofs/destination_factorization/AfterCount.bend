# Compose the arbitrary-table actual scan with full-Ply counting and PR1000's
# arbitrary-U32 inventory frequency/range results.
import Base
import ../../../legal_probe/Chess.bend as Chess
import ../bitboard_membership/PrefixDecomposition.bend as Prefix
import ../bitboard_membership/AllEmittedSound.bend as Squares
import ./Spec.bend as S
import ./Emission.bend as E
import ./Count.bend as C
import ./Scan.bend as Scan

def pair_count(pair: Array<U64> & List<&2,Chess.Ply>,+query: Chess.Ply) -> Nat:
  (table,moves) = pair
  S.count(moves,query)

def actual_key_frequency(+targets: U64,+query: U32) ->
  {Prefix.occurrences(Chess.bit_squares(64n,U64.is_zero(targets),targets,Nil{}),query) ==
    Bool.pick(Nat,U64.test_bit(targets,U32.to_nat(query)),1n,0n) : Nat}:
  Squares.actual_bit_squares_frequency_all_u32(targets,query)

def actual_keys_below64(+targets: U64) ->
  {Squares.all_below64(Chess.bit_squares(64n,U64.is_zero(targets),targets,Nil{})) == True{} : Bool}:
  Squares.actual_bit_squares_below64(targets)

def scan_after_count(table: Array<U64>,+targets: U64,+src: U32,+pawn: Bool,+ep: U32,
  +query: Chess.Ply,+tail: List<&2,Chess.Ply>) ->
  {pair_count(Chess.scan_after(src,pawn,ep,tail,(table,targets)),query) ==
    S.tally(Chess.bit_squares(64n,U64.is_zero(targets),targets,Nil{}),
      src,pawn,ep,query,S.count(tail,query)) : Nat}:
  +keys = Chess.bit_squares(64n,U64.is_zero(targets),targets,Nil{})
  +expanded = S.expand(keys,src,pawn,ep,tail)
  +destinations = Chess.destinations(64n,U64.is_zero(targets),targets,src,pawn,ep,tail)
  +scan_exact: {Chess.scan_after(src,pawn,ep,tail,(table,targets)) ==
    (table,expanded) : Array<U64> & List<&2,Chess.Ply>} =
    Scan.exact(table,targets,src,pawn,ep,tail)
  +projected: {pair_count(Chess.scan_after(src,pawn,ep,tail,(table,targets)),query) ==
    pair_count((table,expanded),query) : Nat} =
    Equal.cong(Array<U64> & List<&2,Chess.Ply>,Nat,pair => pair_count(pair,query),
      Chess.scan_after(src,pawn,ep,tail,(table,targets)),(table,expanded),scan_exact)
  +destination_expanded: {S.count(expanded,query) == S.count(destinations,query) : Nat} =
    Equal.cong(List<&2,Chess.Ply>,Nat,moves => S.count(moves,query),
      expanded,destinations,Equal.sym(List<&2,Chess.Ply>,destinations,expanded,
        E.destinations(64n,U64.is_zero(targets),targets,src,pawn,ep,tail)))
  +counted: {S.count(destinations,query) ==
    S.tally(keys,src,pawn,ep,query,S.count(tail,query)) : Nat} =
    C.destinations(64n,U64.is_zero(targets),targets,src,pawn,ep,query,tail)
  Equal.trans(Nat,
    pair_count(Chess.scan_after(src,pawn,ep,tail,(table,targets)),query),
    S.count(expanded,query),S.tally(keys,src,pawn,ep,query,S.count(tail,query)),
    Equal.trans(Nat,pair_count(Chess.scan_after(src,pawn,ep,tail,(table,targets)),query),
      pair_count((table,expanded),query),S.count(expanded,query),projected,{==}),
    Equal.trans(Nat,S.count(expanded,query),S.count(destinations,query),
      S.tally(keys,src,pawn,ep,query,S.count(tail,query)),destination_expanded,counted))

def duplicate_tail_multiplicity(table: Array<U64>) ->
  {pair_count(Chess.scan_after(12,False{},64,Con{Chess.Ply{12,28,0,0},
    Con{Chess.Ply{12,28,0,0},Nil{}}},(table,U64.zero())),Chess.Ply{12,28,0,0}) == 2n : Nat}:
  Equal.trans(Nat,
    pair_count(Chess.scan_after(12,False{},64,Con{Chess.Ply{12,28,0,0},
      Con{Chess.Ply{12,28,0,0},Nil{}}},(table,U64.zero())),Chess.Ply{12,28,0,0}),
    S.tally(Chess.bit_squares(64n,U64.is_zero(U64.zero()),U64.zero(),Nil{}),
      12,False{},64,Chess.Ply{12,28,0,0},
      S.count(Con{Chess.Ply{12,28,0,0},Con{Chess.Ply{12,28,0,0},Nil{}}},
        Chess.Ply{12,28,0,0})),
    2n,scan_after_count(table,U64.zero(),12,False{},64,Chess.Ply{12,28,0,0},
      Con{Chess.Ply{12,28,0,0},Con{Chess.Ply{12,28,0,0},Nil{}}}),{==})
