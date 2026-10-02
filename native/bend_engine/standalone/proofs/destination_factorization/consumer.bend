import Base
import ../../../legal_probe/Chess.bend as Chess
import ./Spec.bend as S
import ./Factor.bend as F
import ./Emission.bend as E
import ./Count.bend as C
import ./Scan.bend as Scan

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
