import Base
import ./FastCount.bend as Fast
import ../../../legal_probe/Chess.bend as Chess
import ../destination_factorization/AfterCount.bend as After
import ../destination_factorization/Spec.bend as PlySpec
import ../storage/Observe.bend as Observe
import ../table_preservation/Core.bend as Core

def actual_fast_occurrence_count(+xs: List<&2,Chess.Ply>,+c: Observe.Cells,
  +b: Chess.Board,+sensitive: U64,+tail: List<&2,Chess.Ply>,+query: Chess.Ply) ->
  {After.pair_count(Chess.filter_fast(xs,b,sensitive,(Observe.pack(c),tail)),query) ==
    Fast.fast_count_fold(xs,c,b,sensitive,query,PlySpec.count(tail,query)) : Nat}:
  Fast.actual_fast_count(xs,c,b,sensitive,tail,query)

def actual_prepare_occurrence_count(+c: Observe.Cells,+b: Chess.Board,
  +xs: List<&2,Chess.Ply>,+query: Chess.Ply) ->
  {After.pair_count(Chess.filter_prepare(b,(Observe.pack(c),xs)),query) ==
    Fast.prepare_count(c,b,xs,query) : Nat}:
  Fast.actual_prepare_count(c,b,xs,query)

def duplicate_unchecked_occurrences(+c: Observe.Cells,+b: Chess.Board) ->
  {After.pair_count(Chess.filter_fast(
    Con{Chess.Ply{12,28,0,0},Con{Chess.Ply{12,28,0,0},Nil{}}},
    b,U64.zero,(Observe.pack(c),Nil{})),Chess.Ply{12,28,0,0}) == 2n : Nat}:
  Fast.actual_fast_count(
    Con{Chess.Ply{12,28,0,0},Con{Chess.Ply{12,28,0,0},Nil{}}},
    c,b,U64.zero,Nil{},Chess.Ply{12,28,0,0})

def actual_fast_table(+xs: List<&2,Chess.Ply>,+b: Chess.Board,+sensitive: U64,
  r: Array<U64> & List<&2,Chess.Ply>) ->
  {Core.table(List<&2,Chess.Ply>,Chess.filter_fast(xs,b,sensitive,r)) ==
    Core.table(List<&2,Chess.Ply>,r) : Array<U64>}:
  Fast.actual_fast_table_unchanged(xs,b,sensitive,r)

def actual_prepare_table(+b: Chess.Board,r: Array<U64> & List<&2,Chess.Ply>) ->
  {Core.table(List<&2,Chess.Ply>,Chess.filter_prepare(b,r)) ==
    Core.table(List<&2,Chess.Ply>,r) : Array<U64>}:
  Fast.actual_prepare_table_unchanged(b,r)
