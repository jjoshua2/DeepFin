# Production consumer for actual filter_legal full-Ply occurrences and table state.
import Base
import ../../../legal_probe/Chess.bend as Chess
import ../storage/Observe.bend as Observe
import ../destination_factorization/AfterCount.bend as After
import ../destination_factorization/Spec.bend as PlySpec
import ../accepted_castle/Spec.bend as FilterSpec
import ../table_preservation/Core.bend as Core
import ./Count.bend as Count

# Arbitrary candidate occurrences and an arbitrary, duplicate-containing tail.
def actual_filter_occurrence_count(+xs: List<&2,Chess.Ply>,+c: Observe.Cells,
  +b: Chess.Board,+tail: List<&2,Chess.Ply>,+query: Chess.Ply) ->
  {After.pair_count(Chess.filter_legal(xs,b,(Observe.pack(c),tail)),query) ==
    Count.retained_count_fold(xs,c,b,query,PlySpec.count(tail,query)) : Nat}:
  Count.actual_filter_count(xs,c,b,tail,query)

# Two copies of an accepted candidate contribute two matching Ply occurrences.
def duplicate_candidate_occurrences(+c: Observe.Cells,+b: Chess.Board,
  accepted: {FilterSpec.checked(c,b,Chess.Ply{12,28,0,0}) == False{} : Bool}) ->
  {After.pair_count(Chess.filter_legal(
      Con{Chess.Ply{12,28,0,0},Con{Chess.Ply{12,28,0,0},Nil{}}},b,
      (Observe.pack(c),Nil{})),Chess.Ply{12,28,0,0}) == 2n : Nat}:
  +counted = Count.actual_filter_count(
    Con{Chess.Ply{12,28,0,0},Con{Chess.Ply{12,28,0,0},Nil{}}},
    c,b,Nil{},Chess.Ply{12,28,0,0})
  Equal.trans(Nat,
    After.pair_count(Chess.filter_legal(
      Con{Chess.Ply{12,28,0,0},Con{Chess.Ply{12,28,0,0},Nil{}}},b,
      (Observe.pack(c),Nil{})),Chess.Ply{12,28,0,0}),
    Count.retained_count_fold(
      Con{Chess.Ply{12,28,0,0},Con{Chess.Ply{12,28,0,0},Nil{}}},
      c,b,Chess.Ply{12,28,0,0},0n),
    2n,counted,Count.duplicate_fold_count(c,b,accepted))

# The array-preservation theorem is applied to the actual source function and
# its exact input pair; no table equality is assumed as a premise.
def actual_filter_preserves_input_table(+xs: List<&2,Chess.Ply>,+b: Chess.Board,
  r: Array<U64> & List<&2,Chess.Ply>) ->
  {Core.table(List<&2,Chess.Ply>,Chess.filter_legal(xs,b,r)) ==
    Core.table(List<&2,Chess.Ply>,r) : Array<U64>}:
  Count.actual_filter_table_unchanged(xs,b,r)
