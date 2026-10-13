# Occurrence counts for the actual full filter, reusing its checked list specification.
import Base
import ../../../legal_probe/Chess.bend as Chess
import ../destination_factorization/AfterCount.bend as After
import ../destination_factorization/Spec.bend as PlySpec
import ../accepted_castle/Filter.bend as Filter
import ../accepted_castle/Spec.bend as FilterSpec
import ../storage/Observe.bend as Observe
import ../table_preservation/Filters.bend as Table
import ../table_preservation/Core.bend as Core

def retain_count(+reject: Bool,+hit: Bool,total: Nat) -> Nat:
  match reject:
    case True{}: total
    case False{}: PlySpec.bump(hit,total)

# This follows the production accumulator order: process each source move,
# then recurse with its retained occurrence already added to the accumulator.
def retained_count_fold(+xs: List<&2,Chess.Ply>,+c: Observe.Cells,
  +b: Chess.Board,+query: Chess.Ply,total: Nat) -> Nat:
  match xs:
    case Nil{}: total
    case Con{+m,+rest}:
      retained_count_fold(rest,c,b,query,
        retain_count(FilterSpec.checked(c,b,m),PlySpec.same(m,query),total))

def retained_head_count(+reject: Bool,+m: Chess.Ply,+acc: List<&2,Chess.Ply>,
  +query: Chess.Ply) ->
  {PlySpec.count(FilterSpec.retain(reject,m,acc),query) ==
    retain_count(reject,PlySpec.same(m,query),PlySpec.count(acc,query)) : Nat}:
  match reject:
    case True{}: {==}
    case False{}: {==}

# Count the exact specification fold used by the existing checked Filter.full
# theorem. The recursion retains the head into acc before processing the tail.
def full_filter_count(+xs: List<&2,Chess.Ply>,+c: Observe.Cells,
  +b: Chess.Board,+acc: List<&2,Chess.Ply>,+query: Chess.Ply) ->
  {PlySpec.count(FilterSpec.full(xs,c,b,acc),query) ==
    retained_count_fold(xs,c,b,query,PlySpec.count(acc,query)) : Nat}:
  match xs:
    case Nil{}: {==}
    case Con{+m,+rest}:
      +reject = FilterSpec.checked(c,b,m)
      +hit = PlySpec.same(m,query)
      +next_acc = FilterSpec.retain(reject,m,acc)
      +step_count = retained_head_count(reject,m,acc,query)
      +rest_count = full_filter_count(rest,c,b,next_acc,query)
      +lift: {retained_count_fold(rest,c,b,query,PlySpec.count(next_acc,query)) ==
        retained_count_fold(rest,c,b,query,retain_count(reject,hit,PlySpec.count(acc,query))) : Nat} =
        Equal.cong(Nat,Nat,total => retained_count_fold(rest,c,b,query,total),
          PlySpec.count(next_acc,query),retain_count(reject,hit,PlySpec.count(acc,query)),step_count)
      Equal.trans(Nat,PlySpec.count(FilterSpec.full(Con{m,rest},c,b,acc),query),
        retained_count_fold(rest,c,b,query,PlySpec.count(next_acc,query)),
        retained_count_fold(Con{m,rest},c,b,query,PlySpec.count(acc,query)),
        rest_count,lift)

# The source-facing theorem counts the actual Chess.filter_legal result.
# Filter.full supplies its exact accumulator semantics; no move predicate is
# assumed beyond the implementation's in_check(table, make_move(...), turn).
def actual_filter_count(+xs: List<&2,Chess.Ply>,+c: Observe.Cells,
  +b: Chess.Board,+tail: List<&2,Chess.Ply>,+query: Chess.Ply) ->
  {After.pair_count(Chess.filter_legal(xs,b,(Observe.pack(c),tail)),query) ==
    retained_count_fold(xs,c,b,query,PlySpec.count(tail,query)) : Nat}:
  +actual = Filter.full(xs,c,b,tail)
  +projected: {After.pair_count(Chess.filter_legal(xs,b,(Observe.pack(c),tail)),query) ==
    PlySpec.count(FilterSpec.full(xs,c,b,tail),query) : Nat} =
    Equal.trans(Nat,After.pair_count(Chess.filter_legal(xs,b,(Observe.pack(c),tail)),query),
      After.pair_count((Observe.pack(c),FilterSpec.full(xs,c,b,tail)),query),
      PlySpec.count(FilterSpec.full(xs,c,b,tail),query),
      Equal.cong(Array<U64> & List<&2,Chess.Ply>,Nat,
        r => After.pair_count(r,query),
        Chess.filter_legal(xs,b,(Observe.pack(c),tail)),
        (Observe.pack(c),FilterSpec.full(xs,c,b,tail)),actual),{==})
  Equal.trans(Nat,After.pair_count(Chess.filter_legal(xs,b,(Observe.pack(c),tail)),query),
    PlySpec.count(FilterSpec.full(xs,c,b,tail),query),
    retained_count_fold(xs,c,b,query,PlySpec.count(tail,query)),
    projected,full_filter_count(xs,c,b,tail,query))

# Two equal retained candidates contribute twice; the premise is exactly the
# implementation predicate for this move and makes no chess-validity claim.
def duplicate_retain(+reject: Bool) -> Nat:
  retain_count(reject,True{},retain_count(reject,True{},0n))

def duplicate_retain_cong(+a: Bool,+b: Bool,
  e: {a == b : Bool}) ->
  {duplicate_retain(a) == duplicate_retain(b) : Nat}:
  %e : {duplicate_retain(a) == duplicate_retain(_) : Nat}
  {==}

def duplicate_fold_count(+c: Observe.Cells,+b: Chess.Board,
  safe: {FilterSpec.checked(c,b,Chess.Ply{12,28,0,0}) == False{} : Bool}) ->
  {retained_count_fold(Con{Chess.Ply{12,28,0,0},
      Con{Chess.Ply{12,28,0,0},Nil{}}},c,b,Chess.Ply{12,28,0,0},0n) == 2n : Nat}:
  Equal.trans(Nat,
    retained_count_fold(Con{Chess.Ply{12,28,0,0},
      Con{Chess.Ply{12,28,0,0},Nil{}}},c,b,Chess.Ply{12,28,0,0},0n),
    duplicate_retain(False{}),
    2n,
    duplicate_retain_cong(
      FilterSpec.checked(c,b,Chess.Ply{12,28,0,0}),False{},safe),
    {==})

# Reuse the source-checked table projection invariant for arbitrary input pairs.
def actual_filter_table_unchanged(+xs: List<&2,Chess.Ply>,+b: Chess.Board,
  r: Array<U64> & List<&2,Chess.Ply>) ->
  {Core.table(List<&2,Chess.Ply>,Chess.filter_legal(xs,b,r)) ==
    Core.table(List<&2,Chess.Ply>,r) : Array<U64>}:
  Table.full(xs,b,r)
