# Counts actual fast and prepare filters; no equivalence or safety claim.
import Base
import ../../../legal_probe/Chess.bend as Chess
import ../legal_filter_count/Count.bend as Count
import ../destination_factorization/AfterCount.bend as After
import ../destination_factorization/Spec.bend as PlySpec
import ../accepted_castle/Filter.bend as Filter
import ../accepted_castle/Spec.bend as FilterSpec
import ../storage/Observe.bend as Observe
import ../table_preservation/Filters.bend as Table
import ../table_preservation/Core.bend as Core

def fast_step_count(required: Bool,+reject: Bool,+hit: Bool,total: Nat) -> Nat:
  match required:
    case True{}: Count.retain_count(reject,hit,total)
    case False{}: PlySpec.bump(hit,total)

def fast_count_fold(+xs: List<&2,Chess.Ply>,+c: Observe.Cells,+b: Chess.Board,
  +sensitive: U64,+query: Chess.Ply,total: Nat) -> Nat:
  match xs:
    case Nil{}: total
    case Con{+m,+rest}:
      fast_count_fold(rest,c,b,sensitive,query,
        fast_step_count(Chess.filter_requires(sensitive,m),
          FilterSpec.checked(c,b,m),PlySpec.same(m,query),total))

def fast_head_count(+required: Bool,+c: Observe.Cells,+b: Chess.Board,
  +m: Chess.Ply,+acc: List<&2,Chess.Ply>,+query: Chess.Ply) ->
  {PlySpec.count(FilterSpec.fast_step(required,c,b,m,acc),query) ==
    fast_step_count(required,FilterSpec.checked(c,b,m),PlySpec.same(m,query),
      PlySpec.count(acc,query)) : Nat}:
  match required:
    case True{}: Count.retained_head_count(FilterSpec.checked(c,b,m),m,acc,query)
    case False{}: {==}

def fast_spec_count(+xs: List<&2,Chess.Ply>,+c: Observe.Cells,+b: Chess.Board,
  +sensitive: U64,+acc: List<&2,Chess.Ply>,+query: Chess.Ply) ->
  {PlySpec.count(FilterSpec.fast(xs,c,b,sensitive,acc),query) ==
    fast_count_fold(xs,c,b,sensitive,query,PlySpec.count(acc,query)) : Nat}:
  match xs:
    case Nil{}: {==}
    case Con{+m,+rest}:
      +required = Chess.filter_requires(sensitive,m)
      +reject = FilterSpec.checked(c,b,m)
      +hit = PlySpec.same(m,query)
      +next_acc = FilterSpec.fast_step(required,c,b,m,acc)
      +step_count = fast_head_count(required,c,b,m,acc,query)
      +rest_count = fast_spec_count(rest,c,b,sensitive,next_acc,query)
      +lift: {fast_count_fold(rest,c,b,sensitive,query,PlySpec.count(next_acc,query)) ==
        fast_count_fold(rest,c,b,sensitive,query,
          fast_step_count(required,reject,hit,PlySpec.count(acc,query))) : Nat} =
        Equal.cong(Nat,Nat,total => fast_count_fold(rest,c,b,sensitive,query,total),
          PlySpec.count(next_acc,query),
          fast_step_count(required,reject,hit,PlySpec.count(acc,query)),step_count)
      Equal.trans(Nat,PlySpec.count(FilterSpec.fast(Con{m,rest},c,b,sensitive,acc),query),
        fast_count_fold(rest,c,b,sensitive,query,PlySpec.count(next_acc,query)),
        fast_count_fold(Con{m,rest},c,b,sensitive,query,PlySpec.count(acc,query)),
        rest_count,lift)

def actual_fast_count(+xs: List<&2,Chess.Ply>,+c: Observe.Cells,+b: Chess.Board,
  +sensitive: U64,+tail: List<&2,Chess.Ply>,+query: Chess.Ply) ->
  {After.pair_count(Chess.filter_fast(xs,b,sensitive,(Observe.pack(c),tail)),query) ==
    fast_count_fold(xs,c,b,sensitive,query,PlySpec.count(tail,query)) : Nat}:
  +actual = Filter.fast(xs,c,b,sensitive,tail)
  +projected: {After.pair_count(Chess.filter_fast(xs,b,sensitive,(Observe.pack(c),tail)),query) ==
    PlySpec.count(FilterSpec.fast(xs,c,b,sensitive,tail),query) : Nat} =
    Equal.trans(Nat,After.pair_count(Chess.filter_fast(xs,b,sensitive,(Observe.pack(c),tail)),query),
      After.pair_count((Observe.pack(c),FilterSpec.fast(xs,c,b,sensitive,tail)),query),
      PlySpec.count(FilterSpec.fast(xs,c,b,sensitive,tail),query),
      Equal.cong(Array<U64> & List<&2,Chess.Ply>,Nat,r => After.pair_count(r,query),
        Chess.filter_fast(xs,b,sensitive,(Observe.pack(c),tail)),
        (Observe.pack(c),FilterSpec.fast(xs,c,b,sensitive,tail)),actual),{==})
  Equal.trans(Nat,After.pair_count(Chess.filter_fast(xs,b,sensitive,(Observe.pack(c),tail)),query),
    PlySpec.count(FilterSpec.fast(xs,c,b,sensitive,tail),query),
    fast_count_fold(xs,c,b,sensitive,query,PlySpec.count(tail,query)),
    projected,fast_spec_count(xs,c,b,sensitive,tail,query))

def choose_count(check: Bool,+c: Observe.Cells,+b: Chess.Board,
  xs: List<&2,Chess.Ply>,rays: U64,+query: Chess.Ply) -> Nat:
  match check:
    case True{}: Count.retained_count_fold(xs,c,b,query,0n)
    case False{}:
      fast_count_fold(xs,c,b,
        U64.and(Chess.color(b,Chess.get_turn(b)),U64.or(rays,Chess.get_kings(b))),query,0n)

def choose_spec_count(+check: Bool,+c: Observe.Cells,+b: Chess.Board,
  +xs: List<&2,Chess.Ply>,+rays: U64,+query: Chess.Ply) ->
  {PlySpec.count(FilterSpec.choose(check,c,b,xs,rays),query) ==
    choose_count(check,c,b,xs,rays,query) : Nat}:
  match check:
    case True{}: Count.full_filter_count(xs,c,b,Nil{},query)
    case False{}:
      fast_spec_count(xs,c,b,
        U64.and(Chess.color(b,Chess.get_turn(b)),U64.or(rays,Chess.get_kings(b))),Nil{},query)

def prepare_count(+c: Observe.Cells,+b: Chess.Board,xs: List<&2,Chess.Ply>,
  query: Chess.Ply) -> Nat:
  choose_count(FilterSpec.initial(c,b),c,b,xs,FilterSpec.rays(c,b),query)

def actual_prepare_count(+c: Observe.Cells,+b: Chess.Board,
  +xs: List<&2,Chess.Ply>,+query: Chess.Ply) ->
  {After.pair_count(Chess.filter_prepare(b,(Observe.pack(c),xs)),query) ==
    prepare_count(c,b,xs,query) : Nat}:
  +actual = Filter.prepare(c,b,xs)
  +projected: {After.pair_count(Chess.filter_prepare(b,(Observe.pack(c),xs)),query) ==
    PlySpec.count(FilterSpec.prepared(c,b,xs),query) : Nat} =
    Equal.trans(Nat,After.pair_count(Chess.filter_prepare(b,(Observe.pack(c),xs)),query),
      After.pair_count((Observe.pack(c),FilterSpec.prepared(c,b,xs)),query),
      PlySpec.count(FilterSpec.prepared(c,b,xs),query),
      Equal.cong(Array<U64> & List<&2,Chess.Ply>,Nat,r => After.pair_count(r,query),
        Chess.filter_prepare(b,(Observe.pack(c),xs)),
        (Observe.pack(c),FilterSpec.prepared(c,b,xs)),actual),{==})
  Equal.trans(Nat,After.pair_count(Chess.filter_prepare(b,(Observe.pack(c),xs)),query),
    PlySpec.count(FilterSpec.prepared(c,b,xs),query),prepare_count(c,b,xs,query),
    projected,choose_spec_count(FilterSpec.initial(c,b),c,b,xs,FilterSpec.rays(c,b),query))

def actual_fast_table_unchanged(+xs: List<&2,Chess.Ply>,+b: Chess.Board,
  +sensitive: U64,r: Array<U64> & List<&2,Chess.Ply>) ->
  {Core.table(List<&2,Chess.Ply>,Chess.filter_fast(xs,b,sensitive,r)) ==
    Core.table(List<&2,Chess.Ply>,r) : Array<U64>}:
  Table.fast(xs,b,sensitive,r)

def actual_prepare_table_unchanged(+b: Chess.Board,r: Array<U64> & List<&2,Chess.Ply>) ->
  {Core.table(List<&2,Chess.Ply>,Chess.filter_prepare(b,r)) ==
    Core.table(List<&2,Chess.Ply>,r) : Array<U64>}:
  Table.prepare(b,r)
