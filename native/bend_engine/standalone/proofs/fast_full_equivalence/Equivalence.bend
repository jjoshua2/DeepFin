# Conditional equality of the actual filters; no unconditional legality claim.
import Base
import ../../../legal_probe/Chess.bend as Chess
import ../accepted_castle/Spec.bend as S
import ../accepted_castle/Filter.bend as Filter
import ../castle_chain/Spec.bend as C
import ../storage/Observe.bend as Observe

# Only bypassed candidates must have the actual full-filter check answer False.
# This predicate says the full filter would retain them, not chess legality.
def bypass_safe(-c: Observe.Cells,-b: Chess.Board,-sensitive: U64,-m: Chess.Ply) -> Type:
  {Chess.filter_requires(sensitive,m) == False{} : Bool} -> S.safe(c,b,m)

def fast_full_head(required: Bool,+c: Observe.Cells,+b: Chess.Board,
  +sensitive: U64,+m: Chess.Ply,+acc: List<&2,Chess.Ply>,
  same: {Chess.filter_requires(sensitive,m) == required : Bool},
  safe: bypass_safe(c,b,sensitive,m)) ->
  {S.fast_step(required,c,b,m,acc) ==
    S.retain(S.checked(c,b,m),m,acc) : List<&2,Chess.Ply>}:
  match required:
    case True{}: {==}
    case False{}:
      %Equal.sym(Bool,S.checked(c,b,m),False{},safe(same)) :
        {Con{m,acc} == S.retain(_,m,acc) : List<&2,Chess.Ply>}
      {==}

def fast_full_equal(+xs: List<&2,Chess.Ply>,+c: Observe.Cells,+b: Chess.Board,
  +sensitive: U64,+tail: List<&2,Chess.Ply>,
  premise: C.every(m => bypass_safe(c,b,sensitive,m),xs)) ->
  {S.fast(xs,c,b,sensitive,tail) == S.full(xs,c,b,tail) : List<&2,Chess.Ply>}:
  match xs:
    case Nil{}: {==}
    case Con{+m,+rest}:
      (head_safe,rest_safe) = premise
      +required = Chess.filter_requires(sensitive,m)
      +next_fast = S.fast_step(required,c,b,m,tail)
      +next_full = S.retain(S.checked(c,b,m),m,tail)
      +head_eq = fast_full_head(required,c,b,sensitive,m,tail,{==},head_safe)
      +rest_eq = fast_full_equal(rest,c,b,sensitive,next_fast,rest_safe)
      +lift: {S.full(rest,c,b,next_fast) == S.full(rest,c,b,next_full) : List<&2,Chess.Ply>} =
        Equal.cong(List<&2,Chess.Ply>,List<&2,Chess.Ply>,
          acc => S.full(rest,c,b,acc),next_fast,next_full,head_eq)
      Equal.trans(List<&2,Chess.Ply>,
        S.fast(Con{m,rest},c,b,sensitive,tail),
        S.full(rest,c,b,next_fast),S.full(Con{m,rest},c,b,tail),rest_eq,lift)

def pair_lift(+c: Observe.Cells,+xs: List<&2,Chess.Ply>,+ys: List<&2,Chess.Ply>,
  same: {xs == ys : List<&2,Chess.Ply>}) ->
  {(Observe.pack(c),xs) == (Observe.pack(c),ys) : Array<U64> & List<&2,Chess.Ply>}:
  Equal.cong(List<&2,Chess.Ply>,Array<U64> & List<&2,Chess.Ply>,
    moves => (Observe.pack(c),moves),xs,ys,same)

def actual_fast_full(+xs: List<&2,Chess.Ply>,+c: Observe.Cells,+b: Chess.Board,
  +sensitive: U64,+tail: List<&2,Chess.Ply>,
  premise: C.every(m => bypass_safe(c,b,sensitive,m),xs)) ->
  {Chess.filter_fast(xs,b,sensitive,(Observe.pack(c),tail)) ==
    Chess.filter_legal(xs,b,(Observe.pack(c),tail)) : Array<U64> & List<&2,Chess.Ply>}:
  +fast_actual = Filter.fast(xs,c,b,sensitive,tail)
  +full_actual = Filter.full(xs,c,b,tail)
  +spec_eq = fast_full_equal(xs,c,b,sensitive,tail,premise)
  +paired = pair_lift(c,S.fast(xs,c,b,sensitive,tail),S.full(xs,c,b,tail),spec_eq)
  Equal.trans(Array<U64> & List<&2,Chess.Ply>,
    Chess.filter_fast(xs,b,sensitive,(Observe.pack(c),tail)),
    (Observe.pack(c),S.fast(xs,c,b,sensitive,tail)),
    Chess.filter_legal(xs,b,(Observe.pack(c),tail)),fast_actual,
    Equal.trans(Array<U64> & List<&2,Chess.Ply>,
      (Observe.pack(c),S.fast(xs,c,b,sensitive,tail)),
      (Observe.pack(c),S.full(xs,c,b,tail)),
      Chess.filter_legal(xs,b,(Observe.pack(c),tail)),paired,
      Equal.sym(Array<U64> & List<&2,Chess.Ply>,
        Chess.filter_legal(xs,b,(Observe.pack(c),tail)),
        (Observe.pack(c),S.full(xs,c,b,tail)),full_actual)))

def sensitive(+b: Chess.Board,rays: U64) -> U64:
  U64.and(Chess.color(b,Chess.get_turn(b)),U64.or(rays,Chess.get_kings(b)))

def prepared_sensitive(+c: Observe.Cells,+b: Chess.Board) -> U64:
  sensitive(b,S.rays(c,b))

def chosen_full_equal(check: Bool,+c: Observe.Cells,+b: Chess.Board,
  +xs: List<&2,Chess.Ply>,+rays: U64,
  premise: C.every(m => bypass_safe(c,b,sensitive(b,rays),m),xs)) ->
  {S.choose(check,c,b,xs,rays) == S.full(xs,c,b,Nil{}) : List<&2,Chess.Ply>}:
  match check:
    case True{}: {==}
    case False{}: fast_full_equal(xs,c,b,sensitive(b,rays),Nil{},premise)

def actual_prepare_full(+c: Observe.Cells,+b: Chess.Board,+xs: List<&2,Chess.Ply>,
  premise: C.every(m => bypass_safe(c,b,prepared_sensitive(c,b),m),xs)) ->
  {Chess.filter_prepare(b,(Observe.pack(c),xs)) ==
    Chess.filter_legal(xs,b,(Observe.pack(c),Nil{})) : Array<U64> & List<&2,Chess.Ply>}:
  +prepared_actual = Filter.prepare(c,b,xs)
  +full_actual = Filter.full(xs,c,b,Nil{})
  +spec_eq = chosen_full_equal(S.initial(c,b),c,b,xs,S.rays(c,b),premise)
  +paired = pair_lift(c,S.prepared(c,b,xs),S.full(xs,c,b,Nil{}),spec_eq)
  Equal.trans(Array<U64> & List<&2,Chess.Ply>,
    Chess.filter_prepare(b,(Observe.pack(c),xs)),
    (Observe.pack(c),S.prepared(c,b,xs)),
    Chess.filter_legal(xs,b,(Observe.pack(c),Nil{})),prepared_actual,
    Equal.trans(Array<U64> & List<&2,Chess.Ply>,
      (Observe.pack(c),S.prepared(c,b,xs)),(Observe.pack(c),S.full(xs,c,b,Nil{})),
      Chess.filter_legal(xs,b,(Observe.pack(c),Nil{})),paired,
      Equal.sym(Array<U64> & List<&2,Chess.Ply>,
        Chess.filter_legal(xs,b,(Observe.pack(c),Nil{})),
        (Observe.pack(c),S.full(xs,c,b,Nil{})),full_actual)))
