# Exact tag-two projection through the actual prepared full/fast filter.
# Ordinary safety is not assumed or concluded.
import Base
import ./CastleSpec.bend as P
import ../accepted_castle/Spec.bend as S
import ../accepted_castle/Filter.bend as Filter
import ../castle_chain/Spec.bend as C
import ../castle_emission/Spec.bend as E
import ../castle_safety/Spec.bend as M
import ../castle_safety/Mask.bend as Mask
import ../storage/Observe.bend as O
import ../storage/Representation.bend as R
import ../../../legal_probe/Chess.bend as Chess

def ordinary_cons(+m: Chess.Ply,+acc: List<&2,Chess.Ply>,
  plain: C.ordinary(m)) ->
  {P.project(Con{m,acc}) == P.project(acc) : List<&2,Chess.Ply>}:
  Equal.cong(Bool,List<&2,Chess.Ply>,
    tag => P.proj_bit(tag,m,P.project(acc)),
    U32.is_eq(C.flag(m),2),False{},plain)

def ordinary_retain(reject: Bool,+m: Chess.Ply,+acc: List<&2,Chess.Ply>,
  plain: C.ordinary(m)) ->
  {P.project(S.retain(reject,m,acc)) == P.project(acc) : List<&2,Chess.Ply>}:
  match reject:
    case True{}: {==}
    case False{}: ordinary_cons(m,acc,plain)

def castle_retain(reject: Bool,+m: Chess.Ply,+acc: List<&2,Chess.Ply>,
  tagged: {U32.is_eq(C.flag(m),2) == True{} : Bool}) ->
  {P.project(S.retain(reject,m,acc)) ==
    S.retain(reject,m,P.project(acc)) : List<&2,Chess.Ply>}:
  match reject:
    case True{}: {==}
    case False{}:
      Equal.cong(Bool,List<&2,Chess.Ply>,
        tag => P.proj_bit(tag,m,P.project(acc)),
        U32.is_eq(C.flag(m),2),True{},tagged)

# Used to erase the ordinary scan after, rather than before, its actual execution.
def ordinary(xs: List<&2,Chess.Ply>,good: C.every(C.ordinary,xs)) ->
  {P.project(xs) == Nil{} : List<&2,Chess.Ply>}:
  match xs:
    case Nil{}: {==}
    case Con{+m,+rest}:
      (plain,others) = good
      Equal.trans(List<&2,Chess.Ply>,
        P.project(Con{m,rest}),P.project(rest),Nil{},
        ordinary_cons(m,rest,plain),
        ordinary(rest,others))

# The tail induction uses the actual checked accumulator. This local identity
# moves projection across that accumulator and the corresponding input head.
def full_head(tag: Bool,+c: O.Cells,+b: Chess.Board,
  +m: Chess.Ply,+rest: List<&2,Chess.Ply>,+acc: List<&2,Chess.Ply>,
  same: {U32.is_eq(C.flag(m),2) == tag : Bool}) ->
  {S.full(P.project(rest),c,b,
      P.project(S.retain(S.checked(c,b,m),m,acc))) ==
    S.full(P.proj_bit(tag,m,P.project(rest)),c,b,P.project(acc))
      : List<&2,Chess.Ply>}:
  match tag:
    case False{}:
      Equal.cong(List<&2,Chess.Ply>,List<&2,Chess.Ply>,
        tail => S.full(P.project(rest),c,b,tail),
        P.project(S.retain(S.checked(c,b,m),m,acc)),
        P.project(acc),
        ordinary_retain(S.checked(c,b,m),m,acc,same))
    case True{}:
      Equal.cong(List<&2,Chess.Ply>,List<&2,Chess.Ply>,
        tail => S.full(P.project(rest),c,b,tail),
        P.project(S.retain(S.checked(c,b,m),m,acc)),
        S.retain(S.checked(c,b,m),m,P.project(acc)),
        castle_retain(S.checked(c,b,m),m,acc,same))

# No classification or initialization premise is needed for the full fold.
def full(xs: List<&2,Chess.Ply>,+c: O.Cells,+b: Chess.Board,
  +acc: List<&2,Chess.Ply>) ->
  {P.project(S.full(xs,c,b,acc)) ==
    S.full(P.project(xs),c,b,P.project(acc)) : List<&2,Chess.Ply>}:
  match xs:
    case Nil{}: {==}
    case Con{+m,+rest}:
      Equal.trans(List<&2,Chess.Ply>,
        P.project(S.full(rest,c,b,S.retain(S.checked(c,b,m),m,acc))),
        S.full(P.project(rest),c,b,
          P.project(S.retain(S.checked(c,b,m),m,acc))),
        S.full(P.project(Con{m,rest}),c,b,P.project(acc)),
        full(rest,c,b,S.retain(S.checked(c,b,m),m,acc)),
        full_head(U32.is_eq(C.flag(m),2),c,b,m,rest,acc,{==}))

def ordinary_fast(required: Bool,+c: O.Cells,+b: Chess.Board,
  +m: Chess.Ply,+acc: List<&2,Chess.Ply>,plain: C.ordinary(m)) ->
  {P.project(S.fast_step(required,c,b,m,acc)) ==
    P.project(acc) : List<&2,Chess.Ply>}:
  match required:
    case True{}:
      ordinary_retain(S.checked(c,b,m),m,acc,plain)
    case False{}:
      ordinary_cons(m,acc,plain)

# On ordinary moves either fast branch has the same empty projected addition.
# On a generated castle, its owned home king forces the actual checked branch.
def step(+c: O.Cells,+b: Chess.Board,+rays: U64,
  +m: Chess.Ply,+acc: List<&2,Chess.Ply>,part: C.classified(b,m)) ->
  {P.project(S.fast_step(
      Chess.filter_requires(M.sensitive(b,rays),m),c,b,m,acc)) ==
    P.project(S.retain(S.checked(c,b,m),m,acc)) : List<&2,Chess.Ply>}:
  match part:
    case Inl{+plain}:
      Equal.trans(List<&2,Chess.Ply>,
        P.project(S.fast_step(
          Chess.filter_requires(M.sensitive(b,rays),m),c,b,m,acc)),
        P.project(acc),
        P.project(S.retain(S.checked(c,b,m),m,acc)),
        ordinary_fast(
          Chess.filter_requires(M.sensitive(b,rays),m),c,b,m,acc,plain),
        Equal.sym(List<&2,Chess.Ply>,
          P.project(S.retain(S.checked(c,b,m),m,acc)),
          P.project(acc),
          ordinary_retain(S.checked(c,b,m),m,acc,plain)))
    case Inr{castle}:
      Equal.cong(Bool,List<&2,Chess.Ply>,
        required => P.project(S.fast_step(required,c,b,m,acc)),
        Chess.filter_requires(M.sensitive(b,rays),m),True{},
        Mask.certificate(b,m,rays,castle))

def fast(xs: List<&2,Chess.Ply>,+c: O.Cells,+b: Chess.Board,
  +rays: U64,+acc: List<&2,Chess.Ply>,
  good: C.every(m => C.classified(b,m),xs)) ->
  {P.project(S.fast(xs,c,b,M.sensitive(b,rays),acc)) ==
    S.full(P.project(xs),c,b,P.project(acc)) : List<&2,Chess.Ply>}:
  match xs:
    case Nil{}: {==}
    case Con{+m,+rest}:
      (part,others) = good
      Equal.trans(List<&2,Chess.Ply>,
        P.project(S.fast(rest,c,b,M.sensitive(b,rays),
          S.fast_step(Chess.filter_requires(M.sensitive(b,rays),m),
            c,b,m,acc))),
        S.full(P.project(rest),c,b,
          P.project(S.fast_step(
            Chess.filter_requires(M.sensitive(b,rays),m),c,b,m,acc))),
        S.full(P.project(Con{m,rest}),c,b,P.project(acc)),
        fast(rest,c,b,rays,
          S.fast_step(Chess.filter_requires(M.sensitive(b,rays),m),
            c,b,m,acc),
          others),
        Equal.trans(List<&2,Chess.Ply>,
          S.full(P.project(rest),c,b,
            P.project(S.fast_step(
              Chess.filter_requires(M.sensitive(b,rays),m),c,b,m,acc))),
          S.full(P.project(rest),c,b,
            P.project(S.retain(S.checked(c,b,m),m,acc))),
          S.full(P.project(Con{m,rest}),c,b,P.project(acc)),
          Equal.cong(List<&2,Chess.Ply>,List<&2,Chess.Ply>,
            tail => S.full(P.project(rest),c,b,tail),
            P.project(S.fast_step(
              Chess.filter_requires(M.sensitive(b,rays),m),c,b,m,acc)),
            P.project(S.retain(S.checked(c,b,m),m,acc)),
            step(c,b,rays,m,acc,part)),
          full_head(U32.is_eq(C.flag(m),2),c,b,m,rest,acc,{==})))

def choose(check: Bool,+c: O.Cells,+b: Chess.Board,
  +xs: List<&2,Chess.Ply>,+rays: U64,
  good: C.every(m => C.classified(b,m),xs)) ->
  {P.project(S.choose(check,c,b,xs,rays)) ==
    S.full(P.project(xs),c,b,Nil{}) : List<&2,Chess.Ply>}:
  match check:
    case True{}:
      full(xs,c,b,Nil{})
    case False{}:
      fast(xs,c,b,rays,Nil{},good)

def prepared(+c: O.Cells,+b: Chess.Board,+xs: List<&2,Chess.Ply>,
  good: C.every(m => C.classified(b,m),xs)) ->
  {P.project(S.prepared(c,b,xs)) ==
    S.full(P.project(xs),c,b,Nil{}) : List<&2,Chess.Ply>}:
  choose(S.initial(c,b),c,b,xs,S.rays(c,b),good)

def prepared_moves(+c: O.Cells,+b: Chess.Board,+xs: List<&2,Chess.Ply>,
  good: C.every(m => C.classified(b,m),xs)) ->
  {P.project(E.moves(Chess.filter_prepare(b,(O.pack(c),xs)))) ==
    S.full(P.project(xs),c,b,Nil{}) : List<&2,Chess.Ply>}:
  Equal.trans(List<&2,Chess.Ply>,
    P.project(E.moves(Chess.filter_prepare(b,(O.pack(c),xs)))),
    P.project(S.prepared(c,b,xs)),
    S.full(P.project(xs),c,b,Nil{}),
    Equal.cong(Array<U64> & List<&2,Chess.Ply>,List<&2,Chess.Ply>,
      result => P.project(E.moves(result)),
      Chess.filter_prepare(b,(O.pack(c),xs)),
      (O.pack(c),S.prepared(c,b,xs)),
      Filter.prepare(c,b,xs)),
    prepared(c,b,xs,good))

def full_moves(+c: O.Cells,+b: Chess.Board,+xs: List<&2,Chess.Ply>) ->
  {E.moves(Chess.filter_legal(xs,b,(O.pack(c),Nil{}))) ==
    S.full(xs,c,b,Nil{}) : List<&2,Chess.Ply>}:
  Equal.cong(Array<U64> & List<&2,Chess.Ply>,List<&2,Chess.Ply>,
    result => E.moves(result),
    Chess.filter_legal(xs,b,(O.pack(c),Nil{})),
    (O.pack(c),S.full(xs,c,b,Nil{})),
    Filter.full(xs,c,b,Nil{}))

def packed(+c: O.Cells,+b: Chess.Board,+xs: List<&2,Chess.Ply>,
  good: C.every(m => C.classified(b,m),xs)) ->
  {P.project(E.moves(Chess.filter_prepare(b,(O.pack(c),xs)))) ==
    E.moves(Chess.filter_legal(P.project(xs),b,(O.pack(c),Nil{})))
      : List<&2,Chess.Ply>}:
  Equal.trans(List<&2,Chess.Ply>,
    P.project(E.moves(Chess.filter_prepare(b,(O.pack(c),xs)))),
    S.full(P.project(xs),c,b,Nil{}),
    E.moves(Chess.filter_legal(P.project(xs),b,(O.pack(c),Nil{}))),
    prepared_moves(c,b,xs,good),
    Equal.sym(List<&2,Chess.Ply>,
      E.moves(Chess.filter_legal(P.project(xs),b,(O.pack(c),Nil{}))),
      S.full(P.project(xs),c,b,Nil{}),
      full_moves(c,b,P.project(xs))))

def lift(-a: Array<U64>,+b: Chess.Board,+xs: List<&2,Chess.Ply>,
  cert: Exists(O.Cells,c => {O.pack(c) == a : Array<U64>})) ->
  C.every(m => C.classified(b,m),xs) ->
  {P.project(E.moves(Chess.filter_prepare(b,(a,xs)))) ==
    E.moves(Chess.filter_legal(P.project(xs),b,(a,Nil{})))
      : List<&2,Chess.Ply>}:
  (+c,e) = cert
  +same: {O.pack(c) == a : Array<U64>} = e
  %same :
    C.every(m => C.classified(b,m),xs) ->
    {P.project(E.moves(Chess.filter_prepare(b,(_,xs)))) ==
      E.moves(Chess.filter_legal(P.project(xs),b,(a,Nil{})))
        : List<&2,Chess.Ply>}
  %same :
    C.every(m => C.classified(b,m),xs) ->
    {P.project(E.moves(Chess.filter_prepare(b,(O.pack(c),xs)))) ==
      E.moves(Chess.filter_legal(P.project(xs),b,(_,Nil{})))
        : List<&2,Chess.Ply>}
  good => packed(c,b,xs,good)

def actual(a: Array<U64>,+b: Chess.Board,+xs: List<&2,Chess.Ply>,
  good: C.every(m => C.classified(b,m),xs)) ->
  {P.project(E.moves(Chess.filter_prepare(b,(a,xs)))) ==
    E.moves(Chess.filter_legal(P.project(xs),b,(a,Nil{})))
      : List<&2,Chess.Ply>}:
  lift(a,b,xs,R.reify(a))(good)
