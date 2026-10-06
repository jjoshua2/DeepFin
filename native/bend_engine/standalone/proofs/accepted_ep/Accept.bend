# Extract the actual EP check decision in both prepared-filter branches.
import Base
import ../accepted_castle/Spec.bend as S
import ../accepted_castle/Accept.bend as A
import ../accepted_castle/Filter.bend as Filter
import ../accepted_castle/Generated.bend as G
import ../castle_emission/Spec.bend as E
import ../castle_chain/Spec.bend as C
import ../castle_chain/Lists.bend as L
import ../storage/Observe.bend as O
import ../../../legal_probe/Chess.bend as Chess

# The production flag1 forcing fact also occurs in promotion_bypass_preservation/Raw.
def required(+sensitive: U64,+m: Chess.Ply,tag: {C.flag(m) == 1 : U32}) ->
  {Chess.filter_requires(sensitive,m) == True{} : Bool}:
  match m:
    case Chess.Ply{+src,+dst,+promotion,+flag}:
      %Equal.sym(U32,flag,1,tag) : {Chess.filter_requires(sensitive,Chess.Ply{src,dst,promotion,_}) == True{} : Bool}
      {==}

def chosen(check: Bool,+c: O.Cells,+b: Chess.Board,+xs: List<&2,Chess.Ply>,+rays: U64,+m: Chess.Ply,
  tag: {C.flag(m) == 1 : U32},here: E.member(m,S.choose(check,c,b,xs,rays))) -> S.safe(c,b,m):
  match check:
    case True{}:
      L.at(x => S.safe(c,b,x),S.full(xs,c,b,Nil{}),m,A.full(xs,c,b,Nil{},Unit{}),here)
    case False{}:
      +sensitive = U64.and(Chess.color(b,Chess.get_turn(b)),U64.or(rays,Chess.get_kings(b)))
      L.at(x => S.required_safe(c,b,sensitive,x),S.fast(xs,c,b,sensitive,Nil{}),m,
        A.fast(xs,c,b,sensitive,Nil{},Unit{}),here)(required(sensitive,m,tag))

def prepared(+c: O.Cells,+b: Chess.Board,+xs: List<&2,Chess.Ply>,+m: Chess.Ply,
  tag: {C.flag(m) == 1 : U32},here: E.member(m,E.moves(Chess.filter_prepare(b,(O.pack(c),xs))))) -> S.safe(c,b,m):
  chosen(S.initial(c,b),c,b,xs,S.rays(c,b),m,tag,
    A.transport(Chess.filter_prepare(b,(O.pack(c),xs)),(O.pack(c),S.prepared(c,b,xs)),m,
      Filter.prepare(c,b,xs),here))

def known(+c: O.Cells,+b: Chess.Board,+m: Chess.Ply,
  here: E.member(m,E.moves(Chess.legal_moves(O.pack(c),b))),tag: {C.flag(m) == 1 : U32}) -> S.safe(c,b,m):
  prepared(c,b,E.moves(G.candidates(O.pack(c),b)),m,tag,
    A.transport(Chess.legal_moves(O.pack(c),b),
      Chess.filter_prepare(b,(O.pack(c),E.moves(G.candidates(O.pack(c),b)))),m,G.prepared_equal(c,b),here))
