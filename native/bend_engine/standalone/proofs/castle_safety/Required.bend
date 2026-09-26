import Base
import ../../../legal_probe/Chess.bend as Chess
import ../castle_emission/Spec.bend as E
import ../castle_chain/Spec.bend as ChainSpec
import ../ray/Frame.bend as F
import ./Cover.bend as Cover
import ./Spec.bend as S

def king(+b: Chess.Board,+side: Bool,e: {E.guard(b,side) == True{} : Bool}) ->
  {E.king_ok(b) == True{} : Bool}:
  F.left(E.king_ok(b),Bool.and(E.rook_ok(b,side),E.path_clear(b,side)),
    F.right(E.rights_ok(b,side),Bool.and(E.king_ok(b),Bool.and(E.rook_ok(b,side),E.path_clear(b,side))),e))

def source_bound(+white: Bool) ->
  {Nat.is_lt(U32.to_nat(U32.add(Bool.pick(U32,white,0,56),4)),64n) == True{} : Bool}:
  match white:
    case False{}: {==}
    case True{}: {==}

def guard(+b: Chess.Board,+side: Bool,+rays: U64,e: {E.guard(b,side) == True{} : Bool}) ->
  {Chess.filter_requires(S.sensitive(b,rays),E.move(b,side)) == True{} : Bool}:
  Cover.actual_bit(Chess.color(b,Chess.get_turn(b)),Chess.get_kings(b),rays,U32.to_nat(E.src(b)),
    source_bound(U32.is_eq(Chess.get_turn(b),1)),king(b,side,e))

def side(+b: Chess.Board,+ks: Bool,+m: Chess.Ply,+rays: U64,p: ChainSpec.side(b,ks,m)) ->
  {Chess.filter_requires(S.sensitive(b,rays),m) == True{} : Bool}:
  (ok,same) = p
  Equal.trans(Bool,Chess.filter_requires(S.sensitive(b,rays),m),Chess.filter_requires(S.sensitive(b,rays),E.move(b,ks)),True{},
    Equal.cong(Chess.Ply,Bool,x => Chess.filter_requires(S.sensitive(b,rays),x),m,E.move(b,ks),same),guard(b,ks,rays,ok))

def certificate(+b: Chess.Board,+m: Chess.Ply,+rays: U64,p: ChainSpec.castle(b,m)) ->
  {Chess.filter_requires(S.sensitive(b,rays),m) == True{} : Bool}:
  match p:
    case Inl{k}: side(b,True{},m,rays,k)
    case Inr{q}: side(b,False{},m,rays,q)
