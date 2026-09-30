import Base
import ../../../legal_probe/Chess.bend as Chess
import ../castle_emission/Spec.bend as E
import ../castle_chain/Spec.bend as ChainSpec
import ../ray/Frame.bend as F
import ./Spec.bend as S
import ./Bits.bend as Bits

def or_right(+a: Bool,+b: Bool,e: {b == True{} : Bool}) -> {Bool.or(a,b) == True{} : Bool}:
  match a:
    case True{}: {==}
    case False{}: e

def owned(+b: Chess.Board,+rays: U64,+src: U32,+dst: U32,+promotion: U32,+flag: U32,
  bound: {Nat.is_lt(U32.to_nat(src),64n) == True{} : Bool},e: {S.owned_king(b,src) == True{} : Bool}) ->
  {Chess.filter_requires(S.sensitive(b,rays),Chess.Ply{src,dst,promotion,flag}) == True{} : Bool}:
  or_right(U32.is_eq(flag,1),U64.test_bit(S.sensitive(b,rays),U32.to_nat(src)),
    Bits.sensitive(Chess.color(b,Chess.get_turn(b)),Chess.get_kings(b),rays,U32.to_nat(src),bound,e))

def home_bound(+white: Bool) ->
  {Nat.is_lt(U32.to_nat(U32.add(Bool.pick(U32,white,0,56),4)),64n) == True{} : Bool}:
  match white:
    case True{}: {==}
    case False{}: {==}

def king(+b: Chess.Board,+ks: Bool,e: {E.guard(b,ks) == True{} : Bool}) -> {E.king_ok(b) == True{} : Bool}:
  F.left(E.king_ok(b),Bool.and(E.rook_ok(b,ks),E.path_clear(b,ks)),
    F.right(E.rights_ok(b,ks),Bool.and(E.king_ok(b),Bool.and(E.rook_ok(b,ks),E.path_clear(b,ks))),e))

def guarded(+b: Chess.Board,+ks: Bool,+rays: U64,e: {E.guard(b,ks) == True{} : Bool}) ->
  {Chess.filter_requires(S.sensitive(b,rays),E.move(b,ks)) == True{} : Bool}:
  owned(b,rays,E.src(b),E.dst(b,ks),0,2,home_bound(U32.is_eq(Chess.get_turn(b),1)),king(b,ks,e))

def side(+b: Chess.Board,+ks: Bool,+m: Chess.Ply,+rays: U64,p: ChainSpec.side(b,ks,m)) ->
  {Chess.filter_requires(S.sensitive(b,rays),m) == True{} : Bool}:
  (good,same) = p
  Equal.trans(Bool,Chess.filter_requires(S.sensitive(b,rays),m),Chess.filter_requires(S.sensitive(b,rays),E.move(b,ks)),True{},
    Equal.cong(Chess.Ply,Bool,x => Chess.filter_requires(S.sensitive(b,rays),x),m,E.move(b,ks),same),guarded(b,ks,rays,good))

def certificate(+b: Chess.Board,+m: Chess.Ply,+rays: U64,p: ChainSpec.castle(b,m)) ->
  {Chess.filter_requires(S.sensitive(b,rays),m) == True{} : Bool}:
  match p:
    case Inl{k}: side(b,True{},m,rays,k)
    case Inr{q}: side(b,False{},m,rays,q)
