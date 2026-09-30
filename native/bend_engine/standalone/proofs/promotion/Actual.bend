# Bridge actual flag-zero promotion to the unchanged clearing/insertion kernel.
import Base
import ./Spec.bend as S
import ./Encoding.bend as E
import ../move_update/Spec.bend as Move
import ../move_update/Core.bend as Core
import ../move_update/Clear.bend as Clear
import ../move_update/Ordinary.bend as Ordinary
import ../board/Spec.bend as B
import ../board/Actual.bend as BA
import ../../../legal_probe/Chess.bend as Chess

def dispatch(+b: Chess.Board,+src: U32,+dst: U32,put: U32) -> Chess.Board:
  Core.raw(b,Move.removal(U64.bit(U32.to_nat(src)),U64.bit(U32.to_nat(dst))),U64.bit(U32.to_nat(dst)),put,
    U32.is_eq(Chess.get_turn(b),1),U32.xor(Chess.get_turn(b),1),
    Ordinary.rights(b,src,dst,Chess.piece(b,src)),Ordinary.ep(src,dst,Chess.piece(b,src)))

def unfold(+b: Chess.Board,+src: U32,+dst: U32,+c: S.Choice) ->
  {Chess.make_move(b,S.ply(src,dst,c)) == dispatch(b,src,dst,Bool.pick(U32,U32.is_zero(S.tag(c)),Chess.piece(b,src),S.tag(c))) : Chess.Board}:
  match b:
    case Chess.Board{p,n,bi,r,q,k,w,bl,turn,rights,ep}: {==}

def exact(+b: Chess.Board,+src: U32,+dst: U32,+c: S.Choice) ->
  {Chess.make_move(b,S.ply(src,dst,c)) == S.expected(b,src,dst,c) : Chess.Board}:
  Equal.trans(Chess.Board,Chess.make_move(b,S.ply(src,dst,c)),
    dispatch(b,src,dst,Bool.pick(U32,U32.is_zero(S.tag(c)),Chess.piece(b,src),S.tag(c))),S.expected(b,src,dst,c),
    unfold(b,src,dst,c),
    Equal.trans(Chess.Board,dispatch(b,src,dst,Bool.pick(U32,U32.is_zero(S.tag(c)),Chess.piece(b,src),S.tag(c))),
      dispatch(b,src,dst,S.tag(c)),S.expected(b,src,dst,c),
      Equal.cong(U32,Chess.Board,k => dispatch(b,src,dst,k),
        Bool.pick(U32,U32.is_zero(S.tag(c)),Chess.piece(b,src),S.tag(c)),S.tag(c),E.selected(Chess.piece(b,src),c)),
      Core.bridge(b,Move.removal(U64.bit(U32.to_nat(src)),U64.bit(U32.to_nat(dst))),U64.bit(U32.to_nat(dst)),S.piece(c),
        U32.is_eq(Chess.get_turn(b),1),U32.xor(Chess.get_turn(b),1),
        Ordinary.rights(b,src,dst,Chess.piece(b,src)),Ordinary.ep(src,dst,Chess.piece(b,src)))))

def preserves(+b: Chess.Board,+src: U32,+dst: U32,+c: S.Choice,good: {B.valid(b) == True{} : Bool}) ->
  {B.valid(Chess.make_move(b,S.ply(src,dst,c))) == True{} : Bool}:
  Equal.trans(Bool,B.valid(Chess.make_move(b,S.ply(src,dst,c))),B.valid(S.expected(b,src,dst,c)),True{},
    Equal.cong(Chess.Board,Bool,x => B.valid(x),Chess.make_move(b,S.ply(src,dst,c)),S.expected(b,src,dst,c),exact(b,src,dst,c)),
    Equal.trans(Bool,B.valid(S.expected(b,src,dst,c)),
      B.valid(Move.replace(b,U64.bit(U32.to_nat(src)),U64.bit(U32.to_nat(dst)),S.piece(c),U32.is_eq(Chess.get_turn(b),1))),True{},
      BA.metadata(Move.replace(b,U64.bit(U32.to_nat(src)),U64.bit(U32.to_nat(dst)),S.piece(c),U32.is_eq(Chess.get_turn(b),1)),
        U32.xor(Chess.get_turn(b),1),Ordinary.rights(b,src,dst,Chess.piece(b,src)),Ordinary.ep(src,dst,Chess.piece(b,src))),
      Clear.replace(b,U64.bit(U32.to_nat(src)),U64.bit(U32.to_nat(dst)),S.piece(c),U32.is_eq(Chess.get_turn(b),1),good)))
