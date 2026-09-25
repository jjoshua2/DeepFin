# Exact non-castling/non-promotion application; no legality claim.
import Base
import ./Spec.bend as S
import ./Selection.bend as Selection
import ./Clear.bend as Clear
import ./Core.bend as Core
import ../board/Spec.bend as B
import ../board/Actual.bend as BA
import ../../Position.bend as Position
import ../../../legal_probe/Chess.bend as Chess

def rights(+board: Chess.Board,+src: U32,+dst: U32,+kind: U32) -> U32:
  U32.and(Chess.get_rights(board),U32.not(U32.or(U32.or(Chess.corner_right(src),Chess.corner_right(dst)),
    Bool.pick(U32,U32.is_eq(kind,5),Bool.pick(U32,U32.is_eq(Chess.get_turn(board),1),3,12),0))))

def ep(+src: U32,+dst: U32,kind: U32) -> U32:
  Bool.pick(U32,Bool.and(U32.is_zero(kind),U32.is_eq(U32.xor(src,dst),16)),U32.div(U32.add(src,dst),2),64)

def dispatch(+board: Chess.Board,+src: U32,+dst: U32,+kind: U32) -> Chess.Board:
  Core.raw(board,S.removal(U64.bit(U32.to_nat(src)),U64.bit(U32.to_nat(dst))),U64.bit(U32.to_nat(dst)),kind,
    U32.is_eq(Chess.get_turn(board),1),U32.xor(Chess.get_turn(board),1),rights(board,src,dst,kind),ep(src,dst,kind))

def expected(+board: Chess.Board,+src: U32,+dst: U32) -> Chess.Board:
  Core.assembled(board,S.removal(U64.bit(U32.to_nat(src)),U64.bit(U32.to_nat(dst))),U64.bit(U32.to_nat(dst)),Selection.piece(board,src),
    U32.is_eq(Chess.get_turn(board),1),U32.xor(Chess.get_turn(board),1),rights(board,src,dst,B.tag(Selection.piece(board,src))),ep(src,dst,B.tag(Selection.piece(board,src))))

def unfold(+board: Chess.Board,+src: U32,+dst: U32) ->
  {Chess.make_move(board,Chess.Ply{src,dst,0,0}) == dispatch(board,src,dst,Chess.piece(board,src)) : Chess.Board}:
  match board:
    case Chess.Board{p,n,b,r,q,k,w,bl,old_turn,old_rights,old_ep}: {==}

def actual(+board: Chess.Board,+src: U32,+dst: U32) ->
  {Chess.make_move(board,Chess.Ply{src,dst,0,0}) == expected(board,src,dst) : Chess.Board}:
  Equal.trans(Chess.Board,Chess.make_move(board,Chess.Ply{src,dst,0,0}),dispatch(board,src,dst,Chess.piece(board,src)),expected(board,src,dst),
    unfold(board,src,dst),
    Equal.trans(Chess.Board,dispatch(board,src,dst,Chess.piece(board,src)),dispatch(board,src,dst,B.tag(Selection.piece(board,src))),expected(board,src,dst),
      Equal.cong(U32,Chess.Board,kind => dispatch(board,src,dst,kind),Chess.piece(board,src),B.tag(Selection.piece(board,src)),
        Equal.sym(U32,B.tag(Selection.piece(board,src)),Chess.piece(board,src),Selection.actual(board,src))),
      Core.bridge(board,S.removal(U64.bit(U32.to_nat(src)),U64.bit(U32.to_nat(dst))),U64.bit(U32.to_nat(dst)),Selection.piece(board,src),
        U32.is_eq(Chess.get_turn(board),1),U32.xor(Chess.get_turn(board),1),rights(board,src,dst,B.tag(Selection.piece(board,src))),ep(src,dst,B.tag(Selection.piece(board,src))))))

def preserves(+board: Chess.Board,+src: U32,+dst: U32,good: {B.valid(board) == True{} : Bool}) ->
  {B.valid(Chess.make_move(board,Chess.Ply{src,dst,0,0})) == True{} : Bool}:
  Equal.trans(Bool,B.valid(Chess.make_move(board,Chess.Ply{src,dst,0,0})),B.valid(expected(board,src,dst)),True{},
    Equal.cong(Chess.Board,Bool,x => B.valid(x),Chess.make_move(board,Chess.Ply{src,dst,0,0}),expected(board,src,dst),actual(board,src,dst)),
    Equal.trans(Bool,B.valid(expected(board,src,dst)),B.valid(S.replace(board,U64.bit(U32.to_nat(src)),U64.bit(U32.to_nat(dst)),Selection.piece(board,src),U32.is_eq(Chess.get_turn(board),1))),True{},
      BA.metadata(S.replace(board,U64.bit(U32.to_nat(src)),U64.bit(U32.to_nat(dst)),Selection.piece(board,src),U32.is_eq(Chess.get_turn(board),1)),
        U32.xor(Chess.get_turn(board),1),rights(board,src,dst,B.tag(Selection.piece(board,src))),ep(src,dst,B.tag(Selection.piece(board,src)))),
      Clear.replace(board,U64.bit(U32.to_nat(src)),U64.bit(U32.to_nat(dst)),Selection.piece(board,src),U32.is_eq(Chess.get_turn(board),1),good)))
