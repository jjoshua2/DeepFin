# Supplementary actual flag-one update; no legal en-passant preconditions proved.
import Base
import ./Mask.bend as Mask
import ../move_update/Ordinary.bend as Ordinary
import ../move_update/Spec.bend as S
import ../move_update/Selection.bend as Selection
import ../move_update/Clear.bend as Clear
import ../move_update/Core.bend as Core
import ../board/Spec.bend as B
import ../board/Actual.bend as BA
import ../../Position.bend as Position
import ../../../legal_probe/Chess.bend as Chess

def source(+src: U32,+dst: U32) -> U64:
  U64.or(U64.bit(U32.to_nat(src)),U64.bit(U32.to_nat(U32.xor(dst,8))))

def mask(+src: U32,+dst: U32) -> U64:
  Mask.raw(source(src,dst),U64.bit(U32.to_nat(dst)))

def dispatch(+board: Chess.Board,+src: U32,+dst: U32,+kind: U32) -> Chess.Board:
  Core.raw(board,mask(src,dst),U64.bit(U32.to_nat(dst)),kind,
    U32.is_eq(Chess.get_turn(board),1),U32.xor(Chess.get_turn(board),1),Ordinary.rights(board,src,dst,kind),Ordinary.ep(src,dst,kind))

def expected(+board: Chess.Board,+src: U32,+dst: U32) -> Chess.Board:
  Core.assembled(board,mask(src,dst),U64.bit(U32.to_nat(dst)),Selection.piece(board,src),
    U32.is_eq(Chess.get_turn(board),1),U32.xor(Chess.get_turn(board),1),Ordinary.rights(board,src,dst,B.tag(Selection.piece(board,src))),Ordinary.ep(src,dst,B.tag(Selection.piece(board,src))))

def unfold(+board: Chess.Board,+src: U32,+dst: U32) ->
  {Chess.make_move(board,Chess.Ply{src,dst,0,1}) == dispatch(board,src,dst,Chess.piece(board,src)) : Chess.Board}:
  match board:
    case Chess.Board{p,n,b,r,q,k,w,bl,old_turn,old_rights,old_ep}: {==}

def actual(+board: Chess.Board,+src: U32,+dst: U32) ->
  {Chess.make_move(board,Chess.Ply{src,dst,0,1}) == expected(board,src,dst) : Chess.Board}:
  Equal.trans(Chess.Board,Chess.make_move(board,Chess.Ply{src,dst,0,1}),dispatch(board,src,dst,Chess.piece(board,src)),expected(board,src,dst),
    unfold(board,src,dst),
    Equal.trans(Chess.Board,dispatch(board,src,dst,Chess.piece(board,src)),dispatch(board,src,dst,B.tag(Selection.piece(board,src))),expected(board,src,dst),
      Equal.cong(U32,Chess.Board,kind => dispatch(board,src,dst,kind),Chess.piece(board,src),B.tag(Selection.piece(board,src)),
        Equal.sym(U32,B.tag(Selection.piece(board,src)),Chess.piece(board,src),Selection.actual(board,src))),
      Core.bridge(board,mask(src,dst),U64.bit(U32.to_nat(dst)),Selection.piece(board,src),
        U32.is_eq(Chess.get_turn(board),1),U32.xor(Chess.get_turn(board),1),Ordinary.rights(board,src,dst,B.tag(Selection.piece(board,src))),Ordinary.ep(src,dst,B.tag(Selection.piece(board,src))))))

def normalized(+b: Chess.Board,+src: U32,+dst: U32) -> Chess.Board:
  Core.assembled(b,S.removal(source(src,dst),U64.bit(U32.to_nat(dst))),U64.bit(U32.to_nat(dst)),Selection.piece(b,src),
    U32.is_eq(Chess.get_turn(b),1),U32.xor(Chess.get_turn(b),1),Ordinary.rights(b,src,dst,B.tag(Selection.piece(b,src))),Ordinary.ep(src,dst,B.tag(Selection.piece(b,src))))

def normalize(+b: Chess.Board,+src: U32,+dst: U32) -> {expected(b,src,dst) == normalized(b,src,dst) : Chess.Board}:
  Equal.cong(U64,Chess.Board,m => Core.assembled(b,m,U64.bit(U32.to_nat(dst)),Selection.piece(b,src),
    U32.is_eq(Chess.get_turn(b),1),U32.xor(Chess.get_turn(b),1),Ordinary.rights(b,src,dst,B.tag(Selection.piece(b,src))),Ordinary.ep(src,dst,B.tag(Selection.piece(b,src)))),
    mask(src,dst),S.removal(source(src,dst),U64.bit(U32.to_nat(dst))),Mask.same(source(src,dst),U64.bit(U32.to_nat(dst))))

def preserves(+b: Chess.Board,+src: U32,+dst: U32,good: {B.valid(b) == True{} : Bool}) ->
  {B.valid(Chess.make_move(b,Chess.Ply{src,dst,0,1})) == True{} : Bool}:
  Equal.trans(Bool,B.valid(Chess.make_move(b,Chess.Ply{src,dst,0,1})),B.valid(normalized(b,src,dst)),True{},
    Equal.cong(Chess.Board,Bool,x => B.valid(x),Chess.make_move(b,Chess.Ply{src,dst,0,1}),normalized(b,src,dst),
      Equal.trans(Chess.Board,Chess.make_move(b,Chess.Ply{src,dst,0,1}),expected(b,src,dst),normalized(b,src,dst),actual(b,src,dst),normalize(b,src,dst))),
    Equal.trans(Bool,B.valid(normalized(b,src,dst)),B.valid(S.replace(b,source(src,dst),U64.bit(U32.to_nat(dst)),Selection.piece(b,src),U32.is_eq(Chess.get_turn(b),1))),True{},
      BA.metadata(S.replace(b,source(src,dst),U64.bit(U32.to_nat(dst)),Selection.piece(b,src),U32.is_eq(Chess.get_turn(b),1)),
        U32.xor(Chess.get_turn(b),1),Ordinary.rights(b,src,dst,B.tag(Selection.piece(b,src))),Ordinary.ep(src,dst,B.tag(Selection.piece(b,src)))),
      Clear.replace(b,source(src,dst),U64.bit(U32.to_nat(dst)),Selection.piece(b,src),U32.is_eq(Chess.get_turn(b),1),good)))
