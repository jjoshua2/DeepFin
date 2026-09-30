import Base
import ./Spec.bend as S
import ./Fresh.bend as Fresh
import ../board/Spec.bend as B
import ../board/Actual.bend as BA
import ../move_update/Spec.bend as M
import ../move_update/Core.bend as Core
import ../move_update/Clear.bend as Clear
import ../move_update/Ordinary.bend as O
import ../move_update/Selection.bend as Selection
import ../../../legal_probe/Chess.bend as Chess

def raw(+b: Chess.Board,+src: U32,+dst: U32,kind: U32) -> Chess.Board:
  Core.raw(b,S.removal(src,dst),S.target(dst),kind,
    U32.is_eq(Chess.get_turn(b),1),U32.xor(Chess.get_turn(b),1),
    O.rights(b,src,dst,Chess.piece(b,src)),O.ep(src,dst,Chess.piece(b,src)))

def unfold(+b: Chess.Board,+src: U32,+dst: U32) ->
  {Chess.make_move(b,Chess.Ply{src,dst,0,1}) == raw(b,src,dst,Chess.piece(b,src)) : Chess.Board}:
  match b:
    case Chess.Board{p,n,bi,r,q,k,w,bl,turn,rights,ep}: {==}

def actual(+b: Chess.Board,+src: U32,+dst: U32) ->
  {Chess.make_move(b,Chess.Ply{src,dst,0,1}) == S.ep_expected(b,src,dst) : Chess.Board}:
  Equal.trans(Chess.Board,Chess.make_move(b,Chess.Ply{src,dst,0,1}),raw(b,src,dst,Chess.piece(b,src)),S.ep_expected(b,src,dst),
    unfold(b,src,dst),
    Equal.trans(Chess.Board,raw(b,src,dst,Chess.piece(b,src)),raw(b,src,dst,B.tag(Selection.piece(b,src))),S.ep_expected(b,src,dst),
      Equal.cong(U32,Chess.Board,x => raw(b,src,dst,x),Chess.piece(b,src),B.tag(Selection.piece(b,src)),
        Equal.sym(U32,B.tag(Selection.piece(b,src)),Chess.piece(b,src),Selection.actual(b,src))),
      Core.bridge(b,S.removal(src,dst),S.target(dst),Selection.piece(b,src),
        U32.is_eq(Chess.get_turn(b),1),U32.xor(Chess.get_turn(b),1),
        O.rights(b,src,dst,Chess.piece(b,src)),O.ep(src,dst,Chess.piece(b,src)))))

def preserves(+b: Chess.Board,+src: U32,+dst: U32,good: {B.valid(b) == True{} : Bool}) ->
  {B.valid(Chess.make_move(b,Chess.Ply{src,dst,0,1})) == True{} : Bool}:
  Equal.trans(Bool,B.valid(Chess.make_move(b,Chess.Ply{src,dst,0,1})),B.valid(S.ep_expected(b,src,dst)),True{},
    Equal.cong(Chess.Board,Bool,x => B.valid(x),Chess.make_move(b,Chess.Ply{src,dst,0,1}),S.ep_expected(b,src,dst),actual(b,src,dst)),
    Equal.trans(Bool,B.valid(S.ep_expected(b,src,dst)),B.valid(B.insert_mask(M.erase(b,S.removal(src,dst)),S.target(dst),Selection.piece(b,src),U32.is_eq(Chess.get_turn(b),1))),True{},
      BA.metadata(B.insert_mask(M.erase(b,S.removal(src,dst)),S.target(dst),Selection.piece(b,src),U32.is_eq(Chess.get_turn(b),1)),
        U32.xor(Chess.get_turn(b),1),O.rights(b,src,dst,Chess.piece(b,src)),O.ep(src,dst,Chess.piece(b,src))),
      BA.mask_valid(M.erase(b,S.removal(src,dst)),S.target(dst),Selection.piece(b,src),U32.is_eq(Chess.get_turn(b),1),
        Clear.clear(b,S.removal(src,dst),good),Fresh.target(b,U64.or(S.target(src),S.target(U32.xor(dst,8))),S.target(dst)))))
