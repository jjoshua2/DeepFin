import Base
import ./Spec.bend as S
import ../board/Spec.bend as B
import ../board/Actual.bend as BA
import ../move_update/Spec.bend as M
import ../move_update/Clear.bend as Clear
import ../move_update/Core.bend as Core
import ../move_update/Ordinary.bend as O
import ../../../legal_probe/Chess.bend as Chess

def raw(+b: Chess.Board,+src: U32,+dst: U32,p: S.Promotion) -> Chess.Board:
  Core.raw(b,M.removal(S.target(src),S.target(dst)),S.target(dst),S.tag(p),
    U32.is_eq(Chess.get_turn(b),1),U32.xor(Chess.get_turn(b),1),
    O.rights(b,src,dst,Chess.piece(b,src)),O.ep(src,dst,Chess.piece(b,src)))

def unfold(+b: Chess.Board,+src: U32,+dst: U32,+p: S.Promotion) ->
  {Chess.make_move(b,Chess.Ply{src,dst,S.tag(p),0}) == raw(b,src,dst,p) : Chess.Board}:
  match b p:
    case Chess.Board{a,b,c,d,e,f,g,h,t,r,ep} S.Knight{}: {==}
    case Chess.Board{a,b,c,d,e,f,g,h,t,r,ep} S.Bishop{}: {==}
    case Chess.Board{a,b,c,d,e,f,g,h,t,r,ep} S.Rook{}: {==}
    case Chess.Board{a,b,c,d,e,f,g,h,t,r,ep} S.Queen{}: {==}

def actual(+b: Chess.Board,+src: U32,+dst: U32,+p: S.Promotion) ->
  {Chess.make_move(b,Chess.Ply{src,dst,S.tag(p),0}) == S.promotion_expected(b,src,dst,p) : Chess.Board}:
  Equal.trans(Chess.Board,Chess.make_move(b,Chess.Ply{src,dst,S.tag(p),0}),raw(b,src,dst,p),S.promotion_expected(b,src,dst,p),
    unfold(b,src,dst,p),
    Core.bridge(b,M.removal(S.target(src),S.target(dst)),S.target(dst),S.piece(p),
      U32.is_eq(Chess.get_turn(b),1),U32.xor(Chess.get_turn(b),1),
      O.rights(b,src,dst,Chess.piece(b,src)),O.ep(src,dst,Chess.piece(b,src))))

def preserves(+b: Chess.Board,+src: U32,+dst: U32,+p: S.Promotion,good: {B.valid(b) == True{} : Bool}) ->
  {B.valid(Chess.make_move(b,Chess.Ply{src,dst,S.tag(p),0})) == True{} : Bool}:
  Equal.trans(Bool,B.valid(Chess.make_move(b,Chess.Ply{src,dst,S.tag(p),0})),B.valid(S.promotion_expected(b,src,dst,p)),True{},
    Equal.cong(Chess.Board,Bool,x => B.valid(x),Chess.make_move(b,Chess.Ply{src,dst,S.tag(p),0}),S.promotion_expected(b,src,dst,p),actual(b,src,dst,p)),
    Equal.trans(Bool,B.valid(S.promotion_expected(b,src,dst,p)),B.valid(M.replace(b,S.target(src),S.target(dst),S.piece(p),U32.is_eq(Chess.get_turn(b),1))),True{},
      BA.metadata(M.replace(b,S.target(src),S.target(dst),S.piece(p),U32.is_eq(Chess.get_turn(b),1)),
        U32.xor(Chess.get_turn(b),1),O.rights(b,src,dst,Chess.piece(b,src)),O.ep(src,dst,Chess.piece(b,src))),
      Clear.replace(b,S.target(src),S.target(dst),S.piece(p),U32.is_eq(Chess.get_turn(b),1),good)))
