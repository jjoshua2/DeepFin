import Base
import ./Raw.bend as Raw
import ../promotion/Spec.bend as P
import ../promotion/Actual.bend as Actual
import ../ordinary_king_frame/Frame.bend as F
import ../ordinary_king_frame/Enemy.bend as Old
import ../board/Spec.bend as B
import ../ray/Algebra.bend as R
import ../../../legal_probe/Chess.bend as Chess

def enemy_after(+b: Chess.Board,+src: U32,+dst: U32,+choice: P.Choice,+white: Bool,
  canonical: {Chess.get_turn(b) == Bool.to_u32(white) : U32}) ->
  {Old.enemy(Raw.moved(b,src,dst,choice),Chess.get_turn(b)) ==
    U64.and_not(Old.enemy(b,Chess.get_turn(b)),F.removal(src,dst)) : U64}:
  Equal.trans(U64,Old.enemy(Raw.moved(b,src,dst,choice),Chess.get_turn(b)),
    Old.enemy(F.moved(b,src,dst),Chess.get_turn(b)),
    U64.and_not(Old.enemy(b,Chess.get_turn(b)),F.removal(src,dst)),
    Raw.colors(b,src,dst,choice,U32.xor(Chess.get_turn(b),1)),Old.enemy_after(b,src,dst,white,canonical))

def kind_update(+b: Chess.Board,+src: U32,+dst: U32,+choice: P.Choice,+piece: B.Piece) ->
  {Old.bits(Raw.moved(b,src,dst,choice),piece) ==
    Chess.update_piece(Old.bits(b,piece),F.removal(src,dst),U64.bit(U32.to_nat(dst)),U32.is_eq(P.tag(choice),B.tag(piece))) : U64}:
  match b piece:
    case Chess.Board{+p,+n,+bi,+r,+q,+k,+w,+bl,+turn,+rights,+ep} B.Pawn{}:
      +board=F.as_board(Chess.Board{p,n,bi,r,q,k,w,bl,turn,rights,ep})
      Equal.cong(Chess.Board,U64,next => Old.bits(next,B.Pawn{}),Raw.moved(board,src,dst,choice),
        Actual.dispatch(board,src,dst,P.tag(choice)),Raw.unfold(board,src,dst,choice))
    case Chess.Board{+p,+n,+bi,+r,+q,+k,+w,+bl,+turn,+rights,+ep} B.Knight{}:
      +board=F.as_board(Chess.Board{p,n,bi,r,q,k,w,bl,turn,rights,ep})
      Equal.cong(Chess.Board,U64,next => Old.bits(next,B.Knight{}),Raw.moved(board,src,dst,choice),
        Actual.dispatch(board,src,dst,P.tag(choice)),Raw.unfold(board,src,dst,choice))
    case Chess.Board{+p,+n,+bi,+r,+q,+k,+w,+bl,+turn,+rights,+ep} B.Bishop{}:
      +board=F.as_board(Chess.Board{p,n,bi,r,q,k,w,bl,turn,rights,ep})
      Equal.cong(Chess.Board,U64,next => Old.bits(next,B.Bishop{}),Raw.moved(board,src,dst,choice),
        Actual.dispatch(board,src,dst,P.tag(choice)),Raw.unfold(board,src,dst,choice))
    case Chess.Board{+p,+n,+bi,+r,+q,+k,+w,+bl,+turn,+rights,+ep} B.Rook{}:
      +board=F.as_board(Chess.Board{p,n,bi,r,q,k,w,bl,turn,rights,ep})
      Equal.trans(U64,Old.bits(Raw.moved(board,src,dst,choice),B.Rook{}),
        U64.or(Chess.update_piece(Old.bits(board,B.Rook{}),F.removal(src,dst),U64.bit(U32.to_nat(dst)),U32.is_eq(P.tag(choice),3)),U64.zero()),Chess.update_piece(Old.bits(board,B.Rook{}),F.removal(src,dst),U64.bit(U32.to_nat(dst)),U32.is_eq(P.tag(choice),3)),
        Equal.cong(Chess.Board,U64,next => Old.bits(next,B.Rook{}),Raw.moved(board,src,dst,choice),
        Actual.dispatch(board,src,dst,P.tag(choice)),Raw.unfold(board,src,dst,choice)),R.zero(Chess.update_piece(Old.bits(board,B.Rook{}),F.removal(src,dst),U64.bit(U32.to_nat(dst)),U32.is_eq(P.tag(choice),3))))
    case Chess.Board{+p,+n,+bi,+r,+q,+k,+w,+bl,+turn,+rights,+ep} B.Queen{}:
      +board=F.as_board(Chess.Board{p,n,bi,r,q,k,w,bl,turn,rights,ep})
      Equal.cong(Chess.Board,U64,next => Old.bits(next,B.Queen{}),Raw.moved(board,src,dst,choice),
        Actual.dispatch(board,src,dst,P.tag(choice)),Raw.unfold(board,src,dst,choice))
    case Chess.Board{+p,+n,+bi,+r,+q,+k,+w,+bl,+turn,+rights,+ep} B.King{}:
      +board=F.as_board(Chess.Board{p,n,bi,r,q,k,w,bl,turn,rights,ep})
      Equal.cong(Chess.Board,U64,next => Old.bits(next,B.King{}),Raw.moved(board,src,dst,choice),
        Actual.dispatch(board,src,dst,P.tag(choice)),Raw.unfold(board,src,dst,choice))

# All six enemy kind planes follow the actual promotion dispatch. The inserted
# friendly kind is masked out; no equality of unmasked piece boards is asserted.
def kind_after(+b: Chess.Board,+src: U32,+dst: U32,+choice: P.Choice,+piece: B.Piece,+white: Bool,
  canonical: {Chess.get_turn(b) == Bool.to_u32(white) : U32}) ->
  {Old.enemy_kind(Raw.moved(b,src,dst,choice),Chess.get_turn(b),piece) ==
    U64.and_not(Old.enemy_kind(b,Chess.get_turn(b),piece),F.removal(src,dst)) : U64}:
  +post=Raw.moved(b,src,dst,choice)
  +update=Chess.update_piece(Old.bits(b,piece),F.removal(src,dst),U64.bit(U32.to_nat(dst)),U32.is_eq(P.tag(choice),B.tag(piece)))
  Equal.trans(U64,Old.enemy_kind(post,Chess.get_turn(b),piece),U64.and(update,Old.enemy(post,Chess.get_turn(b))),
    U64.and_not(Old.enemy_kind(b,Chess.get_turn(b),piece),F.removal(src,dst)),
    Equal.cong(U64,U64,bits => U64.and(bits,Old.enemy(post,Chess.get_turn(b))),Old.bits(post,piece),update,
      kind_update(b,src,dst,choice,piece)),
    Equal.trans(U64,U64.and(update,Old.enemy(post,Chess.get_turn(b))),
      U64.and(update,U64.and_not(Old.enemy(b,Chess.get_turn(b)),F.removal(src,dst))),
      U64.and_not(Old.enemy_kind(b,Chess.get_turn(b),piece),F.removal(src,dst)),
      Equal.cong(U64,U64,enemy => U64.and(update,enemy),Old.enemy(post,Chess.get_turn(b)),
        U64.and_not(Old.enemy(b,Chess.get_turn(b)),F.removal(src,dst)),enemy_after(b,src,dst,choice,white,canonical)),
      Old.masked_update(Old.bits(b,piece),Old.enemy(b,Chess.get_turn(b)),U64.bit(U32.to_nat(src)),U64.bit(U32.to_nat(dst)),
        U32.is_eq(P.tag(choice),B.tag(piece)))))

def kind_equal(+b: Chess.Board,+src: U32,+dst: U32,+choice: P.Choice,+piece: B.Piece,+white: Bool,
  +canonical: {Chess.get_turn(b) == Bool.to_u32(white) : U32}) ->
  {Old.enemy_kind(Raw.moved(b,src,dst,choice),Chess.get_turn(b),piece) ==
    Old.enemy_kind(F.moved(b,src,dst),Chess.get_turn(b),piece) : U64}:
  Equal.trans(U64,Old.enemy_kind(Raw.moved(b,src,dst,choice),Chess.get_turn(b),piece),
    U64.and_not(Old.enemy_kind(b,Chess.get_turn(b),piece),F.removal(src,dst)),
    Old.enemy_kind(F.moved(b,src,dst),Chess.get_turn(b),piece),kind_after(b,src,dst,choice,piece,white,canonical),
    Equal.sym(U64,Old.enemy_kind(F.moved(b,src,dst),Chess.get_turn(b),piece),
      U64.and_not(Old.enemy_kind(b,Chess.get_turn(b),piece),F.removal(src,dst)),Old.kind_after(b,src,dst,piece,white,canonical)))
