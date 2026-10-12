import Base
import ./Algebra.bend as A
import ./Frame.bend as F
import ../ray/Algebra.bend as R
import ../board/Spec.bend as B
import ../move_update/Core.bend as Core
import ../move_update/Spec.bend as M
import ../move_update/Ordinary.bend as O
import ../occupancy_insertion/Lattice.bend as L
import ../../../legal_probe/Chess.bend as Chess

def bits(+b: Chess.Board,piece: B.Piece) -> U64:
  match piece:
    case B.Pawn{}: Chess.get_pawns(b)
    case B.Knight{}: Chess.get_knights(b)
    case B.Bishop{}: Chess.get_bishops(b)
    case B.Rook{}: Chess.get_rooks(b)
    case B.Queen{}: Chess.get_queens(b)
    case B.King{}: Chess.get_kings(b)
def enemy(+b: Chess.Board,side: U32) -> U64:
  Chess.color(b,U32.xor(side,1))
def enemy_kind(+b: Chess.Board,side: U32,piece: B.Piece) -> U64:
  U64.and(bits(b,piece),enemy(b,side))

def raw_enemy(+b: Chess.Board,+mask: U64,+target: U64,+kind: U32,+side: U32) -> U64:
  enemy(Core.raw(b,mask,target,kind,U32.is_eq(side,1),0,0,0),side)

def enemy_unfold(+b: Chess.Board,+src: U32,+dst: U32) ->
  {enemy(F.moved(b,src,dst),Chess.get_turn(b)) ==
    raw_enemy(b,F.removal(src,dst),U64.bit(U32.to_nat(dst)),Chess.piece(b,src),Chess.get_turn(b)) : U64}:
  match b:
    case Chess.Board{+p,+n,+bi,+r,+q,+k,+w,+bl,+turn,+rights,+ep}:
      +board = F.as_board(Chess.Board{p,n,bi,r,q,k,w,bl,turn,rights,ep})
      Equal.cong(Chess.Board,U64,next => enemy(next,Chess.get_turn(board)),
        F.moved(board,src,dst),O.dispatch(board,src,dst,Chess.piece(board,src)),O.unfold(board,src,dst))

# Unlike original-side ownership, the xor-derived enemy needs side 0 or 1.
def raw_canonical(+b: Chess.Board,+mask: U64,+target: U64,+kind: U32,+white: Bool) ->
  {raw_enemy(b,mask,target,kind,Bool.to_u32(white)) ==
    U64.and_not(enemy(b,Bool.to_u32(white)),mask) : U64}:
  match b white:
    case Chess.Board{p,n,bi,r,q,k,+w,bl,turn,rights,ep} False{}:
      R.zero(U64.and_not(w,mask))
    case Chess.Board{p,n,bi,r,q,k,w,+bl,turn,rights,ep} True{}:
      R.zero(U64.and_not(bl,mask))

def raw_at_side(+b: Chess.Board,+mask: U64,+target: U64,+kind: U32,+side: U32,+white: Bool,
  canonical: {side == Bool.to_u32(white) : U32}) ->
  {raw_enemy(b,mask,target,kind,side) == U64.and_not(enemy(b,side),mask) : U64}:
  %Equal.sym(U32,side,Bool.to_u32(white),canonical) :
    {raw_enemy(b,mask,target,kind,_) == U64.and_not(enemy(b,_),mask) : U64}
  raw_canonical(b,mask,target,kind,white)

def enemy_after(+b: Chess.Board,+src: U32,+dst: U32,+white: Bool,
  canonical: {Chess.get_turn(b) == Bool.to_u32(white) : U32}) ->
  {enemy(F.moved(b,src,dst),Chess.get_turn(b)) ==
    U64.and_not(enemy(b,Chess.get_turn(b)),F.removal(src,dst)) : U64}:
  Equal.trans(U64,enemy(F.moved(b,src,dst),Chess.get_turn(b)),
    raw_enemy(b,F.removal(src,dst),U64.bit(U32.to_nat(dst)),Chess.piece(b,src),Chess.get_turn(b)),
    U64.and_not(enemy(b,Chess.get_turn(b)),F.removal(src,dst)),enemy_unfold(b,src,dst),
    raw_at_side(b,F.removal(src,dst),U64.bit(U32.to_nat(dst)),Chess.piece(b,src),Chess.get_turn(b),white,canonical))

def kind_update(+b: Chess.Board,+src: U32,+dst: U32,+piece: B.Piece) ->
  {bits(F.moved(b,src,dst),piece) ==
    Chess.update_piece(bits(b,piece),F.removal(src,dst),U64.bit(U32.to_nat(dst)),U32.is_eq(Chess.piece(b,src),B.tag(piece))) : U64}:
  match b piece:
    case Chess.Board{+p,+n,+bi,+r,+q,+k,+w,+bl,+turn,+rights,+ep} B.Pawn{}:
      +board = F.as_board(Chess.Board{p,n,bi,r,q,k,w,bl,turn,rights,ep})
      Equal.cong(Chess.Board,U64,next => bits(next,B.Pawn{}),F.moved(board,src,dst),O.dispatch(board,src,dst,Chess.piece(board,src)),O.unfold(board,src,dst))
    case Chess.Board{+p,+n,+bi,+r,+q,+k,+w,+bl,+turn,+rights,+ep} B.Knight{}:
      +board = F.as_board(Chess.Board{p,n,bi,r,q,k,w,bl,turn,rights,ep})
      Equal.cong(Chess.Board,U64,next => bits(next,B.Knight{}),F.moved(board,src,dst),O.dispatch(board,src,dst,Chess.piece(board,src)),O.unfold(board,src,dst))
    case Chess.Board{+p,+n,+bi,+r,+q,+k,+w,+bl,+turn,+rights,+ep} B.Bishop{}:
      +board = F.as_board(Chess.Board{p,n,bi,r,q,k,w,bl,turn,rights,ep})
      Equal.cong(Chess.Board,U64,next => bits(next,B.Bishop{}),F.moved(board,src,dst),O.dispatch(board,src,dst,Chess.piece(board,src)),O.unfold(board,src,dst))
    case Chess.Board{+p,+n,+bi,+r,+q,+k,+w,+bl,+turn,+rights,+ep} B.Rook{}:
      +board = F.as_board(Chess.Board{p,n,bi,r,q,k,w,bl,turn,rights,ep})
      Equal.trans(U64,bits(F.moved(board,src,dst),B.Rook{}),U64.or(Chess.update_piece(bits(board,B.Rook{}),F.removal(src,dst),U64.bit(U32.to_nat(dst)),U32.is_eq(Chess.piece(board,src),3)),U64.zero()),Chess.update_piece(bits(board,B.Rook{}),F.removal(src,dst),U64.bit(U32.to_nat(dst)),U32.is_eq(Chess.piece(board,src),3)),
        Equal.cong(Chess.Board,U64,next => bits(next,B.Rook{}),F.moved(board,src,dst),O.dispatch(board,src,dst,Chess.piece(board,src)),O.unfold(board,src,dst)),R.zero(Chess.update_piece(bits(board,B.Rook{}),F.removal(src,dst),U64.bit(U32.to_nat(dst)),U32.is_eq(Chess.piece(board,src),3))))
    case Chess.Board{+p,+n,+bi,+r,+q,+k,+w,+bl,+turn,+rights,+ep} B.Queen{}:
      +board = F.as_board(Chess.Board{p,n,bi,r,q,k,w,bl,turn,rights,ep})
      Equal.cong(Chess.Board,U64,next => bits(next,B.Queen{}),F.moved(board,src,dst),O.dispatch(board,src,dst,Chess.piece(board,src)),O.unfold(board,src,dst))
    case Chess.Board{+p,+n,+bi,+r,+q,+k,+w,+bl,+turn,+rights,+ep} B.King{}:
      +board = F.as_board(Chess.Board{p,n,bi,r,q,k,w,bl,turn,rights,ep})
      Equal.cong(Chess.Board,U64,next => bits(next,B.King{}),F.moved(board,src,dst),O.dispatch(board,src,dst,Chess.piece(board,src)),O.unfold(board,src,dst))

def masked_update(+piece: U64,+enemy: U64,+source: U64,+target: U64,put: Bool) ->
  {U64.and(Chess.update_piece(piece,M.removal(source,target),target,put),U64.and_not(enemy,M.removal(source,target))) ==
    U64.and_not(U64.and(piece,enemy),M.removal(source,target)) : U64}:
  match put:
    case False{}: A.enemy_none(piece,enemy,source,target)
    case True{}: A.enemy_put(piece,enemy,source,target)

def kind_after(+b: Chess.Board,+src: U32,+dst: U32,+piece: B.Piece,+white: Bool,
  canonical: {Chess.get_turn(b) == Bool.to_u32(white) : U32}) ->
  {enemy_kind(F.moved(b,src,dst),Chess.get_turn(b),piece) ==
    U64.and_not(enemy_kind(b,Chess.get_turn(b),piece),F.removal(src,dst)) : U64}:
  Equal.trans(U64,enemy_kind(F.moved(b,src,dst),Chess.get_turn(b),piece),
    U64.and(Chess.update_piece(bits(b,piece),F.removal(src,dst),U64.bit(U32.to_nat(dst)),U32.is_eq(Chess.piece(b,src),B.tag(piece))),enemy(F.moved(b,src,dst),Chess.get_turn(b))),
    U64.and_not(enemy_kind(b,Chess.get_turn(b),piece),F.removal(src,dst)),
    Equal.cong(U64,U64,updated => U64.and(updated,enemy(F.moved(b,src,dst),Chess.get_turn(b))),
      bits(F.moved(b,src,dst),piece),Chess.update_piece(bits(b,piece),F.removal(src,dst),U64.bit(U32.to_nat(dst)),U32.is_eq(Chess.piece(b,src),B.tag(piece))),kind_update(b,src,dst,piece)),
    Equal.trans(U64,U64.and(Chess.update_piece(bits(b,piece),F.removal(src,dst),U64.bit(U32.to_nat(dst)),U32.is_eq(Chess.piece(b,src),B.tag(piece))),enemy(F.moved(b,src,dst),Chess.get_turn(b))),
      U64.and(Chess.update_piece(bits(b,piece),F.removal(src,dst),U64.bit(U32.to_nat(dst)),U32.is_eq(Chess.piece(b,src),B.tag(piece))),U64.and_not(enemy(b,Chess.get_turn(b)),F.removal(src,dst))),
      U64.and_not(enemy_kind(b,Chess.get_turn(b),piece),F.removal(src,dst)),
      Equal.cong(U64,U64,opponent => U64.and(Chess.update_piece(bits(b,piece),F.removal(src,dst),U64.bit(U32.to_nat(dst)),U32.is_eq(Chess.piece(b,src),B.tag(piece))),opponent),
        enemy(F.moved(b,src,dst),Chess.get_turn(b)),U64.and_not(enemy(b,Chess.get_turn(b)),F.removal(src,dst)),enemy_after(b,src,dst,white,canonical)),
      masked_update(bits(b,piece),enemy(b,Chess.get_turn(b)),U64.bit(U32.to_nat(src)),U64.bit(U32.to_nat(dst)),U32.is_eq(Chess.piece(b,src),B.tag(piece)))))

def kind_subset(+b: Chess.Board,+src: U32,+dst: U32,+piece: B.Piece,+white: Bool,
  canonical: {Chess.get_turn(b) == Bool.to_u32(white) : U32}) ->
  L.subset(enemy_kind(F.moved(b,src,dst),Chess.get_turn(b),piece),enemy_kind(b,Chess.get_turn(b),piece)):
  L.rewrite(U64.and_not(enemy_kind(b,Chess.get_turn(b),piece),F.removal(src,dst)),enemy_kind(b,Chess.get_turn(b),piece),
    enemy_kind(F.moved(b,src,dst),Chess.get_turn(b),piece),enemy_kind(b,Chess.get_turn(b),piece),
    kind_after(b,src,dst,piece,white,canonical),{==},A.erased_subset(enemy_kind(b,Chess.get_turn(b),piece),F.removal(src,dst)))
