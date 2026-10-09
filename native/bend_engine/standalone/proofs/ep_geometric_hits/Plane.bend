# Actual EP colored nonpawn erasure: full src/victim/dst mask, any fixed side.
import Base
import ../en_passant/Actual.bend as EP
import ../en_passant/Mask.bend as Mask
import ../move_update/Core.bend as Core
import ../move_update/Ordinary.bend as O
import ../move_update/Spec.bend as M
import ../ordinary_king_frame/Frame.bend as Old
import ../ordinary_king_frame/Enemy.bend as Enemy
import ../ordinary_king_frame/Algebra.bend as A
import ../candidate_king_destination/Algebra.bend as K
import ../ray/Algebra.bend as R
import ../board/Spec.bend as B
import ../occupancy_insertion/Lattice.bend as L
import ../../../legal_probe/Chess.bend as Chess

def moved(+b: Chess.Board,s: U32,d: U32) -> Chess.Board: Chess.make_move(b,Chess.Ply{s,d,0,1})
def mask(+s: U32,+d: U32) -> U64: EP.mask(s,d)
def bits(+b: Chess.Board,p: B.Piece) -> U64: Enemy.bits(b,p)
def plane(+b: Chess.Board,+p: B.Piece,by: U32) -> U64: U64.and(bits(b,p),Chess.color(b,by))
def put(white: Bool,query: Bool) -> Bool:
  match query:
    case False{}: Bool.not(white)
    case True{}: white

def update(+b: Chess.Board,+src: U32,+dst: U32,+piece: B.Piece) ->
  {bits(moved(b,src,dst),piece) ==
    Chess.update_piece(bits(b,piece),mask(src,dst),U64.bit(U32.to_nat(dst)),U32.is_eq(Chess.piece(b,src),B.tag(piece))) : U64}:
  match b piece:
    case Chess.Board{+p,+n,+bi,+r,+q,+k,+w,+bl,+turn,+rights,+ep} B.Pawn{}:
      +board = Old.as_board(Chess.Board{p,n,bi,r,q,k,w,bl,turn,rights,ep})
      Equal.cong(Chess.Board,U64,next => bits(next,B.Pawn{}),moved(board,src,dst),EP.dispatch(board,src,dst,Chess.piece(board,src)),EP.unfold(board,src,dst))
    case Chess.Board{+p,+n,+bi,+r,+q,+k,+w,+bl,+turn,+rights,+ep} B.Knight{}:
      +board = Old.as_board(Chess.Board{p,n,bi,r,q,k,w,bl,turn,rights,ep})
      Equal.cong(Chess.Board,U64,next => bits(next,B.Knight{}),moved(board,src,dst),EP.dispatch(board,src,dst,Chess.piece(board,src)),EP.unfold(board,src,dst))
    case Chess.Board{+p,+n,+bi,+r,+q,+k,+w,+bl,+turn,+rights,+ep} B.Bishop{}:
      +board = Old.as_board(Chess.Board{p,n,bi,r,q,k,w,bl,turn,rights,ep})
      Equal.cong(Chess.Board,U64,next => bits(next,B.Bishop{}),moved(board,src,dst),EP.dispatch(board,src,dst,Chess.piece(board,src)),EP.unfold(board,src,dst))
    case Chess.Board{+p,+n,+bi,+r,+q,+k,+w,+bl,+turn,+rights,+ep} B.Rook{}:
      +board = Old.as_board(Chess.Board{p,n,bi,r,q,k,w,bl,turn,rights,ep})
      Equal.trans(U64,bits(moved(board,src,dst),B.Rook{}),U64.or(Chess.update_piece(bits(board,B.Rook{}),mask(src,dst),U64.bit(U32.to_nat(dst)),U32.is_eq(Chess.piece(board,src),3)),U64.zero()),Chess.update_piece(bits(board,B.Rook{}),mask(src,dst),U64.bit(U32.to_nat(dst)),U32.is_eq(Chess.piece(board,src),3)),
        Equal.cong(Chess.Board,U64,next => bits(next,B.Rook{}),moved(board,src,dst),EP.dispatch(board,src,dst,Chess.piece(board,src)),EP.unfold(board,src,dst)),R.zero(Chess.update_piece(bits(board,B.Rook{}),mask(src,dst),U64.bit(U32.to_nat(dst)),U32.is_eq(Chess.piece(board,src),3))))
    case Chess.Board{+p,+n,+bi,+r,+q,+k,+w,+bl,+turn,+rights,+ep} B.Queen{}:
      +board = Old.as_board(Chess.Board{p,n,bi,r,q,k,w,bl,turn,rights,ep})
      Equal.cong(Chess.Board,U64,next => bits(next,B.Queen{}),moved(board,src,dst),EP.dispatch(board,src,dst,Chess.piece(board,src)),EP.unfold(board,src,dst))
    case Chess.Board{+p,+n,+bi,+r,+q,+k,+w,+bl,+turn,+rights,+ep} B.King{}:
      +board = Old.as_board(Chess.Board{p,n,bi,r,q,k,w,bl,turn,rights,ep})
      Equal.cong(Chess.Board,U64,next => bits(next,B.King{}),moved(board,src,dst),EP.dispatch(board,src,dst,Chess.piece(board,src)),EP.unfold(board,src,dst))


# Metadata and piece kind do not affect these two actual color rows.
def raw_selected(+b: Chess.Board,+remove: U64,+target: U64,+kind: U32,+white: Bool,+query: Bool,+next_turn: U32,+next_rights: U32,+next_ep: U32) ->
  {Chess.select_u64(query,Chess.get_white(Core.raw(b,remove,target,kind,white,next_turn,next_rights,next_ep)),Chess.get_black(Core.raw(b,remove,target,kind,white,next_turn,next_rights,next_ep))) ==
    Chess.update_piece(Chess.select_u64(query,Chess.get_white(b),Chess.get_black(b)),remove,target,put(white,query)) : U64}:
  match b query:
    case Chess.Board{p,n,bi,r,q,k,w,+bl,t,rr,e} False{}:
      Equal.cong(U64,U64,x => Chess.update_piece(bl,remove,x,Bool.not(white)),U64.or(target,U64.zero()),target,R.zero(target))
    case Chess.Board{p,n,bi,r,q,k,+w,bl,t,rr,e} True{}:
      Equal.cong(U64,U64,x => Chess.update_piece(w,remove,x,white),U64.or(target,U64.zero()),target,R.zero(target))

def color_update(+b: Chess.Board,+s: U32,+d: U32,+by: U32) ->
  {Chess.color(moved(b,s,d),by) == Chess.update_piece(Chess.color(b,by),mask(s,d),U64.bit(U32.to_nat(d)),put(U32.is_eq(Chess.get_turn(b),1),U32.is_eq(by,1))) : U64}:
  Equal.trans(U64,Chess.color(moved(b,s,d),by),Chess.color(EP.dispatch(b,s,d,Chess.piece(b,s)),by),
    Chess.update_piece(Chess.color(b,by),mask(s,d),U64.bit(U32.to_nat(d)),put(U32.is_eq(Chess.get_turn(b),1),U32.is_eq(by,1))),
    Equal.cong(Chess.Board,U64,next => Chess.color(next,by),moved(b,s,d),EP.dispatch(b,s,d,Chess.piece(b,s)),EP.unfold(b,s,d)),
    raw_selected(b,mask(s,d),U64.bit(U32.to_nat(d)),Chess.piece(b,s),U32.is_eq(Chess.get_turn(b),1),U32.is_eq(by,1),U32.xor(Chess.get_turn(b),1),O.rights(b,s,d,Chess.piece(b,s)),O.ep(s,d,Chess.piece(b,s))))

def erasure(+b: Chess.Board,+s: U32,+d: U32,+p: B.Piece,
  kind: {Chess.piece(b,s) == 0 : U32},nonpawn: {U32.is_eq(0,B.tag(p)) == False{} : Bool}) ->
  {bits(moved(b,s,d),p) == U64.and_not(bits(b,p),mask(s,d)) : U64}:
  Equal.trans(U64,bits(moved(b,s,d),p),Chess.update_piece(bits(b,p),mask(s,d),U64.bit(U32.to_nat(d)),U32.is_eq(Chess.piece(b,s),B.tag(p))),
    U64.and_not(bits(b,p),mask(s,d)),update(b,s,d,p),
    Old.no_put(bits(b,p),mask(s,d),U64.bit(U32.to_nat(d)),U32.is_eq(Chess.piece(b,s),B.tag(p)),
      Equal.trans(Bool,U32.is_eq(Chess.piece(b,s),B.tag(p)),U32.is_eq(0,B.tag(p)),False{},
        Equal.cong(U32,Bool,k => U32.is_eq(k,B.tag(p)),Chess.piece(b,s),0,kind),nonpawn)))

# The color's optional target put is masked away by the nonpawn destination clear.
def masked_color(+p: U64,+color: U64,+source: U64,+target: U64,+add: Bool) ->
  {U64.and(U64.and_not(p,M.removal(source,target)),Chess.update_piece(color,M.removal(source,target),target,add)) ==
    U64.and_not(U64.and(p,color),M.removal(source,target)) : U64}:
  Equal.trans(U64,U64.and(U64.and_not(p,M.removal(source,target)),Chess.update_piece(color,M.removal(source,target),target,add)),
    U64.and(Chess.update_piece(color,M.removal(source,target),target,add),U64.and_not(p,M.removal(source,target))),
    U64.and_not(U64.and(p,color),M.removal(source,target)),
    K.and_commute(U64.and_not(p,M.removal(source,target)),Chess.update_piece(color,M.removal(source,target),target,add)),
    Equal.trans(U64,U64.and(Chess.update_piece(color,M.removal(source,target),target,add),U64.and_not(p,M.removal(source,target))),
      U64.and_not(U64.and(color,p),M.removal(source,target)),U64.and_not(U64.and(p,color),M.removal(source,target)),
      Enemy.masked_update(color,p,source,target,add),
      Equal.cong(U64,U64,x => U64.and_not(x,M.removal(source,target)),U64.and(color,p),U64.and(p,color),K.and_commute(color,p))))

def colored_erasure(+b: Chess.Board,+s: U32,+d: U32,+p: B.Piece,+by: U32,
  kind: {Chess.piece(b,s) == 0 : U32},nonpawn: {U32.is_eq(0,B.tag(p)) == False{} : Bool}) ->
  {plane(moved(b,s,d),p,by) == U64.and_not(plane(b,p,by),mask(s,d)) : U64}:
  +old=bits(b,p)
  +color=Chess.color(b,by)
  +remove=mask(s,d)
  +target=U64.bit(U32.to_nat(d))
  +add=put(U32.is_eq(Chess.get_turn(b),1),U32.is_eq(by,1))
  Equal.trans(U64,plane(moved(b,s,d),p,by),U64.and(U64.and_not(old,remove),Chess.color(moved(b,s,d),by)),
    U64.and_not(plane(b,p,by),remove),
    Equal.cong(U64,U64,x => U64.and(x,Chess.color(moved(b,s,d),by)),bits(moved(b,s,d),p),U64.and_not(old,remove),erasure(b,s,d,p,kind,nonpawn)),
    Equal.trans(U64,U64.and(U64.and_not(old,remove),Chess.color(moved(b,s,d),by)),
      U64.and(U64.and_not(old,remove),Chess.update_piece(color,remove,target,add)),U64.and_not(plane(b,p,by),remove),
      Equal.cong(U64,U64,x => U64.and(U64.and_not(old,remove),x),Chess.color(moved(b,s,d),by),Chess.update_piece(color,remove,target,add),color_update(b,s,d,by)),
      Equal.trans(U64,U64.and(U64.and_not(old,remove),Chess.update_piece(color,remove,target,add)),
        U64.and(U64.and_not(old,M.removal(EP.source(s,d),target)),Chess.update_piece(color,M.removal(EP.source(s,d),target),target,add)),
        U64.and_not(plane(b,p,by),remove),
        Equal.cong(U64,U64,m => U64.and(U64.and_not(old,m),Chess.update_piece(color,m,target,add)),remove,M.removal(EP.source(s,d),target),Mask.same(EP.source(s,d),target)),
        Equal.trans(U64,U64.and(U64.and_not(old,M.removal(EP.source(s,d),target)),Chess.update_piece(color,M.removal(EP.source(s,d),target),target,add)),
          U64.and_not(plane(b,p,by),M.removal(EP.source(s,d),target)),U64.and_not(plane(b,p,by),remove),
          masked_color(old,color,EP.source(s,d),target,add),
          Equal.cong(U64,U64,m => U64.and_not(plane(b,p,by),m),M.removal(EP.source(s,d),target),remove,
            Equal.sym(U64,remove,M.removal(EP.source(s,d),target),Mask.same(EP.source(s,d),target)))))))

def subset(+b: Chess.Board,+s: U32,+d: U32,+p: B.Piece,+by: U32,
  kind: {Chess.piece(b,s) == 0 : U32},nonpawn: {U32.is_eq(0,B.tag(p)) == False{} : Bool}) ->
  L.subset(plane(moved(b,s,d),p,by),plane(b,p,by)):
  L.rewrite(U64.and_not(plane(b,p,by),mask(s,d)),plane(b,p,by),plane(moved(b,s,d),p,by),plane(b,p,by),
    colored_erasure(b,s,d,p,by,kind,nonpawn),{==},A.erased_subset(plane(b,p,by),mask(s,d)))
