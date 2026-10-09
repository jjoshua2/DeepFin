import Base
import ./Algebra.bend as A
import ../ray/Algebra.bend as R
import ../ray_clear_disjoint/Algebra.bend as D
import ../move_update/Spec.bend as M
import ../move_update/Core.bend as Core
import ../move_update/Ordinary.bend as O
import ../../../legal_probe/Chess.bend as Chess

def moved(+b: Chess.Board,src: U32,dst: U32) -> Chess.Board:
  Chess.make_move(b,Chess.Ply{src,dst,0,0})
def removal(+src: U32,+dst: U32) -> U64:
  M.removal(U64.bit(U32.to_nat(src)),U64.bit(U32.to_nat(dst)))
def plane(+b: Chess.Board,side: U32) -> U64:
  U64.and(Chess.get_kings(b),Chess.color(b,side))
def as_board(b: Chess.Board) -> Chess.Board:
  b

def kings_update(+b: Chess.Board,+src: U32,+dst: U32) ->
  {Chess.get_kings(moved(b,src,dst)) ==
    Chess.update_piece(Chess.get_kings(b),removal(src,dst),U64.bit(U32.to_nat(dst)),U32.is_eq(Chess.piece(b,src),5)) : U64}:
  match b:
    case Chess.Board{+p,+n,+bi,+r,+q,+k,+w,+bl,+turn,+rights,+ep}:
      +board = as_board(Chess.Board{p,n,bi,r,q,k,w,bl,turn,rights,ep})
      Equal.cong(Chess.Board,U64,Chess.get_kings,
        moved(board,src,dst),O.dispatch(board,src,dst,Chess.piece(board,src)),O.unfold(board,src,dst))

def no_put(+bb: U64,+mask: U64,+target: U64,put: Bool,skip: {put == False{} : Bool}) ->
  {Chess.update_piece(bb,mask,target,put) == U64.and_not(bb,mask) : U64}:
  %Equal.sym(Bool,put,False{},skip) :
    {Chess.update_piece(bb,mask,target,_) == U64.and_not(bb,mask) : U64}
  R.zero(U64.and_not(bb,mask))

def kings_keep(+b: Chess.Board,+src: U32,+dst: U32,
  +source_clear: {U64.and(U64.bit(U32.to_nat(src)),Chess.get_kings(b)) == U64.zero() : U64},
  +target_clear: {U64.and(U64.bit(U32.to_nat(dst)),Chess.get_kings(b)) == U64.zero() : U64},
  not_king: {U32.is_eq(Chess.piece(b,src),5) == False{} : Bool}) ->
  {Chess.get_kings(moved(b,src,dst)) == Chess.get_kings(b) : U64}:
  Equal.trans(U64,Chess.get_kings(moved(b,src,dst)),
    Chess.update_piece(Chess.get_kings(b),removal(src,dst),U64.bit(U32.to_nat(dst)),U32.is_eq(Chess.piece(b,src),5)),
    Chess.get_kings(b),kings_update(b,src,dst),
    Equal.trans(U64,Chess.update_piece(Chess.get_kings(b),removal(src,dst),U64.bit(U32.to_nat(dst)),U32.is_eq(Chess.piece(b,src),5)),
      U64.and_not(Chess.get_kings(b),removal(src,dst)),Chess.get_kings(b),
      no_put(Chess.get_kings(b),removal(src,dst),U64.bit(U32.to_nat(dst)),U32.is_eq(Chess.piece(b,src),5),not_king),
      A.keep(Chess.get_kings(b),removal(src,dst),
        A.removal_clear(U64.bit(U32.to_nat(src)),U64.bit(U32.to_nat(dst)),Chess.get_kings(b),source_clear,target_clear))))

# Query the original side after make_move flips the metadata turn.
def own_raw(+b: Chess.Board,+mask: U64,+target: U64,+kind: U32,+white: Bool,+turn: U32,+rights: U32,+ep: U32) ->
  {Chess.select_u64(white,Chess.get_white(Core.raw(b,mask,target,kind,white,turn,rights,ep)),
    Chess.get_black(Core.raw(b,mask,target,kind,white,turn,rights,ep))) ==
    U64.or(U64.and_not(Chess.select_u64(white,Chess.get_white(b),Chess.get_black(b)),mask),U64.or(target,U64.zero())) : U64}:
  match b white:
    case Chess.Board{p,n,bi,r,q,k,w,bl,oldturn,oldrights,oldep} False{}: {==}
    case Chess.Board{p,n,bi,r,q,k,w,bl,oldturn,oldrights,oldep} True{}: {==}

def own_update(+b: Chess.Board,+src: U32,+dst: U32) ->
  {Chess.color(moved(b,src,dst),Chess.get_turn(b)) ==
    U64.or(U64.and_not(Chess.color(b,Chess.get_turn(b)),removal(src,dst)),U64.bit(U32.to_nat(dst))) : U64}:
  Equal.trans(U64,Chess.color(moved(b,src,dst),Chess.get_turn(b)),
    Chess.color(O.dispatch(b,src,dst,Chess.piece(b,src)),Chess.get_turn(b)),
    U64.or(U64.and_not(Chess.color(b,Chess.get_turn(b)),removal(src,dst)),U64.bit(U32.to_nat(dst))),
    Equal.cong(Chess.Board,U64,board => Chess.color(board,Chess.get_turn(b)),
      moved(b,src,dst),O.dispatch(b,src,dst,Chess.piece(b,src)),O.unfold(b,src,dst)),
    Equal.trans(U64,Chess.color(O.dispatch(b,src,dst,Chess.piece(b,src)),Chess.get_turn(b)),
      U64.or(U64.and_not(Chess.color(b,Chess.get_turn(b)),removal(src,dst)),U64.or(U64.bit(U32.to_nat(dst)),U64.zero())),
      U64.or(U64.and_not(Chess.color(b,Chess.get_turn(b)),removal(src,dst)),U64.bit(U32.to_nat(dst))),
      own_raw(b,removal(src,dst),U64.bit(U32.to_nat(dst)),Chess.piece(b,src),U32.is_eq(Chess.get_turn(b),1),
        U32.xor(Chess.get_turn(b),1),O.rights(b,src,dst,Chess.piece(b,src)),O.ep(src,dst,Chess.piece(b,src))),
      Equal.cong(U64,U64,target => U64.or(U64.and_not(Chess.color(b,Chess.get_turn(b)),removal(src,dst)),target),
        U64.or(U64.bit(U32.to_nat(dst)),U64.zero()),U64.bit(U32.to_nat(dst)),R.zero(U64.bit(U32.to_nat(dst))))))

def plane_keep(+b: Chess.Board,+src: U32,+dst: U32,
  +source_clear: {U64.and(U64.bit(U32.to_nat(src)),Chess.get_kings(b)) == U64.zero() : U64},
  +target_clear: {U64.and(U64.bit(U32.to_nat(dst)),Chess.get_kings(b)) == U64.zero() : U64},
  not_king: {U32.is_eq(Chess.piece(b,src),5) == False{} : Bool}) ->
  {plane(moved(b,src,dst),Chess.get_turn(b)) == plane(b,Chess.get_turn(b)) : U64}:
  Equal.trans(U64,plane(moved(b,src,dst),Chess.get_turn(b)),
    U64.and(Chess.get_kings(b),Chess.color(moved(b,src,dst),Chess.get_turn(b))),plane(b,Chess.get_turn(b)),
    Equal.cong(U64,U64,kings => U64.and(kings,Chess.color(moved(b,src,dst),Chess.get_turn(b))),
      Chess.get_kings(moved(b,src,dst)),Chess.get_kings(b),kings_keep(b,src,dst,source_clear,target_clear,not_king)),
    Equal.trans(U64,U64.and(Chess.get_kings(b),Chess.color(moved(b,src,dst),Chess.get_turn(b))),
      U64.and(Chess.get_kings(b),U64.or(U64.and_not(Chess.color(b,Chess.get_turn(b)),removal(src,dst)),U64.bit(U32.to_nat(dst)))),
      plane(b,Chess.get_turn(b)),
      Equal.cong(U64,U64,own => U64.and(Chess.get_kings(b),own),
        Chess.color(moved(b,src,dst),Chess.get_turn(b)),
        U64.or(U64.and_not(Chess.color(b,Chess.get_turn(b)),removal(src,dst)),U64.bit(U32.to_nat(dst))),own_update(b,src,dst)),
      A.frame(Chess.get_kings(b),Chess.color(b,Chess.get_turn(b)),U64.bit(U32.to_nat(src)),U64.bit(U32.to_nat(dst)),source_clear,target_clear)))

def selected_keep(+b: Chess.Board,+src: U32,+dst: U32,
  source_clear: D.disjoint(U64.bit(U32.to_nat(src)),Chess.get_kings(b)),
  target_clear: D.disjoint(U64.bit(U32.to_nat(dst)),Chess.get_kings(b)),
  not_king: {U32.is_eq(Chess.piece(b,src),5) == False{} : Bool}) ->
  {U64.ctz(plane(moved(b,src,dst),Chess.get_turn(b))) == U64.ctz(plane(b,Chess.get_turn(b))) : U32}:
  Equal.cong(U64,U32,U64.ctz,plane(moved(b,src,dst),Chess.get_turn(b)),plane(b,Chess.get_turn(b)),
    plane_keep(b,src,dst,source_clear,target_clear,not_king))
