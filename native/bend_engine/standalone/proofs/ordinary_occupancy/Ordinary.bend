import Base
import ./Algebra.bend as A
import ../move_update/Ordinary.bend as O
import ../move_update/Core.bend as Core
import ../move_update/Spec.bend as M
import ../ray/Path.bend as P
import ../ray/Spec.bend as S
import ../ray_clear_disjoint/Algebra.bend as D
import ../ray_clear_disjoint/Clear.bend as C
import ../occupancy_insertion/Lattice.bend as L
import ../../Tables.bend as Tables
import ../../../legal_probe/Chess.bend as Chess

def normalized(+board: Chess.Board,+src: U32,+dst: U32) -> U64:
  U64.or(U64.and_not(Chess.occupied(board),U64.bit(U32.to_nat(src))),U64.bit(U32.to_nat(dst)))

def raw_occupied(+board: Chess.Board,+mask: U64,+target: U64,+kind: U32,+white: Bool,+turn: U32,+rights: U32,+ep: U32) ->
  {Chess.occupied(Core.raw(board,mask,target,kind,white,turn,rights,ep)) ==
    U64.or(U64.and_not(Chess.occupied(board),mask),target) : U64}:
  match board:
    case Chess.Board{p,n,b,r,q,k,+w,+bl,oldturn,oldrights,oldep}:
      A.color_updates(w,bl,mask,target,white)

# Total ordinary occupancy identity: no ownership, consistency, turn or range premise.
def normalization(+board: Chess.Board,+src: U32,+dst: U32) ->
  {Chess.occupied(Chess.make_move(board,Chess.Ply{src,dst,0,0})) == normalized(board,src,dst) : U64}:
  Equal.trans(U64,Chess.occupied(Chess.make_move(board,Chess.Ply{src,dst,0,0})),
    Chess.occupied(O.dispatch(board,src,dst,Chess.piece(board,src))),normalized(board,src,dst),
    Equal.cong(Chess.Board,U64,b => Chess.occupied(b),Chess.make_move(board,Chess.Ply{src,dst,0,0}),
      O.dispatch(board,src,dst,Chess.piece(board,src)),O.unfold(board,src,dst)),
    Equal.trans(U64,Chess.occupied(O.dispatch(board,src,dst,Chess.piece(board,src))),
      U64.or(U64.and_not(Chess.occupied(board),M.removal(U64.bit(U32.to_nat(src)),U64.bit(U32.to_nat(dst)))),U64.bit(U32.to_nat(dst))),
      normalized(board,src,dst),
      raw_occupied(board,M.removal(U64.bit(U32.to_nat(src)),U64.bit(U32.to_nat(dst))),U64.bit(U32.to_nat(dst)),Chess.piece(board,src),
        U32.is_eq(Chess.get_turn(board),1),U32.xor(Chess.get_turn(board),1),O.rights(board,src,dst,Chess.piece(board,src)),O.ep(src,dst,Chess.piece(board,src))),
      A.reinsert(Chess.occupied(board),U64.bit(U32.to_nat(src)),U64.bit(U32.to_nat(dst)))))

# Actual post-move occupancy is rewritten into the checked clearing/insertion consumer.
def actual_ray(+n: Nat,+sq: U32,+dir: U32,+xs: List<&2,Nat>,+board: Chess.Board,+src: U32,+dst: U32,+acc: U64,
  route: {S.trace(n,sq,dir,xs) == True{} : Bool},
  empty: D.disjoint(U64.bit(U32.to_nat(src)),P.attack(xs,Chess.occupied(board)))) ->
  L.subset(Tables.ray(n,False{},sq,dir,Chess.occupied(Chess.make_move(board,Chess.Ply{src,dst,0,0})),False{},acc),
    Tables.ray(n,False{},sq,dir,Chess.occupied(board),False{},acc)):
  L.rewrite(Tables.ray(n,False{},sq,dir,normalized(board,src,dst),False{},acc),
    Tables.ray(n,False{},sq,dir,Chess.occupied(board),False{},acc),
    Tables.ray(n,False{},sq,dir,Chess.occupied(Chess.make_move(board,Chess.Ply{src,dst,0,0})),False{},acc),
    Tables.ray(n,False{},sq,dir,Chess.occupied(board),False{},acc),
    Equal.cong(U64,U64,occ => Tables.ray(n,False{},sq,dir,occ,False{},acc),
      Chess.occupied(Chess.make_move(board,Chess.Ply{src,dst,0,0})),normalized(board,src,dst),normalization(board,src,dst)),{==},
    C.clear_then_insert(n,sq,dir,xs,Chess.occupied(board),U64.bit(U32.to_nat(src)),U64.bit(U32.to_nat(dst)),acc,route,empty))
