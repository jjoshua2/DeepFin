# Exact flag-one production update; the old side is queried after turn flips.
import Base
import ../en_passant/Actual.bend as EP
import ../en_passant/Mask.bend as Mask
import ../move_update/Core.bend as Core
import ../move_update/Ordinary.bend as O
import ../ordinary_king_frame/Frame.bend as Old
import ../ordinary_king_frame/Algebra.bend as A
import ../candidate_king_destination/Algebra.bend as Union
import ../ray/Algebra.bend as R
import ../../../legal_probe/Chess.bend as C

def moved(+b: C.Board,s: U32,d: U32) -> C.Board: C.make_move(b,C.Ply{s,d,0,1})
def source(+s: U32,+d: U32) -> U64: EP.source(s,d)
def mask(+s: U32,+d: U32) -> U64: EP.mask(s,d)
def plane(+b: C.Board,t: U32) -> U64: Old.plane(b,t)

def update(+b: C.Board,+s: U32,+d: U32) -> {C.get_kings(moved(b,s,d)) == C.update_piece(C.get_kings(b),mask(s,d),U64.bit(U32.to_nat(d)),U32.is_eq(C.piece(b,s),5)) : U64}:
  match b:
    case C.Board{+p,+n,+bi,+r,+q,+k,+w,+bl,+t,+rr,+e}:
      +board=Old.as_board(C.Board{p,n,bi,r,q,k,w,bl,t,rr,e})
      Equal.cong(C.Board,U64,C.get_kings,moved(board,s,d),EP.dispatch(board,s,d,C.piece(board,s)),EP.unfold(board,s,d))

def kings(+b: C.Board,+s: U32,+d: U32,+sc: {U64.and(U64.bit(U32.to_nat(s)),C.get_kings(b)) == U64.zero() : U64},+dc: {U64.and(U64.bit(U32.to_nat(d)),C.get_kings(b)) == U64.zero() : U64},+vc: {U64.and(U64.bit(U32.to_nat(U32.xor(d,8))),C.get_kings(b)) == U64.zero() : U64},kind: {C.piece(b,s) == 0 : U32}) -> {C.get_kings(moved(b,s,d)) == C.get_kings(b) : U64}:
  +combined=Union.union_clear(U64.bit(U32.to_nat(s)),U64.bit(U32.to_nat(U32.xor(d,8))),C.get_kings(b),sc,vc)
  +raw_clear=Equal.trans(U64,U64.and(mask(s,d),C.get_kings(b)),U64.and(U64.or(U64.or(source(s,d),U64.bit(U32.to_nat(d))),U64.or(U64.bit(U32.to_nat(d)),U64.zero())),C.get_kings(b)),U64.zero(),
    Equal.cong(U64,U64,x => U64.and(x,C.get_kings(b)),mask(s,d),U64.or(U64.or(source(s,d),U64.bit(U32.to_nat(d))),U64.or(U64.bit(U32.to_nat(d)),U64.zero())),Mask.same(source(s,d),U64.bit(U32.to_nat(d)))),A.removal_clear(source(s,d),U64.bit(U32.to_nat(d)),C.get_kings(b),combined,dc))
  Equal.trans(U64,C.get_kings(moved(b,s,d)),C.update_piece(C.get_kings(b),mask(s,d),U64.bit(U32.to_nat(d)),U32.is_eq(C.piece(b,s),5)),C.get_kings(b),update(b,s,d),
    Equal.trans(U64,C.update_piece(C.get_kings(b),mask(s,d),U64.bit(U32.to_nat(d)),U32.is_eq(C.piece(b,s),5)),U64.and_not(C.get_kings(b),mask(s,d)),C.get_kings(b),
      Old.no_put(C.get_kings(b),mask(s,d),U64.bit(U32.to_nat(d)),U32.is_eq(C.piece(b,s),5),Equal.cong(U32,Bool,x => U32.is_eq(x,5),C.piece(b,s),0,kind)),A.keep(C.get_kings(b),mask(s,d),raw_clear)))

def own(+b: C.Board,+s: U32,+d: U32) -> {C.color(moved(b,s,d),C.get_turn(b)) == U64.or(U64.and_not(C.color(b,C.get_turn(b)),mask(s,d)),U64.bit(U32.to_nat(d))) : U64}:
  Equal.trans(U64,C.color(moved(b,s,d),C.get_turn(b)),C.color(EP.dispatch(b,s,d,C.piece(b,s)),C.get_turn(b)),U64.or(U64.and_not(C.color(b,C.get_turn(b)),mask(s,d)),U64.bit(U32.to_nat(d))),
    Equal.cong(C.Board,U64,x => C.color(x,C.get_turn(b)),moved(b,s,d),EP.dispatch(b,s,d,C.piece(b,s)),EP.unfold(b,s,d)),
    Equal.trans(U64,C.color(EP.dispatch(b,s,d,C.piece(b,s)),C.get_turn(b)),U64.or(U64.and_not(C.color(b,C.get_turn(b)),mask(s,d)),U64.or(U64.bit(U32.to_nat(d)),U64.zero())),U64.or(U64.and_not(C.color(b,C.get_turn(b)),mask(s,d)),U64.bit(U32.to_nat(d))),
      Old.own_raw(b,mask(s,d),U64.bit(U32.to_nat(d)),C.piece(b,s),U32.is_eq(C.get_turn(b),1),U32.xor(C.get_turn(b),1),O.rights(b,s,d,C.piece(b,s)),O.ep(s,d,C.piece(b,s))),
      Equal.cong(U64,U64,x => U64.or(U64.and_not(C.color(b,C.get_turn(b)),mask(s,d)),x),U64.or(U64.bit(U32.to_nat(d)),U64.zero()),U64.bit(U32.to_nat(d)),R.zero(U64.bit(U32.to_nat(d))))))

def selected_plane(+b: C.Board,+s: U32,+d: U32,+sc: {U64.and(U64.bit(U32.to_nat(s)),C.get_kings(b)) == U64.zero() : U64},+dc: {U64.and(U64.bit(U32.to_nat(d)),C.get_kings(b)) == U64.zero() : U64},+vc: {U64.and(U64.bit(U32.to_nat(U32.xor(d,8))),C.get_kings(b)) == U64.zero() : U64},kind: {C.piece(b,s) == 0 : U32}) -> {plane(moved(b,s,d),C.get_turn(b)) == plane(b,C.get_turn(b)) : U64}:
  Equal.trans(U64,plane(moved(b,s,d),C.get_turn(b)),U64.and(C.get_kings(b),C.color(moved(b,s,d),C.get_turn(b))),plane(b,C.get_turn(b)),
    Equal.cong(U64,U64,x => U64.and(x,C.color(moved(b,s,d),C.get_turn(b))),C.get_kings(moved(b,s,d)),C.get_kings(b),kings(b,s,d,sc,dc,vc,kind)),
    Equal.trans(U64,U64.and(C.get_kings(b),C.color(moved(b,s,d),C.get_turn(b))),U64.and(C.get_kings(b),U64.or(U64.and_not(C.color(b,C.get_turn(b)),mask(s,d)),U64.bit(U32.to_nat(d)))),plane(b,C.get_turn(b)),
      Equal.cong(U64,U64,x => U64.and(C.get_kings(b),x),C.color(moved(b,s,d),C.get_turn(b)),U64.or(U64.and_not(C.color(b,C.get_turn(b)),mask(s,d)),U64.bit(U32.to_nat(d))),own(b,s,d)),
      Equal.trans(U64,U64.and(C.get_kings(b),U64.or(U64.and_not(C.color(b,C.get_turn(b)),mask(s,d)),U64.bit(U32.to_nat(d)))),U64.and(C.get_kings(b),U64.or(U64.and_not(C.color(b,C.get_turn(b)),U64.or(U64.or(source(s,d),U64.bit(U32.to_nat(d))),U64.or(U64.bit(U32.to_nat(d)),U64.zero()))),U64.bit(U32.to_nat(d)))),plane(b,C.get_turn(b)),
        Equal.cong(U64,U64,x => U64.and(C.get_kings(b),U64.or(U64.and_not(C.color(b,C.get_turn(b)),x),U64.bit(U32.to_nat(d)))),mask(s,d),U64.or(U64.or(source(s,d),U64.bit(U32.to_nat(d))),U64.or(U64.bit(U32.to_nat(d)),U64.zero())),Mask.same(source(s,d),U64.bit(U32.to_nat(d)))),
        A.frame(C.get_kings(b),C.color(b,C.get_turn(b)),source(s,d),U64.bit(U32.to_nat(d)),Union.union_clear(U64.bit(U32.to_nat(s)),U64.bit(U32.to_nat(U32.xor(d,8))),C.get_kings(b),sc,vc),dc))))
