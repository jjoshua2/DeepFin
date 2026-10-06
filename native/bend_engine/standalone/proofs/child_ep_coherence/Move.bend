# Only flag 0/1 and promotion 0; actual field evidence comes from Fields.initialized.
import Base
import ../move_update/Core.bend as Core
import ../move_update/Ordinary.bend as Ordinary
import ../initialized_pawn_double/Spec.bend as D
import ../bitboard_membership/AllEmittedSound.bend as All
import ../bitboard_membership/IndexEquality.bend as Eq
import ../../../legal_probe/Chess.bend as C

def mask(+s: U32,+d: U32,ep: Bool) -> U64:
  U64.or(U64.or(U64.bit(U32.to_nat(s)),U64.bit(U32.to_nat(Bool.pick(U32,ep,U32.xor(d,8),d)))),U64.or(U64.bit(U32.to_nat(d)),U64.zero()))

def midpoint(+s: U32,d: U32) -> U32: U32.div(U32.add(s,d),2)

def dispatch(+b: C.Board,+s: U32,+d: U32,+ep: Bool,+kind: U32) -> C.Board:
  Core.raw(b,mask(s,d,ep),U64.bit(U32.to_nat(d)),kind,D.white(b),U32.xor(C.get_turn(b),1),Ordinary.rights(b,s,d,kind),Ordinary.ep(s,d,kind))

def unfold(+b: C.Board,+s: U32,+d: U32,ep: Bool) -> {C.make_move(b,C.Ply{s,d,0,Bool.to_u32(ep)}) == dispatch(b,s,d,ep,C.piece(b,s)) : C.Board}:
  match b ep:
    case C.Board{p,n,bi,r,q,k,w,bl,t,rights,old_ep} False{}: {==}
    case C.Board{p,n,bi,r,q,k,w,bl,t,rights,old_ep} True{}: {==}

def raw_turn(+b: C.Board,+s: U32,+d: U32,+ep: Bool,+turn: U32) -> C.Board:
  Core.raw(b,mask(s,d,ep),U64.bit(U32.to_nat(d)),0,U32.is_eq(turn,1),U32.xor(turn,1),Ordinary.rights(b,s,d,0),midpoint(s,d))

def next(+b: C.Board,+s: U32,+d: U32,+ep: Bool,+white: Bool) -> C.Board:
  Core.raw(b,mask(s,d,ep),U64.bit(U32.to_nat(d)),0,white,Bool.to_u32(Bool.not(white)),Ordinary.rights(b,s,d,0),midpoint(s,d))

def turn(+b: C.Board,+s: U32,+d: U32,+ep: Bool,white: Bool) -> {raw_turn(b,s,d,ep,Bool.to_u32(white)) == next(b,s,d,ep,white) : C.Board}:
  match white:
    case False{}: {==}
    case True{}: {==}

def zero(+b: C.Board,+s: U32,+d: U32,+ep: Bool,+trigger: {Bool.and(U32.is_zero(C.piece(b,s)),U32.is_eq(U32.xor(s,d),16)) == True{} : Bool}) -> {dispatch(b,s,d,ep,C.piece(b,s)) == raw_turn(b,s,d,ep,C.get_turn(b)) : C.Board}:
  +kind0=Eq.u32_eq_true_same(C.piece(b,s),0,All.bool_and_true_left(U32.is_zero(C.piece(b,s)),U32.is_eq(U32.xor(s,d),16),trigger))
  Equal.trans(C.Board,dispatch(b,s,d,ep,C.piece(b,s)),dispatch(b,s,d,ep,0),raw_turn(b,s,d,ep,C.get_turn(b)),
    Equal.cong(U32,C.Board,x => dispatch(b,s,d,ep,x),C.piece(b,s),0,kind0),
    Equal.cong(U32,C.Board,x => Core.raw(b,mask(s,d,ep),U64.bit(U32.to_nat(d)),0,D.white(b),U32.xor(C.get_turn(b),1),Ordinary.rights(b,s,d,0),x),
      Ordinary.ep(s,d,0),midpoint(s,d),
      %Equal.sym(Bool,U32.is_eq(U32.xor(s,d),16),True{},All.bool_and_true_right(U32.is_zero(C.piece(b,s)),U32.is_eq(U32.xor(s,d),16),trigger)) : {Bool.pick(U32,_,midpoint(s,d),64) == midpoint(s,d) : U32}
      {==}))

def actual(+b: C.Board,+s: U32,+d: U32,+ep: Bool,+white: Bool,+trigger: {Bool.and(U32.is_zero(C.piece(b,s)),U32.is_eq(U32.xor(s,d),16)) == True{} : Bool},canonical: {C.get_turn(b) == Bool.to_u32(white) : U32}) -> {C.make_move(b,C.Ply{s,d,0,Bool.to_u32(ep)}) == next(b,s,d,ep,white) : C.Board}:
  Equal.trans(C.Board,C.make_move(b,C.Ply{s,d,0,Bool.to_u32(ep)}),dispatch(b,s,d,ep,C.piece(b,s)),next(b,s,d,ep,white),unfold(b,s,d,ep),
    Equal.trans(C.Board,dispatch(b,s,d,ep,C.piece(b,s)),raw_turn(b,s,d,ep,C.get_turn(b)),next(b,s,d,ep,white),zero(b,s,d,ep,trigger),
      Equal.trans(C.Board,raw_turn(b,s,d,ep,C.get_turn(b)),raw_turn(b,s,d,ep,Bool.to_u32(white)),next(b,s,d,ep,white),Equal.cong(U32,C.Board,x => raw_turn(b,s,d,ep,x),C.get_turn(b),Bool.to_u32(white),canonical),turn(b,s,d,ep,white))))
