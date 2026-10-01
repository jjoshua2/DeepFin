# Exact projection of actual moves onto the mover's selected king plane.
import Base
import ./Spec.bend as S
import ./Algebra.bend as A
import ../castling/Actual.bend as C
import ../move_update/Core.bend as Core
import ../move_update/Ordinary.bend as O
import ../move_update/Spec.bend as Delete
import ../ray/Algebra.bend as Old
import ../decoder/Bits.bend as Bits
import ../../../legal_probe/Chess.bend as Chess

def white_value(+white: Bool) -> {U32.is_eq(Bool.to_u32(white),1) == white : Bool}:
  match white:
    case False{}: {==}
    case True{}: {==}

def white_turn(+b: Chess.Board,+white: Bool,e: {Chess.get_turn(b) == Bool.to_u32(white) : U32}) ->
  {U32.is_eq(Chess.get_turn(b),1) == white : Bool}:
  Equal.trans(Bool,U32.is_eq(Chess.get_turn(b),1),U32.is_eq(Bool.to_u32(white),1),white,
    Equal.cong(U32,Bool,x => U32.is_eq(x,1),Chess.get_turn(b),Bool.to_u32(white),e),white_value(white))

def raw_castle(+board: Chess.Board,+m: U64,+t: U64,+r: U64,+white: Bool,turn: U32,rights: U32,ep: U32) ->
  {S.kings(C.raw(board,m,t,r,5,white,turn,rights,ep),white) == S.effect(Chess.get_kings(board),S.color(board,white),m,t,r) : U64}:
  match board white:
    case Chess.Board{p,n,b,ro,q,k,w,bl,ot,or,oe} False{}: {==}
    case Chess.Board{p,n,b,ro,q,k,w,bl,ot,or,oe} True{}: {==}

def raw_ordinary(+board: Chess.Board,+m: U64,+t: U64,+white: Bool,turn: U32,rights: U32,ep: U32) ->
  {S.kings(Core.raw(board,m,t,5,white,turn,rights,ep),white) ==
    U64.and(U64.or(U64.and_not(Chess.get_kings(board),m),t),U64.or(U64.and_not(S.color(board,white),m),t)) : U64}:
  match board white:
    case Chess.Board{p,n,b,ro,q,k,w,bl,ot,or,oe} False{}:
      Equal.cong(U64,U64,x => U64.and(U64.or(U64.and_not(k,m),t),U64.or(U64.and_not(bl,m),x)),U64.or(t,U64.zero()),t,Old.zero(t))
    case Chess.Board{p,n,b,ro,q,k,w,bl,ot,or,oe} True{}:
      Equal.cong(U64,U64,x => U64.and(U64.or(U64.and_not(k,m),t),U64.or(U64.and_not(w,m),x)),U64.or(t,U64.zero()),t,Old.zero(t))

def castle_kind(+b: Chess.Board,+src: U32,+dst: U32,+white: Bool,
  e: {Chess.get_turn(b) == Bool.to_u32(white) : U32}) ->
  {S.kings(C.dispatch(b,src,dst,5),white) == S.effect(Chess.get_kings(b),S.color(b,white),C.mask(src,dst),U64.bit(U32.to_nat(dst)),C.rook_to(src,dst)) : U64}:
  Equal.trans(U64,S.kings(C.dispatch(b,src,dst,5),white),
    S.kings(C.raw(b,C.mask(src,dst),U64.bit(U32.to_nat(dst)),C.rook_to(src,dst),5,white,U32.xor(Chess.get_turn(b),1),O.rights(b,src,dst,5),O.ep(src,dst,5)),white),
    S.effect(Chess.get_kings(b),S.color(b,white),C.mask(src,dst),U64.bit(U32.to_nat(dst)),C.rook_to(src,dst)),
    Equal.cong(Bool,U64,w => S.kings(C.raw(b,C.mask(src,dst),U64.bit(U32.to_nat(dst)),C.rook_to(src,dst),5,w,U32.xor(Chess.get_turn(b),1),O.rights(b,src,dst,5),O.ep(src,dst,5)),white),
      U32.is_eq(Chess.get_turn(b),1),white,white_turn(b,white,e)),
    raw_castle(b,C.mask(src,dst),U64.bit(U32.to_nat(dst)),C.rook_to(src,dst),white,U32.xor(Chess.get_turn(b),1),O.rights(b,src,dst,5),O.ep(src,dst,5)))

def ordinary_kind(+b: Chess.Board,+src: U32,+dst: U32,+white: Bool,
  e: {Chess.get_turn(b) == Bool.to_u32(white) : U32}) ->
  {S.kings(O.dispatch(b,src,dst,5),white) ==
    U64.and(U64.or(U64.and_not(Chess.get_kings(b),Delete.removal(U64.bit(U32.to_nat(src)),U64.bit(U32.to_nat(dst)))),U64.bit(U32.to_nat(dst))),
      U64.or(U64.and_not(S.color(b,white),Delete.removal(U64.bit(U32.to_nat(src)),U64.bit(U32.to_nat(dst)))),U64.bit(U32.to_nat(dst)))) : U64}:
  Equal.trans(U64,S.kings(O.dispatch(b,src,dst,5),white),
    S.kings(Core.raw(b,Delete.removal(U64.bit(U32.to_nat(src)),U64.bit(U32.to_nat(dst))),U64.bit(U32.to_nat(dst)),5,white,U32.xor(Chess.get_turn(b),1),O.rights(b,src,dst,5),O.ep(src,dst,5)),white),
    U64.and(U64.or(U64.and_not(Chess.get_kings(b),Delete.removal(U64.bit(U32.to_nat(src)),U64.bit(U32.to_nat(dst)))),U64.bit(U32.to_nat(dst))),
      U64.or(U64.and_not(S.color(b,white),Delete.removal(U64.bit(U32.to_nat(src)),U64.bit(U32.to_nat(dst)))),U64.bit(U32.to_nat(dst)))),
    Equal.cong(Bool,U64,w => S.kings(Core.raw(b,Delete.removal(U64.bit(U32.to_nat(src)),U64.bit(U32.to_nat(dst))),U64.bit(U32.to_nat(dst)),5,w,U32.xor(Chess.get_turn(b),1),O.rights(b,src,dst,5),O.ep(src,dst,5)),white),
      U32.is_eq(Chess.get_turn(b),1),white,white_turn(b,white,e)),
    raw_ordinary(b,Delete.removal(U64.bit(U32.to_nat(src)),U64.bit(U32.to_nat(dst))),U64.bit(U32.to_nat(dst)),white,U32.xor(Chess.get_turn(b),1),O.rights(b,src,dst,5),O.ep(src,dst,5)))

def zero_king(+k: U64) -> {U64.and(k,U64.zero()) == U64.zero() : U64}:
  match k:
    case U64{U32{+kl},U32{+kh}}:
      Equal.trans(U64,U64{U32{Word.and(32n,kl,Word.zero(32n))},U32{Word.and(32n,kh,Word.zero(32n))}},
        U64{U32{Word.zero(32n)},U32{Word.and(32n,kh,Word.zero(32n))}},U64.zero(),
        Equal.cong(Word(32n),U64,x => U64{U32{x},U32{Word.and(32n,kh,Word.zero(32n))}},Word.and(32n,kl,Word.zero(32n)),Word.zero(32n),Bits.and_zero(32n,kl)),
        Equal.cong(Word(32n),U64,x => U64{U32{Word.zero(32n)},U32{x}},Word.and(32n,kh,Word.zero(32n)),Word.zero(32n),Bits.and_zero(32n,kh)))

def plain(+k: U64,+c: U64,+m: U64,+t: U64) ->
  {U64.and(U64.or(U64.and_not(k,m),t),U64.or(U64.and_not(c,m),t)) == U64.or(U64.and_not(U64.and(k,c),m),t) : U64}:
  Equal.trans(U64,U64.and(U64.or(U64.and_not(k,m),t),U64.or(U64.and_not(c,m),t)),S.effect(k,c,m,t,U64.zero()),U64.or(U64.and_not(U64.and(k,c),m),t),
    Equal.cong(U64,U64,x => U64.and(U64.or(U64.and_not(k,m),t),U64.or(U64.and_not(c,m),x)),t,U64.or(t,U64.zero()),Equal.sym(U64,U64.or(t,U64.zero()),t,Old.zero(t))),
    A.projection(k,c,m,t,U64.zero(),zero_king(k)))

def ordinary(+b: Chess.Board,+src: U32,+dst: U32,+white: Bool,
  kind: {Chess.piece(b,src) == 5 : U32},turn: {Chess.get_turn(b) == Bool.to_u32(white) : U32}) ->
  {S.kings(Chess.make_move(b,Chess.Ply{src,dst,0,0}),white) == U64.or(U64.and_not(S.kings(b,white),Delete.removal(U64.bit(U32.to_nat(src)),U64.bit(U32.to_nat(dst)))),U64.bit(U32.to_nat(dst))) : U64}:
  Equal.trans(U64,S.kings(Chess.make_move(b,Chess.Ply{src,dst,0,0}),white),S.kings(O.dispatch(b,src,dst,5),white),U64.or(U64.and_not(S.kings(b,white),Delete.removal(U64.bit(U32.to_nat(src)),U64.bit(U32.to_nat(dst)))),U64.bit(U32.to_nat(dst))),
    Equal.trans(U64,S.kings(Chess.make_move(b,Chess.Ply{src,dst,0,0}),white),S.kings(O.dispatch(b,src,dst,Chess.piece(b,src)),white),S.kings(O.dispatch(b,src,dst,5),white),
      Equal.cong(Chess.Board,U64,x => S.kings(x,white),Chess.make_move(b,Chess.Ply{src,dst,0,0}),O.dispatch(b,src,dst,Chess.piece(b,src)),O.unfold(b,src,dst)),
      Equal.cong(U32,U64,x => S.kings(O.dispatch(b,src,dst,x),white),Chess.piece(b,src),5,kind)),
    Equal.trans(U64,S.kings(O.dispatch(b,src,dst,5),white),
      U64.and(U64.or(U64.and_not(Chess.get_kings(b),Delete.removal(U64.bit(U32.to_nat(src)),U64.bit(U32.to_nat(dst)))),U64.bit(U32.to_nat(dst))),U64.or(U64.and_not(S.color(b,white),Delete.removal(U64.bit(U32.to_nat(src)),U64.bit(U32.to_nat(dst)))),U64.bit(U32.to_nat(dst)))),
      U64.or(U64.and_not(S.kings(b,white),Delete.removal(U64.bit(U32.to_nat(src)),U64.bit(U32.to_nat(dst)))),U64.bit(U32.to_nat(dst))),
      ordinary_kind(b,src,dst,white,turn),plain(Chess.get_kings(b),S.color(b,white),Delete.removal(U64.bit(U32.to_nat(src)),U64.bit(U32.to_nat(dst))),U64.bit(U32.to_nat(dst)))))

def castle(+b: Chess.Board,+src: U32,+dst: U32,+white: Bool,
  kind: {Chess.piece(b,src) == 5 : U32},turn: {Chess.get_turn(b) == Bool.to_u32(white) : U32},
  empty: {U64.and(Chess.get_kings(b),C.rook_to(src,dst)) == U64.zero() : U64}) ->
  {S.kings(Chess.make_move(b,Chess.Ply{src,dst,0,2}),white) == U64.or(U64.and_not(S.kings(b,white),C.mask(src,dst)),U64.bit(U32.to_nat(dst))) : U64}:
  Equal.trans(U64,S.kings(Chess.make_move(b,Chess.Ply{src,dst,0,2}),white),S.kings(C.dispatch(b,src,dst,5),white),U64.or(U64.and_not(S.kings(b,white),C.mask(src,dst)),U64.bit(U32.to_nat(dst))),
    Equal.trans(U64,S.kings(Chess.make_move(b,Chess.Ply{src,dst,0,2}),white),S.kings(C.dispatch(b,src,dst,Chess.piece(b,src)),white),S.kings(C.dispatch(b,src,dst,5),white),
      Equal.cong(Chess.Board,U64,x => S.kings(x,white),Chess.make_move(b,Chess.Ply{src,dst,0,2}),C.dispatch(b,src,dst,Chess.piece(b,src)),C.unfold(b,src,dst)),
      Equal.cong(U32,U64,x => S.kings(C.dispatch(b,src,dst,x),white),Chess.piece(b,src),5,kind)),
    Equal.trans(U64,S.kings(C.dispatch(b,src,dst,5),white),S.effect(Chess.get_kings(b),S.color(b,white),C.mask(src,dst),U64.bit(U32.to_nat(dst)),C.rook_to(src,dst)),
      U64.or(U64.and_not(S.kings(b,white),C.mask(src,dst)),U64.bit(U32.to_nat(dst))),castle_kind(b,src,dst,white,turn),
      A.projection(Chess.get_kings(b),S.color(b,white),C.mask(src,dst),U64.bit(U32.to_nat(dst)),C.rook_to(src,dst),empty)))
