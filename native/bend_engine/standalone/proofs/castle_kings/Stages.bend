# Singleton premises propagated through actual ordinary transit and flag-two update.
import Base
import ./Spec.bend as S
import ./Kind.bend as Kind
import ./Fresh.bend as Fresh
import ./Projection.bend as P
import ../board/Spec.bend as B
import ../move_update/Spec.bend as D
import ../castling/Actual.bend as C
import ../castling/Route.bend as R
import ../castle_emission/Spec.bend as E
import ../castle_emission/Guard.bend as Guard
import ../../../legal_probe/Chess.bend as Chess

def transit_removes_source(+white: Bool,+ks: Bool) ->
  {U64.or(U64.and_not(U64.bit(U32.to_nat(S.src(white))),D.removal(U64.bit(U32.to_nat(S.src(white))),U64.bit(U32.to_nat(S.transit(white,ks))))),U64.bit(U32.to_nat(S.transit(white,ks)))) == U64.bit(U32.to_nat(S.transit(white,ks))) : U64}:
  match white ks:
    case False{} False{}: {==}
    case False{} True{}: {==}
    case True{} False{}: {==}
    case True{} True{}: {==}

def castle_removes_source(+white: Bool,+ks: Bool) ->
  {U64.or(U64.and_not(U64.bit(U32.to_nat(S.src(white))),C.mask(S.src(white),S.dst(white,ks))),U64.bit(U32.to_nat(S.dst(white,ks)))) == U64.bit(U32.to_nat(S.dst(white,ks))) : U64}:
  match white ks:
    case False{} False{}: {==}
    case False{} True{}: {==}
    case True{} False{}: {==}
    case True{} True{}: {==}

def route_rook(+white: Bool,+ks: Bool) ->
  {R.rook(E.choose(white,ks)) == C.rook_to(S.src(white),S.dst(white,ks)) : U64}:
  match white ks:
    case False{} False{}: {==}
    case False{} True{}: {==}
    case True{} False{}: {==}
    case True{} True{}: {==}

def guard_fresh(+b: Chess.Board,+white: Bool,+ks: Bool,
  turn: {Chess.get_turn(b) == Bool.to_u32(white) : U32},guard: {E.guard(b,ks) == True{} : Bool}) ->
  {B.fresh_mask(b,C.rook_to(S.src(white),S.dst(white,ks))) == True{} : Bool}:
  Equal.trans(Bool,B.fresh_mask(b,C.rook_to(S.src(white),S.dst(white,ks))),B.fresh_mask(b,R.rook(E.route(b,ks))),True{},
    Equal.cong(U64,Bool,x => B.fresh_mask(b,x),C.rook_to(S.src(white),S.dst(white,ks)),R.rook(E.route(b,ks)),
      Equal.sym(U64,R.rook(E.route(b,ks)),C.rook_to(S.src(white),S.dst(white,ks)),
        Equal.trans(U64,R.rook(E.route(b,ks)),R.rook(E.choose(white,ks)),C.rook_to(S.src(white),S.dst(white,ks)),
          Equal.cong(Bool,U64,x => R.rook(E.choose(x,ks)),U32.is_eq(Chess.get_turn(b),1),white,P.white_turn(b,white,turn)),route_rook(white,ks)))),
    Guard.fresh(b,ks,guard))

def mid(+b: Chess.Board,+white: Bool,+ks: Bool,
  good: {B.valid(b) == True{} : Bool},turn: {Chess.get_turn(b) == Bool.to_u32(white) : U32},
  +one: {S.kings(b,white) == U64.bit(U32.to_nat(S.src(white))) : U64}) ->
  {S.kings(S.mid(b,white,ks),white) == U64.bit(U32.to_nat(S.transit(white,ks))) : U64}:
  Equal.trans(U64,S.kings(S.mid(b,white,ks),white),
    U64.or(U64.and_not(S.kings(b,white),D.removal(U64.bit(U32.to_nat(S.src(white))),U64.bit(U32.to_nat(S.transit(white,ks))))),U64.bit(U32.to_nat(S.transit(white,ks)))),
    U64.bit(U32.to_nat(S.transit(white,ks))),
    P.ordinary(b,S.src(white),S.transit(white,ks),white,Kind.source_kind(b,white,good,one),turn),
    Equal.trans(U64,U64.or(U64.and_not(S.kings(b,white),D.removal(U64.bit(U32.to_nat(S.src(white))),U64.bit(U32.to_nat(S.transit(white,ks))))),U64.bit(U32.to_nat(S.transit(white,ks)))),
      U64.or(U64.and_not(U64.bit(U32.to_nat(S.src(white))),D.removal(U64.bit(U32.to_nat(S.src(white))),U64.bit(U32.to_nat(S.transit(white,ks))))),U64.bit(U32.to_nat(S.transit(white,ks)))),U64.bit(U32.to_nat(S.transit(white,ks))),
      Equal.cong(U64,U64,x => U64.or(U64.and_not(x,D.removal(U64.bit(U32.to_nat(S.src(white))),U64.bit(U32.to_nat(S.transit(white,ks))))),U64.bit(U32.to_nat(S.transit(white,ks)))),S.kings(b,white),U64.bit(U32.to_nat(S.src(white))),one),transit_removes_source(white,ks)))

def final(+b: Chess.Board,+white: Bool,+ks: Bool,
  +good: {B.valid(b) == True{} : Bool},+turn: {Chess.get_turn(b) == Bool.to_u32(white) : U32},
  +one: {S.kings(b,white) == U64.bit(U32.to_nat(S.src(white))) : U64},guard: {E.guard(b,ks) == True{} : Bool}) ->
  {S.kings(S.final(b,white,ks),white) == U64.bit(U32.to_nat(S.dst(white,ks))) : U64}:
  Equal.trans(U64,S.kings(S.final(b,white,ks),white),
    U64.or(U64.and_not(S.kings(b,white),C.mask(S.src(white),S.dst(white,ks))),U64.bit(U32.to_nat(S.dst(white,ks)))),U64.bit(U32.to_nat(S.dst(white,ks))),
    P.castle(b,S.src(white),S.dst(white,ks),white,Kind.source_kind(b,white,good,one),turn,
      Fresh.board(b,C.rook_to(S.src(white),S.dst(white,ks)),good,guard_fresh(b,white,ks,turn,guard))),
    Equal.trans(U64,U64.or(U64.and_not(S.kings(b,white),C.mask(S.src(white),S.dst(white,ks))),U64.bit(U32.to_nat(S.dst(white,ks)))),
      U64.or(U64.and_not(U64.bit(U32.to_nat(S.src(white))),C.mask(S.src(white),S.dst(white,ks))),U64.bit(U32.to_nat(S.dst(white,ks)))),U64.bit(U32.to_nat(S.dst(white,ks))),
      Equal.cong(U64,U64,x => U64.or(U64.and_not(x,C.mask(S.src(white),S.dst(white,ks))),U64.bit(U32.to_nat(S.dst(white,ks)))),S.kings(b,white),U64.bit(U32.to_nat(S.src(white))),one),castle_removes_source(white,ks)))

def turn_unfold(+b: Chess.Board,+move: Chess.Ply) ->
  {Chess.get_turn(Chess.make_move(b,move)) == U32.xor(Chess.get_turn(b),1) : U32}:
  match b move:
    case Chess.Board{p,n,bi,r,q,k,w,bl,turn,rights,ep} Chess.Ply{src,dst,promo,flag}: {==}

def opposite(+white: Bool) -> {U32.xor(Bool.to_u32(white),1) == Bool.to_u32(Bool.not(white)) : U32}:
  match white:
    case False{}: {==}
    case True{}: {==}

def flip(+b: Chess.Board,+move: Chess.Ply,+white: Bool,e: {Chess.get_turn(b) == Bool.to_u32(white) : U32}) ->
  {Chess.get_turn(Chess.make_move(b,move)) == Bool.to_u32(Bool.not(white)) : U32}:
  Equal.trans(U32,Chess.get_turn(Chess.make_move(b,move)),U32.xor(Chess.get_turn(b),1),Bool.to_u32(Bool.not(white)),turn_unfold(b,move),
    Equal.trans(U32,U32.xor(Chess.get_turn(b),1),U32.xor(Bool.to_u32(white),1),Bool.to_u32(Bool.not(white)),
      Equal.cong(U32,U32,x => U32.xor(x,1),Chess.get_turn(b),Bool.to_u32(white),e),opposite(white)))
