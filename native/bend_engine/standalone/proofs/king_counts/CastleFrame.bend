# Candidate bridge for both actual fixed-color king planes, including flag2.
import Base
import ./CastleFusion.bend as X
import ./Frame.bend as Frame
import ./Selector.bend as S
import ../color_counts/Actual.bend as A
import ../backrank_pawns/Mask.bend as M
import ../board/Spec.bend as B
import ../castle_emission/Spec.bend as E
import ../castle_emission/Guard.bend as G
import ../castle_kings/Fresh.bend as Fresh
import ../castling/Route.bend as R
import ../../../legal_probe/Chess.bend as Chess

def landing(m: Chess.Ply) -> U64:
  Chess.Ply{+s,+d,p,f}=m
  Chess.select_u64(U32.is_eq(f,2),U64.bit(U32.to_nat(U32.div(U32.add(s,d),2))),U64.zero())

# Only parent king/rook-landing clearance is supplied here.
# Neither output equality nor a king count is an input.
def raw(+b: Chess.Board,+m: Chess.Ply,+color: U64,+side: Bool,
  empty: {U64.and(Chess.get_kings(b),landing(m)) == U64.zero() : U64}) ->
  {U64.and(Chess.update_piece(Chess.get_kings(b),M.remove(m),U64.bit(U32.to_nat(A.destination(m))),S.put(b,m)),
    Chess.update_piece(color,M.remove(m),A.added(m),side)) == Frame.model(b,m,color,side) : U64}:
  Chess.Ply{+s,+d,+p,+f}=m
  Equal.trans(U64,
    U64.and(Chess.update_piece(Chess.get_kings(b),M.remove(m),U64.bit(U32.to_nat(A.destination(m))),S.put(b,m)),
      Chess.update_piece(color,M.remove(m),A.added(m),side)),
    U64.and(Chess.update_piece(Chess.get_kings(b),M.remove(m),U64.bit(U32.to_nat(A.destination(m))),S.put(b,m)),
      Chess.update_piece(color,M.remove(m),U64.bit(U32.to_nat(A.destination(m))),side)),
    Frame.model(b,m,color,side),
    X.extra(Chess.get_kings(b),color,M.remove(m),U64.bit(U32.to_nat(A.destination(m))),landing(m),S.put(b,m),side,empty),
    Frame.raw_fusion(b,m,color,side))

def frame(+b: Chess.Board,+m: Chess.Ply,+color: U64,+side: Bool,
  empty: {U64.and(Chess.get_kings(b),landing(m)) == U64.zero() : U64}) ->
  {U64.and(Chess.get_kings(Chess.make_move(b,m)),Chess.update_piece(color,M.remove(m),A.added(m),side)) ==
   Frame.model(b,m,color,side) : U64}:
  Equal.trans(U64,
    U64.and(Chess.get_kings(Chess.make_move(b,m)),Chess.update_piece(color,M.remove(m),A.added(m),side)),
    U64.and(Chess.update_piece(Chess.get_kings(b),M.remove(m),U64.bit(U32.to_nat(A.destination(m))),S.put(b,m)),
      Chess.update_piece(color,M.remove(m),A.added(m),side)),
    Frame.model(b,m,color,side),
    Equal.cong(U64,U64,x => U64.and(x,Chess.update_piece(color,M.remove(m),A.added(m),side)),
      Chess.get_kings(Chess.make_move(b,m)),
      Chess.update_piece(Chess.get_kings(b),M.remove(m),U64.bit(U32.to_nat(A.destination(m))),S.put(b,m)),Frame.king_actual(b,m)),
    raw(b,m,color,side,empty))

def white_actual(+b: Chess.Board,+m: Chess.Ply,
  empty: {U64.and(Chess.get_kings(b),landing(m)) == U64.zero() : U64}) ->
  {U64.and(Chess.get_kings(Chess.make_move(b,m)),Chess.get_white(Chess.make_move(b,m))) ==
   Frame.model(b,m,Chess.get_white(b),U32.is_eq(Chess.get_turn(b),1)) : U64}:
  Equal.trans(U64,U64.and(Chess.get_kings(Chess.make_move(b,m)),Chess.get_white(Chess.make_move(b,m))),
    U64.and(Chess.get_kings(Chess.make_move(b,m)),Chess.update_piece(Chess.get_white(b),M.remove(m),A.added(m),U32.is_eq(Chess.get_turn(b),1))),
    Frame.model(b,m,Chess.get_white(b),U32.is_eq(Chess.get_turn(b),1)),
    Equal.cong(U64,U64,x => U64.and(Chess.get_kings(Chess.make_move(b,m)),x),Chess.get_white(Chess.make_move(b,m)),
      Chess.update_piece(Chess.get_white(b),M.remove(m),A.added(m),U32.is_eq(Chess.get_turn(b),1)),A.white_actual(b,m)),
    frame(b,m,Chess.get_white(b),U32.is_eq(Chess.get_turn(b),1),empty))

def black_actual(+b: Chess.Board,+m: Chess.Ply,
  empty: {U64.and(Chess.get_kings(b),landing(m)) == U64.zero() : U64}) ->
  {U64.and(Chess.get_kings(Chess.make_move(b,m)),Chess.get_black(Chess.make_move(b,m))) ==
   Frame.model(b,m,Chess.get_black(b),Bool.not(U32.is_eq(Chess.get_turn(b),1))) : U64}:
  Equal.trans(U64,U64.and(Chess.get_kings(Chess.make_move(b,m)),Chess.get_black(Chess.make_move(b,m))),
    U64.and(Chess.get_kings(Chess.make_move(b,m)),Chess.update_piece(Chess.get_black(b),M.remove(m),A.added(m),Bool.not(U32.is_eq(Chess.get_turn(b),1)))),
    Frame.model(b,m,Chess.get_black(b),Bool.not(U32.is_eq(Chess.get_turn(b),1))),
    Equal.cong(U64,U64,x => U64.and(Chess.get_kings(Chess.make_move(b,m)),x),Chess.get_black(Chess.make_move(b,m)),
      Chess.update_piece(Chess.get_black(b),M.remove(m),A.added(m),Bool.not(U32.is_eq(Chess.get_turn(b),1))),A.black_actual(b,m)),
    frame(b,m,Chess.get_black(b),Bool.not(U32.is_eq(Chess.get_turn(b),1)),empty))

def route_landing(+white: Bool,+ks: Bool) ->
  {landing(Chess.Ply{U32.add(Bool.pick(U32,white,0,56),4),U32.add(Bool.pick(U32,white,0,56),Bool.pick(U32,ks,6,2)),0,2}) ==
   R.rook(E.choose(white,ks)) : U64}:
  match white ks:
    case False{} False{}: {==}
    case False{} True{}: {==}
    case True{} False{}: {==}
    case True{} True{}: {==}

def guarded_clear(+b: Chess.Board,+ks: Bool,
  rows: {B.valid(b) == True{} : Bool},guard: {E.guard(b,ks) == True{} : Bool}) ->
  {U64.and(Chess.get_kings(b),landing(E.move(b,ks))) == U64.zero() : U64}:
  Equal.trans(U64,U64.and(Chess.get_kings(b),landing(E.move(b,ks))),
    U64.and(Chess.get_kings(b),R.rook(E.route(b,ks))),U64.zero(),
    Equal.cong(U64,U64,x => U64.and(Chess.get_kings(b),x),landing(E.move(b,ks)),R.rook(E.route(b,ks)),
      route_landing(U32.is_eq(Chess.get_turn(b),1),ks)),
    Fresh.board(b,R.rook(E.route(b,ks)),rows,G.fresh(b,ks,guard)))

def both(-b: Chess.Board,-m: Chess.Ply) -> Type:
  {U64.and(Chess.get_kings(Chess.make_move(b,m)),Chess.get_white(Chess.make_move(b,m))) ==
   Frame.model(b,m,Chess.get_white(b),U32.is_eq(Chess.get_turn(b),1)) : U64} &
  {U64.and(Chess.get_kings(Chess.make_move(b,m)),Chess.get_black(Chess.make_move(b,m))) ==
   Frame.model(b,m,Chess.get_black(b),Bool.not(U32.is_eq(Chess.get_turn(b),1))) : U64}

# Public guard-derived bridge: arbitrary board, either route, arbitrary raw turn.
def guarded(+b: Chess.Board,+ks: Bool,
  +rows: {B.valid(b) == True{} : Bool},+guard: {E.guard(b,ks) == True{} : Bool}) -> both(b,E.move(b,ks)):
  +empty=guarded_clear(b,ks,rows,guard)
  (white_actual(b,E.move(b,ks),empty),black_actual(b,E.move(b,ks),empty))
