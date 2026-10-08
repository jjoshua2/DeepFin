# Reuse the checked two-plane fusion with the actual king insertion selector.
import Base
import ../pawn_counts/Fusion.bend as F
import ./Selector.bend as S
import ../color_counts/Actual.bend as A
import ../backrank_pawns/Mask.bend as M
import ../../../legal_probe/Chess.bend as Chess

def model(+b: Chess.Board,+m: Chess.Ply,+color: U64,side: Bool) -> U64:
  Chess.update_piece(U64.and(Chess.get_kings(b),color),M.remove(m),U64.bit(U32.to_nat(A.destination(m))),Bool.and(S.put(b,m),side))
def king_actual(+b: Chess.Board,m: Chess.Ply) ->
  {Chess.get_kings(Chess.make_move(b,m)) == Chess.update_piece(Chess.get_kings(b),M.remove(m),U64.bit(U32.to_nat(A.destination(m))),S.put(b,m)) : U64}:
  match m:
    case Chess.Ply{s,d,p,f}: {==}
def raw_fusion(+b: Chess.Board,m: Chess.Ply,+color: U64,+side: Bool) ->
  {U64.and(Chess.update_piece(Chess.get_kings(b),M.remove(m),U64.bit(U32.to_nat(A.destination(m))),S.put(b,m)),
    Chess.update_piece(color,M.remove(m),U64.bit(U32.to_nat(A.destination(m))),side)) == model(b,m,color,side) : U64}:
  match m:
    case Chess.Ply{+s,+d,+p,+f}:
      F.fused(Chess.get_kings(b),color,U64.bit(U32.to_nat(s)),
        U64.bit(U32.to_nat(Bool.pick(U32,U32.is_eq(f,1),U32.xor(d,8),d))),U64.bit(U32.to_nat(d)),
        Chess.select_u64(U32.is_eq(f,2),U64.bit(U32.to_nat(Bool.pick(U32,U32.is_eq(U32.and(d,7),6),U32.inc(d),U32.sub(d,2)))),U64.zero()),
        S.put(b,Chess.Ply{s,d,p,f}),side)
def frame(+b: Chess.Board,+m: Chess.Ply,+color: U64,+side: Bool,no: {U32.is_eq(A.flag(m),2) == False{} : Bool}) ->
  {U64.and(Chess.get_kings(Chess.make_move(b,m)),Chess.update_piece(color,M.remove(m),A.added(m),side)) == model(b,m,color,side) : U64}:
  %Equal.sym(U64,Chess.get_kings(Chess.make_move(b,m)),Chess.update_piece(Chess.get_kings(b),M.remove(m),U64.bit(U32.to_nat(A.destination(m))),S.put(b,m)),king_actual(b,m)) :
    {U64.and(_,Chess.update_piece(color,M.remove(m),A.added(m),side)) == model(b,m,color,side) : U64}
  %Equal.sym(U64,A.added(m),U64.bit(U32.to_nat(A.destination(m))),A.noncastle_added(m,no)) :
    {U64.and(Chess.update_piece(Chess.get_kings(b),M.remove(m),U64.bit(U32.to_nat(A.destination(m))),S.put(b,m)),
      Chess.update_piece(color,M.remove(m),_,side)) == model(b,m,color,side) : U64}
  raw_fusion(b,m,color,side)
def white_actual(+b: Chess.Board,+m: Chess.Ply,no: {U32.is_eq(A.flag(m),2) == False{} : Bool}) ->
  {U64.and(Chess.get_kings(Chess.make_move(b,m)),Chess.get_white(Chess.make_move(b,m))) ==
    model(b,m,Chess.get_white(b),U32.is_eq(Chess.get_turn(b),1)) : U64}:
  Equal.trans(U64,U64.and(Chess.get_kings(Chess.make_move(b,m)),Chess.get_white(Chess.make_move(b,m))),
    U64.and(Chess.get_kings(Chess.make_move(b,m)),Chess.update_piece(Chess.get_white(b),M.remove(m),A.added(m),U32.is_eq(Chess.get_turn(b),1))),
    model(b,m,Chess.get_white(b),U32.is_eq(Chess.get_turn(b),1)),
    Equal.cong(U64,U64,x => U64.and(Chess.get_kings(Chess.make_move(b,m)),x),Chess.get_white(Chess.make_move(b,m)),
      Chess.update_piece(Chess.get_white(b),M.remove(m),A.added(m),U32.is_eq(Chess.get_turn(b),1)),A.white_actual(b,m)),
    frame(b,m,Chess.get_white(b),U32.is_eq(Chess.get_turn(b),1),no))
def black_actual(+b: Chess.Board,+m: Chess.Ply,no: {U32.is_eq(A.flag(m),2) == False{} : Bool}) ->
  {U64.and(Chess.get_kings(Chess.make_move(b,m)),Chess.get_black(Chess.make_move(b,m))) ==
    model(b,m,Chess.get_black(b),Bool.not(U32.is_eq(Chess.get_turn(b),1))) : U64}:
  Equal.trans(U64,U64.and(Chess.get_kings(Chess.make_move(b,m)),Chess.get_black(Chess.make_move(b,m))),
    U64.and(Chess.get_kings(Chess.make_move(b,m)),Chess.update_piece(Chess.get_black(b),M.remove(m),A.added(m),Bool.not(U32.is_eq(Chess.get_turn(b),1)))),
    model(b,m,Chess.get_black(b),Bool.not(U32.is_eq(Chess.get_turn(b),1))),
    Equal.cong(U64,U64,x => U64.and(Chess.get_kings(Chess.make_move(b,m)),x),Chess.get_black(Chess.make_move(b,m)),
      Chess.update_piece(Chess.get_black(b),M.remove(m),A.added(m),Bool.not(U32.is_eq(Chess.get_turn(b),1))),A.black_actual(b,m)),
    frame(b,m,Chess.get_black(b),Bool.not(U32.is_eq(Chess.get_turn(b),1)),no))
