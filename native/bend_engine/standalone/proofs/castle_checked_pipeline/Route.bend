# Actual producer coordinates selected from the certified original turn.
import Base
import ../../../legal_probe/Chess.bend as Chess
import ../castle_kings/Spec.bend as K
import ../castle_kings/Projection.bend as P
import ../castle_emission/Spec.bend as E

def source(+b: Chess.Board,+white: Bool,turn: {Chess.get_turn(b) == Bool.to_u32(white) : U32}) ->
  {E.src(b) == K.src(white) : U32}:
  Equal.cong(Bool,U32,x => U32.add(Bool.pick(U32,x,0,56),4),U32.is_eq(Chess.get_turn(b),1),white,P.white_turn(b,white,turn))

def destination(+b: Chess.Board,+white: Bool,+ks: Bool,turn: {Chess.get_turn(b) == Bool.to_u32(white) : U32}) ->
  {E.dst(b,ks) == K.dst(white,ks) : U32}:
  Equal.cong(Bool,U32,x => U32.add(Bool.pick(U32,x,0,56),Bool.pick(U32,ks,6,2)),U32.is_eq(Chess.get_turn(b),1),white,P.white_turn(b,white,turn))

def transit(+b: Chess.Board,+white: Bool,+ks: Bool,turn: {Chess.get_turn(b) == Bool.to_u32(white) : U32}) ->
  {E.transit(b,ks) == K.transit(white,ks) : U32}:
  Equal.cong(Bool,U32,x => U32.add(Bool.pick(U32,x,0,56),Bool.pick(U32,ks,5,3)),U32.is_eq(Chess.get_turn(b),1),white,P.white_turn(b,white,turn))

def selected(+b: Chess.Board,+white: Bool,+ks: Bool,-table: Array<U64>,+tail: List<&2,Chess.Ply>,
  +turn: {Chess.get_turn(b) == Bool.to_u32(white) : U32}) ->
  {Chess.castle_side(b,ks,(table,tail)) ==
   Chess.castle_checked(E.guard(b,ks),table,b,K.src(white),K.dst(white,ks),K.transit(white,ks),tail) : Array<U64> & List<&2,Chess.Ply>}:
  %Equal.sym(U32,E.src(b),K.src(white),source(b,white,turn)) :
    {Chess.castle_checked(E.guard(b,ks),table,b,_,E.dst(b,ks),E.transit(b,ks),tail) ==
     Chess.castle_checked(E.guard(b,ks),table,b,K.src(white),K.dst(white,ks),K.transit(white,ks),tail) : Array<U64> & List<&2,Chess.Ply>}
  %Equal.sym(U32,E.dst(b,ks),K.dst(white,ks),destination(b,white,ks,turn)) :
    {Chess.castle_checked(E.guard(b,ks),table,b,K.src(white),_,E.transit(b,ks),tail) ==
     Chess.castle_checked(E.guard(b,ks),table,b,K.src(white),K.dst(white,ks),K.transit(white,ks),tail) : Array<U64> & List<&2,Chess.Ply>}
  Equal.cong(U32,Array<U64> & List<&2,Chess.Ply>,x => Chess.castle_checked(E.guard(b,ks),table,b,K.src(white),K.dst(white,ks),x,tail),
    E.transit(b,ks),K.transit(white,ks),transit(b,white,ks,turn))
