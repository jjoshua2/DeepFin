# Transport derived route certificates to the existing stage-check interface.
import Base
import ./Spec.bend as S
import ../castle_emission/Spec.bend as E
import ../castle_chain/Spec.bend as C
import ../castle_kings/Spec.bend as K
import ../castle_kings/Projection.bend as Projection
import ../../../legal_probe/Chess.bend as Chess

def square(m: Chess.Ply) -> U32:
  Chess.Ply{s,d,p,f} = m
  d

def move(+white: Bool,ks: Bool) -> Chess.Ply:
  Chess.Ply{K.src(white),K.dst(white,ks),0,2}

def route(+b: Chess.Board,+white: Bool,+ks: Bool,
  turn: {Chess.get_turn(b) == Bool.to_u32(white) : U32}) ->
  {E.move(b,ks) == move(white,ks) : Chess.Ply}:
  Equal.cong(Bool,Chess.Ply,w => move(w,ks),U32.is_eq(Chess.get_turn(b),1),white,
    Projection.white_turn(b,white,turn))

def decide(-a: Array<U64>,+b: Chess.Board,+white: Bool,+ks: Bool,+m: Chess.Ply,
  same: {m == move(white,ks) : Chess.Ply},turn: {Chess.get_turn(b) == Bool.to_u32(white) : U32},
  no: {S.answer(Chess.in_check(a,Chess.make_move(b,m),Chess.get_turn(b))) == False{} : Bool}) ->
  {S.answer(Chess.in_check(a,K.final(b,white,ks),Bool.to_u32(white))) == False{} : Bool}:
  %turn : {S.answer(Chess.in_check(a,K.final(b,white,ks),_)) == False{} : Bool}
  %same : {S.answer(Chess.in_check(a,Chess.make_move(b,_),Chess.get_turn(b))) == False{} : Bool}
  no
