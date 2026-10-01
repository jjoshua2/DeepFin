# Equality bridge; the public theorem constructs this query certificate itself.
import Base
import ./Spec.bend as S
import ../castle_kings/Spec.bend as K
import ../initialized_attacks/Spec.bend as G
import ../../../legal_probe/Chess.bend as Chess

def stage(-a: Array<U64>,+b: Chess.Board,+white: Bool,+ks: Bool,
  result: {Chess.in_check(a,K.final(b,white,ks),Bool.to_u32(white)) ==
    (a,G.attacked(K.final(b,white,ks),U32.to_nat(K.dst(white,ks)),Bool.not(white))) : Array<U64> & Bool},
  no: {S.answer(Chess.in_check(a,K.final(b,white,ks),Bool.to_u32(white))) == False{} : Bool}) ->
  {G.attacked(K.final(b,white,ks),U32.to_nat(K.dst(white,ks)),Bool.not(white)) == False{} : Bool}:
  Equal.trans(Bool,G.attacked(K.final(b,white,ks),U32.to_nat(K.dst(white,ks)),Bool.not(white)),
    S.answer(Chess.in_check(a,K.final(b,white,ks),Bool.to_u32(white))),False{},
    Equal.sym(Bool,S.answer(Chess.in_check(a,K.final(b,white,ks),Bool.to_u32(white))),
      G.attacked(K.final(b,white,ks),U32.to_nat(K.dst(white,ks)),Bool.not(white)),
      Equal.cong(Array<U64> & Bool,Bool,S.answer,
        Chess.in_check(a,K.final(b,white,ks),Bool.to_u32(white)),
        (a,G.attacked(K.final(b,white,ks),U32.to_nat(K.dst(white,ks)),Bool.not(white))),result)),no)
