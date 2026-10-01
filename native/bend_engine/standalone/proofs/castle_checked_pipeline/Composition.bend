import Base
import ../../../legal_probe/Chess.bend as Chess
import ../castle_kings/Spec.bend as K
import ../castle_emission/Spec.bend as E
import ../board/Spec.bend as B
import ../initialized_attacks/Build.bend as Init
import ./Spec.bend as S
import ./Certificates.bend as C
import ./Wire.bend as W
import ./Decision.bend as D

def producer(+d: Nat,+seed: U64,+n: Nat,+extra: Nat,+b: Chess.Board,+white: Bool,+ks: Bool,+tail: List<&2,Chess.Ply>,
  +depth: {d == 17n : Nat},+tables: {n == 128n : Nat},+full: {extra == 64n : Nat},
  +good: {B.valid(b) == True{} : Bool},+turn: {Chess.get_turn(b) == Bool.to_u32(white) : U32},
  +one: {K.kings(b,white) == U64.bit(U32.to_nat(K.src(white))) : U64},+guard: {E.guard(b,ks) == True{} : Bool}) ->
  {Chess.castle_side(b,ks,(Init.run(d,seed,n,extra),tail)) == (Init.run(d,seed,n,extra),S.keep(Bool.or(S.check(b,white,ks,K.Initial{}),S.check(b,white,ks,K.Transit{})),S.move(white,ks),tail)) : Array<U64> & List<&2,Chess.Ply>}:
  W.producer(b,white,ks,Init.run(d,seed,n,extra),tail,S.check(b,white,ks,K.Initial{}),S.check(b,white,ks,K.Transit{}),turn,guard,
    C.at(d,seed,n,extra,b,white,ks,K.Initial{},depth,tables,full,good,turn,one,guard),C.at(d,seed,n,extra,b,white,ks,K.Transit{},depth,tables,full,good,turn,one,guard))

def pipeline(+d: Nat,+seed: U64,+n: Nat,+extra: Nat,+b: Chess.Board,+white: Bool,+ks: Bool,
  +depth: {d == 17n : Nat},+tables: {n == 128n : Nat},+full: {extra == 64n : Nat},
  +good: {B.valid(b) == True{} : Bool},+turn: {Chess.get_turn(b) == Bool.to_u32(white) : U32},
  +one: {K.kings(b,white) == U64.bit(U32.to_nat(K.src(white))) : U64},+guard: {E.guard(b,ks) == True{} : Bool}) ->
  {S.run(Init.run(d,seed,n,extra),b,ks) == (Init.run(d,seed,n,extra),S.keep(S.rejected(b,white,ks),S.move(white,ks),Nil{})) : Array<U64> & List<&2,Chess.Ply>}:
  W.pipeline(b,white,ks,Init.run(d,seed,n,extra),Bool.or(S.check(b,white,ks,K.Initial{}),S.check(b,white,ks,K.Transit{})),S.check(b,white,ks,K.Completed{}),
    producer(d,seed,n,extra,b,white,ks,Nil{},depth,tables,full,good,turn,one,guard),
    C.at(d,seed,n,extra,b,white,ks,K.Completed{},depth,tables,full,good,turn,one,guard))

def accepted(+d: Nat,+seed: U64,+n: Nat,+extra: Nat,+b: Chess.Board,+white: Bool,+ks: Bool,+m: Chess.Ply,
  +depth: {d == 17n : Nat},+tables: {n == 128n : Nat},+full: {extra == 64n : Nat},
  +good: {B.valid(b) == True{} : Bool},+turn: {Chess.get_turn(b) == Bool.to_u32(white) : U32},
  +one: {K.kings(b,white) == U64.bit(U32.to_nat(K.src(white))) : U64},+guard: {E.guard(b,ks) == True{} : Bool},
  here: E.member(m,E.moves(S.run(Init.run(d,seed,n,extra),b,ks)))) ->
  {S.check(b,white,ks,K.Initial{}) == False{} : Bool} & ({S.check(b,white,ks,K.Transit{}) == False{} : Bool} & {S.check(b,white,ks,K.Completed{}) == False{} : Bool}):
  D.result(S.run(Init.run(d,seed,n,extra),b,ks),Init.run(d,seed,n,extra),S.check(b,white,ks,K.Initial{}),S.check(b,white,ks,K.Transit{}),S.check(b,white,ks,K.Completed{}),m,S.move(white,ks),
    pipeline(d,seed,n,extra,b,white,ks,depth,tables,full,good,turn,one,guard),here)
