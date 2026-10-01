import Base
import ../../../legal_probe/Chess.bend as Chess
import ../castle_kings/Spec.bend as K
import ../board/Spec.bend as B
import ../castle_emission/Spec.bend as E
import ../initialized_attacks/Build.bend as Init
import ./Spec.bend as S
import ./Runtime.bend as Runtime
import ./Wire.bend as Wire
import ./Bind.bend as Bind
import ./Checks.bend as Checks

def producer(+d: Nat,+seed: U64,+n: Nat,+extra: Nat,+b: Chess.Board,+white: Bool,+ks: Bool,
  +depth: {d == 17n : Nat},+tables: {n == 128n : Nat},+full: {extra == 64n : Nat},
  +good: {B.valid(b) == True{} : Bool},+turn: {Chess.get_turn(b) == Bool.to_u32(white) : U32},
  +one: {K.kings(b,white) == U64.bit(U32.to_nat(K.src(white))) : U64},+guard: {E.guard(b,ks) == True{} : Bool},+tail: List<&2,Chess.Ply>) ->
  {Chess.castle_side(b,ks,(Init.run(d,seed,n,extra),tail)) == (Init.run(d,seed,n,extra),S.before(b,white,ks,tail)) : Array<U64> & List<&2,Chess.Ply>}:
  Equal.trans(Array<U64> & List<&2,Chess.Ply>,
    Chess.castle_side(b,ks,(Init.run(d,seed,n,extra),tail)),
    Chess.castle_checked(True{},Init.run(d,seed,n,extra),b,K.src(white),K.dst(white,ks),K.transit(white,ks),tail),
    (Init.run(d,seed,n,extra),S.before(b,white,ks,tail)),
    Bind.side(Init.run(d,seed,n,extra),b,white,ks,tail,turn,guard),
    Wire.producer(Init.run(d,seed,n,extra),b,K.src(white),K.dst(white,ks),K.transit(white,ks),tail,
      S.checked(b,white,ks,K.Initial{}),S.checked(b,white,ks,K.Transit{}),
      Checks.stage(d,seed,n,extra,b,white,ks,K.Initial{},depth,tables,full,good,turn,one,guard),
      Checks.stage(d,seed,n,extra,b,white,ks,K.Transit{},depth,tables,full,good,turn,one,guard)))

def filtered(+d: Nat,+seed: U64,+n: Nat,+extra: Nat,+b: Chess.Board,+white: Bool,+ks: Bool,
  +depth: {d == 17n : Nat},+tables: {n == 128n : Nat},+full: {extra == 64n : Nat},
  +good: {B.valid(b) == True{} : Bool},+turn: {Chess.get_turn(b) == Bool.to_u32(white) : U32},
  +one: {K.kings(b,white) == U64.bit(U32.to_nat(K.src(white))) : U64},+guard: {E.guard(b,ks) == True{} : Bool}) ->
  {Runtime.run(Init.run(d,seed,n,extra),b,ks) == (Init.run(d,seed,n,extra),S.after(b,white,ks)) : Array<U64> & List<&2,Chess.Ply>}:
  Equal.trans(Array<U64> & List<&2,Chess.Ply>,
    Runtime.run(Init.run(d,seed,n,extra),b,ks),Runtime.finish(b,(Init.run(d,seed,n,extra),S.before(b,white,ks,Nil{}))),
    (Init.run(d,seed,n,extra),S.after(b,white,ks)),
    Equal.cong(Array<U64> & List<&2,Chess.Ply>,Array<U64> & List<&2,Chess.Ply>,r => Runtime.finish(b,r),
      Chess.castle_side(b,ks,(Init.run(d,seed,n,extra),Nil{})),(Init.run(d,seed,n,extra),S.before(b,white,ks,Nil{})),
      producer(d,seed,n,extra,b,white,ks,depth,tables,full,good,turn,one,guard,Nil{})),
    Wire.filter_one(Init.run(d,seed,n,extra),b,S.move(white,ks),S.checked(b,white,ks,K.Initial{}),S.checked(b,white,ks,K.Transit{}),S.checked(b,white,ks,K.Completed{}),
      Checks.stage(d,seed,n,extra,b,white,ks,K.Completed{},depth,tables,full,good,turn,one,guard)))
