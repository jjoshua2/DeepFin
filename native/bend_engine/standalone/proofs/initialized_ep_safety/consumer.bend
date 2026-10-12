# Public bridge for one affine input table with its exact initialization identity.
import Base
import ./Geometry.bend as Safety
import ../board/Spec.bend as Board
import ../ep_king_frame/Frame.bend as F
import ../castle_emission/Spec.bend as E
import ../castle_chain/Spec.bend as C
import ../initialized_attacks/Build.bend as Init
import ../initialized_attacks/Spec.bend as G
import ../../../legal_probe/Chess.bend as Chess
import ../../Position.bend as Position

def lift(-a: Array<U64>,+d: Nat,+seed: U64,+n: Nat,+extra: Nat,+b: Chess.Board,+m: Chess.Ply,+q: Nat,+side: Bool,
  initialized: {a == Init.run(d,seed,n,extra) : Array<U64>},
  depth: {d == 17n : Nat},tables: {n == 128n : Nat},full: {extra == 64n : Nat},
  good: {Board.valid(b) == True{} : Bool},
  coherent: {Position.valid_ep(Chess.get_ep(b),b) == True{} : Bool},
  turn: {Chess.get_turn(b) == Bool.to_u32(side) : U32},
  bound: {Nat.is_lt(q,64n) == True{} : Bool},
  one: {F.plane(b,Bool.to_u32(side)) == U64.bit(q) : U64}) ->
  E.member(m,E.moves(Chess.legal_moves(a,b))) -> {C.flag(m) == 1 : U32} ->
  {G.attacked(Chess.make_move(b,m),q,Bool.not(side)) == False{} : Bool}:
  %Equal.sym(Array<U64>,a,Init.run(d,seed,n,extra),initialized) :
    E.member(m,E.moves(Chess.legal_moves(_,b))) -> {C.flag(m) == 1 : U32} ->
    {G.attacked(Chess.make_move(b,m),q,Bool.not(side)) == False{} : Bool}
  here => tag => Safety.actual(d,seed,n,extra,b,m,q,side,depth,tables,full,good,coherent,turn,bound,one,here,tag)

def use_generated(a: Array<U64>,+d: Nat,+seed: U64,+n: Nat,+extra: Nat,+b: Chess.Board,+m: Chess.Ply,+q: Nat,+side: Bool,
  initialized: {a == Init.run(d,seed,n,extra) : Array<U64>},
  depth: {d == 17n : Nat},tables: {n == 128n : Nat},full: {extra == 64n : Nat},
  good: {Board.valid(b) == True{} : Bool},
  coherent: {Position.valid_ep(Chess.get_ep(b),b) == True{} : Bool},
  turn: {Chess.get_turn(b) == Bool.to_u32(side) : U32},
  bound: {Nat.is_lt(q,64n) == True{} : Bool},
  one: {F.plane(b,Bool.to_u32(side)) == U64.bit(q) : U64},
  here: E.member(m,E.moves(Chess.legal_moves(a,b))),tag: {C.flag(m) == 1 : U32}) ->
  {G.attacked(Chess.make_move(b,m),q,Bool.not(side)) == False{} : Bool}:
  lift(a,d,seed,n,extra,b,m,q,side,initialized,depth,tables,full,good,coherent,turn,bound,one)(here)(tag)
