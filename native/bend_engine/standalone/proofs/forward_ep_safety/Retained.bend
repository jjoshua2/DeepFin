# The actual retained EP proof supplies the geometry fact; no desired answer is assumed.
import Base
import ./Transport.bend as Transport
import ../initialized_ep_safety/consumer.bend as EP
import ../board/Spec.bend as Board
import ../ep_king_frame/Frame.bend as F
import ../castle_emission/Spec.bend as E
import ../castle_chain/Spec.bend as C
import ../initialized_attacks/Build.bend as Init
import ../forward_attacks/Spec.bend as Forward
import ../../../legal_probe/Chess.bend as Chess
import ../../Position.bend as Position

def use_generated(a: Array<U64>,+d: Nat,+seed: U64,+n: Nat,+extra: Nat,+b: Chess.Board,+m: Chess.Ply,+q: Nat,+side: Bool,
  initialized: {a == Init.run(d,seed,n,extra) : Array<U64>},
  depth: {d == 17n : Nat},tables: {n == 128n : Nat},full: {extra == 64n : Nat},
  good: {Board.valid(b) == True{} : Bool},
  coherent: {Position.valid_ep(Chess.get_ep(b),b) == True{} : Bool},
  turn: {Chess.get_turn(b) == Bool.to_u32(side) : U32},
  +bound: {Nat.is_lt(q,64n) == True{} : Bool},
  one: {F.plane(b,Bool.to_u32(side)) == U64.bit(q) : U64},
  here: E.member(m,E.moves(Chess.legal_moves(a,b))),tag: {C.flag(m) == 1 : U32}) ->
  {Forward.attacked(Chess.make_move(b,m),q,Bool.not(side)) == False{} : Bool}:
  Transport.clear(Chess.make_move(b,m),q,Bool.not(side),bound,
    EP.use_generated(a,d,seed,n,extra,b,m,q,side,initialized,depth,tables,full,good,coherent,turn,bound,one,here,tag))
