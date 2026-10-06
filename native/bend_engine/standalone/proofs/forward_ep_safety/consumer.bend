# Public retained-EP forward witness statements on the exact affine input table.
import Base
import ./Retained.bend as Retained
import ./Boundary.bend as Boundary
import ../board/Spec.bend as Board
import ../ep_king_frame/Frame.bend as F
import ../castle_emission/Spec.bend as E
import ../castle_chain/Spec.bend as C
import ../initialized_attacks/Build.bend as Init
import ../forward_attacks/Spec.bend as Forward
import ../accepted_ep/Boards.bend as B
import ../../../legal_probe/Chess.bend as Chess
import ../../Position.bend as Position

def use_generated(a: Array<U64>,+d: Nat,+seed: U64,+n: Nat,+extra: Nat,+b: Chess.Board,+m: Chess.Ply,+q: Nat,+side: Bool,
  initialized: {a == Init.run(d,seed,n,extra) : Array<U64>},
  depth: {d == 17n : Nat},tables: {n == 128n : Nat},full: {extra == 64n : Nat},
  good: {Board.valid(b) == True{} : Bool},
  coherent: {Position.valid_ep(Chess.get_ep(b),b) == True{} : Bool},
  turn: {Chess.get_turn(b) == Bool.to_u32(side) : U32},
  bound: {Nat.is_lt(q,64n) == True{} : Bool},
  one: {F.plane(b,Bool.to_u32(side)) == U64.bit(q) : U64},
  here: E.member(m,E.moves(Chess.legal_moves(a,b))),tag: {C.flag(m) == 1 : U32}) ->
  {Forward.attacked(Chess.make_move(b,m),q,Bool.not(side)) == False{} : Bool}:
  Retained.use_generated(a,d,seed,n,extra,b,m,q,side,initialized,depth,tables,full,good,coherent,turn,bound,one,here,tag)

def discovered_excluded(a: Array<U64>,+d: Nat,+seed: U64,+n: Nat,+extra: Nat,+b: Chess.Board,+m: Chess.Ply,
  initialized: {a == Init.run(d,seed,n,extra) : Array<U64>},
  depth: {d == 17n : Nat},tables: {n == 128n : Nat},full: {extra == 64n : Nat},
  same: {b == B.discovered() : Chess.Board},
  record: {m == Chess.Ply{36,43,0,1} : Chess.Ply},
  here: E.member(m,E.moves(Chess.legal_moves(a,b)))) -> Empty:
  Boundary.discovered_excluded(a,d,seed,n,extra,b,m,initialized,depth,tables,full,same,record,here)
