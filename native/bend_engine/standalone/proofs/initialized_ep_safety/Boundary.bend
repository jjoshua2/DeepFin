# Transport exact board and full-Ply fixtures without expanding their legal list.
import Base
import ./consumer.bend as Safety
import ./Witnesses.bend as W
import ../accepted_ep/Boards.bend as B
import ../board/Spec.bend as Board
import ../ep_king_frame/Frame.bend as F
import ../initialized_attacks/Build.bend as Init
import ../initialized_attacks/Spec.bend as G
import ../castle_emission/Spec.bend as E
import ../castle_chain/Spec.bend as C
import ../u64/Words.bend as Words
import ../../../legal_probe/Chess.bend as Chess
import ../../Position.bend as Position

def premises(+b: Chess.Board) -> Type:
  ({Board.valid(b) == True{} : Bool} &
  ({Position.valid_ep(Chess.get_ep(b),b) == True{} : Bool} &
  ({Chess.get_turn(b) == Bool.to_u32(True{}) : U32} &
  {F.plane(b,Bool.to_u32(True{})) == U64.bit(32n) : U64})))

def certificates(+b: Chess.Board,+m: Chess.Ply,
  same: {b == B.discovered() : Chess.Board},
  record: {m == Chess.Ply{36,43,0,1} : Chess.Ply}) ->
  premises(b) & ({C.flag(m) == 1 : U32} &
  {G.attacked(Chess.make_move(b,m),32n,False{}) == True{} : Bool}):
  %Equal.sym(Chess.Ply,m,Chess.Ply{36,43,0,1},record) :
    premises(b) & ({C.flag(_) == 1 : U32} &
    {G.attacked(Chess.make_move(b,_),32n,False{}) == True{} : Bool})
  %Equal.sym(Chess.Board,b,B.discovered(),same) :
    premises(_) & ({C.flag(Chess.Ply{36,43,0,1}) == 1 : U32} &
    {G.attacked(Chess.make_move(_,Chess.Ply{36,43,0,1}),32n,False{}) == True{} : Bool})
  (W.discovered_parent(),({==},W.discovered_child_attacked()))

def exclude(a: Array<U64>,+d: Nat,+seed: U64,+n: Nat,+extra: Nat,+b: Chess.Board,+m: Chess.Ply,
  initialized: {a == Init.run(d,seed,n,extra) : Array<U64>},
  depth: {d == 17n : Nat},tables: {n == 128n : Nat},full: {extra == 64n : Nat},
  here: E.member(m,E.moves(Chess.legal_moves(a,b))),
  cert: premises(b) & ({C.flag(m) == 1 : U32} &
    {G.attacked(Chess.make_move(b,m),32n,False{}) == True{} : Bool})) -> Empty:
  ((good,(coherent,(turn,one))),(tag,hit)) = cert
  +no: {G.attacked(Chess.make_move(b,m),32n,False{}) == False{} : Bool} =
    Safety.use_generated(a,d,seed,n,extra,b,m,32n,True{},
      initialized,depth,tables,full,good,coherent,turn,{==},one,here,tag)
  Words.false_true(Empty,Equal.trans(Bool,False{},
    G.attacked(Chess.make_move(b,m),32n,False{}),True{},
    Equal.sym(Bool,G.attacked(Chess.make_move(b,m),32n,False{}),False{},no),hit))

def discovered_excluded(a: Array<U64>,+d: Nat,+seed: U64,+n: Nat,+extra: Nat,+b: Chess.Board,+m: Chess.Ply,
  initialized: {a == Init.run(d,seed,n,extra) : Array<U64>},
  depth: {d == 17n : Nat},tables: {n == 128n : Nat},full: {extra == 64n : Nat},
  same: {b == B.discovered() : Chess.Board},
  record: {m == Chess.Ply{36,43,0,1} : Chess.Ply},
  here: E.member(m,E.moves(Chess.legal_moves(a,b)))) -> Empty:
  exclude(a,d,seed,n,extra,b,m,initialized,depth,tables,full,here,certificates(b,m,same,record))
