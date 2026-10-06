# Symbolic fixture identities connect the actual retained theorem to a forward hit.
import Base
import ./Retained.bend as Retained
import ./Transport.bend as Transport
import ../initialized_ep_safety/Boundary.bend as Original
import ../accepted_ep/Boards.bend as B
import ../castle_emission/Spec.bend as E
import ../castle_chain/Spec.bend as C
import ../initialized_attacks/Build.bend as Init
import ../initialized_attacks/Spec.bend as Geometry
import ../forward_attacks/Spec.bend as Forward
import ../u64/Words.bend as Words
import ../../../legal_probe/Chess.bend as Chess

def exclude(a: Array<U64>,+d: Nat,+seed: U64,+n: Nat,+extra: Nat,+b: Chess.Board,+m: Chess.Ply,
  initialized: {a == Init.run(d,seed,n,extra) : Array<U64>},
  depth: {d == 17n : Nat},tables: {n == 128n : Nat},full: {extra == 64n : Nat},
  here: E.member(m,E.moves(Chess.legal_moves(a,b))),
  cert: Original.premises(b) & ({C.flag(m) == 1 : U32} &
    {Geometry.attacked(Chess.make_move(b,m),32n,False{}) == True{} : Bool})) -> Empty:
  ((good,(coherent,(turn,one))),(tag,hit)) = cert
  +no: {Forward.attacked(Chess.make_move(b,m),32n,False{}) == False{} : Bool} =
    Retained.use_generated(a,d,seed,n,extra,b,m,32n,True{},
      initialized,depth,tables,full,good,coherent,turn,{==},one,here,tag)
  +yes: {Forward.attacked(Chess.make_move(b,m),32n,False{}) == True{} : Bool} =
    Transport.hit(Chess.make_move(b,m),32n,False{},{==},hit)
  Words.false_true(Empty,Equal.trans(Bool,False{},
    Forward.attacked(Chess.make_move(b,m),32n,False{}),True{},
    Equal.sym(Bool,Forward.attacked(Chess.make_move(b,m),32n,False{}),False{},no),yes))

def discovered_excluded(a: Array<U64>,+d: Nat,+seed: U64,+n: Nat,+extra: Nat,+b: Chess.Board,+m: Chess.Ply,
  initialized: {a == Init.run(d,seed,n,extra) : Array<U64>},
  depth: {d == 17n : Nat},tables: {n == 128n : Nat},full: {extra == 64n : Nat},
  same: {b == B.discovered() : Chess.Board},
  record: {m == Chess.Ply{36,43,0,1} : Chess.Ply},
  here: E.member(m,E.moves(Chess.legal_moves(a,b)))) -> Empty:
  exclude(a,d,seed,n,extra,b,m,initialized,depth,tables,full,here,Original.certificates(b,m,same,record))
