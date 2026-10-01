# Complete statements, with the existing castling dependency bodies retained.
import Base
import ../generator_contract/castle_consumer.bend as QualifiedCastles
import ../../../legal_probe/Chess.bend as Chess
import ../board/Spec.bend as Board
import ../castle_kings/Spec.bend as K
import ../initialized_attacks/Build.bend as Init
import ../initialized_attacks/Spec.bend as Geometry
import ../generator_contract/CastleSpec.bend as Projection
import ./Spec.bend as Forward
import ./Bridge.bend as Bridge
import ./Actual.bend as Actual
import ./Castles.bend as Castles

def use_geometry(+b: Chess.Board,+target: Nat,+by: Bool,
  bound: {Nat.is_lt(target,64n) == True{} : Bool}) ->
  {Geometry.attacked(b,target,by) == Forward.attacked(b,target,by) : Bool}:
  Bridge.geometry(b,target,by,bound)

def use_initialized_attack(+d: Nat,+seed: U64,+n: Nat,+extra: Nat,+b: Chess.Board,+target: Nat,+by: Bool,
  depth: {d == 17n : Nat},tables: {n == 128n : Nat},full: {extra == 64n : Nat},
  bound: {Nat.is_lt(target,64n) == True{} : Bool}) ->
  {Chess.attacked(Init.run(d,seed,n,extra),b,U32.from_nat(target),Bool.to_u32(by)) ==
    (Init.run(d,seed,n,extra),Forward.attacked(b,target,by)) : Array<U64> & Bool}:
  Actual.attacked(d,seed,n,extra,b,target,by,depth,tables,full,bound)

def use_initialized_check(+d: Nat,+seed: U64,+n: Nat,+extra: Nat,+b: Chess.Board,+target: Nat,+side: Bool,
  depth: {d == 17n : Nat},tables: {n == 128n : Nat},full: {extra == 64n : Nat},
  bound: {Nat.is_lt(target,64n) == True{} : Bool},
  one: {U64.and(Chess.get_kings(b),Chess.color(b,Bool.to_u32(side))) == U64.bit(target) : U64}) ->
  {Chess.in_check(Init.run(d,seed,n,extra),b,Bool.to_u32(side)) ==
    (Init.run(d,seed,n,extra),Forward.attacked(b,target,Bool.not(side))) : Array<U64> & Bool}:
  Actual.checked(d,seed,n,extra,b,target,side,depth,tables,full,bound,one)

def use_forward_castle_subset(+d: Nat,+seed: U64,+n: Nat,+extra: Nat,+b: Chess.Board,+white: Bool,
  depth: {d == 17n : Nat},tables: {n == 128n : Nat},full: {extra == 64n : Nat},
  good: {Board.valid(b) == True{} : Bool},turn: {Chess.get_turn(b) == Bool.to_u32(white) : U32},
  one: {K.kings(b,white) == U64.bit(U32.to_nat(K.src(white))) : U64}) ->
  {Projection.result(Chess.legal_moves(Init.run(d,seed,n,extra),b)) ==
    (Init.run(d,seed,n,extra),Castles.expected(b,white)) : Array<U64> & List<&2,Chess.Ply>}:
  Castles.exact(d,seed,n,extra,b,white,depth,tables,full,good,turn,one)
