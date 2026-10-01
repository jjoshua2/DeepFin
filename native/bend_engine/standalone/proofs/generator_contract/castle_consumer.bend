# Full exact-pair consumer, with inherited law bodies loaded.
import Base
import ../castle_sequence/PROOF.bend as InheritedProof
import ../../../legal_probe/Chess.bend as Chess
import ../board/Spec.bend as B
import ../castle_kings/Spec.bend as K
import ../initialized_attacks/Build.bend as Init
import ./CastleSpec.bend as P
import ./CastleComposition.bend as Composition
import ./CastleUnique.bend as Unique
import ../castle_emission/Spec.bend as E

def use_exact_castling_projection(+d: Nat,+seed: U64,+n: Nat,+extra: Nat,+b: Chess.Board,+white: Bool,
  +depth: {d == 17n : Nat},+tables: {n == 128n : Nat},+full: {extra == 64n : Nat},
  +good: {B.valid(b) == True{} : Bool},+turn: {Chess.get_turn(b) == Bool.to_u32(white) : U32},
  +one: {K.kings(b,white) == U64.bit(U32.to_nat(K.src(white))) : U64}) ->
  {P.result(Chess.legal_moves(Init.run(d,seed,n,extra),b)) ==
    (Init.run(d,seed,n,extra),P.expected(b,white)) : Array<U64> & List<&2,Chess.Ply>}:
  Composition.exact(d,seed,n,extra,b,white,depth,tables,full,good,turn,one)

def use_no_duplicate_castles(+d: Nat,+seed: U64,+n: Nat,+extra: Nat,+b: Chess.Board,+white: Bool,
  +depth: {d == 17n : Nat},+tables: {n == 128n : Nat},+full: {extra == 64n : Nat},
  +good: {B.valid(b) == True{} : Bool},+turn: {Chess.get_turn(b) == Bool.to_u32(white) : U32},
  +one: {K.kings(b,white) == U64.bit(U32.to_nat(K.src(white))) : U64}) ->
  {Unique.unique(P.project(E.moves(Chess.legal_moves(Init.run(d,seed,n,extra),b)))) == True{} : Bool}:
  Composition.no_duplicates(d,seed,n,extra,b,white,depth,tables,full,good,turn,one)
