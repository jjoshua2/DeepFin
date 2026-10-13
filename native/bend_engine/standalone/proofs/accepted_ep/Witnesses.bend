# Nonvacuous public-law instances for both actual EP directions.
import Base
import ./consumer.bend as P
import ./Boards.bend as B
import ../castle_emission/Spec.bend as E
import ../../../legal_probe/Chess.bend as C
def white_member() -> E.member(C.Ply{36,43,0,1},E.moves(C.legal_moves(ALeaf{U64.bit(43n)},B.white()))): Inl{{==}}
def black_member() -> E.member(C.Ply{27,20,0,1},E.moves(C.legal_moves(ALeaf{U64.bit(20n)},B.black()))): Inr{Inr{Inl{{==}}}}
def white_pair() ->
  {C.in_check(ALeaf{U64.bit(43n)},C.make_move(B.white(),C.Ply{36,43,0,1}),1) ==
    (ALeaf{U64.bit(43n)},False{}) : Array<U64> & Bool}:
  P.use_generated(ALeaf{U64.bit(43n)},B.white(),C.Ply{36,43,0,1},white_member(),{==})
def black_pair() ->
  {C.in_check(ALeaf{U64.bit(20n)},C.make_move(B.black(),C.Ply{27,20,0,1}),0) ==
    (ALeaf{U64.bit(20n)},False{}) : Array<U64> & Bool}:
  P.use_generated(ALeaf{U64.bit(20n)},B.black(),C.Ply{27,20,0,1},black_member(),{==})
def discovered_member() -> E.member(C.Ply{36,43,0,1},E.moves(C.legal_moves(ALeaf{U64.bit(43n)},B.discovered()))): Inl{{==}}
def discovered_pair() ->
  {C.in_check(ALeaf{U64.bit(43n)},C.make_move(B.discovered(),C.Ply{36,43,0,1}),1) ==
    (ALeaf{U64.bit(43n)},False{}) : Array<U64> & Bool}:
  P.use_generated(ALeaf{U64.bit(43n)},B.discovered(),C.Ply{36,43,0,1},discovered_member(),{==})
