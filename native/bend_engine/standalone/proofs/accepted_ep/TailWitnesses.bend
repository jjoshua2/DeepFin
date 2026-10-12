# Candidate duplicates are checked; an existing raw accumulator is not revisited.
import Base
import ./Boards.bend as B
import ../destination_factorization/Spec.bend as Count
import ../castle_emission/Spec.bend as E
import ../../../legal_probe/Chess.bend as C
def ep() -> C.Ply: C.Ply{36,43,0,1}
def twice() -> List<&2,C.Ply>: Con{ep(),Con{ep(),Nil{}}}
def table(-T: Type,r: Array<U64> & T) -> Array<U64>:
  (a,x)=r
  a
def prepared() -> Array<U64> & List<&2,C.Ply>:
  C.filter_prepare(B.discovered(),(ALeaf{U64.bit(43n)},twice()))
def prepared_duplicates() -> {Count.count(E.moves(prepared()),ep()) == 2n : Nat}: {==}
def prepared_table() -> {table(List<&2,C.Ply>,prepared()) == ALeaf{U64.bit(43n)} : Array<U64>}: {==}
def wrong_promotion_absent() -> {Count.count(E.moves(prepared()),C.Ply{36,43,1,1}) == 0n : Nat}: {==}
def wrong_flag_absent() -> {Count.count(E.moves(prepared()),C.Ply{36,43,0,0}) == 0n : Nat}: {==}
def rejected_duplicates() ->
  {Count.count(E.moves(C.filter_prepare(B.discovered(),(ALeaf{U64.bit(39n)},twice()))),ep()) == 0n : Nat}: {==}
def raw_tail() -> Array<U64> & List<&2,C.Ply>:
  C.filter_legal(Nil{},B.discovered(),(ALeaf{U64.bit(39n)},twice()))
def raw_tail_pair() -> {raw_tail() == (ALeaf{U64.bit(39n)},twice()) : Array<U64> & List<&2,C.Ply>}: {==}
def raw_tail_duplicates() -> {Count.count(E.moves(raw_tail()),ep()) == 2n : Nat}: {==}
def raw_tail_check_true() ->
  {C.in_check(ALeaf{U64.bit(39n)},C.make_move(B.discovered(),ep()),1) ==
    (ALeaf{U64.bit(39n)},True{}) : Array<U64> & Bool}: {==}
