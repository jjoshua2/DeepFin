import Base
import ./PROOF.bend as Proof
import ./LAWS.bend as Laws
import ./Destination.bend as D
import ../castle_emission/Spec.bend as E
import ../castle_chain/Spec.bend as C
import ../board/Spec.bend as B
import ../castle_kings/Spec.bend as K
import ../initialized_attacks/Build.bend as Init
import ../initialized_attacks/Spec.bend as G
import ../../../legal_probe/Chess.bend as Chess
import ../../Position.bend as Position

def use_prepared(a: Array<U64>,+b: Chess.Board,+xs: List<&2,Chess.Ply>,+m: Chess.Ply,
  castle: C.castle(b,m),here: E.member(m,E.moves(Chess.filter_prepare(b,(a,xs))))) ->
  {Chess.in_check(a,Chess.make_move(b,m),Chess.get_turn(b)) == (a,False{}) : Array<U64> & Bool}:
  Laws.prepared_castle_check_returns_false(a,b,xs,m,castle,here)

def use_generated(a: Array<U64>,+b: Chess.Board,+m: Chess.Ply,
  here: E.member(m,E.moves(Chess.legal_moves(a,b))),tag: {C.flag(m) == 2 : U32}) ->
  {Chess.in_check(a,Chess.make_move(b,m),Chess.get_turn(b)) == (a,False{}) : Array<U64> & Bool}:
  Laws.generated_castle_check_returns_false(a,b,m,here,tag)

def use_geometry(+d: Nat,+seed: U64,+n: Nat,+extra: Nat,+b: Chess.Board,+white: Bool,+m: Chess.Ply,
  depth: {d == 17n : Nat},tables: {n == 128n : Nat},full: {extra == 64n : Nat},
  good: {B.valid(b) == True{} : Bool},turn: {Chess.get_turn(b) == Bool.to_u32(white) : U32},
  one: {K.kings(b,white) == U64.bit(U32.to_nat(K.src(white))) : U64},
  here: E.member(m,E.moves(Chess.legal_moves(Init.run(d,seed,n,extra),b))),tag: {C.flag(m) == 2 : U32}) ->
  {G.attacked(Chess.make_move(b,m),U32.to_nat(D.square(m)),Bool.not(white)) == False{} : Bool}:
  Laws.initialized_generated_castle_destination_is_unattacked(d,seed,n,extra,b,white,m,
    depth,tables,full,good,turn,one,here,tag)

# Satisfiable membership witnesses use an arbitrary tiny table, not correct geometry.
def board() -> Chess.Board:
  Position.metadata(Position.put(5,True{},4,Position.put(3,True{},0,Position.put(3,True{},7,Position.empty()))),1,3,64)

def kingside_really_retained() ->
  {E.moves(Chess.legal_moves(Array.new(U64,0n,U64.zero()),board())) ==
    Con{Chess.Ply{4,6,0,2},Con{Chess.Ply{4,2,0,2},Nil{}}} : List<&2,Chess.Ply>}:
  {==}

def kingside_returns_false_pair() ->
  {Chess.in_check(Array.new(U64,0n,U64.zero()),Chess.make_move(board(),Chess.Ply{4,6,0,2}),1) ==
    (Array.new(U64,0n,U64.zero()),False{}) : Array<U64> & Bool}:
  Laws.generated_castle_check_returns_false(Array.new(U64,0n,U64.zero()),board(),Chess.Ply{4,6,0,2},Inl{{==}},{==})

def queenside_returns_false_pair() ->
  {Chess.in_check(Array.new(U64,0n,U64.zero()),Chess.make_move(board(),Chess.Ply{4,2,0,2}),1) ==
    (Array.new(U64,0n,U64.zero()),False{}) : Array<U64> & Bool}:
  Laws.generated_castle_check_returns_false(Array.new(U64,0n,U64.zero()),board(),Chess.Ply{4,2,0,2},Inr{Inl{{==}}},{==})
