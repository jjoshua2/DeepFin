import Base
import ./PROOF.bend as Proof
import ./LAWS.bend as Laws
import ./Spec.bend as S
import ../castle_emission/Spec.bend as E
import ../board/Spec.bend as B
import ../../../legal_probe/Chess.bend as Chess
import ../../Position.bend as Position

def ordinary(keys: List<&2,U32>,+b: Chess.Board,table: Array<U64>) ->
  S.every(S.ordinary,E.moves(Chess.scan(keys,b,(table,Nil{})))):
  Laws.ordinary_scan_excludes_castling(keys,b,table)

def filter(+b: Chess.Board,table: Array<U64>,+xs: List<&2,Chess.Ply>,+m: Chess.Ply,
  here: E.member(m,E.moves(Chess.filter_prepare(b,(table,xs))))) -> E.member(m,xs):
  Laws.final_filter_preserves_member_origin(b,table,xs,m,here)

def suffix(+b: Chess.Board,table: Array<U64>,+xs: List<&2,Chess.Ply>,+m: Chess.Ply,
  here: E.member(m,E.moves(S.suffix(b,(table,xs))))) -> E.member(m,xs) | S.castle(b,m):
  Laws.filtered_castling_suffix_origin(b,table,xs,m,here)

def provenance(table: Array<U64>,+b: Chess.Board,+m: Chess.Ply,
  here: E.member(m,E.moves(Chess.legal_moves(table,b))),tag: {S.flag(m) == 2 : U32}) -> S.castle(b,m):
  Laws.legal_castle_member_has_producer(table,b,m,here,tag)

def preserves(table: Array<U64>,+b: Chess.Board,+m: Chess.Ply,
  good: {B.valid(b) == True{} : Bool},here: E.member(m,E.moves(Chess.legal_moves(table,b))),tag: {S.flag(m) == 2 : U32}) ->
  {B.valid(Chess.make_move(b,m)) == True{} : Bool}:
  Laws.legal_castle_member_preserves_representation(table,b,m,good,here,tag)

# Satisfiability witnesses use a small arbitrary zero table, not certified attacks.
# The runtime reference suite uses real Tables.build instead.
def board() -> Chess.Board:
  Position.metadata(Position.put(5,True{},4,Position.put(3,True{},0,Position.put(3,True{},7,Position.empty()))),1,3,64)

def both_emitted_in_order() ->
  {E.moves(Chess.legal_moves(Array.new(U64,0n,U64.zero()),board())) ==
    Con{Chess.Ply{4,6,0,2},Con{Chess.Ply{4,2,0,2},Nil{}}} : List<&2,Chess.Ply>}:
  {==}

def kingside_certificate() -> S.castle(board(),Chess.Ply{4,6,0,2}):
  Laws.legal_castle_member_has_producer(Array.new(U64,0n,U64.zero()),board(),Chess.Ply{4,6,0,2},Inl{{==}},{==})

def queenside_consistent() -> {B.valid(Chess.make_move(board(),Chess.Ply{4,2,0,2})) == True{} : Bool}:
  Laws.legal_castle_member_preserves_representation(Array.new(U64,0n,U64.zero()),board(),Chess.Ply{4,2,0,2},{==},Inr{Inl{{==}}},{==})

# A final filter alone is NOT a validation of arbitrary caller-supplied castling.
def untrusted_tail_can_survive_filter() ->
  {E.moves(Chess.filter_prepare(Position.empty(),(Array.new(U64,0n,U64.zero()),Con{Chess.Ply{4,6,0,2},Nil{}}))) ==
    Con{Chess.Ply{4,6,0,2},Nil{}} : List<&2,Chess.Ply>}:
  {==}

def empty_board_has_no_generated_castle() ->
  {E.moves(Chess.legal_moves(Array.new(U64,0n,U64.zero()),Position.empty())) == Nil{} : List<&2,Chess.Ply>}:
  {==}
