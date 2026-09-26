import Base
import ./PROOF.bend as Proof
import ./LAWS.bend as Laws
import ./Spec.bend as S
import ../castle_emission/Spec.bend as E
import ../board/Spec.bend as B
import ../../../legal_probe/Chess.bend as Chess
import ../../Position.bend as Position

def scan(+xs: List<&2,U32>,+b: Chess.Board,r: Array<U64> & List<&2,Chess.Ply>,tail: S.all(S.not_castle,E.moves(r))) -> S.all(S.not_castle,E.moves(Chess.scan(xs,b,r))):
  Laws.scan_excludes_castling(xs,b,r,tail)

def filter(+b: Chess.Board,r: Array<U64> & List<&2,Chess.Ply>,-P: @-m: Chess.Ply -> Type,p: S.all(P,E.moves(r))) -> S.all(P,E.moves(Chess.filter_prepare(b,r))):
  Laws.final_filter_preserves_predicate(b,r,P,p)

def both(+b: Chess.Board,table: Array<U64>,+tail: List<&2,Chess.Ply>) -> S.all(m => S.origin(b,tail,m),E.moves(S.both(b,(table,tail)))):
  Laws.both_castling_sides_provenance(b,table,tail)

def filtered(+b: Chess.Board,table: Array<U64>,+tail: List<&2,Chess.Ply>) -> S.all(m => S.origin(b,tail,m),E.moves(S.filtered(b,(table,tail)))):
  Laws.filtered_castling_provenance(b,table,tail)

def provenance(table: Array<U64>,+b: Chess.Board,+m: Chess.Ply,here: E.member(m,E.moves(Chess.legal_moves(table,b))),castle: {S.flag(m) == 2 : U32}) -> S.certified(b,m):
  Laws.legal_castle_guard_and_route(table,b,m,here,castle)

def preserves(table: Array<U64>,+b: Chess.Board,+m: Chess.Ply,good: {B.valid(b) == True{} : Bool},here: E.member(m,E.moves(Chess.legal_moves(table,b))),castle: {S.flag(m) == 2 : U32}) -> {B.valid(Chess.make_move(b,m)) == True{} : Bool}:
  Laws.legal_castle_preserves_representation(table,b,m,good,here,castle)

# Only a satisfiability witness: this tiny arbitrary table is NOT an attack oracle.
def board() -> Chess.Board:
  Position.metadata(Position.put(3,True{},0,Position.put(3,True{},7,Position.put(5,True{},4,Position.empty()))),1,3,64)

def nonempty_both() -> {E.moves(Chess.legal_moves(Array.new(U64,0n,U64.zero()),board())) == Con{Chess.Ply{4,6,0,2},Con{Chess.Ply{4,2,0,2},Nil{}}} : List<&2,Chess.Ply>}:
  {==}

def king_origin() -> S.certified(board(),Chess.Ply{4,6,0,2}):
  Laws.legal_castle_guard_and_route(Array.new(U64,0n,U64.zero()),board(),Chess.Ply{4,6,0,2},Inl{{==}},{==})

def queen_preserves() -> {B.valid(Chess.make_move(board(),Chess.Ply{4,2,0,2})) == True{} : Bool}:
  Laws.legal_castle_preserves_representation(Array.new(U64,0n,U64.zero()),board(),Chess.Ply{4,2,0,2},{==},Inr{Inl{{==}}},{==})

# An arbitrary caller tail is not magically certified by composition/filtering.
def old_fake_castle_survives() -> {E.moves(S.filtered(Position.empty(),(Array.new(U64,0n,U64.zero()),Con{Chess.Ply{4,6,0,2},Nil{}}))) == Con{Chess.Ply{4,6,0,2},Nil{}} : List<&2,Chess.Ply>}:
  {==}

def old_guard_is_false() -> {E.guard(Position.empty(),True{}) == False{} : Bool}:
  {==}

# Full legal_moves owns its initial empty list; the fake tail cannot enter there.
def full_empty_has_no_castle() -> {E.moves(Chess.legal_moves(Array.new(U64,0n,U64.zero()),Position.empty())) == Nil{} : List<&2,Chess.Ply>}:
  {==}
