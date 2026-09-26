import Base
import ./Boundaries.bend as Boundaries
import ./PROOF.bend as Proof
import ./LAWS.bend as Laws
import ./Spec.bend as S
import ../castle_emission/Spec.bend as E
import ../castle_chain/Spec.bend as C
import ../castle_chain/consumer.bend as Parent
import ../../../legal_probe/Chess.bend as Chess
import ../../Position.bend as Position

def guarded(+b: Chess.Board,+side: Bool,+rays: U64,ok: {E.guard(b,side) == True{} : Bool}) ->
  {Chess.filter_requires(S.sensitive(b,rays),E.move(b,side)) == True{} : Bool}:
  Laws.guarded_castle_requires_full_check(b,side,rays,ok)

def generated(table: Array<U64>,+b: Chess.Board,+m: Chess.Ply,+rays: U64,
  here: E.member(m,E.moves(Chess.legal_moves(table,b))),tag: {C.flag(m) == 2 : U32}) ->
  {Chess.filter_requires(S.sensitive(b,rays),m) == True{} : Bool}:
  Laws.generated_castle_requires_full_check(table,b,m,rays,here,tag)

def step(table: Array<U64>,+b: Chess.Board,+m: Chess.Ply,+rays: U64,r: Array<U64> & List<&2,Chess.Ply>,
  here: E.member(m,E.moves(Chess.legal_moves(table,b))),tag: {C.flag(m) == 2 : U32}) ->
  {Chess.filter_fast_step(Chess.filter_requires(S.sensitive(b,rays),m),b,m,r) == Chess.filter_step(b,m,r) : Array<U64> & List<&2,Chess.Ply>}:
  Laws.generated_castle_fast_step_equals_checked_step(table,b,m,rays,r,here,tag)

def blockers(+b: Chess.Board,+xs: List<&2,Chess.Ply>,table: Array<U64>,+rays: U64,good: C.every(m => C.castle(b,m),xs)) ->
  {Chess.filter_blockers(b,xs,(table,rays)) == Chess.filter_legal(xs,b,(table,Nil{})) : Array<U64> & List<&2,Chess.Ply>}:
  Laws.guarded_castling_blocker_filter_equals_full_filter(b,xs,table,rays,good)

# Satisfiable symbolic-table witnesses, not assertions of correct attack tables.
def generated_even_with_zero_rays() ->
  {Chess.filter_requires(S.sensitive(Parent.board(),U64.zero()),Chess.Ply{4,6,0,2}) == True{} : Bool}:
  Laws.generated_castle_requires_full_check(Array.new(U64,0n,U64.zero()),Parent.board(),Chess.Ply{4,6,0,2},U64.zero(),Inl{{==}},{==})

# A raw flag-two value alone does not force the full check without source membership.
def raw_tag_alone_is_insufficient() ->
  {Chess.filter_requires(S.sensitive(Position.empty(),U64.zero()),Chess.Ply{4,6,0,2}) == False{} : Bool}:
  {==}

# List equivalence handles duplicates and checks, rather than assuming unique castles.
def two_copies() -> List<&2,Chess.Ply>:
  Con{Chess.Ply{4,6,0,2},Con{Chess.Ply{4,6,0,2},Nil{}}}

def duplicate_list_full_pair(table: Array<U64>,+rays: U64) ->
  {Chess.filter_blockers(Parent.board(),two_copies(),(table,rays)) == Chess.filter_legal(two_copies(),Parent.board(),(table,Nil{})) : Array<U64> & List<&2,Chess.Ply>}:
  Laws.guarded_castling_blocker_filter_equals_full_filter(Parent.board(),two_copies(),table,rays,
    (Inl{({==},{==})},(Inl{({==},{==})},Unit{})))
