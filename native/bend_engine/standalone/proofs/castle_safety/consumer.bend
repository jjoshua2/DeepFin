import Base
import ./PROOF.bend as Proof
import ./LAWS.bend as Laws
import ./Spec.bend as S
import ./Forced.bend as Force
import ../castle_emission/Spec.bend as E
import ../castle_chain/Spec.bend as ChainSpec
import ../../../legal_probe/Chess.bend as Chess
import ../../Position.bend as Position

def owned(+b: Chess.Board,+rays: U64,+src: U32,+dst: U32,+promotion: U32,+flag: U32,
  bound: {Nat.is_lt(U32.to_nat(src),64n) == True{} : Bool},present: {S.owned_king(b,src) == True{} : Bool}) ->
  {Chess.filter_requires(S.sensitive(b,rays),Chess.Ply{src,dst,promotion,flag}) == True{} : Bool}:
  Laws.owned_king_source_requires_check(b,rays,src,dst,promotion,flag,bound,present)

def blockers(+b: Chess.Board,+moves: List<&2,Chess.Ply>,table: Array<U64>,+rays: U64) ->
  {Chess.filter_blockers(b,moves,(table,rays)) == Chess.filter_fast(moves,b,S.sensitive(b,rays),(table,Nil{})) : Array<U64> & List<&2,Chess.Ply>}:
  Laws.blocker_stage_uses_sensitive_mask(b,moves,table,rays)

def required(table: Array<U64>,+b: Chess.Board,+m: Chess.Ply,+rays: U64,
  here: E.member(m,E.moves(Chess.legal_moves(table,b))),tag: {ChainSpec.flag(m) == 2 : U32}) ->
  {Chess.filter_requires(S.sensitive(b,rays),m) == True{} : Bool}:
  Laws.generated_castle_requires_check(table,b,m,rays,here,tag)

def checked(generation_table: Array<U64>,+b: Chess.Board,+m: Chess.Ply,+rays: U64,
  here: E.member(m,E.moves(Chess.legal_moves(generation_table,b))),tag: {ChainSpec.flag(m) == 2 : U32},
  filter_table: Array<U64>,+acc: List<&2,Chess.Ply>) ->
  {Chess.filter_fast_step(Chess.filter_requires(S.sensitive(b,rays),m),b,m,(filter_table,acc)) ==
    Chess.filter_after(m,acc,Chess.in_check(filter_table,Chess.make_move(b,m),Chess.get_turn(b))) : Array<U64> & List<&2,Chess.Ply>}:
  Laws.generated_castle_takes_checked_step(generation_table,b,m,rays,here,tag,filter_table,acc)

def rejects(+m: Chess.Ply,+acc: List<&2,Chess.Ply>,table: Array<U64>) ->
  {Chess.filter_after(m,acc,(table,True{})) == (table,acc) : Array<U64> & List<&2,Chess.Ply>}:
  Laws.positive_check_rejects_current_move(m,acc,table)

# Small arbitrary-table satisfiability witness, not a certified attack table.
def board() -> Chess.Board:
  Position.metadata(Position.put(5,True{},4,Position.put(3,True{},0,Position.put(3,True{},7,Position.empty()))),1,3,64)

def generated_with_no_rays() ->
  {Chess.filter_requires(S.sensitive(board(),U64.zero()),Chess.Ply{4,6,0,2}) == True{} : Bool}:
  Laws.generated_castle_requires_check(Array.new(U64,0n,U64.zero()),board(),Chess.Ply{4,6,0,2},U64.zero(),Inl{{==}},{==})

def high_limb_king() ->
  {Chess.filter_requires(S.sensitive(Position.metadata(Position.put(5,False{},63,Position.empty()),0,0,64),U64.zero()),Chess.Ply{63,62,0,0}) == True{} : Bool}:
  Laws.owned_king_source_requires_check(Position.metadata(Position.put(5,False{},63,Position.empty()),0,0,64),U64.zero(),63,62,0,0,{==},{==})

# Flag2 alone is NOT enough: raw unowned/empty-source moves can take the shortcut.
def raw_castle_can_skip() ->
  {Chess.filter_requires(S.sensitive(Position.empty(),U64.zero()),Chess.Ply{4,6,0,2}) == False{} : Bool}:
  {==}

# Rejection preserves arbitrary preexisting entries, even an equal move value.
def old_equal_entry_survives_rejection() ->
  {E.moves(Chess.filter_after(Chess.Ply{4,6,0,2},Con{Chess.Ply{4,6,0,2},Nil{}},(Array.new(U64,0n,U64.zero()),True{}))) ==
    Con{Chess.Ply{4,6,0,2},Nil{}} : List<&2,Chess.Ply>}:
  {==}

def complete_generator(table: Array<U64>,+b: Chess.Board) ->
  {Chess.legal_moves(table,b) == Force.legal_moves(table,b) : Array<U64> & List<&2,Chess.Ply>}:
  Laws.complete_generator_equals_forced_castle_checks(table,b)
