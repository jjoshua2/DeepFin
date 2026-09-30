import Base
import ../../../legal_probe/Chess.bend as Chess
import ../castle_emission/Spec.bend as E
import ../castle_chain/Spec.bend as ChainSpec
import ../castle_chain/Chain.bend as Chain
import ./Spec.bend as S
import ./Mask.bend as Mask
import ./Step.bend as Step

def generated(table: Array<U64>,+b: Chess.Board,+m: Chess.Ply,+rays: U64,
  here: E.member(m,E.moves(Chess.legal_moves(table,b))),tag: {ChainSpec.flag(m) == 2 : U32}) ->
  {Chess.filter_requires(S.sensitive(b,rays),m) == True{} : Bool}:
  Mask.certificate(b,m,rays,Chain.legal_member(table,b,m,here,tag))

def generated_step(generation_table: Array<U64>,+b: Chess.Board,+m: Chess.Ply,+rays: U64,
  here: E.member(m,E.moves(Chess.legal_moves(generation_table,b))),tag: {ChainSpec.flag(m) == 2 : U32},
  filter_table: Array<U64>,+acc: List<&2,Chess.Ply>) ->
  {Chess.filter_fast_step(Chess.filter_requires(S.sensitive(b,rays),m),b,m,(filter_table,acc)) ==
    Chess.filter_after(m,acc,Chess.in_check(filter_table,Chess.make_move(b,m),Chess.get_turn(b))) : Array<U64> & List<&2,Chess.Ply>}:
  Step.checked(b,m,rays,filter_table,acc,generated(generation_table,b,m,rays,here,tag))
