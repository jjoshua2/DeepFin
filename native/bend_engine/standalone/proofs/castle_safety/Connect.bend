import Base
import ../../../legal_probe/Chess.bend as Chess
import ../castle_emission/Spec.bend as E
import ../castle_chain/Spec.bend as ChainSpec
import ../castle_chain/Chain.bend as Chain
import ./Spec.bend as S
import ./Required.bend as R
import ./Filter.bend as F

def generated(table: Array<U64>,+b: Chess.Board,+m: Chess.Ply,+rays: U64,
  here: E.member(m,E.moves(Chess.legal_moves(table,b))),tag: {ChainSpec.flag(m) == 2 : U32}) ->
  {Chess.filter_requires(S.sensitive(b,rays),m) == True{} : Bool}:
  R.certificate(b,m,rays,Chain.legal_member(table,b,m,here,tag))

def step(table: Array<U64>,+b: Chess.Board,+m: Chess.Ply,+rays: U64,r: Array<U64> & List<&2,Chess.Ply>,
  here: E.member(m,E.moves(Chess.legal_moves(table,b))),tag: {ChainSpec.flag(m) == 2 : U32}) ->
  {Chess.filter_fast_step(Chess.filter_requires(S.sensitive(b,rays),m),b,m,r) == Chess.filter_step(b,m,r) : Array<U64> & List<&2,Chess.Ply>}:
  F.step(Chess.filter_requires(S.sensitive(b,rays),m),b,m,r,generated(table,b,m,rays,here,tag))
