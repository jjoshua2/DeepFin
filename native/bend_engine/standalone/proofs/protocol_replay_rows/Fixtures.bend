# Executing-path fixtures; full Ply flags, failed lookup and recursive failure.
import Base
import ./Spec.bend as S
import ./Find.bend as Find
import ./consumer.bend as Consumer
import ../fen_validated_full/Root.bend as Root
import ../initialized_attacks/Build.bend as Init
import ../../Position.bend as Pos
import ../../Protocol.bend as Protocol
import ../../../legal_probe/Chess.bend as Chess

def actual_replay(a: Array<U64>,+maybe: Maybe<&2,Pos.Game>,+texts: List<&2,String>,
  good: Root.rows(maybe)) -> S.replay(a,maybe,texts):
  Consumer.use_replay(a,maybe,texts,good)

def actual_fen_replay(a: Array<U64>,+d: Nat,+seed: U64,+n: Nat,+extra: Nat,
  +pieces: String,+turn: String,+castle: String,+target: String,+half: String,+fulltext: String,
  +texts: List<&2,String>,
  initialized: {a == Init.run(d,seed,n,extra) : Array<U64>},
  depth: {d == 17n : Nat},tables: {n == 128n : Nat},full: {extra == 64n : Nat}) ->
  S.fen_replay(a,d,seed,n,extra,pieces,turn,castle,target,half,fulltext,texts):
  Consumer.use_fen_replay(a,d,seed,n,extra,pieces,turn,castle,target,half,fulltext,texts,
    initialized,depth,tables,full)

# The text is the same for both flags. Selection retains the first exact full Ply.
def first_flag() ->
  {Pos.find_move(Con{Chess.Ply{12,28,0,1},Con{Chess.Ply{12,28,0,0},Nil{}}},"e2e4") ==
    Some{Chess.Ply{12,28,0,1}} : Maybe<&2,Chess.Ply>}: {==}
def duplicate_first() ->
  {Pos.find_move(Con{Chess.Ply{12,28,0,1},Con{Chess.Ply{12,28,0,1},Con{Chess.Ply{12,28,0,0},Nil{}}}},"e2e4") ==
    Some{Chess.Ply{12,28,0,1}} : Maybe<&2,Chess.Ply>}: {==}
def promotion_text() ->
  {Pos.find_move(Con{Chess.Ply{48,56,1,0},Con{Chess.Ply{48,56,4,0},Nil{}}},"a7a8q") ==
    Some{Chess.Ply{48,56,4,0}} : Maybe<&2,Chess.Ply>}: {==}
def failed_lookup(+g: Pos.Game) ->
  {Protocol.move_found(g,"e2e5",(ALeaf{U64.zero},Con{Chess.Ply{12,28,0,1},Nil{}})) ==
    (ALeaf{U64.zero},None{}) : Array<U64> & Maybe<&2,Pos.Game>}: {==}
def initial_none(+texts: List<&2,String>,a: Array<U64>) ->
  S.replay(a,None{},texts): Consumer.use_replay(a,None{},texts,Unit{})
def none_commands() ->
  {Protocol.moves(Con{"e2e4",Con{"e2e4",Con{"invalid",Nil{}}}},
    (ANode{ALeaf{U64{7,9}},ALeaf{U64{11,13}}},None{})) ==
    (ANode{ALeaf{U64{7,9}},ALeaf{U64{11,13}}},None{}) : Array<U64> & Maybe<&2,Pos.Game>}: {==}

# This is the actual legal-move selector and apply, including counters/history.
def first_success() ->
  {Root.value(Protocol.apply_one("e2e4",Some{Pos.start()},ALeaf{U64.zero})) ==
    Some{Pos.apply(Pos.start(),Chess.Ply{12,28,0,0})} : Maybe<&2,Pos.Game>}: {==}
def second_invalid() ->
  {Protocol.moves(Con{"e2e4",Con{"invalid",Nil{}}},(ALeaf{U64.zero},Some{Pos.start()})) ==
    (ALeaf{U64.zero},None{}) : Array<U64> & Maybe<&2,Pos.Game>}: {==}
