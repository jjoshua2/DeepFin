# Actual public singleton consumer, plus exact first-match full-Ply witnesses.
import Base
import ./Spec.bend as S
import ./consumer.bend as Consumer
import ../initialized_attacks/Build.bend as Init
import ../../Position.bend as Pos
import ../../../legal_probe/Chess.bend as Chess

def actual_fen_one(a: Array<U64>,+d: Nat,+seed: U64,+n: Nat,+extra: Nat,
  +pieces: String,+turn: String,+castle: String,+target: String,+half: String,+fulltext: String,+text: String,
  initialized: {a == Init.run(d,seed,n,extra) : Array<U64>},
  depth: {d == 17n : Nat},tables: {n == 128n : Nat},full: {extra == 64n : Nat},
  rows: S.rows(Pos.fen(pieces,turn,castle,target,half,fulltext))) ->
  S.singleton(a,Pos.fen(pieces,turn,castle,target,half,fulltext),text):
  Consumer.use_fen_singleton(a,d,seed,n,extra,pieces,turn,castle,target,half,fulltext,text,
    initialized,depth,tables,full,rows)

def first(+m: Chess.Ply,+tail: List<&2,Chess.Ply>,+text: String,
  yes: {String.eq(Pos.move_text(m),text) == True{} : Bool}) ->
  {Pos.find_move(Con{m,tail},text) == Some{m} : Maybe<&2,Chess.Ply>}:
  %Equal.sym(Bool,String.eq(Pos.move_text(m),text),True{},yes) :
    {Pos.chosen(_,m,Pos.find_move(tail,text)) == Some{m} : Maybe<&2,Chess.Ply>}
  {==}

# Both records have identical move text; the first complete flag wins.
def ep_first(+tail: List<&2,Chess.Ply>) ->
  {Pos.find_move(Con{Chess.Ply{36,43,0,1},Con{Chess.Ply{36,43,0,0},tail}},
    Pos.move_text(Chess.Ply{36,43,0,1})) == Some{Chess.Ply{36,43,0,1}} : Maybe<&2,Chess.Ply>}:
  first(Chess.Ply{36,43,0,1},Con{Chess.Ply{36,43,0,0},tail},Pos.move_text(Chess.Ply{36,43,0,1}),{==})

def promotion_first(+tail: List<&2,Chess.Ply>) ->
  {Pos.find_move(Con{Chess.Ply{48,56,4,0},Con{Chess.Ply{48,56,4,0},tail}},
    Pos.move_text(Chess.Ply{48,56,4,0})) == Some{Chess.Ply{48,56,4,0}} : Maybe<&2,Chess.Ply>}:
  first(Chess.Ply{48,56,4,0},Con{Chess.Ply{48,56,4,0},tail},Pos.move_text(Chess.Ply{48,56,4,0}),{==})
