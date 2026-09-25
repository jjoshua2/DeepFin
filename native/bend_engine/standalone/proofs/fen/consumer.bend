# Public six-field FEN composition; exact lexical acceptance is not claimed.
import Base
import ../../Position.bend as P
import ../../Text.bend as Text
import ../../../legal_probe/Chess.bend as Chess
import ./Spec.bend as S

import ./LAWS.bend as Laws
import ./PROOF.bend as Proof
import ../board/Spec.bend as Board

def consistent(+pieces: String,+turn: String,+castle: String,+target: String,+half: String,+full: String) ->
  {S.game_consistent(P.fen(pieces,turn,castle,target,half,full)) == True{} : Bool}:
  Laws.fen_result_is_consistent(pieces,turn,castle,target,half,full)

def reject_any_tail(+turn: String,+castle: String,+target: String,+half: String,+full: String) ->
  {P.fen("x7/8/8/8/8/8/8/8",turn,castle,target,half,full) == None{} : Maybe<&2,P.Game>}:
  Laws.rejected_placement_rejects_fen("x7/8/8/8/8/8/8/8",turn,castle,target,half,full,{==})

def supplied_fields(+pieces: String,+turn: String,+castle: String,+target: String,+half: String,+full: String,
  +b: Chess.Board,+t: U32,+r: U32,+e: U32,+h: U32,+f: U32,
  pb: {S.placement(pieces) == Some{b} : Maybe<&2,Chess.Board>},
  pt: {P.side(turn) == Some{t} : Maybe<&2,U32>},
  pr: {P.rights(castle) == Some{r} : Maybe<&2,U32>},
  pe: {P.ep(target) == Some{e} : Maybe<&2,U32>},
  ph: {Text.number(half) == Some{h} : Maybe<&2,U32>},
  pf: {Text.number(full) == Some{f} : Maybe<&2,U32>},
  hb: {U32.is_le(h,1000000) == True{} : Bool},fp: {U32.is_gt(f,0) == True{} : Bool},fb: {U32.is_le(f,1000000) == True{} : Bool}) ->
  {P.fen(pieces,turn,castle,target,half,full) == Some{P.Game{P.metadata(b,t,r,e),h,f,Nil{}}} : Maybe<&2,P.Game>}:
  Laws.validated_fields_construct_exact_game(pieces,turn,castle,target,half,full,b,t,r,e,h,f,pb,pt,pr,pe,ph,pf,hb,fp,fb)

def nonempty_exact() ->
  {P.fen("P7/8/8/8/8/8/8/7n","b","Kq","h8","0017","00042") ==
    Some{P.Game{P.metadata(P.put(1,False{},7,P.put(0,True{},56,P.empty())),0,9,63),17,42,Nil{}}} : Maybe<&2,P.Game>}:
  Laws.validated_fields_construct_exact_game("P7/8/8/8/8/8/8/7n","b","Kq","h8","0017","00042",
    P.put(1,False{},7,P.put(0,True{},56,P.empty())),0,9,63,17,42,{==},{==},{==},{==},{==},{==},{==},{==},{==})

def empty_is_accepted_not_legal() ->
  {P.fen("8/8/8/8/8/8/8/8","w","-","-","0","1") == Some{P.Game{P.empty(),0,1,Nil{}}} : Maybe<&2,P.Game>}:
  {==}

def invalid_fullmove() ->
  {P.fen("8/8/8/8/8/8/8/8","w","-","-","0","0") == None{} : Maybe<&2,P.Game>}:
  {==}
