# Public six-field FEN composition; exact lexical acceptance is not claimed.
import Base
import ../../Position.bend as P
import ../../Text.bend as Text
import ../../../legal_probe/Chess.bend as Chess
import ./Spec.bend as S

law fen_result_is_consistent:
  for +pieces: String
  for +turn: String
  for +castle: String
  for +target: String
  for +half: String
  for +full: String
  {S.game_consistent(P.fen(pieces,turn,castle,target,half,full)) == True{} : Bool}

law rejected_placement_rejects_fen:
  for +pieces: String
  for +turn: String
  for +castle: String
  for +target: String
  for +half: String
  for +full: String
  for rejected: {S.placement(pieces) == None{} : Maybe<&2,Chess.Board>}
  {P.fen(pieces,turn,castle,target,half,full) == None{} : Maybe<&2,P.Game>}

law validated_fields_construct_exact_game:
  for +pieces: String
  for +turn: String
  for +castle: String
  for +target: String
  for +half: String
  for +full: String
  for +b: Chess.Board
  for +t: U32
  for +r: U32
  for +e: U32
  for +h: U32
  for +f: U32
  for pb: {S.placement(pieces) == Some{b} : Maybe<&2,Chess.Board>}
  for pt: {P.side(turn) == Some{t} : Maybe<&2,U32>}
  for pr: {P.rights(castle) == Some{r} : Maybe<&2,U32>}
  for pe: {P.ep(target) == Some{e} : Maybe<&2,U32>}
  for ph: {Text.number(half) == Some{h} : Maybe<&2,U32>}
  for pf: {Text.number(full) == Some{f} : Maybe<&2,U32>}
  for hb: {U32.is_le(h,1000000) == True{} : Bool}
  for fp: {U32.is_gt(f,0) == True{} : Bool}
  for fb: {U32.is_le(f,1000000) == True{} : Bool}
  {P.fen(pieces,turn,castle,target,half,full) == Some{P.Game{P.metadata(b,t,r,e),h,f,Nil{}}} : Maybe<&2,P.Game>}
