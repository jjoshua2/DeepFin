# Connect the actual public six-field parser; metadata validation remains separate.
import Base
import ./Spec.bend as S
import ../frontier/LAWS.bend as FrontierLaws
import ../frontier/PROOF.bend as FrontierProof
import ../frontier/Spec.bend as Frontier
import ../board/Spec.bend as B
import ../board/Actual.bend as Board
import ../../Position.bend as Position
import ../../Text.bend as Text
import ../../../legal_probe/Chess.bend as Chess

def scalar(m: Maybe<&2,U32>,f: U32 -> Maybe<&2,Position.Game>,
  rule: @x: U32 -> {S.game_consistent(f(x)) == True{} : Bool}) ->
  {S.game_consistent(Maybe.bind(&2,U32,Position.Game,m,f)) == True{} : Bool}:
  match m:
    case None{}: {==}
    case Some{x}: rule(x)

def advance(b: Chess.Board,turn: String,castle: String,target: String,half: String,full: String) -> Maybe<&2,Position.Game>:
  Maybe.bind(&2,U32,Position.Game,Position.side(turn),+t => Maybe.bind(&2,U32,Position.Game,Position.rights(castle),+r => Maybe.bind(&2,U32,Position.Game,Position.ep(target),+e => Maybe.bind(&2,U32,Position.Game,Text.number(half),+h0 => Maybe.bind(&2,U32,Position.Game,Text.bounded(h0),+h => Maybe.bind(&2,U32,Position.Game,Text.number(full),+f0 => Maybe.bind(&2,U32,Position.Game,Text.positive(f0),+f1 => Maybe.bind(&2,U32,Position.Game,Text.bounded(f1),+f => Maybe.pure(&2,Position.Game,Position.Game{Position.metadata(b,t,r,e),h,f,Nil{}})))))))))

def from_board(+b: Chess.Board,+turn: String,+castle: String,+target: String,+half: String,+full: String,
  +good: {B.valid(b) == True{} : Bool}) -> {S.game_consistent(advance(b,turn,castle,target,half,full)) == True{} : Bool}:
  scalar(Position.side(turn),+t => Maybe.bind(&2,U32,Position.Game,Position.rights(castle),+r => Maybe.bind(&2,U32,Position.Game,Position.ep(target),+e => Maybe.bind(&2,U32,Position.Game,Text.number(half),+h0 => Maybe.bind(&2,U32,Position.Game,Text.bounded(h0),+h => Maybe.bind(&2,U32,Position.Game,Text.number(full),+f0 => Maybe.bind(&2,U32,Position.Game,Text.positive(f0),+f1 => Maybe.bind(&2,U32,Position.Game,Text.bounded(f1),+f => Maybe.pure(&2,Position.Game,Position.Game{Position.metadata(b,t,r,e),h,f,Nil{}})))))))),+t =>
    scalar(Position.rights(castle),+r => Maybe.bind(&2,U32,Position.Game,Position.ep(target),+e => Maybe.bind(&2,U32,Position.Game,Text.number(half),+h0 => Maybe.bind(&2,U32,Position.Game,Text.bounded(h0),+h => Maybe.bind(&2,U32,Position.Game,Text.number(full),+f0 => Maybe.bind(&2,U32,Position.Game,Text.positive(f0),+f1 => Maybe.bind(&2,U32,Position.Game,Text.bounded(f1),+f => Maybe.pure(&2,Position.Game,Position.Game{Position.metadata(b,t,r,e),h,f,Nil{}}))))))),+r =>
    scalar(Position.ep(target),+e => Maybe.bind(&2,U32,Position.Game,Text.number(half),+h0 => Maybe.bind(&2,U32,Position.Game,Text.bounded(h0),+h => Maybe.bind(&2,U32,Position.Game,Text.number(full),+f0 => Maybe.bind(&2,U32,Position.Game,Text.positive(f0),+f1 => Maybe.bind(&2,U32,Position.Game,Text.bounded(f1),+f => Maybe.pure(&2,Position.Game,Position.Game{Position.metadata(b,t,r,e),h,f,Nil{}})))))),+e =>
    scalar(Text.number(half),+h0 => Maybe.bind(&2,U32,Position.Game,Text.bounded(h0),+h => Maybe.bind(&2,U32,Position.Game,Text.number(full),+f0 => Maybe.bind(&2,U32,Position.Game,Text.positive(f0),+f1 => Maybe.bind(&2,U32,Position.Game,Text.bounded(f1),+f => Maybe.pure(&2,Position.Game,Position.Game{Position.metadata(b,t,r,e),h,f,Nil{}}))))),+h0 =>
    scalar(Text.bounded(h0),+h => Maybe.bind(&2,U32,Position.Game,Text.number(full),+f0 => Maybe.bind(&2,U32,Position.Game,Text.positive(f0),+f1 => Maybe.bind(&2,U32,Position.Game,Text.bounded(f1),+f => Maybe.pure(&2,Position.Game,Position.Game{Position.metadata(b,t,r,e),h,f,Nil{}})))),+h =>
    scalar(Text.number(full),+f0 => Maybe.bind(&2,U32,Position.Game,Text.positive(f0),+f1 => Maybe.bind(&2,U32,Position.Game,Text.bounded(f1),+f => Maybe.pure(&2,Position.Game,Position.Game{Position.metadata(b,t,r,e),h,f,Nil{}}))),+f0 =>
    scalar(Text.positive(f0),+f1 => Maybe.bind(&2,U32,Position.Game,Text.bounded(f1),+f => Maybe.pure(&2,Position.Game,Position.Game{Position.metadata(b,t,r,e),h,f,Nil{}})),+f1 =>
    scalar(Text.bounded(f1),+f => Maybe.pure(&2,Position.Game,Position.Game{Position.metadata(b,t,r,e),h,f,Nil{}}),+f =>
    Equal.trans(Bool,B.valid(Position.metadata(b,t,r,e)),B.valid(b),True{},Board.metadata(b,t,r,e),good)))))))))

def from_layout(+m: Maybe<&2,Chess.Board>,+turn: String,+castle: String,+target: String,+half: String,+full: String,
  good: {Frontier.result_valid(m) == True{} : Bool}) ->
  {S.game_consistent(Maybe.bind(&2,Chess.Board,Position.Game,m,b => advance(b,turn,castle,target,half,full))) == True{} : Bool}:
  match m:
    case None{}: {==}
    case Some{b}: from_board(b,turn,castle,target,half,full,good)

def actual(+pieces: String,+turn: String,+castle: String,+target: String,+half: String,+full: String) ->
  {S.game_consistent(Position.fen(pieces,turn,castle,target,half,full)) == True{} : Bool}:
  from_layout(Position.layout_result(Position.layout(pieces,Frontier.initial())),turn,castle,target,half,full,FrontierLaws.accepted_placement_has_consistent_board(pieces))
