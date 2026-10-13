import Base
import ../fen_validated_full/Root.bend as Root
import ../fen_validated_full/Turn.bend as Turn
import ../child_metadata/Range.bend as Range
import ../board/Spec.bend as B
import ../storage/Observe.bend as O
import ../table_preservation/Core.bend as T
import ../initialized_attacks/Build.bend as Init
import ../../Position.bend as Pos
import ../../Protocol.bend as Protocol
import ../../../legal_probe/Chess.bend as Chess

def ready(-b: Chess.Board) -> Type:
  {B.valid(b) == True{} : Bool} & (Turn.canonical(Chess.get_turn(b)) & Range.ep(Chess.get_ep(b)))
def parent(out: Maybe<&2,Pos.Game>) -> Type:
  match out:
    case None{}: Unit
    case Some{g}: {B.valid(Pos.board(g)) == True{} : Bool} & Turn.canonical(Chess.get_turn(Pos.board(g)))
def output(out: Maybe<&2,Pos.Game>) -> Type:
  match out:
    case None{}: Unit
    case Some{g}: ready(Pos.board(g))
def weaken(+out: Maybe<&2,Pos.Game>,good: output(out)) -> parent(out):
  match out:
    case None{}: Unit{}
    case Some{g}:
      (rows,(canonical,bound))=good
      (rows,canonical)
def accepted(+out: Maybe<&2,Pos.Game>,good: Root.accepted(out)) -> parent(out):
  match out:
    case None{}: Unit{}
    case Some{g}:
      (rows,(canonical,valid))=good
      (rows,canonical)
def root(-c: O.Cells,-r: Array<U64> & Maybe<&2,Pos.Game>) -> Type:
  {T.table(Maybe<&2,Pos.Game>,r) == O.pack(c) : Array<U64>} & output(Root.value(r))
def replay(-a: Array<U64>,-maybe: Maybe<&2,Pos.Game>,-text: String,-tail: List<&2,String>) -> Type:
  {T.table(Maybe<&2,Pos.Game>,Protocol.moves(Con{text,tail},(a,maybe))) == a : Array<U64>} &
    output(Root.value(Protocol.moves(Con{text,tail},(a,maybe))))
def initialized(-a: Array<U64>,-d: Nat,-seed: U64,-n: Nat,-extra: Nat,
  -r: Array<U64> & Maybe<&2,Pos.Game>) -> Type:
  {T.table(Maybe<&2,Pos.Game>,r) == a : Array<U64>} &
    ({T.table(Maybe<&2,Pos.Game>,r) == Init.run(d,seed,n,extra) : Array<U64>} &
      ({d == 17n : Nat} & ({n == 128n : Nat} & ({extra == 64n : Nat} & output(Root.value(r))))))
def fen_replay(-a: Array<U64>,-d: Nat,-seed: U64,-n: Nat,-extra: Nat,
  -pieces: String,-turn: String,-castle: String,-target: String,-half: String,-fulltext: String,
  -text: String,-tail: List<&2,String>) -> Type:
  initialized(a,d,seed,n,extra,
    Protocol.moves(Con{text,tail},Protocol.validate(Pos.fen(pieces,turn,castle,target,half,fulltext),a)))
