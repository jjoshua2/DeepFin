import Base
import ../protocol_replay_metadata/Spec.bend as Metadata
import ../child_ep_coherence/Coherence.bend as EP
import ../fen_validated_full/Root.bend as Root
import ../storage/Observe.bend as O
import ../table_preservation/Core.bend as T
import ../initialized_attacks/Build.bend as Init
import ../../Position.bend as Pos
import ../../Protocol.bend as Protocol
import ../../../legal_probe/Chess.bend as Chess

def ready(-b: Chess.Board) -> Type:
  Metadata.ready(b) & {EP.valid(b) == True{} : Bool}
def output(out: Maybe<&2,Pos.Game>) -> Type:
  match out:
    case None{}: Unit
    case Some{g}: ready(Pos.board(g))
def ep_output(out: Maybe<&2,Pos.Game>) -> Type:
  match out:
    case None{}: Unit
    case Some{g}: {EP.valid(Pos.board(g)) == True{} : Bool}
def weaken(+out: Maybe<&2,Pos.Game>,good: output(out)) -> Metadata.parent(out):
  match out:
    case None{}: Unit{}
    case Some{g}:
      (metadata,coherent)=good
      Metadata.weaken(Some{g},metadata)
def root(-c: O.Cells,-r: Array<U64> & Maybe<&2,Pos.Game>) -> Type:
  {T.table(Maybe<&2,Pos.Game>,r) == O.pack(c) : Array<U64>} & output(Root.value(r))
def initialized(-a: Array<U64>,-d: Nat,-seed: U64,-n: Nat,-extra: Nat,
  -r: Array<U64> & Maybe<&2,Pos.Game>) -> Type:
  {T.table(Maybe<&2,Pos.Game>,r) == a : Array<U64>} &
    ({T.table(Maybe<&2,Pos.Game>,r) == Init.run(d,seed,n,extra) : Array<U64>} &
      ({d == 17n : Nat} & ({n == 128n : Nat} & ({extra == 64n : Nat} & output(Root.value(r))))))
def replay(-a: Array<U64>,-d: Nat,-seed: U64,-n: Nat,-extra: Nat,
  -maybe: Maybe<&2,Pos.Game>,-text: String,-tail: List<&2,String>) -> Type:
  initialized(a,d,seed,n,extra,Protocol.moves(Con{text,tail},(a,maybe)))
def fen_replay(-a: Array<U64>,-d: Nat,-seed: U64,-n: Nat,-extra: Nat,
  -pieces: String,-turn: String,-castle: String,-target: String,-half: String,-fulltext: String,
  -text: String,-tail: List<&2,String>) -> Type:
  initialized(a,d,seed,n,extra,
    Protocol.moves(Con{text,tail},Protocol.validate(Pos.fen(pieces,turn,castle,target,half,fulltext),a)))
