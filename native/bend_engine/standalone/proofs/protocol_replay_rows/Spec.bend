import Base
import ../fen_validated_full/Root.bend as Root
import ../storage/Observe.bend as O
import ../table_preservation/Core.bend as T
import ../initialized_attacks/Build.bend as Init
import ../../Position.bend as Pos
import ../../Protocol.bend as Protocol
import ../../../legal_probe/Chess.bend as Chess

def selected(-P: Chess.Ply -> Type,out: Maybe<&2,Chess.Ply>) -> Type:
  match out:
    case None{}: Unit
    case Some{m}: P(m) & Unit

def root(-c: O.Cells,-r: Array<U64> & Maybe<&2,Pos.Game>) -> Type:
  {T.table(Maybe<&2,Pos.Game>,r) == O.pack(c) : Array<U64>} & Root.rows(Root.value(r))

def replay(-a: Array<U64>,-maybe: Maybe<&2,Pos.Game>,-texts: List<&2,String>) -> Type:
  {T.table(Maybe<&2,Pos.Game>,Protocol.moves(texts,(a,maybe))) == a : Array<U64>} &
    Root.rows(Root.value(Protocol.moves(texts,(a,maybe))))

# Both exact input storage and its existing initialized-table contract are retained.
def initialized(-a: Array<U64>,-d: Nat,-seed: U64,-n: Nat,-extra: Nat,
  -r: Array<U64> & Maybe<&2,Pos.Game>) -> Type:
  {T.table(Maybe<&2,Pos.Game>,r) == a : Array<U64>} &
    ({T.table(Maybe<&2,Pos.Game>,r) == Init.run(d,seed,n,extra) : Array<U64>} &
      ({d == 17n : Nat} & ({n == 128n : Nat} & ({extra == 64n : Nat} & Root.rows(Root.value(r))))))

def fen_replay(-a: Array<U64>,-d: Nat,-seed: U64,-n: Nat,-extra: Nat,
  -pieces: String,-turn: String,-castle: String,-target: String,-half: String,-fulltext: String,
  -texts: List<&2,String>) -> Type:
  initialized(a,d,seed,n,extra,
    Protocol.moves(texts,Protocol.validate(Pos.fen(pieces,turn,castle,target,half,fulltext),a)))
