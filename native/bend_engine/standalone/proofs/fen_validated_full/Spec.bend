import Base
import ./Root.bend as Root
import ./Replay.bend as Replay
import ../storage/Observe.bend as O
import ../accepted_castle/Generated.bend as Actual
import ../../Position.bend as Pos
import ../../Protocol.bend as Protocol
import ../../../legal_probe/Chess.bend as Chess

# Parsed row partition is a separate remaining premise, never inferred from Pos.valid.
def rows(out: Maybe<&2,Pos.Game>) -> Type:
  Root.rows(out)
def accepted(out: Maybe<&2,Pos.Game>) -> Type:
  Root.accepted(out)
def root(-c: O.Cells,-r: Array<U64> & Maybe<&2,Pos.Game>) -> Type:
  Root.root(c,r)

# Execute actual candidates (including both castle producers), then full filtering.
def full_prepare(+b: Chess.Board,r: Array<U64> & List<&2,Chess.Ply>) -> Array<U64> & List<&2,Chess.Ply>:
  (table,+xs)=r
  Chess.filter_legal(xs,b,(table,Nil{}))
def full_moves(a: Array<U64>,+b: Chess.Board) -> Array<U64> & List<&2,Chess.Ply>:
  full_prepare(b,Actual.candidates(a,b))
def full_one(+text: String,maybe: Maybe<&2,Pos.Game>,a: Array<U64>) -> Array<U64> & Maybe<&2,Pos.Game>:
  match maybe:
    case None{}: (a,None{})
    case Some{+g}: Protocol.move_found(g,text,full_moves(a,Pos.board(g)))
def full_after(+text: String,r: Array<U64> & Maybe<&2,Pos.Game>) -> Array<U64> & Maybe<&2,Pos.Game>:
  (a,g)=r
  full_one(text,g,a)
def one_after(+text: String,r: Array<U64> & Maybe<&2,Pos.Game>) -> Array<U64> & Maybe<&2,Pos.Game>:
  Replay.one_after(text,r)
def step(-a: Array<U64>,-maybe: Maybe<&2,Pos.Game>,-text: String) -> Type:
  {one_after(text,Protocol.validate(maybe,a)) == full_after(text,Protocol.validate(maybe,a)) :
    Array<U64> & Maybe<&2,Pos.Game>}
def singleton(-a: Array<U64>,-maybe: Maybe<&2,Pos.Game>,-text: String) -> Type:
  {Protocol.moves(Con{text,Nil{}},Protocol.validate(maybe,a)) ==
    full_after(text,Protocol.validate(maybe,a)) : Array<U64> & Maybe<&2,Pos.Game>}
