# Small structural connection to the actual singleton moves branch.
import Base
import ../../Position.bend as Pos
import ../../Protocol.bend as Protocol

def one_after(+text: String,r: Array<U64> & Maybe<&2,Pos.Game>) -> Array<U64> & Maybe<&2,Pos.Game>:
  (a,g)=r
  Protocol.apply_one(text,g,a)

def singleton(-finish: @r: Array<U64> & Maybe<&2,Pos.Game> -> Array<U64> & Maybe<&2,Pos.Game>,
  +text: String,r: Array<U64> & Maybe<&2,Pos.Game>,
  proof: {one_after(text,r) == finish(r) : Array<U64> & Maybe<&2,Pos.Game>}) ->
  {Protocol.moves(Con{text,Nil{}},r) == finish(r) : Array<U64> & Maybe<&2,Pos.Game>}:
  (a,g)=r
  proof
