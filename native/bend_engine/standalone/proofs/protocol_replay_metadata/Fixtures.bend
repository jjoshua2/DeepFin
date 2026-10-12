import Base
import ./Spec.bend as S
import ./consumer.bend as Consumer
import ../fen_validated_full/Root.bend as Root
import ../initialized_attacks/Build.bend as Init
import ../../Position.bend as Pos
import ../../Protocol.bend as Protocol
import ../../../legal_probe/Chess.bend as Chess

def actual_replay(a: Array<U64>,+maybe: Maybe<&2,Pos.Game>,+text: String,+tail: List<&2,String>,
  good: S.parent(maybe)) -> S.replay(a,maybe,text,tail):
  Consumer.use_replay(a,maybe,text,tail,good)
def actual_fen_replay(a: Array<U64>,+d: Nat,+seed: U64,+n: Nat,+extra: Nat,
  +pieces: String,+turn: String,+castle: String,+target: String,+half: String,+fulltext: String,
  +text: String,+tail: List<&2,String>,
  initialized: {a == Init.run(d,seed,n,extra) : Array<U64>},
  depth: {d == 17n : Nat},tables: {n == 128n : Nat},full: {extra == 64n : Nat}) ->
  S.fen_replay(a,d,seed,n,extra,pieces,turn,castle,target,half,fulltext,text,tail):
  Consumer.use_fen_replay(a,d,seed,n,extra,pieces,turn,castle,target,half,fulltext,text,tail,
    initialized,depth,tables,full)

def raw() -> Pos.Game:
  Pos.Game{Pos.metadata(Pos.board(Pos.start()),1,15,65),0,1,Nil{}}
def raw_parent() -> S.parent(Some{raw()}): ({==},{==})
def arbitrary_bad_ep(+text: String,+tail: List<&2,String>) ->
  S.replay(ALeaf{U64.zero},Some{raw()},text,tail):
  Consumer.use_replay(ALeaf{U64.zero},Some{raw()},text,tail,raw_parent())
def initial_none(a: Array<U64>,+text: String,+tail: List<&2,String>) ->
  S.replay(a,None{},text,tail): Consumer.use_replay(a,None{},text,tail,Unit{})

def observed_ep(out: Maybe<&2,Pos.Game>) -> U32:
  match out:
    case None{}: 64
    case Some{g}: Chess.get_ep(Pos.board(g))
def observed_turn(out: Maybe<&2,Pos.Game>) -> U32:
  match out:
    case None{}: 0
    case Some{g}: Chess.get_turn(Pos.board(g))

# Empty replay keeps invalid input EP; nonempty replay's first actual move replaces it.
def empty_keeps_bad_ep() ->
  {observed_ep(Root.value(Protocol.moves(Nil{},(ALeaf{U64.zero},Some{raw()})))) == 65 : U32}: {==}
def first_ep() ->
  {observed_ep(Root.value(Protocol.moves(Con{"e2e4",Nil{}},(ALeaf{U64.zero},Some{raw()})))) == 20 : U32}: {==}
def first_turn() ->
  {observed_turn(Root.value(Protocol.moves(Con{"e2e4",Nil{}},(ALeaf{U64.zero},Some{raw()})))) == 0 : U32}: {==}

# Two real moves restore the initial side; the theorem claims canonicality, not fixed parity.
def two_successes() ->
  {Protocol.moves(Con{"e2e4",Con{"e7e5",Nil{}}},(ALeaf{U64.zero},Some{raw()})) ==
    (ALeaf{U64.zero},Some{Pos.apply(Pos.apply(raw(),Chess.Ply{12,28,0,0}),Chess.Ply{52,36,0,0})}) :
    Array<U64> & Maybe<&2,Pos.Game>}: {==}
def second_turn() ->
  {observed_turn(Root.value(Protocol.moves(Con{"e2e4",Con{"e7e5",Nil{}}},(ALeaf{U64.zero},Some{raw()})))) == 1 : U32}: {==}
def second_ep() ->
  {observed_ep(Root.value(Protocol.moves(Con{"e2e4",Con{"e7e5",Nil{}}},(ALeaf{U64.zero},Some{raw()})))) == 44 : U32}: {==}
def first_invalid() ->
  {Protocol.moves(Con{"invalid",Nil{}},(ALeaf{U64.zero},Some{raw()})) ==
    (ALeaf{U64.zero},None{}) : Array<U64> & Maybe<&2,Pos.Game>}: {==}
def second_invalid() ->
  {Protocol.moves(Con{"e2e4",Con{"invalid",Nil{}}},(ALeaf{U64.zero},Some{raw()})) ==
    (ALeaf{U64.zero},None{}) : Array<U64> & Maybe<&2,Pos.Game>}: {==}
