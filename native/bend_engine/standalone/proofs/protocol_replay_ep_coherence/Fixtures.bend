import Base
import ./Spec.bend as S
import ./Apply.bend as Apply
import ./consumer.bend as Consumer
import ../protocol_replay_metadata/Spec.bend as Metadata
import ../protocol_replay_metadata/Fixtures.bend as Prior
import ../protocol_replay_rows/Find.bend as Find
import ../protocol_replay_rows/Spec.bend as Selected
import ../child_ep_coherence/Coherence.bend as EP
import ../child_ep_coherence/Witnesses.bend as Witness
import ../fen_validated_full/Root.bend as Root
import ../castle_chain/Spec.bend as C
import ../castle_emission/Spec.bend as E
import ../initialized_attacks/Build.bend as Init
import ../../Position.bend as Pos
import ../../Protocol.bend as Protocol
import ../../../legal_probe/Chess.bend as Chess

def actual_initialized(a: Array<U64>,+d: Nat,+seed: U64,+n: Nat,+extra: Nat,
  +maybe: Maybe<&2,Pos.Game>,+text: String,+tail: List<&2,String>,good: Metadata.parent(maybe),
  built: {a == Init.run(d,seed,n,extra) : Array<U64>},
  depth: {d == 17n : Nat},tables: {n == 128n : Nat},full: {extra == 64n : Nat}) ->
  S.replay(a,d,seed,n,extra,maybe,text,tail):
  Consumer.use_replay(a,d,seed,n,extra,maybe,text,tail,good,built,depth,tables,full)

# No parent EP bound or coherence is supplied, including the raw EP65 Game.
def raw_ep65(a: Array<U64>,+d: Nat,+seed: U64,+n: Nat,+extra: Nat,+text: String,+tail: List<&2,String>,
  built: {a == Init.run(d,seed,n,extra) : Array<U64>},
  depth: {d == 17n : Nat},tables: {n == 128n : Nat},full: {extra == 64n : Nat}) ->
  S.replay(a,d,seed,n,extra,Some{Prior.raw()},text,tail):
  Consumer.use_replay(a,d,seed,n,extra,Some{Prior.raw()},text,tail,Prior.raw_parent(),built,depth,tables,full)
def raw_parent_incoherent() -> {EP.valid(Pos.board(Prior.raw())) == False{} : Bool}: {==}

# Duplicate full fields and shared text remain attached to the actual first match.
def full_moves() -> List<&2,Chess.Ply>:
  Chess.Ply{8,24,0,1} <> Chess.Ply{8,24,0,0} <> Chess.Ply{8,24,0,1} <> Nil{}
def full_fields() -> C.every(m => {EP.child(Witness.white(24),m) == True{} : Bool},full_moves()):
  (Witness.raw_ep_double(),(Witness.white_double(),(Witness.raw_ep_double(),Unit{})))
def first_full_flag() -> {Pos.find_move(full_moves(),"a2a4") == Some{Chess.Ply{8,24,0,1}} : Maybe<&2,Chess.Ply>}: {==}
def selected_coherent() -> Selected.selected(m => {EP.child(Witness.white(24),m) == True{} : Bool},Pos.find_move(full_moves(),"a2a4")):
  Find.all(m => {EP.child(Witness.white(24),m) == True{} : Bool},full_moves(),"a2a4",full_fields())
def selected_applied() -> S.ep_output(Protocol.move_apply(Pos.Game{Witness.white(24),0,1,Nil{}},Pos.find_move(full_moves(),"a2a4"))):
  Apply.moved(Pos.Game{Witness.white(24),0,1,Nil{}},Pos.find_move(full_moves(),"a2a4"),selected_coherent())

# The initialized premise cannot be dropped for arbitrary arrays; retain the old jump witness.
def arbitrary_jump_incoherent() -> {EP.child(Witness.jump(),Chess.Ply{32,48,0,0}) == False{} : Bool}:
  Witness.arbitrary_jump_incoherent()
def observed_valid(out: Maybe<&2,Pos.Game>) -> Bool:
  match out:
    case None{}: True{}
    case Some{g}: EP.valid(Pos.board(g))
def empty_retains_ep65() ->
  {observed_valid(Root.value(Protocol.moves(Nil{},(ALeaf{U64.zero},Some{Prior.raw()})))) == False{} : Bool}: {==}
