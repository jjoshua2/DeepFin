import Base
import ./Spec.bend as S
import ./consumer.bend as Kings
import ../pawn_counts/Initialized.bend as Prior
import ../pawn_counts/Spec.bend as Pawns
import ../backrank_pawns/Spec.bend as Backrank
import ../color_counts/Spec.bend as Colors
import ../castle_rights/Spec.bend as Rights
import ../protocol_replay_metadata/Spec.bend as Metadata
import ../protocol_replay_ep_coherence/Spec.bend as EP
import ../storage/Observe.bend as O
import ../storage/Representation.bend as R
import ../initialized_attacks/Build.bend as Init
import ../../Position.bend as Pos
import ../../../legal_probe/Chess.bend as Chess

def complete(-a: Array<U64>,-d: Nat,-seed: U64,-n: Nat,-extra: Nat,-maybe: Maybe<&2,Pos.Game>,-text: String,-tail: List<&2,String>) -> Type:
  Prior.complete(a,d,seed,n,extra,maybe,text,tail) & S.replay(a,d,seed,n,extra,maybe,Con{text,tail})
def duplicate_metadata(+maybe: Maybe<&2,Pos.Game>,good: Metadata.parent(maybe)) -> Metadata.parent(maybe) & Metadata.parent(maybe):
  match maybe:
    case None{}: (Unit{},Unit{})
    case Some{+g}:
      (+rows,canonical)=good
      +same: {Chess.get_turn(Pos.board(g)) == Bool.to_u32(U32.is_eq(Chess.get_turn(Pos.board(g)),1)) : U32}=canonical
      ((rows,same),(rows,same))
def finish(+c: O.Cells,+d: Nat,+seed: U64,+n: Nat,+extra: Nat,+maybe: Maybe<&2,Pos.Game>,+text: String,+tail: List<&2,String>,
  metadatas: Metadata.parent(maybe) & Metadata.parent(maybe),zeros: Rights.empty_rights(maybe) & Rights.empty_rights(maybe),
  coherent: EP.ep_output(maybe),pawns: Backrank.good(maybe),colors: Colors.bounded(maybe),counts: Pawns.bounded(maybe),
  +known: {O.pack(c) == Init.run(d,seed,n,extra) : Array<U64>},+depth: {d == 17n : Nat},+tables: {n == 128n : Nat},+full: {extra == 64n : Nat}) ->
  complete(O.pack(c),d,seed,n,extra,maybe,text,tail):
  (m1,m2)=metadatas
  (z1,z2)=zeros
  (Prior.use_replay(O.pack(c),d,seed,n,extra,maybe,text,tail,m1,z1,pawns,colors,counts,known,depth,tables,full),
    Kings.use_replay(O.pack(c),d,seed,n,extra,maybe,Con{text,tail},m2,coherent,z2,known,depth,tables,full))
def lift(-a: Array<U64>,+d: Nat,+seed: U64,+n: Nat,+extra: Nat,+maybe: Maybe<&2,Pos.Game>,+text: String,+tail: List<&2,String>,
  metadata: Metadata.parent(maybe),coherent: EP.ep_output(maybe),zero: Rights.empty_rights(maybe),pawns: Backrank.good(maybe),colors: Colors.bounded(maybe),counts: Pawns.bounded(maybe),
  built: {a == Init.run(d,seed,n,extra) : Array<U64>},depth: {d == 17n : Nat},tables: {n == 128n : Nat},full: {extra == 64n : Nat},
  cert: Exists(O.Cells,c => {O.pack(c) == a : Array<U64>})) -> complete(a,d,seed,n,extra,maybe,text,tail):
  (+c,e)=cert
  +same: {O.pack(c) == a : Array<U64>}=e
  known: {O.pack(c) == Init.run(d,seed,n,extra) : Array<U64>}=Equal.trans(Array<U64>,O.pack(c),a,Init.run(d,seed,n,extra),same,built)
  %same : complete(_,d,seed,n,extra,maybe,text,tail)
  finish(c,d,seed,n,extra,maybe,text,tail,duplicate_metadata(maybe,metadata),Prior.duplicate_rights(maybe,zero),coherent,pawns,colors,counts,known,depth,tables,full)
def use_replay(a: Array<U64>,+d: Nat,+seed: U64,+n: Nat,+extra: Nat,+maybe: Maybe<&2,Pos.Game>,+text: String,+tail: List<&2,String>,
  metadata: Metadata.parent(maybe),coherent: EP.ep_output(maybe),zero: Rights.empty_rights(maybe),pawns: Backrank.good(maybe),colors: Colors.bounded(maybe),counts: Pawns.bounded(maybe),
  built: {a == Init.run(d,seed,n,extra) : Array<U64>},depth: {d == 17n : Nat},tables: {n == 128n : Nat},full: {extra == 64n : Nat}) ->
  complete(a,d,seed,n,extra,maybe,text,tail):
  lift(a,d,seed,n,extra,maybe,text,tail,metadata,coherent,zero,pawns,colors,counts,built,depth,tables,full,R.reify(a))
