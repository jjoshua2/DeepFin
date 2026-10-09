import Base
import ./Spec.bend as S
import ./consumer.bend as Counts
import ../backrank_pawns/Spec.bend as Pawns
import ../color_counts/Initialized.bend as Prior
import ../color_counts/Spec.bend as Colors
import ../castle_rights/Spec.bend as Rights
import ../protocol_replay_metadata/Spec.bend as Metadata
import ../storage/Observe.bend as O
import ../storage/Representation.bend as R
import ../initialized_attacks/Build.bend as Init
import ../../Position.bend as Pos
import ../../../legal_probe/Chess.bend as Chess

def complete(-a: Array<U64>,-d: Nat,-seed: U64,-n: Nat,-extra: Nat,-maybe: Maybe<&2,Pos.Game>,-text: String,-tail: List<&2,String>) -> Type:
  Prior.complete(a,d,seed,n,extra,maybe,text,tail) & S.replay(a,maybe,Con{text,tail})
def duplicate_rights(+maybe: Maybe<&2,Pos.Game>,zero: Rights.empty_rights(maybe)) ->
  Rights.empty_rights(maybe) & Rights.empty_rights(maybe):
  match maybe:
    case None{}: (Unit{},Unit{})
    case Some{+g}:
      +same: {Chess.get_rights(Pos.board(g)) == 0 : U32}=zero
      (same,same)

def finish(+c: O.Cells,+d: Nat,+seed: U64,+n: Nat,+extra: Nat,+maybe: Maybe<&2,Pos.Game>,+text: String,+tail: List<&2,String>,
  metadata: Metadata.parent(maybe),zeros: Rights.empty_rights(maybe) & Rights.empty_rights(maybe),pawns: Pawns.good(maybe),colors: Colors.bounded(maybe),counts: S.bounded(maybe),
  known: {O.pack(c) == Init.run(d,seed,n,extra) : Array<U64>},depth: {d == 17n : Nat},tables: {n == 128n : Nat},full: {extra == 64n : Nat}) ->
  complete(O.pack(c),d,seed,n,extra,maybe,text,tail):
  (first,second)=zeros
  (Prior.use_replay(O.pack(c),d,seed,n,extra,maybe,text,tail,metadata,first,pawns,colors,known,depth,tables,full),
    Counts.use_replay(O.pack(c),maybe,Con{text,tail},S.good_from(maybe,second,counts)))

def lift(-a: Array<U64>,+d: Nat,+seed: U64,+n: Nat,+extra: Nat,+maybe: Maybe<&2,Pos.Game>,+text: String,+tail: List<&2,String>,
  metadata: Metadata.parent(maybe),zero: Rights.empty_rights(maybe),pawns: Pawns.good(maybe),colors: Colors.bounded(maybe),counts: S.bounded(maybe),
  built: {a == Init.run(d,seed,n,extra) : Array<U64>},depth: {d == 17n : Nat},tables: {n == 128n : Nat},full: {extra == 64n : Nat},
  cert: Exists(O.Cells,c => {O.pack(c) == a : Array<U64>})) -> complete(a,d,seed,n,extra,maybe,text,tail):
  (+c,e)=cert
  +same: {O.pack(c) == a : Array<U64>}=e
  known: {O.pack(c) == Init.run(d,seed,n,extra) : Array<U64>}=Equal.trans(Array<U64>,O.pack(c),a,Init.run(d,seed,n,extra),same,built)
  %same : complete(_,d,seed,n,extra,maybe,text,tail)
  finish(c,d,seed,n,extra,maybe,text,tail,metadata,duplicate_rights(maybe,zero),pawns,colors,counts,known,depth,tables,full)
def use_replay(a: Array<U64>,+d: Nat,+seed: U64,+n: Nat,+extra: Nat,+maybe: Maybe<&2,Pos.Game>,+text: String,+tail: List<&2,String>,
  metadata: Metadata.parent(maybe),zero: Rights.empty_rights(maybe),pawns: Pawns.good(maybe),colors: Colors.bounded(maybe),counts: S.bounded(maybe),
  built: {a == Init.run(d,seed,n,extra) : Array<U64>},depth: {d == 17n : Nat},tables: {n == 128n : Nat},full: {extra == 64n : Nat}) ->
  complete(a,d,seed,n,extra,maybe,text,tail):
  lift(a,d,seed,n,extra,maybe,text,tail,metadata,zero,pawns,colors,counts,built,depth,tables,full,R.reify(a))
