import Base
import ./Spec.bend as S
import ./consumer.bend as Pawns
import ../castle_rights/Spec.bend as Rights
import ../castle_rights/Initialized.bend as Prior
import ../protocol_replay_metadata/Spec.bend as Metadata
import ../storage/Observe.bend as O
import ../storage/Representation.bend as R
import ../initialized_attacks/Build.bend as Init
import ../../Position.bend as Pos

def complete(-a: Array<U64>,-d: Nat,-seed: U64,-n: Nat,-extra: Nat,
  -maybe: Maybe<&2,Pos.Game>,-text: String,-tail: List<&2,String>) -> Type:
  Prior.complete(a,d,seed,n,extra,maybe,text,tail) & S.replay(a,maybe,Con{text,tail})
def lift(-a: Array<U64>,+d: Nat,+seed: U64,+n: Nat,+extra: Nat,
  +maybe: Maybe<&2,Pos.Game>,+text: String,+tail: List<&2,String>,
  metadata: Metadata.parent(maybe),zero: Rights.empty_rights(maybe),pawns: S.good(maybe),
  built: {a == Init.run(d,seed,n,extra) : Array<U64>},
  depth: {d == 17n : Nat},tables: {n == 128n : Nat},full: {extra == 64n : Nat},
  cert: Exists(O.Cells,c => {O.pack(c) == a : Array<U64>})) -> complete(a,d,seed,n,extra,maybe,text,tail):
  (+c,e)=cert
  +same: {O.pack(c) == a : Array<U64>}=e
  known: {O.pack(c) == Init.run(d,seed,n,extra) : Array<U64>}=
    Equal.trans(Array<U64>,O.pack(c),a,Init.run(d,seed,n,extra),same,built)
  %same : complete(_,d,seed,n,extra,maybe,text,tail)
  (Prior.use_replay(O.pack(c),d,seed,n,extra,maybe,text,tail,metadata,zero,known,depth,tables,full),
    Pawns.use_replay(O.pack(c),maybe,Con{text,tail},pawns))
def use_replay(a: Array<U64>,+d: Nat,+seed: U64,+n: Nat,+extra: Nat,
  +maybe: Maybe<&2,Pos.Game>,+text: String,+tail: List<&2,String>,
  metadata: Metadata.parent(maybe),zero: Rights.empty_rights(maybe),pawns: S.good(maybe),
  built: {a == Init.run(d,seed,n,extra) : Array<U64>},
  depth: {d == 17n : Nat},tables: {n == 128n : Nat},full: {extra == 64n : Nat}) -> complete(a,d,seed,n,extra,maybe,text,tail):
  lift(a,d,seed,n,extra,maybe,text,tail,metadata,zero,pawns,built,depth,tables,full,R.reify(a))
