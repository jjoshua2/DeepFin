import Base
import ./Spec.bend as S
import ./Replay.bend as Replay
import ../fen_validated_full/Root.bend as Root
import ../parsed_fen_rows/Rows.bend as Rows
import ../storage/Observe.bend as O
import ../storage/Representation.bend as R
import ../table_preservation/Core.bend as T
import ../initialized_attacks/Build.bend as Init
import ../../Position.bend as Pos
import ../../Protocol.bend as Protocol

def replay_lift(-a: Array<U64>,+maybe: Maybe<&2,Pos.Game>,+texts: List<&2,String>,
  good: Root.rows(maybe),cert: Exists(O.Cells,c => {O.pack(c) == a : Array<U64>})) ->
  S.replay(a,maybe,texts):
  (+c,e)=cert
  +same: {O.pack(c) == a : Array<U64>}=e
  %same : S.replay(_,maybe,texts)
  Replay.moves(c,texts,(O.pack(c),maybe),({==},good))

# Arbitrary arrays and raw Games are covered with only conditional input rows.
def use_replay(a: Array<U64>,+maybe: Maybe<&2,Pos.Game>,+texts: List<&2,String>,
  good: Root.rows(maybe)) -> S.replay(a,maybe,texts):
  replay_lift(a,maybe,texts,good,R.reify(a))

def initialized(+c: O.Cells,+d: Nat,+seed: U64,+n: Nat,+extra: Nat,
  -r: Array<U64> & Maybe<&2,Pos.Game>,
  built: {O.pack(c) == Init.run(d,seed,n,extra) : Array<U64>},
  depth: {d == 17n : Nat},tables: {n == 128n : Nat},full: {extra == 64n : Nat},
  checked: S.root(c,r)) -> S.initialized(O.pack(c),d,seed,n,extra,r):
  (+kept,good)=checked
  (kept,(Equal.trans(Array<U64>,T.table(Maybe<&2,Pos.Game>,r),O.pack(c),Init.run(d,seed,n,extra),kept,built),
    (depth,(tables,(full,good)))))

def fen_lift(-a: Array<U64>,+d: Nat,+seed: U64,+n: Nat,+extra: Nat,
  +pieces: String,+turn: String,+castle: String,+target: String,+half: String,+fulltext: String,
  +texts: List<&2,String>,
  built: {a == Init.run(d,seed,n,extra) : Array<U64>},
  depth: {d == 17n : Nat},tables: {n == 128n : Nat},full: {extra == 64n : Nat},
  cert: Exists(O.Cells,c => {O.pack(c) == a : Array<U64>})) ->
  S.fen_replay(a,d,seed,n,extra,pieces,turn,castle,target,half,fulltext,texts):
  (+c,e)=cert
  +same: {O.pack(c) == a : Array<U64>}=e
  +known: {O.pack(c) == Init.run(d,seed,n,extra) : Array<U64>}=
    Equal.trans(Array<U64>,O.pack(c),a,Init.run(d,seed,n,extra),same,built)
  %same : S.fen_replay(_,d,seed,n,extra,pieces,turn,castle,target,half,fulltext,texts)
  initialized(c,d,seed,n,extra,
    Protocol.moves(texts,Protocol.validate(Pos.fen(pieces,turn,castle,target,half,fulltext),O.pack(c))),
    known,depth,tables,full,
    Replay.initial(c,Pos.fen(pieces,turn,castle,target,half,fulltext),texts,
      Rows.fen(pieces,turn,castle,target,half,fulltext)))

# The actual parser supplies rows internally; the initialized contract is explicit.
def use_fen_replay(a: Array<U64>,+d: Nat,+seed: U64,+n: Nat,+extra: Nat,
  +pieces: String,+turn: String,+castle: String,+target: String,+half: String,+fulltext: String,
  +texts: List<&2,String>,
  initialized: {a == Init.run(d,seed,n,extra) : Array<U64>},
  depth: {d == 17n : Nat},tables: {n == 128n : Nat},full: {extra == 64n : Nat}) ->
  S.fen_replay(a,d,seed,n,extra,pieces,turn,castle,target,half,fulltext,texts):
  fen_lift(a,d,seed,n,extra,pieces,turn,castle,target,half,fulltext,texts,
    initialized,depth,tables,full,R.reify(a))
