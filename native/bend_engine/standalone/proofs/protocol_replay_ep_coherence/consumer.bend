import Base
import ./Spec.bend as S
import ./Replay.bend as Replay
import ../protocol_replay_metadata/Spec.bend as Metadata
import ../fen_validated_full/Turn.bend as Turn
import ../fen_validated_full/Validate.bend as Validate
import ../parsed_fen_rows/Rows.bend as Rows
import ../storage/Observe.bend as O
import ../storage/Representation.bend as R
import ../table_preservation/Core.bend as T
import ../initialized_attacks/Build.bend as Init
import ../../Position.bend as Pos
import ../../Protocol.bend as Protocol

def initialized(+c: O.Cells,+d: Nat,+seed: U64,+n: Nat,+extra: Nat,
  -r: Array<U64> & Maybe<&2,Pos.Game>,
  built: {O.pack(c) == Init.run(d,seed,n,extra) : Array<U64>},
  depth: {d == 17n : Nat},tables: {n == 128n : Nat},full: {extra == 64n : Nat},
  checked: S.root(c,r)) -> S.initialized(O.pack(c),d,seed,n,extra,r):
  (+kept,good)=checked
  (kept,(Equal.trans(Array<U64>,T.table(Maybe<&2,Pos.Game>,r),O.pack(c),Init.run(d,seed,n,extra),kept,built),
    (depth,(tables,(full,good)))))

def replay_lift(-a: Array<U64>,+d: Nat,+seed: U64,+n: Nat,+extra: Nat,
  +maybe: Maybe<&2,Pos.Game>,+text: String,+tail: List<&2,String>,good: Metadata.parent(maybe),
  built: {a == Init.run(d,seed,n,extra) : Array<U64>},
  +depth: {d == 17n : Nat},tables: {n == 128n : Nat},+full: {extra == 64n : Nat},
  cert: Exists(O.Cells,c => {O.pack(c) == a : Array<U64>})) -> S.replay(a,d,seed,n,extra,maybe,text,tail):
  (+c,e)=cert
  +same: {O.pack(c) == a : Array<U64>}=e
  +known: {O.pack(c) == Init.run(d,seed,n,extra) : Array<U64>}=
    Equal.trans(Array<U64>,O.pack(c),a,Init.run(d,seed,n,extra),same,built)
  %same : S.replay(_,d,seed,n,extra,maybe,text,tail)
  initialized(c,d,seed,n,extra,Protocol.moves(Con{text,tail},(O.pack(c),maybe)),known,depth,tables,full,
    Replay.first(c,d,seed,n,extra,maybe,text,tail,known,depth,full,good))

# The caller supplies exact initialization and parent rows+canonical turn only.
def use_replay(a: Array<U64>,+d: Nat,+seed: U64,+n: Nat,+extra: Nat,
  +maybe: Maybe<&2,Pos.Game>,+text: String,+tail: List<&2,String>,good: Metadata.parent(maybe),
  built: {a == Init.run(d,seed,n,extra) : Array<U64>},
  depth: {d == 17n : Nat},tables: {n == 128n : Nat},full: {extra == 64n : Nat}) ->
  S.replay(a,d,seed,n,extra,maybe,text,tail):
  replay_lift(a,d,seed,n,extra,maybe,text,tail,good,built,depth,tables,full,R.reify(a))

def fen_lift(-a: Array<U64>,+d: Nat,+seed: U64,+n: Nat,+extra: Nat,
  +pieces: String,+turn: String,+castle: String,+target: String,+half: String,+fulltext: String,
  +text: String,+tail: List<&2,String>,
  built: {a == Init.run(d,seed,n,extra) : Array<U64>},
  +depth: {d == 17n : Nat},tables: {n == 128n : Nat},+full: {extra == 64n : Nat},
  cert: Exists(O.Cells,c => {O.pack(c) == a : Array<U64>})) ->
  S.fen_replay(a,d,seed,n,extra,pieces,turn,castle,target,half,fulltext,text,tail):
  (+c,e)=cert
  +same: {O.pack(c) == a : Array<U64>}=e
  +known: {O.pack(c) == Init.run(d,seed,n,extra) : Array<U64>}=
    Equal.trans(Array<U64>,O.pack(c),a,Init.run(d,seed,n,extra),same,built)
  %same : S.fen_replay(_,d,seed,n,extra,pieces,turn,castle,target,half,fulltext,text,tail)
  initialized(c,d,seed,n,extra,
    Protocol.moves(Con{text,tail},Protocol.validate(Pos.fen(pieces,turn,castle,target,half,fulltext),O.pack(c))),
    known,depth,tables,full,
    Replay.validated(c,d,seed,n,extra,text,tail,
      Protocol.validate(Pos.fen(pieces,turn,castle,target,half,fulltext),O.pack(c)),
      Validate.known(c,Pos.fen(pieces,turn,castle,target,half,fulltext),
        Rows.fen(pieces,turn,castle,target,half,fulltext),Turn.fen(pieces,turn,castle,target,half,fulltext)),known,depth,full))

def use_fen_replay(a: Array<U64>,+d: Nat,+seed: U64,+n: Nat,+extra: Nat,
  +pieces: String,+turn: String,+castle: String,+target: String,+half: String,+fulltext: String,
  +text: String,+tail: List<&2,String>,
  built: {a == Init.run(d,seed,n,extra) : Array<U64>},
  depth: {d == 17n : Nat},tables: {n == 128n : Nat},full: {extra == 64n : Nat}) ->
  S.fen_replay(a,d,seed,n,extra,pieces,turn,castle,target,half,fulltext,text,tail):
  fen_lift(a,d,seed,n,extra,pieces,turn,castle,target,half,fulltext,text,tail,built,depth,tables,full,R.reify(a))
