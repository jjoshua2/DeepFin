import Base
import ./Spec.bend as S
import ./Replay.bend as Replay
import ../fen_validated_full/Root.bend as Root
import ../fen_validated_full/Turn.bend as Turn
import ../fen_validated_full/Validate.bend as Validate
import ../parsed_fen_rows/Rows.bend as Rows
import ../storage/Observe.bend as O
import ../storage/Representation.bend as R
import ../table_preservation/Core.bend as T
import ../initialized_attacks/Build.bend as Init
import ../../Position.bend as Pos
import ../../Protocol.bend as Protocol

def replay_lift(-a: Array<U64>,+maybe: Maybe<&2,Pos.Game>,+text: String,+tail: List<&2,String>,
  good: S.parent(maybe),cert: Exists(O.Cells,c => {O.pack(c) == a : Array<U64>})) ->
  S.replay(a,maybe,text,tail):
  (+c,e)=cert
  +same: {O.pack(c) == a : Array<U64>}=e
  %same : S.replay(_,maybe,text,tail)
  Replay.first(c,maybe,text,tail,good)

# The nonempty command shape is explicit; no input EP range is supplied.
def use_replay(a: Array<U64>,+maybe: Maybe<&2,Pos.Game>,+text: String,+tail: List<&2,String>,
  good: S.parent(maybe)) -> S.replay(a,maybe,text,tail):
  replay_lift(a,maybe,text,tail,good,R.reify(a))

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
  +text: String,+tail: List<&2,String>,
  built: {a == Init.run(d,seed,n,extra) : Array<U64>},
  depth: {d == 17n : Nat},tables: {n == 128n : Nat},full: {extra == 64n : Nat},
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
    Replay.validated(c,text,tail,Protocol.validate(Pos.fen(pieces,turn,castle,target,half,fulltext),O.pack(c)),
      Validate.known(c,Pos.fen(pieces,turn,castle,target,half,fulltext),
        Rows.fen(pieces,turn,castle,target,half,fulltext),Turn.fen(pieces,turn,castle,target,half,fulltext))))

def use_fen_replay(a: Array<U64>,+d: Nat,+seed: U64,+n: Nat,+extra: Nat,
  +pieces: String,+turn: String,+castle: String,+target: String,+half: String,+fulltext: String,
  +text: String,+tail: List<&2,String>,
  initialized: {a == Init.run(d,seed,n,extra) : Array<U64>},
  depth: {d == 17n : Nat},tables: {n == 128n : Nat},full: {extra == 64n : Nat}) ->
  S.fen_replay(a,d,seed,n,extra,pieces,turn,castle,target,half,fulltext,text,tail):
  fen_lift(a,d,seed,n,extra,pieces,turn,castle,target,half,fulltext,text,tail,
    initialized,depth,tables,full,R.reify(a))
