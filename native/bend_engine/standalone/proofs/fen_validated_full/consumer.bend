# No caller white, canonical-turn or king-presence certificate.
import Base
import ./Spec.bend as S
import ./Turn.bend as Turn
import ./Validate.bend as Validate
import ./Apply.bend as Apply
import ../storage/Observe.bend as O
import ../storage/Representation.bend as R
import ../initialized_attacks/Build.bend as Init
import ../../Position.bend as Pos
import ../../Protocol.bend as Protocol
import ../../../legal_probe/Chess.bend as Chess

def step_lift(-a: Array<U64>,+d: Nat,+seed: U64,+n: Nat,+extra: Nat,
  +pieces: String,+turn: String,+castle: String,+target: String,+half: String,+fulltext: String,+text: String,
  initialized: {a == Init.run(d,seed,n,extra) : Array<U64>},
  depth: {d == 17n : Nat},tables: {n == 128n : Nat},full: {extra == 64n : Nat},
  rows: S.rows(Pos.fen(pieces,turn,castle,target,half,fulltext)),
  cert: Exists(O.Cells,c => {O.pack(c) == a : Array<U64>})) ->
  S.step(a,Pos.fen(pieces,turn,castle,target,half,fulltext),text):
  (+c,e)=cert
  +same: {O.pack(c) == a : Array<U64>}=e
  +built: {O.pack(c) == Init.run(d,seed,n,extra) : Array<U64>}=
    Equal.trans(Array<U64>,O.pack(c),a,Init.run(d,seed,n,extra),same,initialized)
  %same : S.step(_,Pos.fen(pieces,turn,castle,target,half,fulltext),text)
  Apply.result(c,d,seed,n,extra,text,
    Protocol.validate(Pos.fen(pieces,turn,castle,target,half,fulltext),O.pack(c)),
    built,depth,tables,full,
    Validate.known(c,Pos.fen(pieces,turn,castle,target,half,fulltext),rows,
      Turn.fen(pieces,turn,castle,target,half,fulltext)))

def use_fen_step(a: Array<U64>,+d: Nat,+seed: U64,+n: Nat,+extra: Nat,
  +pieces: String,+turn: String,+castle: String,+target: String,+half: String,+fulltext: String,+text: String,
  initialized: {a == Init.run(d,seed,n,extra) : Array<U64>},
  depth: {d == 17n : Nat},tables: {n == 128n : Nat},full: {extra == 64n : Nat},
  rows: S.rows(Pos.fen(pieces,turn,castle,target,half,fulltext))) ->
  S.step(a,Pos.fen(pieces,turn,castle,target,half,fulltext),text):
  step_lift(a,d,seed,n,extra,pieces,turn,castle,target,half,fulltext,text,
    initialized,depth,tables,full,rows,R.reify(a))

def singleton_lift(-a: Array<U64>,+d: Nat,+seed: U64,+n: Nat,+extra: Nat,
  +pieces: String,+turn: String,+castle: String,+target: String,+half: String,+fulltext: String,+text: String,
  initialized: {a == Init.run(d,seed,n,extra) : Array<U64>},
  depth: {d == 17n : Nat},tables: {n == 128n : Nat},full: {extra == 64n : Nat},
  rows: S.rows(Pos.fen(pieces,turn,castle,target,half,fulltext)),
  cert: Exists(O.Cells,c => {O.pack(c) == a : Array<U64>})) ->
  S.singleton(a,Pos.fen(pieces,turn,castle,target,half,fulltext),text):
  (+c,e)=cert
  +same: {O.pack(c) == a : Array<U64>}=e
  +built: {O.pack(c) == Init.run(d,seed,n,extra) : Array<U64>}=
    Equal.trans(Array<U64>,O.pack(c),a,Init.run(d,seed,n,extra),same,initialized)
  %same : S.singleton(_,Pos.fen(pieces,turn,castle,target,half,fulltext),text)
  Apply.singleton(text,Protocol.validate(Pos.fen(pieces,turn,castle,target,half,fulltext),O.pack(c)),
    step_lift(O.pack(c),d,seed,n,extra,pieces,turn,castle,target,half,fulltext,text,
      built,depth,tables,full,rows,(c,{==})))

def use_fen_singleton(a: Array<U64>,+d: Nat,+seed: U64,+n: Nat,+extra: Nat,
  +pieces: String,+turn: String,+castle: String,+target: String,+half: String,+fulltext: String,+text: String,
  initialized: {a == Init.run(d,seed,n,extra) : Array<U64>},
  depth: {d == 17n : Nat},tables: {n == 128n : Nat},full: {extra == 64n : Nat},
  rows: S.rows(Pos.fen(pieces,turn,castle,target,half,fulltext))) ->
  S.singleton(a,Pos.fen(pieces,turn,castle,target,half,fulltext),text):
  singleton_lift(a,d,seed,n,extra,pieces,turn,castle,target,half,fulltext,text,
    initialized,depth,tables,full,rows,R.reify(a))
