# Actual FEN consumers require no caller parsed-board certificate.
import Base
import ./Rows.bend as Rows
import ../fen_validated_full/consumer.bend as Prior
import ../fen_validated_full/Spec.bend as S
import ../initialized_attacks/Build.bend as Init
import ../../Position.bend as Pos

def use_fen_step(a: Array<U64>,+d: Nat,+seed: U64,+n: Nat,+extra: Nat,
  +pieces: String,+turn: String,+castle: String,+target: String,+half: String,+fulltext: String,+text: String,
  initialized: {a == Init.run(d,seed,n,extra) : Array<U64>},
  depth: {d == 17n : Nat},tables: {n == 128n : Nat},full: {extra == 64n : Nat}) ->
  S.step(a,Pos.fen(pieces,turn,castle,target,half,fulltext),text):
  Prior.use_fen_step(a,d,seed,n,extra,pieces,turn,castle,target,half,fulltext,text,
    initialized,depth,tables,full,Rows.fen(pieces,turn,castle,target,half,fulltext))

def use_fen_singleton(a: Array<U64>,+d: Nat,+seed: U64,+n: Nat,+extra: Nat,
  +pieces: String,+turn: String,+castle: String,+target: String,+half: String,+fulltext: String,+text: String,
  initialized: {a == Init.run(d,seed,n,extra) : Array<U64>},
  depth: {d == 17n : Nat},tables: {n == 128n : Nat},full: {extra == 64n : Nat}) ->
  S.singleton(a,Pos.fen(pieces,turn,castle,target,half,fulltext),text):
  Prior.use_fen_singleton(a,d,seed,n,extra,pieces,turn,castle,target,half,fulltext,text,
    initialized,depth,tables,full,Rows.fen(pieces,turn,castle,target,half,fulltext))
