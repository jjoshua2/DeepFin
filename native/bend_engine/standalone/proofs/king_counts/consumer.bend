import Base
import ./Actual.bend as K
import ./Emission.bend as Emission
import ./Replay.bend as Replay
import ./Spec.bend as S
import ../board/Spec.bend as Board
import ../protocol_replay_metadata/Spec.bend as Metadata
import ../protocol_replay_ep_coherence/Spec.bend as EP
import ../castle_rights/Spec.bend as Rights
import ../storage/Observe.bend as O
import ../storage/Representation.bend as R
import ../initialized_attacks/Build.bend as Init
import ../table_preservation/Core.bend as T
import ../castle_emission/Spec.bend as E
import ../../Position.bend as Pos
import ../../Protocol.bend as Protocol
import ../../../legal_probe/Chess.bend as Chess

def use_child(table: Array<U64>,+b: Chess.Board,+m: Chess.Ply,rows: {Board.valid(b) == True{} : Bool},
  coherent: {Pos.valid_ep(Chess.get_ep(b),b) == True{} : Bool},zero: {Chess.get_rights(b) == 0 : U32},
  here: E.member(m,E.moves(Chess.legal_moves(table,b)))) -> K.preserved(b,m):
  Emission.legal_member(table,b,m,rows,coherent,zero,here)
def finish(+c: O.Cells,+d: Nat,+seed: U64,+n: Nat,+extra: Nat,+w: Nat,+bl: Nat,-r: Array<U64> & Maybe<&2,Pos.Game>,
  +known: {O.pack(c) == Init.run(d,seed,n,extra) : Array<U64>},depth: {d == 17n : Nat},tables: {n == 128n : Nat},full: {extra == 64n : Nat},
  checked: S.root(c,w,bl,r)) -> S.result(O.pack(c),d,seed,n,extra,w,bl,r):
  (+same,good)=checked
  (same,(Equal.trans(Array<U64>,T.table(Maybe<&2,Pos.Game>,r),O.pack(c),Init.run(d,seed,n,extra),same,known),(depth,(tables,(full,good)))))
def lift(-a: Array<U64>,+d: Nat,+seed: U64,+n: Nat,+extra: Nat,+maybe: Maybe<&2,Pos.Game>,+xs: List<&2,String>,
  metadata: Metadata.parent(maybe),coherent: EP.ep_output(maybe),zero: Rights.empty_rights(maybe),
  built: {a == Init.run(d,seed,n,extra) : Array<U64>},+depth: {d == 17n : Nat},tables: {n == 128n : Nat},+full: {extra == 64n : Nat},
  cert: Exists(O.Cells,c => {O.pack(c) == a : Array<U64>})) -> S.replay(a,d,seed,n,extra,maybe,xs):
  (+c,e)=cert
  +same: {O.pack(c) == a : Array<U64>}=e
  +known: {O.pack(c) == Init.run(d,seed,n,extra) : Array<U64>}=Equal.trans(Array<U64>,O.pack(c),a,Init.run(d,seed,n,extra),same,built)
  %same : S.replay(_,d,seed,n,extra,maybe,xs)
  finish(c,d,seed,n,extra,S.white_limit(maybe),S.black_limit(maybe),Protocol.moves(xs,(O.pack(c),maybe)),known,depth,tables,full,
    Replay.tail(c,d,seed,n,extra,S.white_limit(maybe),S.black_limit(maybe),xs,(O.pack(c),maybe),known,depth,full,({==},S.initial(maybe,metadata,coherent,zero))))
def use_replay(a: Array<U64>,+d: Nat,+seed: U64,+n: Nat,+extra: Nat,+maybe: Maybe<&2,Pos.Game>,+xs: List<&2,String>,
  metadata: Metadata.parent(maybe),coherent: EP.ep_output(maybe),zero: Rights.empty_rights(maybe),
  built: {a == Init.run(d,seed,n,extra) : Array<U64>},depth: {d == 17n : Nat},tables: {n == 128n : Nat},full: {extra == 64n : Nat}) ->
  S.replay(a,d,seed,n,extra,maybe,xs):
  lift(a,d,seed,n,extra,maybe,xs,metadata,coherent,zero,built,depth,tables,full,R.reify(a))
