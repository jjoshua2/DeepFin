import Base
import ./Actual.bend as K
import ../protocol_replay_metadata/Spec.bend as Metadata
import ../protocol_replay_ep_coherence/Spec.bend as EP
import ../castle_rights/Spec.bend as Rights
import ../castle_rights/Rights.bend as RightsMove
import ../fen_validated_full/Root.bend as Value
import ../storage/Observe.bend as O
import ../table_preservation/Core.bend as T
import ../initialized_attacks/Build.bend as Init
import ../../Position.bend as Pos
import ../../Protocol.bend as Protocol
import ../../../legal_probe/Chess.bend as Chess

def white_limit(maybe: Maybe<&2,Pos.Game>) -> Nat:
  match maybe:
    case None{}: 0n
    case Some{g}: K.white(Pos.board(g))
def black_limit(maybe: Maybe<&2,Pos.Game>) -> Nat:
  match maybe:
    case None{}: 0n
    case Some{g}: K.black(Pos.board(g))
def board_counts(-w: Nat,-bl: Nat,-b: Chess.Board) -> Type:
  {K.white(b) == w : Nat} & {K.black(b) == bl : Nat}
def counts(-w: Nat,-bl: Nat,maybe: Maybe<&2,Pos.Game>) -> Type:
  match maybe:
    case None{}: Unit
    case Some{g}: board_counts(w,bl,Pos.board(g))
def naked(-w: Nat,-bl: Nat,-maybe: Maybe<&2,Pos.Game>) -> Type:
  Rights.empty_rights(maybe) & counts(w,bl,maybe)
def state(-w: Nat,-bl: Nat,-maybe: Maybe<&2,Pos.Game>) -> Type:
  Metadata.parent(maybe) & (EP.ep_output(maybe) & naked(w,bl,maybe))
def child_counts(-w: Nat,-bl: Nat,-b: Chess.Board) -> Type:
  {Chess.get_rights(b) == 0 : U32} & board_counts(w,bl,b)
def root(-c: O.Cells,-w: Nat,-bl: Nat,-r: Array<U64> & Maybe<&2,Pos.Game>) -> Type:
  {T.table(Maybe<&2,Pos.Game>,r) == O.pack(c) : Array<U64>} & state(w,bl,Value.value(r))
def naked_root(-c: O.Cells,-w: Nat,-bl: Nat,-r: Array<U64> & Maybe<&2,Pos.Game>) -> Type:
  {T.table(Maybe<&2,Pos.Game>,r) == O.pack(c) : Array<U64>} & naked(w,bl,Value.value(r))
def changed(+b: Chess.Board,+m: Chess.Ply,+w: Nat,+bl: Nat,zero: {Chess.get_rights(b) == 0 : U32},
  before: board_counts(w,bl,b),preserved: K.preserved(b,m)) -> child_counts(w,bl,Chess.make_move(b,m)):
  (pw,pb)=before
  (cw,cb)=preserved
  (RightsMove.zero(b,m,zero),(Equal.trans(Nat,K.white(Chess.make_move(b,m)),K.white(b),w,cw,pw),Equal.trans(Nat,K.black(Chess.make_move(b,m)),K.black(b),bl,cb,pb)))
def combine(+w: Nat,+bl: Nat,+maybe: Maybe<&2,Pos.Game>,ready: EP.output(maybe),good: naked(w,bl,maybe)) -> state(w,bl,maybe):
  match maybe:
    case None{}: (Unit{},(Unit{},good))
    case Some{g}:
      ((rows,(canonical,bound)),coherent)=ready
      ((rows,canonical),(coherent,good))
def joined(+c: O.Cells,+w: Nat,+bl: Nat,r: Array<U64> & Maybe<&2,Pos.Game>,
  checked: naked_root(c,w,bl,r),ready: EP.root(c,r)) -> root(c,w,bl,r):
  (kept,good)=checked
  (same,metadata)=ready
  (kept,combine(w,bl,Value.value(r),metadata,good))
def initial(+maybe: Maybe<&2,Pos.Game>,metadata: Metadata.parent(maybe),coherent: EP.ep_output(maybe),zero: Rights.empty_rights(maybe)) ->
  state(white_limit(maybe),black_limit(maybe),maybe):
  match maybe:
    case None{}: (metadata,(coherent,(zero,Unit{})))
    case Some{g}: (metadata,(coherent,(zero,({==},{==}))))
def result(-a: Array<U64>,-d: Nat,-seed: U64,-n: Nat,-extra: Nat,-w: Nat,-bl: Nat,-r: Array<U64> & Maybe<&2,Pos.Game>) -> Type:
  {T.table(Maybe<&2,Pos.Game>,r) == a : Array<U64>} &
    ({T.table(Maybe<&2,Pos.Game>,r) == Init.run(d,seed,n,extra) : Array<U64>} &
      ({d == 17n : Nat} & ({n == 128n : Nat} & ({extra == 64n : Nat} & state(w,bl,Value.value(r))))))
def replay(-a: Array<U64>,-d: Nat,-seed: U64,-n: Nat,-extra: Nat,-maybe: Maybe<&2,Pos.Game>,-xs: List<&2,String>) -> Type:
  result(a,d,seed,n,extra,white_limit(maybe),black_limit(maybe),Protocol.moves(xs,(a,maybe)))
