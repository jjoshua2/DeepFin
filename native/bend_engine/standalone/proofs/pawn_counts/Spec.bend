import Base
import ./Actual.bend as A
import ./Bounds.bend as B
import ../color_counts/Naturals.bend as N
import ../castle_rights/Rights.bend as Rights
import ../castle_rights/Spec.bend as RightsSpec
import ../fen_validated_full/Root.bend as Value
import ../storage/Observe.bend as O
import ../table_preservation/Core.bend as T
import ../../Position.bend as Pos
import ../../Protocol.bend as Protocol
import ../../../legal_probe/Chess.bend as Chess

def board_good(-b: Chess.Board) -> Type:
  {Chess.get_rights(b) == 0 : U32} & B.valid(b)
def good(maybe: Maybe<&2,Pos.Game>) -> Type:
  match maybe:
    case None{}: Unit
    case Some{g}: board_good(Pos.board(g))
def bounded(maybe: Maybe<&2,Pos.Game>) -> Type:
  match maybe:
    case None{}: Unit
    case Some{g}: B.valid(Pos.board(g))
def good_from(+maybe: Maybe<&2,Pos.Game>,zero: RightsSpec.empty_rights(maybe),bounds: bounded(maybe)) -> good(maybe):
  match maybe:
    case None{}: Unit{}
    case Some{g}: (zero,bounds)
def white_limit(maybe: Maybe<&2,Pos.Game>) -> Nat:
  match maybe:
    case None{}: 0n
    case Some{g}: A.white(Pos.board(g))
def black_limit(maybe: Maybe<&2,Pos.Game>) -> Nat:
  match maybe:
    case None{}: 0n
    case Some{g}: A.black(Pos.board(g))
def board_within(-w: Nat,-bl: Nat,-b: Chess.Board) -> Type:
  {Nat.is_le(A.white(b),w) == True{} : Bool} & {Nat.is_le(A.black(b),bl) == True{} : Bool}
def child_state(-w: Nat,-bl: Nat,-b: Chess.Board) -> Type:
  board_good(b) & board_within(w,bl,b)
def within(-w: Nat,-bl: Nat,maybe: Maybe<&2,Pos.Game>) -> Type:
  match maybe:
    case None{}: Unit
    case Some{g}: board_within(w,bl,Pos.board(g))
def state(-w: Nat,-bl: Nat,-maybe: Maybe<&2,Pos.Game>) -> Type:
  good(maybe) & within(w,bl,maybe)
def root(-c: O.Cells,-w: Nat,-bl: Nat,-r: Array<U64> & Maybe<&2,Pos.Game>) -> Type:
  {T.table(Maybe<&2,Pos.Game>,r) == O.pack(c) : Array<U64>} & state(w,bl,Value.value(r))
def result(-a: Array<U64>,-w: Nat,-bl: Nat,-r: Array<U64> & Maybe<&2,Pos.Game>) -> Type:
  {T.table(Maybe<&2,Pos.Game>,r) == a : Array<U64>} & state(w,bl,Value.value(r))
def replay(-a: Array<U64>,-maybe: Maybe<&2,Pos.Game>,-xs: List<&2,String>) -> Type:
  result(a,white_limit(maybe),black_limit(maybe),Protocol.moves(xs,(a,maybe)))
def initial(+maybe: Maybe<&2,Pos.Game>,parent: good(maybe)) -> state(white_limit(maybe),black_limit(maybe),maybe):
  match maybe:
    case None{}: (parent,Unit{})
    case Some{g}: (parent,(N.reflexive(A.white(Pos.board(g))),N.reflexive(A.black(Pos.board(g)))))
def within_changed(+b: Chess.Board,+m: Chess.Ply,+w: Nat,+bl: Nat,prior: board_within(w,bl,b),changes: A.decreases(b,m)) ->
  board_within(w,bl,Chess.make_move(b,m)):
  (pw,pb)=prior
  (cw,cb)=changes
  (N.transitive(A.white(Chess.make_move(b,m)),A.white(b),w,cw,pw),
    N.transitive(A.black(Chess.make_move(b,m)),A.black(b),bl,cb,pb))
def changed(+b: Chess.Board,+m: Chess.Ply,+w: Nat,+bl: Nat,parent: board_good(b),prior: board_within(w,bl,b),changes: A.decreases(b,m)) ->
  board_good(Chess.make_move(b,m)) & board_within(w,bl,Chess.make_move(b,m)):
  (zero,bounds)=parent
  (+cw,+cb)=changes
  ((Rights.zero(b,m,zero),B.post(b,m,bounds,(cw,cb))),within_changed(b,m,w,bl,prior,(cw,cb)))
