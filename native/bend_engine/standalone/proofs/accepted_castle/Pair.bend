# Recover the whole result pair using the existing table-preservation proof.
import Base
import ./Spec.bend as S
import ../table_preservation/Core.bend as C
import ../table_preservation/Checks.bend as Checks
import ../table_preservation/Queries.bend as Queries
import ../storage/Observe.bend as O
import ../../../legal_probe/Chess.bend as Chess

def boolean(-a: Array<U64>,r: Array<U64> & Bool,
  kept: {C.table(Bool,r) == a : Array<U64>}) ->
  {r == (a,S.answer(r)) : Array<U64> & Bool}:
  (t,+v) = r
  %kept : {(t,v) == (_,v) : Array<U64> & Bool}
  {==}

def word(-a: Array<U64>,r: Array<U64> & U64,
  kept: {C.table(U64,r) == a : Array<U64>}) ->
  {r == (a,S.mask(r)) : Array<U64> & U64}:
  (t,+v) = r
  %kept : {(t,v) == (_,v) : Array<U64> & U64}
  {==}

def check(+c: O.Cells,+b: Chess.Board,+side: U32) ->
  {Chess.in_check(O.pack(c),b,side) == (O.pack(c),S.answer(Chess.in_check(O.pack(c),b,side))) : Array<U64> & Bool}:
  boolean(O.pack(c),Chess.in_check(O.pack(c),b,side),Checks.check(O.pack(c),b,side))

def attack(+c: O.Cells,+b: Chess.Board) ->
  {Chess.attack(4,O.pack(c),U64.ctz(U64.and(Chess.get_kings(b),Chess.color(b,Chess.get_turn(b)))),Chess.get_turn(b),Chess.occupied(b)) == (O.pack(c),S.rays(c,b)) : Array<U64> & U64}:
  +sq = U64.ctz(U64.and(Chess.get_kings(b),Chess.color(b,Chess.get_turn(b))))
  word(O.pack(c),Chess.attack(4,O.pack(c),sq,Chess.get_turn(b),Chess.occupied(b)),Queries.attack(4,O.pack(c),sq,Chess.get_turn(b),Chess.occupied(b)))

def false_check(+c: O.Cells,+b: Chess.Board,+m: Chess.Ply,no: S.safe(c,b,m)) ->
  {Chess.in_check(O.pack(c),Chess.make_move(b,m),Chess.get_turn(b)) == (O.pack(c),False{}) : Array<U64> & Bool}:
  Equal.trans(Array<U64> & Bool,Chess.in_check(O.pack(c),Chess.make_move(b,m),Chess.get_turn(b)),
    (O.pack(c),S.checked(c,b,m)),(O.pack(c),False{}),
    check(c,Chess.make_move(b,m),Chess.get_turn(b)),
    Equal.cong(Bool,Array<U64> & Bool,v => (O.pack(c),v),S.checked(c,b,m),False{},no))
