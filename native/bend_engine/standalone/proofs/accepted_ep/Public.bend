# Consume the actual affine array once, preserving complete value identity.
import Base
import ./Accept.bend as Accept
import ../accepted_castle/Pair.bend as Pair
import ../castle_emission/Spec.bend as E
import ../castle_chain/Spec.bend as C
import ../storage/Observe.bend as O
import ../storage/Representation.bend as R
import ../../../legal_probe/Chess.bend as Chess

def prepared_lift(-a: Array<U64>,+b: Chess.Board,+xs: List<&2,Chess.Ply>,+m: Chess.Ply,
  cert: Exists(O.Cells,c => {O.pack(c) == a : Array<U64>})) ->
  {C.flag(m) == 1 : U32} -> E.member(m,E.moves(Chess.filter_prepare(b,(a,xs)))) ->
  {Chess.in_check(a,Chess.make_move(b,m),Chess.get_turn(b)) == (a,False{}) : Array<U64> & Bool}:
  (+c,e) = cert
  +same: {O.pack(c) == a : Array<U64>} = e
  %same : {C.flag(m) == 1 : U32} -> E.member(m,E.moves(Chess.filter_prepare(b,(_,xs)))) ->
    {Chess.in_check(a,Chess.make_move(b,m),Chess.get_turn(b)) == (a,False{}) : Array<U64> & Bool}
  %same : {C.flag(m) == 1 : U32} -> E.member(m,E.moves(Chess.filter_prepare(b,(O.pack(c),xs)))) ->
    {Chess.in_check(_,Chess.make_move(b,m),Chess.get_turn(b)) == (a,False{}) : Array<U64> & Bool}
  %same : {C.flag(m) == 1 : U32} -> E.member(m,E.moves(Chess.filter_prepare(b,(O.pack(c),xs)))) ->
    {Chess.in_check(O.pack(c),Chess.make_move(b,m),Chess.get_turn(b)) == (_,False{}) : Array<U64> & Bool}
  tag => here => Pair.false_check(c,b,m,Accept.prepared(c,b,xs,m,tag,here))

def prepared(a: Array<U64>,+b: Chess.Board,+xs: List<&2,Chess.Ply>,+m: Chess.Ply,
  tag: {C.flag(m) == 1 : U32},here: E.member(m,E.moves(Chess.filter_prepare(b,(a,xs))))) ->
  {Chess.in_check(a,Chess.make_move(b,m),Chess.get_turn(b)) == (a,False{}) : Array<U64> & Bool}:
  prepared_lift(a,b,xs,m,R.reify(a))(tag)(here)

def legal_lift(-a: Array<U64>,+b: Chess.Board,+m: Chess.Ply,
  cert: Exists(O.Cells,c => {O.pack(c) == a : Array<U64>})) ->
  E.member(m,E.moves(Chess.legal_moves(a,b))) -> {C.flag(m) == 1 : U32} ->
  {Chess.in_check(a,Chess.make_move(b,m),Chess.get_turn(b)) == (a,False{}) : Array<U64> & Bool}:
  (+c,e) = cert
  +same: {O.pack(c) == a : Array<U64>} = e
  %same : E.member(m,E.moves(Chess.legal_moves(_,b))) -> {C.flag(m) == 1 : U32} ->
    {Chess.in_check(a,Chess.make_move(b,m),Chess.get_turn(b)) == (a,False{}) : Array<U64> & Bool}
  %same : E.member(m,E.moves(Chess.legal_moves(O.pack(c),b))) -> {C.flag(m) == 1 : U32} ->
    {Chess.in_check(_,Chess.make_move(b,m),Chess.get_turn(b)) == (a,False{}) : Array<U64> & Bool}
  %same : E.member(m,E.moves(Chess.legal_moves(O.pack(c),b))) -> {C.flag(m) == 1 : U32} ->
    {Chess.in_check(O.pack(c),Chess.make_move(b,m),Chess.get_turn(b)) == (_,False{}) : Array<U64> & Bool}
  here => tag => Pair.false_check(c,b,m,Accept.known(c,b,m,here,tag))

def legal(a: Array<U64>,+b: Chess.Board,+m: Chess.Ply,
  here: E.member(m,E.moves(Chess.legal_moves(a,b))),tag: {C.flag(m) == 1 : U32}) ->
  {Chess.in_check(a,Chess.make_move(b,m),Chess.get_turn(b)) == (a,False{}) : Array<U64> & Bool}:
  legal_lift(a,b,m,R.reify(a))(here)(tag)
