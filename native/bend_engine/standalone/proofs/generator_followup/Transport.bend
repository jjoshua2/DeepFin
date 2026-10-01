import Base
import ./Runtime.bend as Run
import ../table_preservation/Core.bend as C
import ../table_preservation/PROOF.bend as StorageProof
import ../table_preservation/LAWS.bend as Storage
import ../storage/Observe.bend as O
import ../storage/Representation.bend as R
import ../../../legal_probe/Chess.bend as Chess

def pair(-a: Array<U64>,+q: Chess.Board,+side: U32,r: Array<U64> & List<&2,Chess.Ply>,
  kept: {C.table(List<&2,Chess.Ply>,r) == a : Array<U64>}) ->
  {Run.follow(q,side,r) == Run.attach(Run.moves(r),Chess.in_check(a,q,side)) : Array<U64> & (List<&2,Chess.Ply> & Bool)}:
  (t,+ms) = r
  %Equal.sym(Array<U64>,t,a,kept) :
    {Run.follow(q,side,(_,ms)) == Run.attach(ms,Chess.in_check(a,q,side)) : Array<U64> & (List<&2,Chess.Ply> & Bool)}
  {==}

def known(+c: O.Cells,+b: Chess.Board,+q: Chess.Board,+side: U32) ->
  {Run.run(O.pack(c),b,q,side) == Run.attach(Run.moves(Chess.legal_moves(O.pack(c),b)),Chess.in_check(O.pack(c),q,side)) : Array<U64> & (List<&2,Chess.Ply> & Bool)}:
  pair(O.pack(c),q,side,Chess.legal_moves(O.pack(c),b),Storage.legal_moves_preserves_table(O.pack(c),b))

def lift(-a: Array<U64>,+b: Chess.Board,+q: Chess.Board,+side: U32,
  cert: Exists(O.Cells,c => {O.pack(c) == a : Array<U64>})) ->
  {Run.run(a,b,q,side) == Run.attach(Run.moves(Chess.legal_moves(a,b)),Chess.in_check(a,q,side)) : Array<U64> & (List<&2,Chess.Ply> & Bool)}:
  (+c,e) = cert
  +same: {O.pack(c) == a : Array<U64>} = e
  %same : {Run.run(_,b,q,side) == Run.attach(Run.moves(Chess.legal_moves(a,b)),Chess.in_check(a,q,side)) : Array<U64> & (List<&2,Chess.Ply> & Bool)}
  %same : {Run.run(O.pack(c),b,q,side) == Run.attach(Run.moves(Chess.legal_moves(_,b)),Chess.in_check(a,q,side)) : Array<U64> & (List<&2,Chess.Ply> & Bool)}
  %same : {Run.run(O.pack(c),b,q,side) == Run.attach(Run.moves(Chess.legal_moves(O.pack(c),b)),Chess.in_check(_,q,side)) : Array<U64> & (List<&2,Chess.Ply> & Bool)}
  known(c,b,q,side)

def actual(a: Array<U64>,+b: Chess.Board,+q: Chess.Board,+side: U32) ->
  {Run.run(a,b,q,side) == Run.attach(Run.moves(Chess.legal_moves(a,b)),Chess.in_check(a,q,side)) : Array<U64> & (List<&2,Chess.Ply> & Bool)}:
  lift(a,b,q,side,R.reify(a))
