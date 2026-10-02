# Actual scan_after: exact pair equality for an arbitrary affine table and tail.
import Base
import ../../../legal_probe/Chess.bend as Chess
import ../storage/Observe.bend as O
import ../storage/Representation.bend as R
import ./Spec.bend as S
import ./Emission.bend as E

def cells(+c: O.Cells,+targets: U64,+src: U32,+pawn: Bool,+ep: U32,+tail: List<&2,Chess.Ply>) ->
  {Chess.scan_after(src,pawn,ep,tail,(O.pack(c),targets)) ==
    (O.pack(c),S.expand(Chess.bit_squares(64n,U64.is_zero(targets),targets,Nil{}),src,pawn,ep,tail)) : Array<U64> & List<&2,Chess.Ply>}:
  Equal.cong(List<&2,Chess.Ply>,Array<U64> & List<&2,Chess.Ply>,ms => (O.pack(c),ms),
    Chess.destinations(64n,U64.is_zero(targets),targets,src,pawn,ep,tail),
    S.expand(Chess.bit_squares(64n,U64.is_zero(targets),targets,Nil{}),src,pawn,ep,tail),
    E.destinations(64n,U64.is_zero(targets),targets,src,pawn,ep,tail))

def lift(-table: Array<U64>,+targets: U64,+src: U32,+pawn: Bool,+ep: U32,+tail: List<&2,Chess.Ply>,
  cert: Exists(O.Cells,c => {O.pack(c) == table : Array<U64>})) ->
  {Chess.scan_after(src,pawn,ep,tail,(table,targets)) ==
    (table,S.expand(Chess.bit_squares(64n,U64.is_zero(targets),targets,Nil{}),src,pawn,ep,tail)) : Array<U64> & List<&2,Chess.Ply>}:
  (+c,e) = cert
  +same: {O.pack(c) == table : Array<U64>} = e
  %same : {Chess.scan_after(src,pawn,ep,tail,(_,targets)) ==
    (table,S.expand(Chess.bit_squares(64n,U64.is_zero(targets),targets,Nil{}),src,pawn,ep,tail)) : Array<U64> & List<&2,Chess.Ply>}
  %same : {Chess.scan_after(src,pawn,ep,tail,(O.pack(c),targets)) ==
    (_,S.expand(Chess.bit_squares(64n,U64.is_zero(targets),targets,Nil{}),src,pawn,ep,tail)) : Array<U64> & List<&2,Chess.Ply>}
  cells(c,targets,src,pawn,ep,tail)

def exact(table: Array<U64>,+targets: U64,+src: U32,+pawn: Bool,+ep: U32,+tail: List<&2,Chess.Ply>) ->
  {Chess.scan_after(src,pawn,ep,tail,(table,targets)) ==
    (table,S.expand(Chess.bit_squares(64n,U64.is_zero(targets),targets,Nil{}),src,pawn,ep,tail)) : Array<U64> & List<&2,Chess.Ply>}:
  lift(table,targets,src,pawn,ep,tail,R.reify(table))
