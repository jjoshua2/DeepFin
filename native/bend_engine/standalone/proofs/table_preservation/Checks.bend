# Composition of actual attack callbacks; only storage identity is specified here.
import Base
import ./Core.bend as C
import ./Queries.bend as Q
import ../storage/Observe.bend as O
import ../../../legal_probe/Chess.bend as Chess

def bishop(+b: Chess.Board,+by: U32,+hits: U64,r: Array<U64> & U64) ->
  {C.table(Bool,Chess.attacked_bishop(b,by,hits,r)) == C.table(U64,r) : Array<U64>}:
  (table,ba) = r
  {==}

def rook(+b: Chess.Board,+sq: U32,+by: U32,+hits: U64,r: Array<U64> & U64) ->
  {C.table(Bool,Chess.attacked_rook(b,sq,by,hits,r)) == C.table(U64,r) : Array<U64>}:
  (table,+mask) = r
  +new_hits = U64.or(hits,U64.and(mask,U64.or(Chess.get_rooks(b),Chess.get_queens(b))))
  C.apply(Bool,a => Chess.attacked_rook(b,sq,by,hits,(a,mask)),table,+c =>
    Equal.trans(Array<U64>,C.table(Bool,Chess.attacked_rook(b,sq,by,hits,(O.pack(c),mask))),
      C.table(U64,Chess.attack(2,O.pack(c),sq,by,Chess.occupied(b))),O.pack(c),
      bishop(b,by,new_hits,Chess.attack(2,O.pack(c),sq,by,Chess.occupied(b))),
      Q.attack(2,O.pack(c),sq,by,Chess.occupied(b))))

def king(+b: Chess.Board,+sq: U32,+by: U32,+hits: U64,r: Array<U64> & U64) ->
  {C.table(Bool,Chess.attacked_king(b,sq,by,hits,r)) == C.table(U64,r) : Array<U64>}:
  (table,+mask) = r
  +new_hits = U64.or(hits,U64.and(mask,Chess.get_kings(b)))
  C.apply(Bool,a => Chess.attacked_king(b,sq,by,hits,(a,mask)),table,+c =>
    Equal.trans(Array<U64>,C.table(Bool,Chess.attacked_king(b,sq,by,hits,(O.pack(c),mask))),
      C.table(U64,Chess.attack(3,O.pack(c),sq,by,Chess.occupied(b))),O.pack(c),
      rook(b,sq,by,new_hits,Chess.attack(3,O.pack(c),sq,by,Chess.occupied(b))),
      Q.attack(3,O.pack(c),sq,by,Chess.occupied(b))))

def knight(+b: Chess.Board,+sq: U32,+by: U32,+hits: U64,r: Array<U64> & U64) ->
  {C.table(Bool,Chess.attacked_knight(b,sq,by,hits,r)) == C.table(U64,r) : Array<U64>}:
  (table,+mask) = r
  +new_hits = U64.or(hits,U64.and(mask,Chess.get_knights(b)))
  C.apply(Bool,a => Chess.attacked_knight(b,sq,by,hits,(a,mask)),table,+c =>
    Equal.trans(Array<U64>,C.table(Bool,Chess.attacked_knight(b,sq,by,hits,(O.pack(c),mask))),
      C.table(U64,Chess.attack(5,O.pack(c),sq,by,Chess.occupied(b))),O.pack(c),
      king(b,sq,by,new_hits,Chess.attack(5,O.pack(c),sq,by,Chess.occupied(b))),
      Q.attack(5,O.pack(c),sq,by,Chess.occupied(b))))

def pawn(+b: Chess.Board,+sq: U32,+by: U32,r: Array<U64> & U64) ->
  {C.table(Bool,Chess.attacked_pawn(b,sq,by,r)) == C.table(U64,r) : Array<U64>}:
  (table,+mask) = r
  C.apply(Bool,a => Chess.attacked_pawn(b,sq,by,(a,mask)),table,+c =>
    Equal.trans(Array<U64>,C.table(Bool,Chess.attacked_pawn(b,sq,by,(O.pack(c),mask))),
      C.table(U64,Chess.attack(1,O.pack(c),sq,by,Chess.occupied(b))),O.pack(c),
      knight(b,sq,by,U64.and(mask,Chess.get_pawns(b)),Chess.attack(1,O.pack(c),sq,by,Chess.occupied(b))),
      Q.attack(1,O.pack(c),sq,by,Chess.occupied(b))))

def attacked(table: Array<U64>,+b: Chess.Board,+sq: U32,+by: U32) ->
  {C.table(Bool,Chess.attacked(table,b,sq,by)) == table : Array<U64>}:
  C.apply(Bool,a => Chess.attacked(a,b,sq,by),table,+c =>
    Equal.trans(Array<U64>,C.table(Bool,Chess.attacked(O.pack(c),b,sq,by)),
      C.table(U64,Chess.attack(0,O.pack(c),sq,U32.xor(by,1),Chess.occupied(b))),O.pack(c),
      pawn(b,sq,by,Chess.attack(0,O.pack(c),sq,U32.xor(by,1),Chess.occupied(b))),
      Q.attack(0,O.pack(c),sq,U32.xor(by,1),Chess.occupied(b))))

def check(table: Array<U64>,+b: Chess.Board,+side: U32) ->
  {C.table(Bool,Chess.in_check(table,b,side)) == table : Array<U64>}:
  attacked(table,b,U64.ctz(U64.and(Chess.get_kings(b),Chess.color(b,side))),U32.xor(side,1))
