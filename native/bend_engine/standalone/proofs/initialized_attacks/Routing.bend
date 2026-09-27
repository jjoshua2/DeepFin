# Small actual-source dispatch bridges, checked without normalized table construction.
import Base
import ../../../legal_probe/Chess.bend as Chess

def join(a: U64,r: Array<U64> & U64) -> Array<U64> & U64:
  (table,b) = r
  (table,U64.or(a,b))

def queen_tail(+q: U32,+occ: U64,r: Array<U64> & U64) -> Array<U64> & U64:
  (table,rook) = r
  join(rook,Chess.slide(table,U32.add(64,q),occ))

def rook(table: Array<U64>,+q: U32,side: U32,occ: U64) ->
  {Chess.attack(3,table,q,side,occ) == Chess.slide(table,q,occ) : Array<U64> & U64}:
  {==}

def bishop(table: Array<U64>,+q: U32,side: U32,occ: U64) ->
  {Chess.attack(2,table,q,side,occ) == Chess.slide(table,U32.add(64,q),occ) : Array<U64> & U64}:
  {==}

def join_actual(a: U64,r: Array<U64> & U64) ->
  {Chess.or_result(a,r) == join(a,r) : Array<U64> & U64}:
  (table,b) = r
  {==}

def queen_tail_actual(+q: U32,+occ: U64,r: Array<U64> & U64) ->
  {Chess.queen_after(q,occ,r) == queen_tail(q,occ,r) : Array<U64> & U64}:
  (table,rook) = r
  join_actual(rook,Chess.slide(table,U32.add(64,q),occ))

def queen(table: Array<U64>,+q: U32,side: U32,+occ: U64) ->
  {Chess.attack(4,table,q,side,occ) == queen_tail(q,occ,Chess.slide(table,q,occ)) : Array<U64> & U64}:
  queen_tail_actual(q,occ,Chess.slide(table,q,occ))
