# Check every production reduction callback; actual arrays are never copied.
import Base
import ../../../legal_probe/Chess.bend as Chess
import ./Spec.bend as S
import ./Queries.bend as Q
import ./Reduction.bend as R

def bishop(+b: Chess.Board,+by: U32,+pa: U64,+na: U64,+ka: U64,+ra: U64,r: Array<U64> & U64) ->
  {Chess.attacked_bishop(b,by,R.prefix(b,pa,na,ka,ra),r) == Q.bishop(b,by,pa,na,ka,ra,r) : Array<U64> & Bool}:
  (table,ba) = r
  +m = S.masks(pa,na,ka,ra,ba)
  %Equal.sym(Bool,Bool.not(U64.is_zero(U64.and(R.hits(b,m),Chess.color(b,by)))),S.present(b,by,m),R.decision(b,by,m)) :
    {(table,_) == (table,S.present(b,by,m)) : Array<U64> & Bool}
  {==}

def rook(+b: Chess.Board,+sq: U32,+by: U32,+pa: U64,+na: U64,+ka: U64,r: Array<U64> & U64) ->
  {Chess.attacked_rook(b,sq,by,U64.or(U64.or(U64.and(pa,Chess.get_pawns(b)),U64.and(na,Chess.get_knights(b))),U64.and(ka,Chess.get_kings(b))),r) == Q.rook(b,sq,by,pa,na,ka,r) : Array<U64> & Bool}:
  (table,ra) = r
  bishop(b,by,pa,na,ka,ra,Chess.attack(2,table,sq,by,Chess.occupied(b)))

def king(+b: Chess.Board,+sq: U32,+by: U32,+pa: U64,+na: U64,r: Array<U64> & U64) ->
  {Chess.attacked_king(b,sq,by,U64.or(U64.and(pa,Chess.get_pawns(b)),U64.and(na,Chess.get_knights(b))),r) == Q.king(b,sq,by,pa,na,r) : Array<U64> & Bool}:
  (table,ka) = r
  rook(b,sq,by,pa,na,ka,Chess.attack(3,table,sq,by,Chess.occupied(b)))

def knight(+b: Chess.Board,+sq: U32,+by: U32,+pa: U64,r: Array<U64> & U64) ->
  {Chess.attacked_knight(b,sq,by,U64.and(pa,Chess.get_pawns(b)),r) == Q.knight(b,sq,by,pa,r) : Array<U64> & Bool}:
  (table,na) = r
  king(b,sq,by,pa,na,Chess.attack(5,table,sq,by,Chess.occupied(b)))

def pawn(+b: Chess.Board,+sq: U32,+by: U32,r: Array<U64> & U64) ->
  {Chess.attacked_pawn(b,sq,by,r) == Q.pawn(b,sq,by,r) : Array<U64> & Bool}:
  (table,pa) = r
  knight(b,sq,by,pa,Chess.attack(1,table,sq,by,Chess.occupied(b)))

def attacked(table: Array<U64>,+b: Chess.Board,+sq: U32,+by: U32) ->
  {Chess.attacked(table,b,sq,by) == Q.run(table,b,sq,by) : Array<U64> & Bool}:
  pawn(b,sq,by,Chess.attack(0,table,sq,U32.xor(by,1),Chess.occupied(b)))
