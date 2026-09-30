# Independent observation reduction over the actual five state-threaded mask queries.
# Mask generation remains a separate specification obligation.
import Base
import ../../../legal_probe/Chess.bend as Chess
import ./Spec.bend as S

def bishop(+b: Chess.Board,by: U32,pa: U64,na: U64,ka: U64,ra: U64,r: Array<U64> & U64) -> Array<U64> & Bool:
  (table,ba) = r
  (table,S.present(b,by,S.Masks{pa,na,ka,ra,ba}))

def rook(+b: Chess.Board,+sq: U32,+by: U32,pa: U64,na: U64,ka: U64,r: Array<U64> & U64) -> Array<U64> & Bool:
  (table,ra) = r
  bishop(b,by,pa,na,ka,ra,Chess.attack(2,table,sq,by,Chess.occupied(b)))

def king(+b: Chess.Board,+sq: U32,+by: U32,pa: U64,na: U64,r: Array<U64> & U64) -> Array<U64> & Bool:
  (table,ka) = r
  rook(b,sq,by,pa,na,ka,Chess.attack(3,table,sq,by,Chess.occupied(b)))

def knight(+b: Chess.Board,+sq: U32,+by: U32,pa: U64,r: Array<U64> & U64) -> Array<U64> & Bool:
  (table,na) = r
  king(b,sq,by,pa,na,Chess.attack(5,table,sq,by,Chess.occupied(b)))

def pawn(+b: Chess.Board,+sq: U32,+by: U32,r: Array<U64> & U64) -> Array<U64> & Bool:
  (table,pa) = r
  knight(b,sq,by,pa,Chess.attack(1,table,sq,by,Chess.occupied(b)))

def run(table: Array<U64>,+b: Chess.Board,+sq: U32,+by: U32) -> Array<U64> & Bool:
  pawn(b,sq,by,Chess.attack(0,table,sq,U32.xor(by,1),Chess.occupied(b)))
