# Unvisited squares: all lower ranks and the current rank at/after the cursor.
import Base
import ../../Position.bend as P
import ../../../legal_probe/Chess.bend as Chess
import ../board/Spec.bend as B
import ../parser/Spec.bend as Parser

def row(+n: Nat,+at: Nat,+file: Nat,+rank: Nat,+r: Nat) -> Word(n):
  match n:
    case 0n: WNil{}
    case 1n+p:
      WCon{Bool.or(Nat.is_lt(r,rank),Bool.and(Nat.is_eq(r,rank),Nat.is_le(file,at))),row(p,1n+at,file,rank,r)}

def half(+f: Nat,+r: Nat,+base: Nat) -> Word(32n):
  Word.join(8n,24n,row(8n,0n,f,r,base),
    Word.join(8n,16n,row(8n,0n,f,r,1n+base),
      Word.join(8n,8n,row(8n,0n,f,r,2n+base),row(8n,0n,f,r,3n+base))))

def pending(+file: U32,+rank: U32) -> U64:
  U64{U32{half(U32.to_nat(file),U32.to_nat(rank),0n)},U32{half(U32.to_nat(file),U32.to_nat(rank),4n)}}

def bounds(+file: U32,+rank: U32) -> Bool:
  Bool.and(Nat.is_lt(U32.to_nat(file),9n),Nat.is_lt(U32.to_nat(rank),8n))

def state(+b: Chess.Board,+file: U32,+rank: U32) -> Bool:
  Bool.and(bounds(file,rank),Bool.and(B.valid(b),B.fresh_mask(b,pending(file,rank))))

def safe(l: P.Layout) -> Bool:
  P.Layout{b,file,rank,live} = l
  Bool.pick(Bool,live,state(b,file,rank),True{})

def initial() -> P.Layout:
  P.Layout{P.empty(),0,7,True{}}

def square(+file: U32,+rank: U32) -> U32:
  U32.add(file,U32.mul(rank,8))

def digit(+c: U32) -> Bool:
  Bool.and(U32.is_ge(c,49),U32.is_le(c,56))

# These observations expose the useful invariant, not a success flag alone.
def ready(l: P.Layout) -> Bool:
  P.Layout{b,file,rank,live} = l
  state(b,file,rank)

def target(l: P.Layout) -> U32:
  P.Layout{b,file,rank,live} = l
  square(file,rank)

def typed(p: B.Piece,white: Bool) -> String:
  SCon{Chr{Parser.letter(p,white)},SNil{}}

def fresh_target(+l: P.Layout) -> Bool:
  Bool.and(B.fresh(Parser.board(l),target(l)),Nat.is_lt(U32.to_nat(target(l)),64n))

def result_valid(result: Maybe<&2,Chess.Board>) -> Bool:
  match result:
    case None{}: True{}
    case Some{b}: B.valid(b)
