# Structural count algebra. Ownership pays for at most one inserted bit.
import Base
import ./Naturals.bend as N
import ../u64/Words.bend as Old
import ../decoder/Bits.bend as B
import ../ray/Algebra.bend as A

def and_le(+n: Nat,+a: Word(n),+mask: Word(n)) ->
  {Nat.is_le(Word.count(n,Word.and(n,a,mask)),Word.count(n,a)) == True{} : Bool}:
  match n:
    case 0n: {==}
    case 1n+p:
      match a mask:
        case WCon{False{},t} WCon{False{},m}: and_le(p,t,m)
        case WCon{False{},t} WCon{True{},m}: and_le(p,t,m)
        case WCon{True{},+t} WCon{False{},+m}:
          Old.le_succ(Word.count(p,Word.and(p,t,m)),Word.count(p,t),and_le(p,t,m))
        case WCon{True{},t} WCon{True{},m}: and_le(p,t,m)

# A deletion mask containing one owned point strictly pays one unit of count.
def owned_delete(+n: Nat,+a: Word(n),+remove: Word(n),+i: Nat,
  owned: {B.get(n,a,i) == True{} : Bool},erased: {B.get(n,remove,i) == True{} : Bool}) ->
  {Nat.is_le(1n+Word.count(n,Word.and(n,a,Word.not(n,remove))),Word.count(n,a)) == True{} : Bool}:
  match n:
    case 0n:
      Old.false_true({Nat.is_le(1n+Word.count(0n,Word.and(0n,a,Word.not(0n,remove))),Word.count(0n,a)) == True{} : Bool},owned)
    case 1n+p:
      match a remove i:
        case WCon{False{},t} WCon{r,rt} 0n:
          Old.false_true({Nat.is_le(1n+Word.count(1n+p,Word.and(1n+p,a,Word.not(1n+p,remove))),Word.count(1n+p,a)) == True{} : Bool},owned)
        case WCon{True{},t} WCon{False{},rt} 0n:
          Old.false_true({Nat.is_le(1n+Word.count(1n+p,Word.and(1n+p,a,Word.not(1n+p,remove))),Word.count(1n+p,a)) == True{} : Bool},erased)
        case WCon{True{},t} WCon{True{},rt} 0n: and_le(p,t,Word.not(p,rt))
        case WCon{False{},t} WCon{False{},rt} 1n+j: owned_delete(p,t,rt,j,owned,erased)
        case WCon{False{},t} WCon{True{},rt} 1n+j: owned_delete(p,t,rt,j,owned,erased)
        case WCon{True{},t} WCon{False{},rt} 1n+j: owned_delete(p,t,rt,j,owned,erased)
        case WCon{True{},t} WCon{True{},rt} 1n+j: and_le(p,t,Word.not(p,rt))

def insert_le(+n: Nat,+a: Word(n),+i: Nat) ->
  {Nat.is_le(Word.count(n,Word.or(n,a,B.bit(n,i))),1n+Word.count(n,a)) == True{} : Bool}:
  match n:
    case 0n: {==}
    case 1n+p:
      match a i:
        case WCon{False{},+t} 0n:
          %Equal.sym(Word(p),Word.or(p,t,Word.zero(p)),t,A.word_zero(p,t)) : {Nat.is_le(1n+Word.count(p,_),1n+Word.count(p,t)) == True{} : Bool}
          N.reflexive(Word.count(p,t))
        case WCon{True{},+t} 0n:
          %Equal.sym(Word(p),Word.or(p,t,Word.zero(p)),t,A.word_zero(p,t)) : {Nat.is_le(1n+Word.count(p,_),2n+Word.count(p,t)) == True{} : Bool}
          Old.le_succ(Word.count(p,t),Word.count(p,t),N.reflexive(Word.count(p,t)))
        case WCon{False{},t} 1n+j: insert_le(p,t,j)
        case WCon{True{},t} 1n+j: insert_le(p,t,j)

def replace_le(+n: Nat,+a: Word(n),+remove: Word(n),+src: Nat,+dst: Nat,
  owned: {B.get(n,a,src) == True{} : Bool},erased: {B.get(n,remove,src) == True{} : Bool}) ->
  {Nat.is_le(Word.count(n,Word.or(n,Word.and(n,a,Word.not(n,remove)),B.bit(n,dst))),Word.count(n,a)) == True{} : Bool}:
  +kept=Word.and(n,a,Word.not(n,remove))
  N.transitive(Word.count(n,Word.or(n,kept,B.bit(n,dst))),1n+Word.count(n,kept),Word.count(n,a),
    insert_le(n,kept,dst),owned_delete(n,a,remove,src,owned,erased))
