# Exact structural bit relocation counts, independent of bitboard enumeration.
import Base
import ../decoder/Bits.bend as B
import ../u64/Words.bend as Old
import ../ray/Algebra.bend as A
import ../pawn_counts/Intersection.bend as I

def full(+n: Nat,+a: Word(n)) ->
  {Word.and(n,a,Word.not(n,Word.zero(n))) == a : Word(n)}:
  match n:
    case 0n:
      match a:
        case WNil{}: {==}
    case 1n+p:
      match a:
        case WCon{False{},+t}:
          Equal.cong(Word(p),Word(1n+p),x => WCon{False{},x},Word.and(p,t,Word.not(p,Word.zero(p))),t,full(p,t))
        case WCon{True{},+t}:
          Equal.cong(Word(p),Word(1n+p),x => WCon{True{},x},Word.and(p,t,Word.not(p,Word.zero(p))),t,full(p,t))

def erase(+n: Nat,+a: Word(n),+i: Nat,owned: {B.get(n,a,i) == True{} : Bool}) ->
  {1n+Word.count(n,Word.and(n,a,Word.not(n,B.bit(n,i)))) == Word.count(n,a) : Nat}:
  match n:
    case 0n: Old.false_true({1n+Word.count(0n,Word.and(0n,a,Word.not(0n,B.bit(0n,i)))) == Word.count(0n,a) : Nat},owned)
    case 1n+p:
      match a i:
        case WCon{False{},t} 0n:
          Old.false_true({1n+Word.count(1n+p,Word.and(1n+p,a,Word.not(1n+p,B.bit(1n+p,0n)))) == Word.count(1n+p,a) : Nat},owned)
        case WCon{True{},+t} 0n:
          Equal.cong(Word(p),Nat,x => 1n+Word.count(p,x),Word.and(p,t,Word.not(p,Word.zero(p))),t,full(p,t))
        case WCon{False{},t} 1n+j: erase(p,t,j,owned)
        case WCon{True{},+t} 1n+j:
          Equal.cong(Nat,Nat,x => 1n+x,1n+Word.count(p,Word.and(p,t,Word.not(p,B.bit(p,j)))),Word.count(p,t),erase(p,t,j,owned))

def insert(+n: Nat,+a: Word(n),+i: Nat,below: {Nat.is_lt(i,n) == True{} : Bool},fresh: {B.get(n,a,i) == False{} : Bool}) ->
  {Word.count(n,Word.or(n,a,B.bit(n,i))) == 1n+Word.count(n,a) : Nat}:
  match n:
    case 0n: B.impossible({Word.count(0n,Word.or(0n,a,B.bit(0n,i))) == 1n+Word.count(0n,a) : Nat},i,below)
    case 1n+p:
      match a i:
        case WCon{False{},+t} 0n:
          Equal.cong(Word(p),Nat,x => 1n+Word.count(p,x),Word.or(p,t,Word.zero(p)),t,A.word_zero(p,t))
        case WCon{True{},t} 0n:
          Old.false_true({Word.count(1n+p,Word.or(1n+p,a,B.bit(1n+p,0n))) == 1n+Word.count(1n+p,a) : Nat},Equal.sym(Bool,True{},False{},fresh))
        case WCon{False{},t} 1n+j: insert(p,t,j,below,fresh)
        case WCon{True{},+t} 1n+j:
          Equal.cong(Nat,Nat,x => 1n+x,Word.count(p,Word.or(p,t,B.bit(p,j))),1n+Word.count(p,t),insert(p,t,j,below,fresh))

def and_false(+other: Bool) -> {Bool.and(False{},other) == False{} : Bool}:
  match other:
    case False{}: {==}
    case True{}: {==}

def cleared_fresh(+n: Nat,+a: Word(n),+src: Nat,+dst: Nat,fresh: {B.get(n,a,dst) == False{} : Bool}) ->
  {B.get(n,Word.and(n,a,Word.not(n,B.bit(n,src))),dst) == False{} : Bool}:
  Equal.trans(Bool,B.get(n,Word.and(n,a,Word.not(n,B.bit(n,src))),dst),
    Bool.and(B.get(n,a,dst),B.get(n,Word.not(n,B.bit(n,src)),dst)),False{},
    I.get_and(n,a,Word.not(n,B.bit(n,src)),dst),
    Equal.trans(Bool,Bool.and(B.get(n,a,dst),B.get(n,Word.not(n,B.bit(n,src)),dst)),
      Bool.and(False{},B.get(n,Word.not(n,B.bit(n,src)),dst)),False{},
      Equal.cong(Bool,Bool,x => Bool.and(x,B.get(n,Word.not(n,B.bit(n,src)),dst)),B.get(n,a,dst),False{},fresh),
      and_false(B.get(n,Word.not(n,B.bit(n,src)),dst))))

def relocate(+n: Nat,+a: Word(n),+src: Nat,+dst: Nat,
  below: {Nat.is_lt(dst,n) == True{} : Bool},owned: {B.get(n,a,src) == True{} : Bool},fresh: {B.get(n,a,dst) == False{} : Bool}) ->
  {Word.count(n,Word.or(n,Word.and(n,a,Word.not(n,B.bit(n,src))),B.bit(n,dst))) == Word.count(n,a) : Nat}:
  Equal.trans(Nat,Word.count(n,Word.or(n,Word.and(n,a,Word.not(n,B.bit(n,src))),B.bit(n,dst))),
    1n+Word.count(n,Word.and(n,a,Word.not(n,B.bit(n,src)))),Word.count(n,a),
    insert(n,Word.and(n,a,Word.not(n,B.bit(n,src))),dst,below,cleared_fresh(n,a,src,dst,fresh)),
    erase(n,a,src,owned))

def erase_empty(+n: Nat,+a: Word(n),+i: Nat,fresh: {B.get(n,a,i) == False{} : Bool}) ->
  {Word.count(n,Word.and(n,a,Word.not(n,B.bit(n,i)))) == Word.count(n,a) : Nat}:
  match n:
    case 0n: {==}
    case 1n+p:
      match a i:
        case WCon{False{},+t} 0n:
          Equal.cong(Word(p),Nat,x => Word.count(p,x),Word.and(p,t,Word.not(p,Word.zero(p))),t,full(p,t))
        case WCon{True{},t} 0n:
          Old.false_true({Word.count(1n+p,Word.and(1n+p,a,Word.not(1n+p,B.bit(1n+p,0n)))) == Word.count(1n+p,a) : Nat},Equal.sym(Bool,True{},False{},fresh))
        case WCon{False{},t} 1n+j: erase_empty(p,t,j,fresh)
        case WCon{True{},+t} 1n+j:
          Equal.cong(Nat,Nat,x => 1n+x,Word.count(p,Word.and(p,t,Word.not(p,B.bit(p,j)))),Word.count(p,t),erase_empty(p,t,j,fresh))
