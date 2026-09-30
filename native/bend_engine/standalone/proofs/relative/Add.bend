# Carry-balance proofs for actual unsigned Word/U32 addition.
# Only the local Boolean full-adder table is case-split; whole operands remain arbitrary.
import Base
import ../successor/Borrow.bend as Borrow
import ../u64/Words.bend as Old

def bit(b: Bool) -> Nat:
  match b:
    case False{}: 0n
    case True{}: 1n

def sum(pair: Bool & Bool) -> Bool:
  (s,c) = pair
  s

def carry_bit(a: Bool,b: Bool,c: Bool) -> Bool:
  Borrow.carry_pair(Bool.full_add(a,b,c))

def sum_bit(a: Bool,b: Bool,c: Bool) -> Bool:
  sum(Bool.full_add(a,b,c))

def carry(n: Nat,a: Word(n),b: Word(n),c: Bool) -> Bool:
  match n:
    case 0n: c
    case 1n+p:
      match a b:
        case WCon{ab,at} WCon{bb,bt}: carry(p,at,bt,carry_bit(ab,bb,c))

# Addition carries ca/cb and subtraction carries q/r satisfy q+ca = r+cb.
def balance(ca: Bool,cb: Bool,q: Bool,r: Bool) -> Bool:
  Nat.is_eq(Nat.add(bit(q),bit(ca)),Nat.add(bit(r),bit(cb)))

def step(+a: Bool,+x: Bool,+y: Bool,+ca: Bool,+cb: Bool,+q: Bool,+r: Bool,
  e: {balance(ca,cb,q,r) == True{} : Bool}) ->
  {balance(carry_bit(a,x,ca),carry_bit(a,y,cb),
    Borrow.carry_bit(sum_bit(a,x,ca),sum_bit(a,y,cb),q),Borrow.carry_bit(x,y,r)) == True{} : Bool}:
  match a x y ca cb q r:
    case False{} False{} False{} False{} False{} False{} False{}: {==}
    case False{} False{} False{} False{} False{} False{} True{}: e
    case False{} False{} False{} False{} False{} True{} False{}: e
    case False{} False{} False{} False{} False{} True{} True{}: {==}
    case False{} False{} False{} False{} True{} False{} False{}: {==}
    case False{} False{} False{} False{} True{} False{} True{}: e
    case False{} False{} False{} False{} True{} True{} False{}: {==}
    case False{} False{} False{} False{} True{} True{} True{}: e
    case False{} False{} False{} True{} False{} False{} False{}: e
    case False{} False{} False{} True{} False{} False{} True{}: {==}
    case False{} False{} False{} True{} False{} True{} False{}: e
    case False{} False{} False{} True{} False{} True{} True{}: {==}
    case False{} False{} False{} True{} True{} False{} False{}: {==}
    case False{} False{} False{} True{} True{} False{} True{}: e
    case False{} False{} False{} True{} True{} True{} False{}: e
    case False{} False{} False{} True{} True{} True{} True{}: {==}
    case False{} False{} True{} False{} False{} False{} False{}: {==}
    case False{} False{} True{} False{} False{} False{} True{}: {==}
    case False{} False{} True{} False{} False{} True{} False{}: {==}
    case False{} False{} True{} False{} False{} True{} True{}: {==}
    case False{} False{} True{} False{} True{} False{} False{}: e
    case False{} False{} True{} False{} True{} False{} True{}: e
    case False{} False{} True{} False{} True{} True{} False{}: {==}
    case False{} False{} True{} False{} True{} True{} True{}: {==}
    case False{} False{} True{} True{} False{} False{} False{}: {==}
    case False{} False{} True{} True{} False{} False{} True{}: {==}
    case False{} False{} True{} True{} False{} True{} False{}: e
    case False{} False{} True{} True{} False{} True{} True{}: e
    case False{} False{} True{} True{} True{} False{} False{}: {==}
    case False{} False{} True{} True{} True{} False{} True{}: {==}
    case False{} False{} True{} True{} True{} True{} False{}: {==}
    case False{} False{} True{} True{} True{} True{} True{}: {==}
    case False{} True{} False{} False{} False{} False{} False{}: {==}
    case False{} True{} False{} False{} False{} False{} True{}: {==}
    case False{} True{} False{} False{} False{} True{} False{}: {==}
    case False{} True{} False{} False{} False{} True{} True{}: {==}
    case False{} True{} False{} False{} True{} False{} False{}: e
    case False{} True{} False{} False{} True{} False{} True{}: e
    case False{} True{} False{} False{} True{} True{} False{}: {==}
    case False{} True{} False{} False{} True{} True{} True{}: {==}
    case False{} True{} False{} True{} False{} False{} False{}: {==}
    case False{} True{} False{} True{} False{} False{} True{}: {==}
    case False{} True{} False{} True{} False{} True{} False{}: e
    case False{} True{} False{} True{} False{} True{} True{}: e
    case False{} True{} False{} True{} True{} False{} False{}: {==}
    case False{} True{} False{} True{} True{} False{} True{}: {==}
    case False{} True{} False{} True{} True{} True{} False{}: {==}
    case False{} True{} False{} True{} True{} True{} True{}: {==}
    case False{} True{} True{} False{} False{} False{} False{}: {==}
    case False{} True{} True{} False{} False{} False{} True{}: e
    case False{} True{} True{} False{} False{} True{} False{}: e
    case False{} True{} True{} False{} False{} True{} True{}: {==}
    case False{} True{} True{} False{} True{} False{} False{}: {==}
    case False{} True{} True{} False{} True{} False{} True{}: e
    case False{} True{} True{} False{} True{} True{} False{}: {==}
    case False{} True{} True{} False{} True{} True{} True{}: e
    case False{} True{} True{} True{} False{} False{} False{}: e
    case False{} True{} True{} True{} False{} False{} True{}: {==}
    case False{} True{} True{} True{} False{} True{} False{}: e
    case False{} True{} True{} True{} False{} True{} True{}: {==}
    case False{} True{} True{} True{} True{} False{} False{}: {==}
    case False{} True{} True{} True{} True{} False{} True{}: e
    case False{} True{} True{} True{} True{} True{} False{}: e
    case False{} True{} True{} True{} True{} True{} True{}: {==}
    case True{} False{} False{} False{} False{} False{} False{}: {==}
    case True{} False{} False{} False{} False{} False{} True{}: e
    case True{} False{} False{} False{} False{} True{} False{}: e
    case True{} False{} False{} False{} False{} True{} True{}: {==}
    case True{} False{} False{} False{} True{} False{} False{}: {==}
    case True{} False{} False{} False{} True{} False{} True{}: e
    case True{} False{} False{} False{} True{} True{} False{}: {==}
    case True{} False{} False{} False{} True{} True{} True{}: e
    case True{} False{} False{} True{} False{} False{} False{}: e
    case True{} False{} False{} True{} False{} False{} True{}: {==}
    case True{} False{} False{} True{} False{} True{} False{}: e
    case True{} False{} False{} True{} False{} True{} True{}: {==}
    case True{} False{} False{} True{} True{} False{} False{}: {==}
    case True{} False{} False{} True{} True{} False{} True{}: e
    case True{} False{} False{} True{} True{} True{} False{}: e
    case True{} False{} False{} True{} True{} True{} True{}: {==}
    case True{} False{} True{} False{} False{} False{} False{}: {==}
    case True{} False{} True{} False{} False{} False{} True{}: {==}
    case True{} False{} True{} False{} False{} True{} False{}: {==}
    case True{} False{} True{} False{} False{} True{} True{}: {==}
    case True{} False{} True{} False{} True{} False{} False{}: e
    case True{} False{} True{} False{} True{} False{} True{}: e
    case True{} False{} True{} False{} True{} True{} False{}: {==}
    case True{} False{} True{} False{} True{} True{} True{}: {==}
    case True{} False{} True{} True{} False{} False{} False{}: {==}
    case True{} False{} True{} True{} False{} False{} True{}: {==}
    case True{} False{} True{} True{} False{} True{} False{}: e
    case True{} False{} True{} True{} False{} True{} True{}: e
    case True{} False{} True{} True{} True{} False{} False{}: {==}
    case True{} False{} True{} True{} True{} False{} True{}: {==}
    case True{} False{} True{} True{} True{} True{} False{}: {==}
    case True{} False{} True{} True{} True{} True{} True{}: {==}
    case True{} True{} False{} False{} False{} False{} False{}: {==}
    case True{} True{} False{} False{} False{} False{} True{}: {==}
    case True{} True{} False{} False{} False{} True{} False{}: {==}
    case True{} True{} False{} False{} False{} True{} True{}: {==}
    case True{} True{} False{} False{} True{} False{} False{}: e
    case True{} True{} False{} False{} True{} False{} True{}: e
    case True{} True{} False{} False{} True{} True{} False{}: {==}
    case True{} True{} False{} False{} True{} True{} True{}: {==}
    case True{} True{} False{} True{} False{} False{} False{}: {==}
    case True{} True{} False{} True{} False{} False{} True{}: {==}
    case True{} True{} False{} True{} False{} True{} False{}: e
    case True{} True{} False{} True{} False{} True{} True{}: e
    case True{} True{} False{} True{} True{} False{} False{}: {==}
    case True{} True{} False{} True{} True{} False{} True{}: {==}
    case True{} True{} False{} True{} True{} True{} False{}: {==}
    case True{} True{} False{} True{} True{} True{} True{}: {==}
    case True{} True{} True{} False{} False{} False{} False{}: {==}
    case True{} True{} True{} False{} False{} False{} True{}: e
    case True{} True{} True{} False{} False{} True{} False{}: e
    case True{} True{} True{} False{} False{} True{} True{}: {==}
    case True{} True{} True{} False{} True{} False{} False{}: {==}
    case True{} True{} True{} False{} True{} False{} True{}: e
    case True{} True{} True{} False{} True{} True{} False{}: {==}
    case True{} True{} True{} False{} True{} True{} True{}: e
    case True{} True{} True{} True{} False{} False{} False{}: e
    case True{} True{} True{} True{} False{} False{} True{}: {==}
    case True{} True{} True{} True{} False{} True{} False{}: e
    case True{} True{} True{} True{} False{} True{} True{}: {==}
    case True{} True{} True{} True{} True{} False{} False{}: {==}
    case True{} True{} True{} True{} True{} False{} True{}: e
    case True{} True{} True{} True{} True{} True{} False{}: e
    case True{} True{} True{} True{} True{} True{} True{}: {==}

def conserved(+n: Nat,+a: Word(n),+x: Word(n),+y: Word(n),+ca: Bool,+cb: Bool,+q: Bool,+r: Bool,
  e: {balance(ca,cb,q,r) == True{} : Bool}) ->
  {balance(carry(n,a,x,ca),carry(n,a,y,cb),
    Borrow.carry(n,Word.adc(n,a,x,False{},ca),Word.adc(n,a,y,False{},cb),q),
    Borrow.carry(n,x,y,r)) == True{} : Bool}:
  match n:
    case 0n: e
    case 1n+p:
      match a x y ca cb:
        case WCon{False{},at} WCon{False{},xt} WCon{False{},yt} False{} False{}:
          conserved(p,at,xt,yt,False{},False{},Borrow.carry_bit(False{},False{},q),Borrow.carry_bit(False{},False{},r),step(False{},False{},False{},False{},False{},q,r,e))
        case WCon{False{},at} WCon{False{},xt} WCon{False{},yt} False{} True{}:
          conserved(p,at,xt,yt,False{},False{},Borrow.carry_bit(False{},True{},q),Borrow.carry_bit(False{},False{},r),step(False{},False{},False{},False{},True{},q,r,e))
        case WCon{False{},at} WCon{False{},xt} WCon{False{},yt} True{} False{}:
          conserved(p,at,xt,yt,False{},False{},Borrow.carry_bit(True{},False{},q),Borrow.carry_bit(False{},False{},r),step(False{},False{},False{},True{},False{},q,r,e))
        case WCon{False{},at} WCon{False{},xt} WCon{False{},yt} True{} True{}:
          conserved(p,at,xt,yt,False{},False{},Borrow.carry_bit(True{},True{},q),Borrow.carry_bit(False{},False{},r),step(False{},False{},False{},True{},True{},q,r,e))
        case WCon{False{},at} WCon{False{},xt} WCon{True{},yt} False{} False{}:
          conserved(p,at,xt,yt,False{},False{},Borrow.carry_bit(False{},True{},q),Borrow.carry_bit(False{},True{},r),step(False{},False{},True{},False{},False{},q,r,e))
        case WCon{False{},at} WCon{False{},xt} WCon{True{},yt} False{} True{}:
          conserved(p,at,xt,yt,False{},True{},Borrow.carry_bit(False{},False{},q),Borrow.carry_bit(False{},True{},r),step(False{},False{},True{},False{},True{},q,r,e))
        case WCon{False{},at} WCon{False{},xt} WCon{True{},yt} True{} False{}:
          conserved(p,at,xt,yt,False{},False{},Borrow.carry_bit(True{},True{},q),Borrow.carry_bit(False{},True{},r),step(False{},False{},True{},True{},False{},q,r,e))
        case WCon{False{},at} WCon{False{},xt} WCon{True{},yt} True{} True{}:
          conserved(p,at,xt,yt,False{},True{},Borrow.carry_bit(True{},False{},q),Borrow.carry_bit(False{},True{},r),step(False{},False{},True{},True{},True{},q,r,e))
        case WCon{False{},at} WCon{True{},xt} WCon{False{},yt} False{} False{}:
          conserved(p,at,xt,yt,False{},False{},Borrow.carry_bit(True{},False{},q),Borrow.carry_bit(True{},False{},r),step(False{},True{},False{},False{},False{},q,r,e))
        case WCon{False{},at} WCon{True{},xt} WCon{False{},yt} False{} True{}:
          conserved(p,at,xt,yt,False{},False{},Borrow.carry_bit(True{},True{},q),Borrow.carry_bit(True{},False{},r),step(False{},True{},False{},False{},True{},q,r,e))
        case WCon{False{},at} WCon{True{},xt} WCon{False{},yt} True{} False{}:
          conserved(p,at,xt,yt,True{},False{},Borrow.carry_bit(False{},False{},q),Borrow.carry_bit(True{},False{},r),step(False{},True{},False{},True{},False{},q,r,e))
        case WCon{False{},at} WCon{True{},xt} WCon{False{},yt} True{} True{}:
          conserved(p,at,xt,yt,True{},False{},Borrow.carry_bit(False{},True{},q),Borrow.carry_bit(True{},False{},r),step(False{},True{},False{},True{},True{},q,r,e))
        case WCon{False{},at} WCon{True{},xt} WCon{True{},yt} False{} False{}:
          conserved(p,at,xt,yt,False{},False{},Borrow.carry_bit(True{},True{},q),Borrow.carry_bit(True{},True{},r),step(False{},True{},True{},False{},False{},q,r,e))
        case WCon{False{},at} WCon{True{},xt} WCon{True{},yt} False{} True{}:
          conserved(p,at,xt,yt,False{},True{},Borrow.carry_bit(True{},False{},q),Borrow.carry_bit(True{},True{},r),step(False{},True{},True{},False{},True{},q,r,e))
        case WCon{False{},at} WCon{True{},xt} WCon{True{},yt} True{} False{}:
          conserved(p,at,xt,yt,True{},False{},Borrow.carry_bit(False{},True{},q),Borrow.carry_bit(True{},True{},r),step(False{},True{},True{},True{},False{},q,r,e))
        case WCon{False{},at} WCon{True{},xt} WCon{True{},yt} True{} True{}:
          conserved(p,at,xt,yt,True{},True{},Borrow.carry_bit(False{},False{},q),Borrow.carry_bit(True{},True{},r),step(False{},True{},True{},True{},True{},q,r,e))
        case WCon{True{},at} WCon{False{},xt} WCon{False{},yt} False{} False{}:
          conserved(p,at,xt,yt,False{},False{},Borrow.carry_bit(True{},True{},q),Borrow.carry_bit(False{},False{},r),step(True{},False{},False{},False{},False{},q,r,e))
        case WCon{True{},at} WCon{False{},xt} WCon{False{},yt} False{} True{}:
          conserved(p,at,xt,yt,False{},True{},Borrow.carry_bit(True{},False{},q),Borrow.carry_bit(False{},False{},r),step(True{},False{},False{},False{},True{},q,r,e))
        case WCon{True{},at} WCon{False{},xt} WCon{False{},yt} True{} False{}:
          conserved(p,at,xt,yt,True{},False{},Borrow.carry_bit(False{},True{},q),Borrow.carry_bit(False{},False{},r),step(True{},False{},False{},True{},False{},q,r,e))
        case WCon{True{},at} WCon{False{},xt} WCon{False{},yt} True{} True{}:
          conserved(p,at,xt,yt,True{},True{},Borrow.carry_bit(False{},False{},q),Borrow.carry_bit(False{},False{},r),step(True{},False{},False{},True{},True{},q,r,e))
        case WCon{True{},at} WCon{False{},xt} WCon{True{},yt} False{} False{}:
          conserved(p,at,xt,yt,False{},True{},Borrow.carry_bit(True{},False{},q),Borrow.carry_bit(False{},True{},r),step(True{},False{},True{},False{},False{},q,r,e))
        case WCon{True{},at} WCon{False{},xt} WCon{True{},yt} False{} True{}:
          conserved(p,at,xt,yt,False{},True{},Borrow.carry_bit(True{},True{},q),Borrow.carry_bit(False{},True{},r),step(True{},False{},True{},False{},True{},q,r,e))
        case WCon{True{},at} WCon{False{},xt} WCon{True{},yt} True{} False{}:
          conserved(p,at,xt,yt,True{},True{},Borrow.carry_bit(False{},False{},q),Borrow.carry_bit(False{},True{},r),step(True{},False{},True{},True{},False{},q,r,e))
        case WCon{True{},at} WCon{False{},xt} WCon{True{},yt} True{} True{}:
          conserved(p,at,xt,yt,True{},True{},Borrow.carry_bit(False{},True{},q),Borrow.carry_bit(False{},True{},r),step(True{},False{},True{},True{},True{},q,r,e))
        case WCon{True{},at} WCon{True{},xt} WCon{False{},yt} False{} False{}:
          conserved(p,at,xt,yt,True{},False{},Borrow.carry_bit(False{},True{},q),Borrow.carry_bit(True{},False{},r),step(True{},True{},False{},False{},False{},q,r,e))
        case WCon{True{},at} WCon{True{},xt} WCon{False{},yt} False{} True{}:
          conserved(p,at,xt,yt,True{},True{},Borrow.carry_bit(False{},False{},q),Borrow.carry_bit(True{},False{},r),step(True{},True{},False{},False{},True{},q,r,e))
        case WCon{True{},at} WCon{True{},xt} WCon{False{},yt} True{} False{}:
          conserved(p,at,xt,yt,True{},False{},Borrow.carry_bit(True{},True{},q),Borrow.carry_bit(True{},False{},r),step(True{},True{},False{},True{},False{},q,r,e))
        case WCon{True{},at} WCon{True{},xt} WCon{False{},yt} True{} True{}:
          conserved(p,at,xt,yt,True{},True{},Borrow.carry_bit(True{},False{},q),Borrow.carry_bit(True{},False{},r),step(True{},True{},False{},True{},True{},q,r,e))
        case WCon{True{},at} WCon{True{},xt} WCon{True{},yt} False{} False{}:
          conserved(p,at,xt,yt,True{},True{},Borrow.carry_bit(False{},False{},q),Borrow.carry_bit(True{},True{},r),step(True{},True{},True{},False{},False{},q,r,e))
        case WCon{True{},at} WCon{True{},xt} WCon{True{},yt} False{} True{}:
          conserved(p,at,xt,yt,True{},True{},Borrow.carry_bit(False{},True{},q),Borrow.carry_bit(True{},True{},r),step(True{},True{},True{},False{},True{},q,r,e))
        case WCon{True{},at} WCon{True{},xt} WCon{True{},yt} True{} False{}:
          conserved(p,at,xt,yt,True{},True{},Borrow.carry_bit(True{},False{},q),Borrow.carry_bit(True{},True{},r),step(True{},True{},True{},True{},False{},q,r,e))
        case WCon{True{},at} WCon{True{},xt} WCon{True{},yt} True{} True{}:
          conserved(p,at,xt,yt,True{},True{},Borrow.carry_bit(True{},True{},q),Borrow.carry_bit(True{},True{},r),step(True{},True{},True{},True{},True{},q,r,e))

def zero_balance(+ca: Bool,+q: Bool,+e: {balance(ca,False{},q,False{}) == True{} : Bool}) ->
  {ca == False{} : Bool} & {q == False{} : Bool}:
  match ca q:
    case False{} False{}: ({==},{==})
    case False{} True{}: ({==},Equal.sym(Bool,False{},True{},e))
    case True{} False{}: (Equal.sym(Bool,False{},True{},e),{==})
    case True{} True{}: (Equal.sym(Bool,False{},True{},e),Equal.sym(Bool,False{},True{},e))

def not_false(+b: Bool,e: {Bool.not(b) == False{} : Bool}) -> {b == True{} : Bool}:
  match b:
    case False{}: Equal.sym(Bool,True{},False{},e)
    case True{}: {==}

def lt_carry(+n: Nat,+x: Word(n),+y: Word(n),e: {Cmp.is_lt(Word.cmp(n,x,y)) == True{} : Bool}) ->
  {Borrow.carry(n,x,y,True{}) == False{} : Bool}:
  Equal.trans(Bool,Borrow.carry(n,x,y,True{}),Bool.not(Cmp.is_lt(Word.cmp(n,x,y))),False{},
    Borrow.outgoing(n,x,y),Equal.cong(Bool,Bool,b => Bool.not(b),Cmp.is_lt(Word.cmp(n,x,y)),True{},e))

def carry_lt(+n: Nat,+x: Word(n),+y: Word(n),e: {Borrow.carry(n,x,y,True{}) == False{} : Bool}) ->
  {Cmp.is_lt(Word.cmp(n,x,y)) == True{} : Bool}:
  not_false(Cmp.is_lt(Word.cmp(n,x,y)),Equal.trans(Bool,Bool.not(Cmp.is_lt(Word.cmp(n,x,y))),Borrow.carry(n,x,y,True{}),False{},
    Equal.sym(Bool,Borrow.carry(n,x,y,True{}),Bool.not(Cmp.is_lt(Word.cmp(n,x,y))),Borrow.outgoing(n,x,y)),e))

def increasing_finish(+n: Nat,+a: Word(n),+x: Word(n),+y: Word(n),
  pair: {carry(n,a,x,False{}) == False{} : Bool}
    & {Borrow.carry(n,Word.add(n,a,x),Word.add(n,a,y),True{}) == False{} : Bool}) ->
  {carry(n,a,x,False{}) == False{} : Bool}
  & {Cmp.is_lt(Word.cmp(n,Word.add(n,a,x),Word.add(n,a,y))) == True{} : Bool}:
  (nc,nq) = pair
  (nc,carry_lt(n,Word.add(n,a,x),Word.add(n,a,y),nq))

def increasing(+n: Nat,+a: Word(n),+x: Word(n),+y: Word(n),
  safe: {carry(n,a,y,False{}) == False{} : Bool},
  lt: {Cmp.is_lt(Word.cmp(n,x,y)) == True{} : Bool}) ->
  {carry(n,a,x,False{}) == False{} : Bool}
  & {Cmp.is_lt(Word.cmp(n,Word.add(n,a,x),Word.add(n,a,y))) == True{} : Bool}:
  +cx = carry(n,a,x,False{})
  +cy = carry(n,a,y,False{})
  +q = Borrow.carry(n,Word.add(n,a,x),Word.add(n,a,y),True{})
  +r = Borrow.carry(n,x,y,True{})
  +rb = lt_carry(n,x,y,lt)
  +cert = Equal.trans(Bool,balance(cx,False{},q,False{}),balance(cx,cy,q,r),True{},
    Equal.trans(Bool,balance(cx,False{},q,False{}),balance(cx,cy,q,False{}),balance(cx,cy,q,r),
      Equal.cong(Bool,Bool,b => balance(cx,b,q,False{}),False{},cy,Equal.sym(Bool,cy,False{},safe)),
      Equal.cong(Bool,Bool,b => balance(cx,cy,q,b),False{},r,Equal.sym(Bool,r,False{},rb))),
    conserved(n,a,x,y,False{},False{},True{},True{},{==}))
  increasing_finish(n,a,x,y,zero_balance(cx,q,cert))

# Subtracting an addend from a ripple sum complements its overflow carry.
def inverse_carry(+n: Nat,+a: Word(n),+b: Word(n),+c: Bool) ->
  {Borrow.carry(n,Word.adc(n,a,b,False{},c),a,Bool.not(c)) == Bool.not(carry(n,a,b,c)) : Bool}:
  match n:
    case 0n: {==}
    case 1n+p:
      match a b c:
        case WCon{False{},at} WCon{False{},bt} False{}: inverse_carry(p,at,bt,False{})
        case WCon{False{},at} WCon{False{},bt} True{}: inverse_carry(p,at,bt,False{})
        case WCon{False{},at} WCon{True{},bt} False{}: inverse_carry(p,at,bt,False{})
        case WCon{False{},at} WCon{True{},bt} True{}: inverse_carry(p,at,bt,True{})
        case WCon{True{},at} WCon{False{},bt} False{}: inverse_carry(p,at,bt,False{})
        case WCon{True{},at} WCon{False{},bt} True{}: inverse_carry(p,at,bt,True{})
        case WCon{True{},at} WCon{True{},bt} False{}: inverse_carry(p,at,bt,True{})
        case WCon{True{},at} WCon{True{},bt} True{}: inverse_carry(p,at,bt,True{})

def not_injective(+a: Bool,+b: Bool,e: {Bool.not(a) == Bool.not(b) : Bool}) -> {a == b : Bool}:
  match a b:
    case False{} False{}: {==}
    case False{} True{}: Equal.sym(Bool,True{},False{},e)
    case True{} False{}: Equal.sym(Bool,False{},True{},e)
    case True{} True{}: {==}

def overflow(+n: Nat,+a: Word(n),+b: Word(n)) ->
  {carry(n,a,b,False{}) == Cmp.is_lt(Word.cmp(n,Word.add(n,a,b),a)) : Bool}:
  not_injective(carry(n,a,b,False{}),Cmp.is_lt(Word.cmp(n,Word.add(n,a,b),a)),
    Equal.trans(Bool,Bool.not(carry(n,a,b,False{})),Borrow.carry(n,Word.add(n,a,b),a,True{}),
      Bool.not(Cmp.is_lt(Word.cmp(n,Word.add(n,a,b),a))),
      Equal.sym(Bool,Borrow.carry(n,Word.add(n,a,b),a,True{}),Bool.not(carry(n,a,b,False{})),inverse_carry(n,a,b,False{})),
      Borrow.outgoing(n,Word.add(n,a,b),a)))

def carry32(a: U32,b: U32) -> Bool:
  match a b:
    case U32{x} U32{y}: carry(32n,x,y,False{})

def overflow32(+a: U32,+b: U32) -> {carry32(a,b) == U32.is_lt(U32.add(a,b),a) : Bool}:
  match a b:
    case U32{x} U32{y}: overflow(32n,x,y)

def flag_zero(+b: Bool,e: {U32.add(U32.add(0,0),Bool.to_u32(b)) == 0 : U32}) -> {b == False{} : Bool}:
  match b:
    case False{}: {==}
    case True{}:
      Equal.sym(Bool,False{},True{},Equal.cong(U32,Bool,x => U32.is_eq(x,0),1,0,e))

def safe_from_wide(+a: U32,+b: U32,+end: U32,
  e: {U64.add(U64.from_u32(a),U64.from_u32(b)) == U64.from_u32(end) : U64}) ->
  {carry32(a,b) == False{} : Bool}:
  Equal.trans(Bool,carry32(a,b),U32.is_lt(U32.add(a,b),a),False{},overflow32(a,b),
    flag_zero(U32.is_lt(U32.add(a,b),a),Equal.cong(U64,U32,w => U64.high(w),
      U64.add(U64.from_u32(a),U64.from_u32(b)),U64.from_u32(end),e)))

def increasing32(+a: U32,+x: U32,+y: U32,
  safe: {carry32(a,y) == False{} : Bool},lt: {U32.is_lt(x,y) == True{} : Bool}) ->
  {carry32(a,x) == False{} : Bool} & {U32.is_lt(U32.add(a,x),U32.add(a,y)) == True{} : Bool}:
  match a x y:
    case U32{aw} U32{xw} U32{yw}: increasing(32n,aw,xw,yw,safe,lt)

def no_wrap(+a: U32,+b: U32,e: {carry32(a,b) == False{} : Bool}) ->
  {U32.is_lt(U32.add(a,b),a) == False{} : Bool}:
  Equal.trans(Bool,U32.is_lt(U32.add(a,b),a),carry32(a,b),False{},
    Equal.sym(Bool,carry32(a,b),U32.is_lt(U32.add(a,b),a),overflow32(a,b)),e)

def widening(+a: U32,+b: U32,e: {carry32(a,b) == False{} : Bool}) ->
  {U64.from_u32(U32.add(a,b)) == U64.add(U64.from_u32(a),U64.from_u32(b)) : U64}:
  Equal.cong(Bool,U64,c => U64{U32.add(a,b),U32.add(U32.add(0,0),Bool.to_u32(c))},
    False{},U32.is_lt(U32.add(a,b),a),Equal.sym(Bool,U32.is_lt(U32.add(a,b),a),False{},no_wrap(a,b,e)))
