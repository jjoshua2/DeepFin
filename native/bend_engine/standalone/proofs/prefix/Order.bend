# Bridge actual unsigned comparison to mathematical order; no assumed arithmetic.
import Base
import ../successor/Value.bend as Value
import ../layout/Low.bend as Low
import ../u64/Words.bend as Old

def head(+b: Bool, +n: Nat) -> Nat:
  Bool.pick(Nat,b,1n+Nat.double(n),Nat.double(n))

def cmp_heads(+x: Nat, +y: Nat, +a: Bool, +b: Bool) ->
  {Nat.cmp(head(a,x),head(b,y)) == Word.cmp.fin(a,b,Nat.cmp(x,y)) : Cmp}:
  match x y:
    case 0n 0n:
      match a b:
        case False{} False{}: {==}
        case False{} True{}: {==}
        case True{} False{}: {==}
        case True{} True{}: {==}
    case 0n 1n+q:
      match a b:
        case False{} False{}: {==}
        case False{} True{}: {==}
        case True{} False{}: {==}
        case True{} True{}: {==}
    case 1n+p 0n:
      match a b:
        case False{} False{}: {==}
        case False{} True{}: {==}
        case True{} False{}: {==}
        case True{} True{}: {==}
    case 1n+p 1n+q:
      match a b:
        case False{} False{}: cmp_heads(p,q,False{},False{})
        case False{} True{}: cmp_heads(p,q,False{},True{})
        case True{} False{}: cmp_heads(p,q,True{},False{})
        case True{} True{}: cmp_heads(p,q,True{},True{})

def word(+n: Nat, +a: Word(n), +b: Word(n)) ->
  {Word.cmp(n,a,b) == Nat.cmp(Word.to_nat(n,a),Word.to_nat(n,b)) : Cmp}:
  match n:
    case 0n: {==}
    case 1n+p:
      match a b:
        case WCon{ab,at} WCon{bb,bt}:
          %Equal.sym(Nat,Word.to_nat(1n+p,WCon{ab,at}),head(ab,Word.to_nat(p,at)),Value.head_value(ab,p,at)) :
            {Word.cmp(1n+p,WCon{ab,at},WCon{bb,bt}) == Nat.cmp(_,Word.to_nat(1n+p,WCon{bb,bt})) : Cmp}
          %Equal.sym(Nat,Word.to_nat(1n+p,WCon{bb,bt}),head(bb,Word.to_nat(p,bt)),Value.head_value(bb,p,bt)) :
            {Word.cmp(1n+p,WCon{ab,at},WCon{bb,bt}) == Nat.cmp(head(ab,Word.to_nat(p,at)),_) : Cmp}
          Equal.trans(Cmp,Word.cmp(1n+p,WCon{ab,at},WCon{bb,bt}),
            Word.cmp.fin(ab,bb,Nat.cmp(Word.to_nat(p,at),Word.to_nat(p,bt))),
            Nat.cmp(head(ab,Word.to_nat(p,at)),head(bb,Word.to_nat(p,bt))),
            Equal.cong(Cmp,Cmp,c => Word.cmp.fin(ab,bb,c),Word.cmp(p,at,bt),Nat.cmp(Word.to_nat(p,at),Word.to_nat(p,bt)),word(p,at,bt)),
            Equal.sym(Cmp,Nat.cmp(head(ab,Word.to_nat(p,at)),head(bb,Word.to_nat(p,bt))),
              Word.cmp.fin(ab,bb,Nat.cmp(Word.to_nat(p,at),Word.to_nat(p,bt))),cmp_heads(Word.to_nat(p,at),Word.to_nat(p,bt),ab,bb)))

def observed(+a: U32, +b: U32) ->
  {U32.is_le(a,b) == Nat.is_le(U32.to_nat(a),U32.to_nat(b)) : Bool}:
  match a b:
    case U32{x} U32{y}:
      Equal.cong(Cmp,Bool,c => Cmp.is_le(c),Word.cmp(32n,x,y),Nat.cmp(Word.to_nat(32n,x),Word.to_nat(32n,y)),word(32n,x,y))

def nat_refl(n: Nat) -> {Nat.is_le(n,n) == True{} : Bool}:
  match n:
    case 0n: {==}
    case 1n+p: nat_refl(p)

def nat_trans(+a: Nat, +b: Nat, +c: Nat,
  ab: {Nat.is_le(a,b) == True{} : Bool}, bc: {Nat.is_le(b,c) == True{} : Bool}) ->
  {Nat.is_le(a,c) == True{} : Bool}:
  match a b c:
    case 0n _ _: Low.zero_le(c)
    case 1n+p 0n _: Old.false_true({Nat.is_le(1n+p,c) == True{} : Bool},ab)
    case 1n+p 1n+q 0n: Old.false_true({Nat.is_le(1n+p,0n) == True{} : Bool},bc)
    case 1n+p 1n+q 1n+r: nat_trans(p,q,r,ab,bc)

def to_nat(+a: U32, +b: U32, e: {U32.is_le(a,b) == True{} : Bool}) ->
  {Nat.is_le(U32.to_nat(a),U32.to_nat(b)) == True{} : Bool}:
  Equal.trans(Bool,Nat.is_le(U32.to_nat(a),U32.to_nat(b)),U32.is_le(a,b),True{},
    Equal.sym(Bool,U32.is_le(a,b),Nat.is_le(U32.to_nat(a),U32.to_nat(b)),observed(a,b)),e)

def refl(+a: U32) -> {U32.is_le(a,a) == True{} : Bool}:
  Equal.trans(Bool,U32.is_le(a,a),Nat.is_le(U32.to_nat(a),U32.to_nat(a)),True{},observed(a,a),nat_refl(U32.to_nat(a)))

def trans(+a: U32, +b: U32, +c: U32,
  ab: {U32.is_le(a,b) == True{} : Bool}, bc: {U32.is_le(b,c) == True{} : Bool}) ->
  {U32.is_le(a,c) == True{} : Bool}:
  Equal.trans(Bool,U32.is_le(a,c),Nat.is_le(U32.to_nat(a),U32.to_nat(c)),True{},observed(a,c),
    nat_trans(U32.to_nat(a),U32.to_nat(b),U32.to_nat(c),to_nat(a,b,ab),to_nat(b,c,bc)))

def weak(c: Cmp, e: {Cmp.is_lt(c) == True{} : Bool}) -> {Cmp.is_le(c) == True{} : Bool}:
  match c:
    case LT{}: {==}
    case EQ{}: {==}
    case GT{}: e

def lt_le(+a: U32, +b: U32, e: {U32.is_lt(a,b) == True{} : Bool}) -> {U32.is_le(a,b) == True{} : Bool}:
  weak(U32.cmp(a,b),e)
