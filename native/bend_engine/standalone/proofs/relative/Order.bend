# Unsigned total-order composition and Nat-observation bridge.
import Base
import ../normalization/Compare.bend as Compare
import ../u64/Words.bend as Old
import ./Add.bend as Add

def coherent(ab: Cmp,bc: Cmp,ac: Cmp) -> Bool:
  match ab bc ac:
    case LT{} LT{} LT{}: True{}
    case LT{} LT{} EQ{}: False{}
    case LT{} LT{} GT{}: False{}
    case LT{} EQ{} LT{}: True{}
    case LT{} EQ{} EQ{}: False{}
    case LT{} EQ{} GT{}: False{}
    case LT{} GT{} LT{}: True{}
    case LT{} GT{} EQ{}: True{}
    case LT{} GT{} GT{}: True{}
    case EQ{} LT{} LT{}: True{}
    case EQ{} LT{} EQ{}: False{}
    case EQ{} LT{} GT{}: False{}
    case EQ{} EQ{} LT{}: False{}
    case EQ{} EQ{} EQ{}: True{}
    case EQ{} EQ{} GT{}: False{}
    case EQ{} GT{} LT{}: False{}
    case EQ{} GT{} EQ{}: False{}
    case EQ{} GT{} GT{}: True{}
    case GT{} LT{} LT{}: True{}
    case GT{} LT{} EQ{}: True{}
    case GT{} LT{} GT{}: True{}
    case GT{} EQ{} LT{}: False{}
    case GT{} EQ{} EQ{}: False{}
    case GT{} EQ{} GT{}: True{}
    case GT{} GT{} LT{}: False{}
    case GT{} GT{} EQ{}: False{}
    case GT{} GT{} GT{}: True{}

def step(+a: Bool,+b: Bool,+c: Bool,+ab: Cmp,+bc: Cmp,+ac: Cmp,
  e: {coherent(ab,bc,ac) == True{} : Bool}) ->
  {coherent(Word.cmp.fin(a,b,ab),Word.cmp.fin(b,c,bc),Word.cmp.fin(a,c,ac)) == True{} : Bool}:
  match a b c ab bc ac:
    case False{} False{} False{} LT{} LT{} LT{}: {==}
    case False{} False{} False{} LT{} LT{} EQ{}: e
    case False{} False{} False{} LT{} LT{} GT{}: e
    case False{} False{} False{} LT{} EQ{} LT{}: {==}
    case False{} False{} False{} LT{} EQ{} EQ{}: e
    case False{} False{} False{} LT{} EQ{} GT{}: e
    case False{} False{} False{} LT{} GT{} LT{}: {==}
    case False{} False{} False{} LT{} GT{} EQ{}: {==}
    case False{} False{} False{} LT{} GT{} GT{}: {==}
    case False{} False{} False{} EQ{} LT{} LT{}: {==}
    case False{} False{} False{} EQ{} LT{} EQ{}: e
    case False{} False{} False{} EQ{} LT{} GT{}: e
    case False{} False{} False{} EQ{} EQ{} LT{}: e
    case False{} False{} False{} EQ{} EQ{} EQ{}: {==}
    case False{} False{} False{} EQ{} EQ{} GT{}: e
    case False{} False{} False{} EQ{} GT{} LT{}: e
    case False{} False{} False{} EQ{} GT{} EQ{}: e
    case False{} False{} False{} EQ{} GT{} GT{}: {==}
    case False{} False{} False{} GT{} LT{} LT{}: {==}
    case False{} False{} False{} GT{} LT{} EQ{}: {==}
    case False{} False{} False{} GT{} LT{} GT{}: {==}
    case False{} False{} False{} GT{} EQ{} LT{}: e
    case False{} False{} False{} GT{} EQ{} EQ{}: e
    case False{} False{} False{} GT{} EQ{} GT{}: {==}
    case False{} False{} False{} GT{} GT{} LT{}: e
    case False{} False{} False{} GT{} GT{} EQ{}: e
    case False{} False{} False{} GT{} GT{} GT{}: {==}
    case False{} False{} True{} LT{} LT{} LT{}: {==}
    case False{} False{} True{} LT{} LT{} EQ{}: {==}
    case False{} False{} True{} LT{} LT{} GT{}: e
    case False{} False{} True{} LT{} EQ{} LT{}: {==}
    case False{} False{} True{} LT{} EQ{} EQ{}: {==}
    case False{} False{} True{} LT{} EQ{} GT{}: e
    case False{} False{} True{} LT{} GT{} LT{}: {==}
    case False{} False{} True{} LT{} GT{} EQ{}: {==}
    case False{} False{} True{} LT{} GT{} GT{}: {==}
    case False{} False{} True{} EQ{} LT{} LT{}: {==}
    case False{} False{} True{} EQ{} LT{} EQ{}: {==}
    case False{} False{} True{} EQ{} LT{} GT{}: e
    case False{} False{} True{} EQ{} EQ{} LT{}: {==}
    case False{} False{} True{} EQ{} EQ{} EQ{}: {==}
    case False{} False{} True{} EQ{} EQ{} GT{}: e
    case False{} False{} True{} EQ{} GT{} LT{}: e
    case False{} False{} True{} EQ{} GT{} EQ{}: e
    case False{} False{} True{} EQ{} GT{} GT{}: {==}
    case False{} False{} True{} GT{} LT{} LT{}: {==}
    case False{} False{} True{} GT{} LT{} EQ{}: {==}
    case False{} False{} True{} GT{} LT{} GT{}: {==}
    case False{} False{} True{} GT{} EQ{} LT{}: {==}
    case False{} False{} True{} GT{} EQ{} EQ{}: {==}
    case False{} False{} True{} GT{} EQ{} GT{}: {==}
    case False{} False{} True{} GT{} GT{} LT{}: e
    case False{} False{} True{} GT{} GT{} EQ{}: e
    case False{} False{} True{} GT{} GT{} GT{}: {==}
    case False{} True{} False{} LT{} LT{} LT{}: {==}
    case False{} True{} False{} LT{} LT{} EQ{}: e
    case False{} True{} False{} LT{} LT{} GT{}: e
    case False{} True{} False{} LT{} EQ{} LT{}: {==}
    case False{} True{} False{} LT{} EQ{} EQ{}: {==}
    case False{} True{} False{} LT{} EQ{} GT{}: {==}
    case False{} True{} False{} LT{} GT{} LT{}: {==}
    case False{} True{} False{} LT{} GT{} EQ{}: {==}
    case False{} True{} False{} LT{} GT{} GT{}: {==}
    case False{} True{} False{} EQ{} LT{} LT{}: {==}
    case False{} True{} False{} EQ{} LT{} EQ{}: e
    case False{} True{} False{} EQ{} LT{} GT{}: e
    case False{} True{} False{} EQ{} EQ{} LT{}: {==}
    case False{} True{} False{} EQ{} EQ{} EQ{}: {==}
    case False{} True{} False{} EQ{} EQ{} GT{}: {==}
    case False{} True{} False{} EQ{} GT{} LT{}: {==}
    case False{} True{} False{} EQ{} GT{} EQ{}: {==}
    case False{} True{} False{} EQ{} GT{} GT{}: {==}
    case False{} True{} False{} GT{} LT{} LT{}: {==}
    case False{} True{} False{} GT{} LT{} EQ{}: {==}
    case False{} True{} False{} GT{} LT{} GT{}: {==}
    case False{} True{} False{} GT{} EQ{} LT{}: e
    case False{} True{} False{} GT{} EQ{} EQ{}: e
    case False{} True{} False{} GT{} EQ{} GT{}: {==}
    case False{} True{} False{} GT{} GT{} LT{}: e
    case False{} True{} False{} GT{} GT{} EQ{}: e
    case False{} True{} False{} GT{} GT{} GT{}: {==}
    case False{} True{} True{} LT{} LT{} LT{}: {==}
    case False{} True{} True{} LT{} LT{} EQ{}: {==}
    case False{} True{} True{} LT{} LT{} GT{}: e
    case False{} True{} True{} LT{} EQ{} LT{}: {==}
    case False{} True{} True{} LT{} EQ{} EQ{}: {==}
    case False{} True{} True{} LT{} EQ{} GT{}: e
    case False{} True{} True{} LT{} GT{} LT{}: {==}
    case False{} True{} True{} LT{} GT{} EQ{}: {==}
    case False{} True{} True{} LT{} GT{} GT{}: {==}
    case False{} True{} True{} EQ{} LT{} LT{}: {==}
    case False{} True{} True{} EQ{} LT{} EQ{}: {==}
    case False{} True{} True{} EQ{} LT{} GT{}: e
    case False{} True{} True{} EQ{} EQ{} LT{}: {==}
    case False{} True{} True{} EQ{} EQ{} EQ{}: {==}
    case False{} True{} True{} EQ{} EQ{} GT{}: e
    case False{} True{} True{} EQ{} GT{} LT{}: {==}
    case False{} True{} True{} EQ{} GT{} EQ{}: {==}
    case False{} True{} True{} EQ{} GT{} GT{}: {==}
    case False{} True{} True{} GT{} LT{} LT{}: {==}
    case False{} True{} True{} GT{} LT{} EQ{}: {==}
    case False{} True{} True{} GT{} LT{} GT{}: {==}
    case False{} True{} True{} GT{} EQ{} LT{}: e
    case False{} True{} True{} GT{} EQ{} EQ{}: e
    case False{} True{} True{} GT{} EQ{} GT{}: {==}
    case False{} True{} True{} GT{} GT{} LT{}: e
    case False{} True{} True{} GT{} GT{} EQ{}: e
    case False{} True{} True{} GT{} GT{} GT{}: {==}
    case True{} False{} False{} LT{} LT{} LT{}: {==}
    case True{} False{} False{} LT{} LT{} EQ{}: e
    case True{} False{} False{} LT{} LT{} GT{}: e
    case True{} False{} False{} LT{} EQ{} LT{}: {==}
    case True{} False{} False{} LT{} EQ{} EQ{}: e
    case True{} False{} False{} LT{} EQ{} GT{}: e
    case True{} False{} False{} LT{} GT{} LT{}: {==}
    case True{} False{} False{} LT{} GT{} EQ{}: {==}
    case True{} False{} False{} LT{} GT{} GT{}: {==}
    case True{} False{} False{} EQ{} LT{} LT{}: {==}
    case True{} False{} False{} EQ{} LT{} EQ{}: {==}
    case True{} False{} False{} EQ{} LT{} GT{}: {==}
    case True{} False{} False{} EQ{} EQ{} LT{}: e
    case True{} False{} False{} EQ{} EQ{} EQ{}: {==}
    case True{} False{} False{} EQ{} EQ{} GT{}: {==}
    case True{} False{} False{} EQ{} GT{} LT{}: e
    case True{} False{} False{} EQ{} GT{} EQ{}: {==}
    case True{} False{} False{} EQ{} GT{} GT{}: {==}
    case True{} False{} False{} GT{} LT{} LT{}: {==}
    case True{} False{} False{} GT{} LT{} EQ{}: {==}
    case True{} False{} False{} GT{} LT{} GT{}: {==}
    case True{} False{} False{} GT{} EQ{} LT{}: e
    case True{} False{} False{} GT{} EQ{} EQ{}: {==}
    case True{} False{} False{} GT{} EQ{} GT{}: {==}
    case True{} False{} False{} GT{} GT{} LT{}: e
    case True{} False{} False{} GT{} GT{} EQ{}: {==}
    case True{} False{} False{} GT{} GT{} GT{}: {==}
    case True{} False{} True{} LT{} LT{} LT{}: {==}
    case True{} False{} True{} LT{} LT{} EQ{}: e
    case True{} False{} True{} LT{} LT{} GT{}: e
    case True{} False{} True{} LT{} EQ{} LT{}: {==}
    case True{} False{} True{} LT{} EQ{} EQ{}: e
    case True{} False{} True{} LT{} EQ{} GT{}: e
    case True{} False{} True{} LT{} GT{} LT{}: {==}
    case True{} False{} True{} LT{} GT{} EQ{}: {==}
    case True{} False{} True{} LT{} GT{} GT{}: {==}
    case True{} False{} True{} EQ{} LT{} LT{}: {==}
    case True{} False{} True{} EQ{} LT{} EQ{}: {==}
    case True{} False{} True{} EQ{} LT{} GT{}: {==}
    case True{} False{} True{} EQ{} EQ{} LT{}: {==}
    case True{} False{} True{} EQ{} EQ{} EQ{}: {==}
    case True{} False{} True{} EQ{} EQ{} GT{}: {==}
    case True{} False{} True{} EQ{} GT{} LT{}: e
    case True{} False{} True{} EQ{} GT{} EQ{}: e
    case True{} False{} True{} EQ{} GT{} GT{}: {==}
    case True{} False{} True{} GT{} LT{} LT{}: {==}
    case True{} False{} True{} GT{} LT{} EQ{}: {==}
    case True{} False{} True{} GT{} LT{} GT{}: {==}
    case True{} False{} True{} GT{} EQ{} LT{}: {==}
    case True{} False{} True{} GT{} EQ{} EQ{}: {==}
    case True{} False{} True{} GT{} EQ{} GT{}: {==}
    case True{} False{} True{} GT{} GT{} LT{}: e
    case True{} False{} True{} GT{} GT{} EQ{}: e
    case True{} False{} True{} GT{} GT{} GT{}: {==}
    case True{} True{} False{} LT{} LT{} LT{}: {==}
    case True{} True{} False{} LT{} LT{} EQ{}: e
    case True{} True{} False{} LT{} LT{} GT{}: e
    case True{} True{} False{} LT{} EQ{} LT{}: {==}
    case True{} True{} False{} LT{} EQ{} EQ{}: {==}
    case True{} True{} False{} LT{} EQ{} GT{}: {==}
    case True{} True{} False{} LT{} GT{} LT{}: {==}
    case True{} True{} False{} LT{} GT{} EQ{}: {==}
    case True{} True{} False{} LT{} GT{} GT{}: {==}
    case True{} True{} False{} EQ{} LT{} LT{}: {==}
    case True{} True{} False{} EQ{} LT{} EQ{}: e
    case True{} True{} False{} EQ{} LT{} GT{}: e
    case True{} True{} False{} EQ{} EQ{} LT{}: e
    case True{} True{} False{} EQ{} EQ{} EQ{}: {==}
    case True{} True{} False{} EQ{} EQ{} GT{}: {==}
    case True{} True{} False{} EQ{} GT{} LT{}: e
    case True{} True{} False{} EQ{} GT{} EQ{}: {==}
    case True{} True{} False{} EQ{} GT{} GT{}: {==}
    case True{} True{} False{} GT{} LT{} LT{}: {==}
    case True{} True{} False{} GT{} LT{} EQ{}: {==}
    case True{} True{} False{} GT{} LT{} GT{}: {==}
    case True{} True{} False{} GT{} EQ{} LT{}: e
    case True{} True{} False{} GT{} EQ{} EQ{}: {==}
    case True{} True{} False{} GT{} EQ{} GT{}: {==}
    case True{} True{} False{} GT{} GT{} LT{}: e
    case True{} True{} False{} GT{} GT{} EQ{}: {==}
    case True{} True{} False{} GT{} GT{} GT{}: {==}
    case True{} True{} True{} LT{} LT{} LT{}: {==}
    case True{} True{} True{} LT{} LT{} EQ{}: e
    case True{} True{} True{} LT{} LT{} GT{}: e
    case True{} True{} True{} LT{} EQ{} LT{}: {==}
    case True{} True{} True{} LT{} EQ{} EQ{}: e
    case True{} True{} True{} LT{} EQ{} GT{}: e
    case True{} True{} True{} LT{} GT{} LT{}: {==}
    case True{} True{} True{} LT{} GT{} EQ{}: {==}
    case True{} True{} True{} LT{} GT{} GT{}: {==}
    case True{} True{} True{} EQ{} LT{} LT{}: {==}
    case True{} True{} True{} EQ{} LT{} EQ{}: e
    case True{} True{} True{} EQ{} LT{} GT{}: e
    case True{} True{} True{} EQ{} EQ{} LT{}: e
    case True{} True{} True{} EQ{} EQ{} EQ{}: {==}
    case True{} True{} True{} EQ{} EQ{} GT{}: e
    case True{} True{} True{} EQ{} GT{} LT{}: e
    case True{} True{} True{} EQ{} GT{} EQ{}: e
    case True{} True{} True{} EQ{} GT{} GT{}: {==}
    case True{} True{} True{} GT{} LT{} LT{}: {==}
    case True{} True{} True{} GT{} LT{} EQ{}: {==}
    case True{} True{} True{} GT{} LT{} GT{}: {==}
    case True{} True{} True{} GT{} EQ{} LT{}: e
    case True{} True{} True{} GT{} EQ{} EQ{}: e
    case True{} True{} True{} GT{} EQ{} GT{}: {==}
    case True{} True{} True{} GT{} GT{} LT{}: e
    case True{} True{} True{} GT{} GT{} EQ{}: e
    case True{} True{} True{} GT{} GT{} GT{}: {==}

def chain(+n: Nat,+a: Word(n),+b: Word(n),+c: Word(n)) ->
  {coherent(Word.cmp(n,a,b),Word.cmp(n,b,c),Word.cmp(n,a,c)) == True{} : Bool}:
  match n:
    case 0n: {==}
    case 1n+p:
      match a b c:
        case WCon{ah,at} WCon{bh,bt} WCon{ch,ct}:
          step(ah,bh,ch,Word.cmp(p,at,bt),Word.cmp(p,bt,ct),Word.cmp(p,at,ct),chain(p,at,bt,ct))

def lt_lt_finish(+ab: Cmp,+bc: Cmp,+ac: Cmp,
  one: {Cmp.is_lt(ab) == True{} : Bool},two: {Cmp.is_lt(bc) == True{} : Bool},
  e: {coherent(ab,bc,ac) == True{} : Bool}) -> {Cmp.is_lt(ac) == True{} : Bool}:
  match ab bc ac:
    case LT{} LT{} LT{}: {==}
    case LT{} LT{} EQ{}: e
    case LT{} LT{} GT{}: e
    case LT{} EQ{} LT{}: {==}
    case LT{} EQ{} EQ{}: two
    case LT{} EQ{} GT{}: two
    case LT{} GT{} LT{}: {==}
    case LT{} GT{} EQ{}: two
    case LT{} GT{} GT{}: two
    case EQ{} LT{} LT{}: {==}
    case EQ{} LT{} EQ{}: one
    case EQ{} LT{} GT{}: one
    case EQ{} EQ{} LT{}: {==}
    case EQ{} EQ{} EQ{}: one
    case EQ{} EQ{} GT{}: one
    case EQ{} GT{} LT{}: {==}
    case EQ{} GT{} EQ{}: one
    case EQ{} GT{} GT{}: one
    case GT{} LT{} LT{}: {==}
    case GT{} LT{} EQ{}: one
    case GT{} LT{} GT{}: one
    case GT{} EQ{} LT{}: {==}
    case GT{} EQ{} EQ{}: one
    case GT{} EQ{} GT{}: one
    case GT{} GT{} LT{}: {==}
    case GT{} GT{} EQ{}: one
    case GT{} GT{} GT{}: one

def lt_lt(+a: U32,+b: U32,+c: U32,
  one: {U32.is_lt(a,b) == True{} : Bool},two: {U32.is_lt(b,c) == True{} : Bool}) ->
  {U32.is_lt(a,c) == True{} : Bool}:
  match a b c:
    case U32{x} U32{y} U32{z}:
      lt_lt_finish(Word.cmp(32n,x,y),Word.cmp(32n,y,z),Word.cmp(32n,x,z),one,two,chain(32n,x,y,z))

def lt_le_finish(+ab: Cmp,+bc: Cmp,+ac: Cmp,
  one: {Cmp.is_lt(ab) == True{} : Bool},two: {Cmp.is_le(bc) == True{} : Bool},
  e: {coherent(ab,bc,ac) == True{} : Bool}) -> {Cmp.is_lt(ac) == True{} : Bool}:
  match ab bc ac:
    case LT{} LT{} LT{}: {==}
    case LT{} LT{} EQ{}: e
    case LT{} LT{} GT{}: e
    case LT{} EQ{} LT{}: {==}
    case LT{} EQ{} EQ{}: e
    case LT{} EQ{} GT{}: e
    case LT{} GT{} LT{}: {==}
    case LT{} GT{} EQ{}: two
    case LT{} GT{} GT{}: two
    case EQ{} LT{} LT{}: {==}
    case EQ{} LT{} EQ{}: one
    case EQ{} LT{} GT{}: one
    case EQ{} EQ{} LT{}: {==}
    case EQ{} EQ{} EQ{}: one
    case EQ{} EQ{} GT{}: one
    case EQ{} GT{} LT{}: {==}
    case EQ{} GT{} EQ{}: one
    case EQ{} GT{} GT{}: one
    case GT{} LT{} LT{}: {==}
    case GT{} LT{} EQ{}: one
    case GT{} LT{} GT{}: one
    case GT{} EQ{} LT{}: {==}
    case GT{} EQ{} EQ{}: one
    case GT{} EQ{} GT{}: one
    case GT{} GT{} LT{}: {==}
    case GT{} GT{} EQ{}: one
    case GT{} GT{} GT{}: one

def lt_le(+a: U32,+b: U32,+c: U32,
  one: {U32.is_lt(a,b) == True{} : Bool},two: {U32.is_le(b,c) == True{} : Bool}) ->
  {U32.is_lt(a,c) == True{} : Bool}:
  match a b c:
    case U32{x} U32{y} U32{z}:
      lt_le_finish(Word.cmp(32n,x,y),Word.cmp(32n,y,z),Word.cmp(32n,x,z),one,two,chain(32n,x,y,z))

def le_lt_finish(+ab: Cmp,+bc: Cmp,+ac: Cmp,
  one: {Cmp.is_le(ab) == True{} : Bool},two: {Cmp.is_lt(bc) == True{} : Bool},
  e: {coherent(ab,bc,ac) == True{} : Bool}) -> {Cmp.is_lt(ac) == True{} : Bool}:
  match ab bc ac:
    case LT{} LT{} LT{}: {==}
    case LT{} LT{} EQ{}: e
    case LT{} LT{} GT{}: e
    case LT{} EQ{} LT{}: {==}
    case LT{} EQ{} EQ{}: two
    case LT{} EQ{} GT{}: two
    case LT{} GT{} LT{}: {==}
    case LT{} GT{} EQ{}: two
    case LT{} GT{} GT{}: two
    case EQ{} LT{} LT{}: {==}
    case EQ{} LT{} EQ{}: e
    case EQ{} LT{} GT{}: e
    case EQ{} EQ{} LT{}: {==}
    case EQ{} EQ{} EQ{}: two
    case EQ{} EQ{} GT{}: two
    case EQ{} GT{} LT{}: {==}
    case EQ{} GT{} EQ{}: two
    case EQ{} GT{} GT{}: two
    case GT{} LT{} LT{}: {==}
    case GT{} LT{} EQ{}: one
    case GT{} LT{} GT{}: one
    case GT{} EQ{} LT{}: {==}
    case GT{} EQ{} EQ{}: one
    case GT{} EQ{} GT{}: one
    case GT{} GT{} LT{}: {==}
    case GT{} GT{} EQ{}: one
    case GT{} GT{} GT{}: one

def le_lt(+a: U32,+b: U32,+c: U32,
  one: {U32.is_le(a,b) == True{} : Bool},two: {U32.is_lt(b,c) == True{} : Bool}) ->
  {U32.is_lt(a,c) == True{} : Bool}:
  match a b c:
    case U32{x} U32{y} U32{z}:
      le_lt_finish(Word.cmp(32n,x,y),Word.cmp(32n,y,z),Word.cmp(32n,x,z),one,two,chain(32n,x,y,z))

def lt_ne_finish(+c: Cmp,e: {Cmp.is_lt(c) == True{} : Bool}) -> {Cmp.is_eq(c) == False{} : Bool}:
  match c:
    case LT{}: {==}
    case EQ{}: Equal.sym(Bool,False{},True{},e)
    case GT{}: {==}

def lt_ne(+a: U32,+b: U32,e: {U32.is_lt(a,b) == True{} : Bool}) -> {U32.is_eq(a,b) == False{} : Bool}:
  lt_ne_finish(U32.cmp(a,b),e)

# Comparison of binary natural observations uses symbolic tails, never 2^32 unary expansion.
def digit(b: Bool,n: Nat) -> Nat:
  Nat.add(Add.bit(b),Nat.double(n))

def digits(+a: Bool,+b: Bool,+x: Nat,+y: Nat) ->
  {Nat.cmp(digit(a,x),digit(b,y)) == Word.cmp.fin(a,b,Nat.cmp(x,y)) : Cmp}:
  match a b x y:
    case False{} False{} 0n 0n: {==}
    case False{} False{} 0n 1n+q: {==}
    case False{} False{} 1n+p 0n: {==}
    case False{} False{} 1n+p 1n+q: digits(False{},False{},p,q)
    case False{} True{} 0n 0n: {==}
    case False{} True{} 0n 1n+q: {==}
    case False{} True{} 1n+p 0n: {==}
    case False{} True{} 1n+p 1n+q: digits(False{},True{},p,q)
    case True{} False{} 0n 0n: {==}
    case True{} False{} 0n 1n+q: {==}
    case True{} False{} 1n+p 0n: {==}
    case True{} False{} 1n+p 1n+q: digits(True{},False{},p,q)
    case True{} True{} 0n 0n: {==}
    case True{} True{} 0n 1n+q: {==}
    case True{} True{} 1n+p 0n: {==}
    case True{} True{} 1n+p 1n+q: digits(True{},True{},p,q)

def observed(+n: Nat,+a: Word(n),+b: Word(n)) ->
  {Word.cmp(n,a,b) == Nat.cmp(Word.to_nat(n,a),Word.to_nat(n,b)) : Cmp}:
  match n:
    case 0n: {==}
    case 1n+p:
      match a b:
        case WCon{False{},at} WCon{False{},bt}:
          Equal.trans(Cmp,Word.cmp.fin(False{},False{},Word.cmp(p,at,bt)),
            Word.cmp.fin(False{},False{},Nat.cmp(Word.to_nat(p,at),Word.to_nat(p,bt))),
            Nat.cmp(digit(False{},Word.to_nat(p,at)),digit(False{},Word.to_nat(p,bt))),
            Equal.cong(Cmp,Cmp,c => Word.cmp.fin(False{},False{},c),Word.cmp(p,at,bt),
              Nat.cmp(Word.to_nat(p,at),Word.to_nat(p,bt)),observed(p,at,bt)),
            Equal.sym(Cmp,Nat.cmp(digit(False{},Word.to_nat(p,at)),digit(False{},Word.to_nat(p,bt))),
              Word.cmp.fin(False{},False{},Nat.cmp(Word.to_nat(p,at),Word.to_nat(p,bt))),
              digits(False{},False{},Word.to_nat(p,at),Word.to_nat(p,bt))))
        case WCon{False{},at} WCon{True{},bt}:
          Equal.trans(Cmp,Word.cmp.fin(False{},True{},Word.cmp(p,at,bt)),
            Word.cmp.fin(False{},True{},Nat.cmp(Word.to_nat(p,at),Word.to_nat(p,bt))),
            Nat.cmp(digit(False{},Word.to_nat(p,at)),digit(True{},Word.to_nat(p,bt))),
            Equal.cong(Cmp,Cmp,c => Word.cmp.fin(False{},True{},c),Word.cmp(p,at,bt),
              Nat.cmp(Word.to_nat(p,at),Word.to_nat(p,bt)),observed(p,at,bt)),
            Equal.sym(Cmp,Nat.cmp(digit(False{},Word.to_nat(p,at)),digit(True{},Word.to_nat(p,bt))),
              Word.cmp.fin(False{},True{},Nat.cmp(Word.to_nat(p,at),Word.to_nat(p,bt))),
              digits(False{},True{},Word.to_nat(p,at),Word.to_nat(p,bt))))
        case WCon{True{},at} WCon{False{},bt}:
          Equal.trans(Cmp,Word.cmp.fin(True{},False{},Word.cmp(p,at,bt)),
            Word.cmp.fin(True{},False{},Nat.cmp(Word.to_nat(p,at),Word.to_nat(p,bt))),
            Nat.cmp(digit(True{},Word.to_nat(p,at)),digit(False{},Word.to_nat(p,bt))),
            Equal.cong(Cmp,Cmp,c => Word.cmp.fin(True{},False{},c),Word.cmp(p,at,bt),
              Nat.cmp(Word.to_nat(p,at),Word.to_nat(p,bt)),observed(p,at,bt)),
            Equal.sym(Cmp,Nat.cmp(digit(True{},Word.to_nat(p,at)),digit(False{},Word.to_nat(p,bt))),
              Word.cmp.fin(True{},False{},Nat.cmp(Word.to_nat(p,at),Word.to_nat(p,bt))),
              digits(True{},False{},Word.to_nat(p,at),Word.to_nat(p,bt))))
        case WCon{True{},at} WCon{True{},bt}:
          Equal.trans(Cmp,Word.cmp.fin(True{},True{},Word.cmp(p,at,bt)),
            Word.cmp.fin(True{},True{},Nat.cmp(Word.to_nat(p,at),Word.to_nat(p,bt))),
            Nat.cmp(digit(True{},Word.to_nat(p,at)),digit(True{},Word.to_nat(p,bt))),
            Equal.cong(Cmp,Cmp,c => Word.cmp.fin(True{},True{},c),Word.cmp(p,at,bt),
              Nat.cmp(Word.to_nat(p,at),Word.to_nat(p,bt)),observed(p,at,bt)),
            Equal.sym(Cmp,Nat.cmp(digit(True{},Word.to_nat(p,at)),digit(True{},Word.to_nat(p,bt))),
              Word.cmp.fin(True{},True{},Nat.cmp(Word.to_nat(p,at),Word.to_nat(p,bt))),
              digits(True{},True{},Word.to_nat(p,at),Word.to_nat(p,bt))))

def from_nat_lt(+a: U32,+b: U32,e: {Nat.is_lt(U32.to_nat(a),U32.to_nat(b)) == True{} : Bool}) ->
  {U32.is_lt(a,b) == True{} : Bool}:
  match a b:
    case U32{x} U32{y}:
      Equal.trans(Bool,U32.is_lt(U32{x},U32{y}),Nat.is_lt(Word.to_nat(32n,x),Word.to_nat(32n,y)),True{},
        Equal.cong(Cmp,Bool,c => Cmp.is_lt(c),Word.cmp(32n,x,y),Nat.cmp(Word.to_nat(32n,x),Word.to_nat(32n,y)),observed(32n,x,y)),e)


def reverse_finish(+ab: Cmp,+ba: Cmp,e: {Cmp.is_lt(ba) == False{} : Bool},
  consistent: {coherent(ab,ba,EQ{}) == True{} : Bool}) -> {Cmp.is_le(ab) == True{} : Bool}:
  match ab ba:
    case LT{} LT{}: {==}
    case LT{} EQ{}: {==}
    case LT{} GT{}: {==}
    case EQ{} LT{}: {==}
    case EQ{} EQ{}: {==}
    case EQ{} GT{}: {==}
    case GT{} LT{}: Equal.sym(Bool,True{},False{},e)
    case GT{} EQ{}: consistent
    case GT{} GT{}: consistent

def reverse_le(+a: U32,+b: U32,e: {U32.is_lt(b,a) == False{} : Bool}) -> {U32.is_le(a,b) == True{} : Bool}:
  match a b:
    case U32{x} U32{y}:
      reverse_finish(Word.cmp(32n,x,y),Word.cmp(32n,y,x),e,
        Equal.trans(Bool,coherent(Word.cmp(32n,x,y),Word.cmp(32n,y,x),EQ{}),
          coherent(Word.cmp(32n,x,y),Word.cmp(32n,y,x),Word.cmp(32n,x,x)),True{},
          Equal.cong(Cmp,Bool,c => coherent(Word.cmp(32n,x,y),Word.cmp(32n,y,x),c),EQ{},Word.cmp(32n,x,x),
            Equal.sym(Cmp,Word.cmp(32n,x,x),EQ{},Compare.reflexive(32n,x))),chain(32n,x,y,x)))
