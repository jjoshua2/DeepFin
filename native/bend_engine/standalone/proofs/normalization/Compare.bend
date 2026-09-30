# Unsigned structural comparison, without expanding unary machine constants.
import Base
import ../u64/Words.bend as Old


def reflexive(+n: Nat, +w: Word(n)) -> {Word.cmp(n,w,w) == EQ{} : Cmp}:
  match n:
    case 0n: {==}
    case 1n+p:
      match w:
        case WCon{False{},t}:
          Equal.cong(Cmp,Cmp,c => Word.cmp.fin(False{},False{},c),Word.cmp(p,t,t),EQ{},reflexive(p,t))
        case WCon{True{},t}:
          Equal.cong(Cmp,Cmp,c => Word.cmp.fin(True{},True{},c),Word.cmp(p,t,t),EQ{},reflexive(p,t))

def zero_comparison(+p: Nat, +t: Word(p), same: {t == Word.zero(p) : Word(p)}) ->
  {Word.cmp(p,t,Word.zero(p)) == EQ{} : Cmp}:
  Equal.trans(Cmp,Word.cmp(p,t,Word.zero(p)),Word.cmp(p,Word.zero(p),Word.zero(p)),EQ{},
    Equal.cong(Word(p),Cmp,x => Word.cmp(p,x,Word.zero(p)),t,Word.zero(p),same),reflexive(p,Word.zero(p)))

def tail_le(+a: Bool, +b: Bool, c: Cmp,
  e: {Cmp.is_le(Word.cmp.fin(a,b,c)) == True{} : Bool}) ->
  {Cmp.is_le(c) == True{} : Bool}:
  match c:
    case LT{}: {==}
    case EQ{}: {==}
    case GT{}: e

def zero_finish(+p: Nat, +b: Bool, +t: Word(p),
  same: {t == Word.zero(p) : Word(p)},
  e: {Cmp.is_le(Word.cmp(1n+p,WCon{b,t},Word.zero(1n+p))) == True{} : Bool}) ->
  {WCon{b,t} == Word.zero(1n+p) : Word(1n+p)}:
  match b:
    case False{}:
      Equal.cong(Word(p),Word(1n+p),x => WCon{False{},x},t,Word.zero(p),same)
    case True{}:
      Old.false_true({WCon{True{},t} == Word.zero(1n+p) : Word(1n+p)},
        Equal.trans(Bool,False{},Cmp.is_le(Word.cmp.fin(True{},False{},Word.cmp(p,t,Word.zero(p)))),True{},
          Equal.cong(Cmp,Bool,c => Cmp.is_le(Word.cmp.fin(True{},False{},c)),
            EQ{},Word.cmp(p,t,Word.zero(p)),Equal.sym(Cmp,Word.cmp(p,t,Word.zero(p)),EQ{},zero_comparison(p,t,same))),e))

def le_zero(+n: Nat, +w: Word(n),
  +e: {Cmp.is_le(Word.cmp(n,w,Word.zero(n))) == True{} : Bool}) ->
  {w == Word.zero(n) : Word(n)}:
  match n:
    case 0n:
      match w:
        case WNil{}: {==}
    case 1n+p:
      match w:
        case WCon{b,t}:
          zero_finish(p,b,t,le_zero(p,t,tail_le(b,False{},Word.cmp(p,t,Word.zero(p)),e)),e)

def tail_of_below_one(+b: Bool, c: Cmp,
  e: {Cmp.is_lt(Word.cmp.fin(b,True{},c)) == True{} : Bool}) ->
  {Cmp.is_le(c) == True{} : Bool}:
  match c:
    case LT{}: {==}
    case EQ{}: {==}
    case GT{}: e

def strip_low(+b: Bool, c: Cmp,
  e: {Cmp.is_lt(Word.cmp.fin(b,False{},c)) == True{} : Bool}) ->
  {Cmp.is_lt(c) == True{} : Bool}:
  match b c:
    case False{} LT{}: {==}
    case True{} LT{}: {==}
    case False{} EQ{}: e
    case True{} EQ{}: e
    case False{} GT{}: e
    case True{} GT{}: e

def below_one_finish(+p: Nat, +b: Bool, +t: Word(p),
  same: {t == Word.zero(p) : Word(p)},
  e: {Cmp.is_lt(Word.cmp(1n+p,WCon{b,t},WCon{True{},Word.zero(p)})) == True{} : Bool}) ->
  {WCon{b,t} == Word.zero(1n+p) : Word(1n+p)}:
  match b:
    case False{}:
      Equal.cong(Word(p),Word(1n+p),x => WCon{False{},x},t,Word.zero(p),same)
    case True{}:
      Old.false_true({WCon{True{},t} == Word.zero(1n+p) : Word(1n+p)},
        Equal.trans(Bool,False{},Cmp.is_lt(Word.cmp.fin(True{},True{},Word.cmp(p,t,Word.zero(p)))),True{},
          Equal.cong(Cmp,Bool,c => Cmp.is_lt(Word.cmp.fin(True{},True{},c)),
            EQ{},Word.cmp(p,t,Word.zero(p)),Equal.sym(Cmp,Word.cmp(p,t,Word.zero(p)),EQ{},zero_comparison(p,t,same))),e))
