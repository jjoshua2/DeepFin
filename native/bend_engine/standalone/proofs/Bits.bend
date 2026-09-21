import Base
import ./u64/Words.bend as U64Words

# PEXT ignores every input bit outside its mask, at arbitrary word width.
def pext_masked(+n: Nat, +x: Word(n), +mask: Word(n)) ->
  {Word.pext(n, Word.and(n, x, mask), mask) == Word.pext(n, x, mask) : Word(n)}:
  match n:
    case 0n: {==}
    case 1n+p:
      match x mask:
        case WCon{False{}, t} WCon{False{}, m}:
          %pext_masked(p, t, m) : {Word.pext(1n+p, Word.and(1n+p, WCon{False{}, t}, WCon{False{}, m}), WCon{False{}, m}) == Word.shr.pad(p, _) : Word(1n+p)}
          {==}
        case WCon{True{}, t} WCon{False{}, m}:
          %pext_masked(p, t, m) : {Word.pext(1n+p, Word.and(1n+p, WCon{True{}, t}, WCon{False{}, m}), WCon{False{}, m}) == Word.shr.pad(p, _) : Word(1n+p)}
          {==}
        case WCon{False{}, t} WCon{True{}, m}:
          %pext_masked(p, t, m) : {Word.pext(1n+p, Word.and(1n+p, WCon{False{}, t}, WCon{True{}, m}), WCon{True{}, m}) == WCon{False{}, _} : Word(1n+p)}
          {==}
        case WCon{True{}, t} WCon{True{}, m}:
          %pext_masked(p, t, m) : {Word.pext(1n+p, Word.and(1n+p, WCon{True{}, t}, WCon{True{}, m}), WCon{True{}, m}) == WCon{True{}, _} : Word(1n+p)}
          {==}

def to_and(+x: U64, +y: U64) ->
  {U64.to_word(U64.and(x, y)) == Word.and(64n, U64.to_word(x), U64.to_word(y)) : Word(64n)}:
  %U64Words.from_to(x) : {U64.to_word(U64.and(_, y)) == Word.and(64n, U64.to_word(x), U64.to_word(y)) : Word(64n)}
  %U64Words.from_to(y) : {U64.to_word(U64.and(U64.from_word(U64.to_word(x)), _)) == Word.and(64n, U64.to_word(x), U64.to_word(y)) : Word(64n)}
  %U64Words.pack_and(U64.to_word(x), U64.to_word(y)) : {U64.to_word(_) == Word.and(64n, U64.to_word(x), U64.to_word(y)) : Word(64n)}
  U64Words.to_from(Word.and(64n, U64.to_word(x), U64.to_word(y)))

def u64_pext_masked(+x: U64, +mask: U64) ->
  {U64.pext(U64.and(x, mask), mask) == U64.pext(x, mask) : U64}:
  %pext_masked(64n, U64.to_word(x), U64.to_word(mask)) : {U64.pext(U64.and(x, mask), mask) == U64.from_word(_) : U64}
  %to_and(x, mask) : {U64.pext(U64.and(x, mask), mask) == U64.from_word(Word.pext(64n, _, U64.to_word(mask))) : U64}
  {==}

# Zero upper bits are the precise precondition for lossless U32 indexing.
def zero_high(+x: U64, e: {U64.high(x) == 0 : U32}) ->
  {U64.from_u32(U64.low(x)) == x : U64}:
  match x:
    case U64{lo, hi}:
      %e : {U64{lo, _} == U64{lo, hi} : U64}
      {==}

# Reflection bridges: a checked Boolean predicate becomes equality evidence.
def and_left(+a: Bool, b: Bool, e: {Bool.and(a, b) == True{} : Bool}) ->
  {a == True{} : Bool}:
  match a:
    case False{}: U64Words.false_true({False{} == True{} : Bool}, e)
    case True{}: {==}

def and_right(a: Bool, +b: Bool, e: {Bool.and(a, b) == True{} : Bool}) ->
  {b == True{} : Bool}:
  match a:
    case False{}: U64Words.false_true({b == True{} : Bool}, e)
    case True{}: e

def fin_tail(a: Bool, b: Bool, +c: Cmp, e: {Cmp.is_eq(Word.cmp.fin(a, b, c)) == True{} : Bool}) ->
  {Cmp.is_eq(c) == True{} : Bool}:
  match c:
    case LT{}: U64Words.false_true({False{} == True{} : Bool}, e)
    case EQ{}: {==}
    case GT{}: U64Words.false_true({False{} == True{} : Bool}, e)

def fin_head(c: Cmp, +a: Bool, +b: Bool, e: {Cmp.is_eq(Word.cmp.fin(a, b, c)) == True{} : Bool}) ->
  {a == b : Bool}:
  match c:
    case LT{}: U64Words.false_true({a == b : Bool}, e)
    case GT{}: U64Words.false_true({a == b : Bool}, e)
    case EQ{}:
      match a b:
        case False{} False{}: {==}
        case True{} True{}: {==}
        case False{} True{}: U64Words.false_true({False{} == True{} : Bool}, e)
        case True{} False{}: U64Words.false_true({True{} == False{} : Bool}, e)

def word_eq(+n: Nat, +x: Word(n), +y: Word(n), e: {Cmp.is_eq(Word.cmp(n, x, y)) == True{} : Bool}) ->
  {x == y : Word(n)}:
  match n:
    case 0n:
      match x y:
        case WNil{} WNil{}: {==}
    case 1n+p:
      match x y:
        case WCon{False{}, at} WCon{False{}, bt}:
          %word_eq(p, at, bt, fin_tail(False{}, False{}, Word.cmp(p, at, bt), e)) : {WCon{False{}, at} == WCon{False{}, _} : Word(1n+p)}
          {==}
        case WCon{True{}, at} WCon{True{}, bt}:
          %word_eq(p, at, bt, fin_tail(True{}, True{}, Word.cmp(p, at, bt), e)) : {WCon{True{}, at} == WCon{True{}, _} : Word(1n+p)}
          {==}
        case WCon{False{}, at} WCon{True{}, bt}:
          U64Words.false_true({WCon{False{}, at} == WCon{True{}, bt} : Word(1n+p)},
            fin_head(Word.cmp(p, at, bt), False{}, True{}, e))
        case WCon{True{}, at} WCon{False{}, bt}:
          U64Words.false_true({WCon{True{}, at} == WCon{False{}, bt} : Word(1n+p)},
            Equal.sym(Bool, True{}, False{}, fin_head(Word.cmp(p, at, bt), True{}, False{}, e)))

def u32_eq(+a: U32, +b: U32, e: {U32.is_eq(a, b) == True{} : Bool}) -> {a == b : U32}:
  match a b:
    case U32{x} U32{y}:
      %word_eq(32n, x, y, e) : {U32{x} == U32{_} : U32}
      {==}


# Consume one conjunction certificate; its two conclusions stay affine.
def and_elim(-A: Type, +a: Bool, +b: Bool,
  e: {Bool.and(a, b) == True{} : Bool}, k: {a == True{} : Bool} -> {b == True{} : Bool} -> A) -> A:
  match a:
    case False{}: U64Words.false_true(A, e)
    case True{}: k({==}, e)

def not_lt_zero(-A: Type, n: Nat, e: {Nat.is_lt(n, 0n) == True{} : Bool}) -> A:
  match n:
    case 0n: U64Words.false_true(A, e)
    case 1n+p: U64Words.false_true(A, e)

def and_intro(+a: Bool, +b: Bool, ea: {a == True{} : Bool}, eb: {b == True{} : Bool}) -> {Bool.and(a, b) == True{} : Bool}:
  match a:
    case False{}: U64Words.false_true({False{} == True{} : Bool}, ea)
    case True{}: eb
