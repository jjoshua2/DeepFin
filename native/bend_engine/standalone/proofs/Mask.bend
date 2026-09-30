import Base

# Structural induction over arbitrary word width; no enumerated U64 inputs.
def absorb(+n: Nat, +x: Word(n), +mask: Word(n)) ->
  {Word.and(n, Word.and(n, x, mask), mask) == Word.and(n, x, mask) : Word(n)}:
  match n:
    case 0n: {==}
    case 1n+p:
      match x mask:
        case WCon{False{}, t} WCon{False{}, m}:
          %absorb(p, t, m) : {WCon{False{}, Word.and(p, Word.and(p, t, m), m)} == WCon{False{}, _} : Word(1n+p)}
          {==}
        case WCon{False{}, t} WCon{True{}, m}:
          %absorb(p, t, m) : {WCon{False{}, Word.and(p, Word.and(p, t, m), m)} == WCon{False{}, _} : Word(1n+p)}
          {==}
        case WCon{True{}, t} WCon{False{}, m}:
          %absorb(p, t, m) : {WCon{False{}, Word.and(p, Word.and(p, t, m), m)} == WCon{False{}, _} : Word(1n+p)}
          {==}
        case WCon{True{}, t} WCon{True{}, m}:
          %absorb(p, t, m) : {WCon{True{}, Word.and(p, Word.and(p, t, m), m)} == WCon{True{}, _} : Word(1n+p)}
          {==}

def zero_left(+n: Nat, +mask: Word(n)) ->
  {Word.and(n, Word.zero(n), mask) == Word.zero(n) : Word(n)}:
  match n:
    case 0n: {==}
    case 1n+p:
      match mask:
        case WCon{a, m}:
          %zero_left(p, m) : {WCon{False{}, Word.and(p, Word.zero(p), m)} == WCon{False{}, _} : Word(1n+p)}
          {==}

def zero_right(+n: Nat, +x: Word(n)) ->
  {Word.and(n, x, Word.zero(n)) == Word.zero(n) : Word(n)}:
  match n:
    case 0n: {==}
    case 1n+p:
      match x:
        case WCon{False{}, t}:
          %zero_right(p, t) : {WCon{False{}, Word.and(p, t, Word.zero(p))} == WCon{False{}, _} : Word(1n+p)}
          {==}
        case WCon{True{}, t}:
          %zero_right(p, t) : {WCon{False{}, Word.and(p, t, Word.zero(p))} == WCon{False{}, _} : Word(1n+p)}
          {==}

def u64_absorb(+x: U64, +mask: U64) ->
  {U64.and(U64.and(x, mask), mask) == U64.and(x, mask) : U64}:
  match x mask:
    case U64{U32{lo}, U32{hi}} U64{U32{ml}, U32{mh}}:
      %absorb(32n, lo, ml) : {U64.and(U64.and(U64{U32{lo}, U32{hi}}, U64{U32{ml}, U32{mh}}), U64{U32{ml}, U32{mh}}) == U64{U32{_}, U32{Word.and(32n, hi, mh)}} : U64}
      %absorb(32n, hi, mh) : {U64.and(U64.and(U64{U32{lo}, U32{hi}}, U64{U32{ml}, U32{mh}}), U64{U32{ml}, U32{mh}}) == U64{U32{Word.and(32n, Word.and(32n, lo, ml), ml)}, U32{_}} : U64}
      {==}

def u64_zero_left(mask: U64) -> {U64.and(U64.zero(), mask) == U64.zero() : U64}:
  match mask:
    case U64{U32{lo}, U32{hi}}:
      %zero_left(32n, lo) : {U64.and(U64.zero(), U64{U32{lo}, U32{hi}}) == U64{U32{_}, 0} : U64}
      %zero_left(32n, hi) : {U64.and(U64.zero(), U64{U32{lo}, U32{hi}}) == U64{U32{Word.and(32n, Word.zero(32n), lo)}, U32{_}} : U64}
      {==}

def u64_zero_right(x: U64) -> {U64.and(x, U64.zero()) == U64.zero() : U64}:
  match x:
    case U64{U32{lo}, U32{hi}}:
      %zero_right(32n, lo) : {U64.and(U64{U32{lo}, U32{hi}}, U64.zero()) == U64{U32{_}, 0} : U64}
      %zero_right(32n, hi) : {U64.and(U64{U32{lo}, U32{hi}}, U64.zero()) == U64{U32{Word.and(32n, lo, Word.zero(32n))}, U32{_}} : U64}
      {==}
