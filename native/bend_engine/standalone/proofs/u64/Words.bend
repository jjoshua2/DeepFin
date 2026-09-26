import Base

# Dropping a padding bit recovers every bit of the original word.
def trim_pad(n: Nat, w: Word(n)) -> {Word.trim(n, Word.shr.pad(n, w)) == w : Word(n)}:
  match n:
    case 0n:
      match w:
        case WNil{}:
          {==}
    case 1n+p:
      match w:
        case WCon{b, t}:
          %trim_pad(p, t) : {WCon{b, Word.trim(p, Word.shr.pad(p, t))} == WCon{b, _} : Word(1n+p)}
          {==}

def pdep_pad(+n: Nat, +mask: Word(n), +w: Word(n)) ->
  {Word.pdep(1n+n, WCon{False{}, mask}, Word.shr.pad(n, w))
    == WCon{False{}, Word.pdep(n, mask, w)} : Word(1n+n)}:
  %trim_pad(n, w) : {Word.pdep(1n+n, WCon{False{}, mask}, Word.shr.pad(n, w))
    == WCon{False{}, Word.pdep(n, mask, _)} : Word(1n+n)}
  {==}

def pdep_pext(+n: Nat, +x: Word(n), +mask: Word(n)) ->
  {Word.pdep(n, mask, Word.pext(n, x, mask)) == Word.and(n, x, mask) : Word(n)}:
  match n:
    case 0n:
      {==}
    case 1n+p:
      match x mask:
        case WCon{False{}, t} WCon{False{}, m}:
          %pdep_pext(p, t, m) : {Word.pdep(1n+p, WCon{False{}, m}, Word.shr.pad(p, Word.pext(p, t, m))) == WCon{False{}, _} : Word(1n+p)}
          pdep_pad(p, m, Word.pext(p, t, m))
        case WCon{True{}, t} WCon{False{}, m}:
          %pdep_pext(p, t, m) : {Word.pdep(1n+p, WCon{False{}, m}, Word.shr.pad(p, Word.pext(p, t, m))) == WCon{False{}, _} : Word(1n+p)}
          pdep_pad(p, m, Word.pext(p, t, m))
        case WCon{False{}, t} WCon{True{}, m}:
          %pdep_pext(p, t, m) : {WCon{False{}, Word.pdep(p, m, Word.pext(p, t, m))} == WCon{False{}, _} : Word(1n+p)}
          {==}
        case WCon{True{}, t} WCon{True{}, m}:
          %pdep_pext(p, t, m) : {WCon{True{}, Word.pdep(p, m, Word.pext(p, t, m))} == WCon{True{}, _} : Word(1n+p)}
          {==}

# The two-U32 representation is an exact split/join, not a truncated word.
def take_join(+n: Nat, -m: Nat, +a: Word(n), +b: Word(m)) ->
  {Word.take(n, m, Word.join(n, m, a, b)) == a : Word(n)}:
  match n:
    case 0n:
      match a:
        case WNil{}:
          {==}
    case 1n+p:
      match a:
        case WCon{h, t}:
          %take_join(p, m, t, b) : {WCon{h, Word.take(p, m, Word.join(p, m, t, b))} == WCon{h, _} : Word(1n+p)}
          {==}

def drop_join(+n: Nat, -m: Nat, +a: Word(n), +b: Word(m)) ->
  {Word.drop(n, m, Word.join(n, m, a, b)) == b : Word(m)}:
  match n:
    case 0n:
      {==}
    case 1n+p:
      match a:
        case WCon{h, t}:
          drop_join(p, m, t, b)

def join_split(+n: Nat, -m: Nat, +w: Word(Nat.add(n, m))) ->
  {Word.join(n, m, Word.take(n, m, w), Word.drop(n, m, w)) == w : Word(Nat.add(n, m))}:
  match n:
    case 0n:
      {==}
    case 1n+p:
      match w:
        case WCon{h, t}:
          %join_split(p, m, t) : {WCon{h, Word.join(p, m, Word.take(p, m, t), Word.drop(p, m, t))} == WCon{h, _} : Word(Nat.add(1n+p, m))}
          {==}

def from_to(+x: U64) -> {U64.from_word(U64.to_word(x)) == x : U64}:
  match x:
    case U64{U32{lo}, U32{hi}}:
      %take_join(32n, 32n, lo, hi) : {U64.from_word(Word.join(32n, 32n, lo, hi)) == U64{U32{_}, U32{hi}} : U64}
      %drop_join(32n, 32n, lo, hi) : {U64.from_word(Word.join(32n, 32n, lo, hi)) == U64{U32{Word.take(32n, 32n, Word.join(32n, 32n, lo, hi))}, U32{_}} : U64}
      {==}

def to_from(w: Word(64n)) -> {U64.to_word(U64.from_word(w)) == w : Word(64n)}:
  join_split(32n, 32n, w)

def take_and(+n: Nat, +m: Nat, +x: Word(Nat.add(n, m)), +y: Word(Nat.add(n, m))) ->
  {Word.take(n, m, Word.and(Nat.add(n, m), x, y))
    == Word.and(n, Word.take(n, m, x), Word.take(n, m, y)) : Word(n)}:
  match n:
    case 0n:
      {==}
    case 1n+p:
      match x y:
        case WCon{a, at} WCon{b, bt}:
          %take_and(p, m, at, bt) : {WCon{Bool.and(a, b), Word.take(p, m, Word.and(Nat.add(p, m), at, bt))} == WCon{Bool.and(a, b), _} : Word(1n+p)}
          {==}

def drop_and(+n: Nat, +m: Nat, +x: Word(Nat.add(n, m)), +y: Word(Nat.add(n, m))) ->
  {Word.drop(n, m, Word.and(Nat.add(n, m), x, y))
    == Word.and(m, Word.drop(n, m, x), Word.drop(n, m, y)) : Word(m)}:
  match n:
    case 0n:
      {==}
    case 1n+p:
      match x y:
        case WCon{a, at} WCon{b, bt}:
          drop_and(p, m, at, bt)

def pack_and(+x: Word(64n), +y: Word(64n)) ->
  {U64.from_word(Word.and(64n, x, y)) == U64.and(U64.from_word(x), U64.from_word(y)) : U64}:
  %take_and(32n, 32n, x, y) : {U64.from_word(Word.and(64n, x, y)) == U64{U32{_}, U32{Word.and(32n, Word.drop(32n, 32n, x), Word.drop(32n, 32n, y))}} : U64}
  %drop_and(32n, 32n, x, y) : {U64.from_word(Word.and(64n, x, y)) == U64{U32{Word.take(32n, 32n, Word.and(64n, x, y))}, U32{_}} : U64}
  {==}

# A false Boolean equality cannot manufacture a certificate.
def false_true(-A: Type, e: {False{} == True{} : Bool}) -> A:
  %e : Bool.pick(Type, _, A, Unit)
  Unit{}

def le_succ(a: Nat, b: Nat, e: {Nat.is_le(a, b) == True{} : Bool}) ->
  {Nat.is_le(a, 1n+b) == True{} : Bool}:
  match a b:
    case 0n _:
      {==}
    case 1n+ap 0n:
      false_true({Nat.is_le(1n+ap, 1n) == True{} : Bool}, e)
    case 1n+ap 1n+bp:
      le_succ(ap, bp, e)

def count_bound(+n: Nat, +w: Word(n)) -> {Nat.is_le(Word.count(n, w), n) == True{} : Bool}:
  match n:
    case 0n:
      {==}
    case 1n+p:
      match w:
        case WCon{False{}, t}:
          le_succ(Word.count(p, t), p, count_bound(p, t))
        case WCon{True{}, t}:
          count_bound(p, t)

# Only 65 possible counts, not enumeration of the 2^64 possible inputs.
# These closed cases bridge a proved Nat bound to the existing U32 comparator.
def count_u32_bound(k: Nat, e: {Nat.is_le(k, 64n) == True{} : Bool}) ->
  {U32.is_le(U32.from_nat(k), 64) == True{} : Bool}:
  match k:
    case 0n:
      {==}
    case 1n:
      {==}
    case 2n:
      {==}
    case 3n:
      {==}
    case 4n:
      {==}
    case 5n:
      {==}
    case 6n:
      {==}
    case 7n:
      {==}
    case 8n:
      {==}
    case 9n:
      {==}
    case 10n:
      {==}
    case 11n:
      {==}
    case 12n:
      {==}
    case 13n:
      {==}
    case 14n:
      {==}
    case 15n:
      {==}
    case 16n:
      {==}
    case 17n:
      {==}
    case 18n:
      {==}
    case 19n:
      {==}
    case 20n:
      {==}
    case 21n:
      {==}
    case 22n:
      {==}
    case 23n:
      {==}
    case 24n:
      {==}
    case 25n:
      {==}
    case 26n:
      {==}
    case 27n:
      {==}
    case 28n:
      {==}
    case 29n:
      {==}
    case 30n:
      {==}
    case 31n:
      {==}
    case 32n:
      {==}
    case 33n:
      {==}
    case 34n:
      {==}
    case 35n:
      {==}
    case 36n:
      {==}
    case 37n:
      {==}
    case 38n:
      {==}
    case 39n:
      {==}
    case 40n:
      {==}
    case 41n:
      {==}
    case 42n:
      {==}
    case 43n:
      {==}
    case 44n:
      {==}
    case 45n:
      {==}
    case 46n:
      {==}
    case 47n:
      {==}
    case 48n:
      {==}
    case 49n:
      {==}
    case 50n:
      {==}
    case 51n:
      {==}
    case 52n:
      {==}
    case 53n:
      {==}
    case 54n:
      {==}
    case 55n:
      {==}
    case 56n:
      {==}
    case 57n:
      {==}
    case 58n:
      {==}
    case 59n:
      {==}
    case 60n:
      {==}
    case 61n:
      {==}
    case 62n:
      {==}
    case 63n:
      {==}
    case 64n:
      {==}
    case 65n+p:
      false_true({U32.is_le(U32.from_nat(65n+p), 64) == True{} : Bool}, e)

def pad_zero(n: Nat) -> {Word.shr.pad(n, Word.zero(n)) == Word.zero(1n+n) : Word(1n+n)}:
  match n:
    case 0n:
      {==}
    case 1n+p:
      %pad_zero(p) : {WCon{False{}, Word.shr.pad(p, Word.zero(p))} == WCon{False{}, _} : Word(2n+p)}
      {==}

def pext_zero_mask(+n: Nat, +x: Word(n)) -> {Word.pext(n, x, Word.zero(n)) == Word.zero(n) : Word(n)}:
  match n:
    case 0n:
      {==}
    case 1n+p:
      match x:
        case WCon{b, t}:
          %pad_zero(p) : {Word.shr.pad(p, Word.pext(p, t, Word.zero(p))) == _ : Word(1n+p)}
          %pext_zero_mask(p, t) : {Word.shr.pad(p, Word.pext(p, t, Word.zero(p))) == Word.shr.pad(p, _) : Word(1n+p)}
          {==}

def pdep_zero_mask(+n: Nat, +x: Word(n)) -> {Word.pdep(n, Word.zero(n), x) == Word.zero(n) : Word(n)}:
  match n:
    case 0n:
      {==}
    case 1n+p:
      %pdep_zero_mask(p, Word.trim(p, x)) : {WCon{False{}, Word.pdep(p, Word.zero(p), Word.trim(p, x))} == WCon{False{}, _} : Word(1n+p)}
      {==}

def not_twice(n: Nat, x: Word(n)) -> {Word.not(n, Word.not(n, x)) == x : Word(n)}:
  match n:
    case 0n:
      match x:
        case WNil{}:
          {==}
    case 1n+p:
      match x:
        case WCon{False{}, t}:
          %not_twice(p, t) : {WCon{False{}, Word.not(p, Word.not(p, t))} == WCon{False{}, _} : Word(1n+p)}
          {==}
        case WCon{True{}, t}:
          %not_twice(p, t) : {WCon{True{}, Word.not(p, Word.not(p, t))} == WCon{True{}, _} : Word(1n+p)}
          {==}

def xor_twice(n: Nat, x: Word(n), y: Word(n)) -> {Word.xor(n, Word.xor(n, x, y), y) == x : Word(n)}:
  match n:
    case 0n:
      match x:
        case WNil{}:
          {==}
    case 1n+p:
      match x y:
        case WCon{False{}, t} WCon{False{}, m}:
          %xor_twice(p, t, m) : {WCon{False{}, Word.xor(p, Word.xor(p, t, m), m)} == WCon{False{}, _} : Word(1n+p)}
          {==}
        case WCon{False{}, t} WCon{True{}, m}:
          %xor_twice(p, t, m) : {WCon{False{}, Word.xor(p, Word.xor(p, t, m), m)} == WCon{False{}, _} : Word(1n+p)}
          {==}
        case WCon{True{}, t} WCon{False{}, m}:
          %xor_twice(p, t, m) : {WCon{True{}, Word.xor(p, Word.xor(p, t, m), m)} == WCon{True{}, _} : Word(1n+p)}
          {==}
        case WCon{True{}, t} WCon{True{}, m}:
          %xor_twice(p, t, m) : {WCon{True{}, Word.xor(p, Word.xor(p, t, m), m)} == WCon{True{}, _} : Word(1n+p)}
          {==}
