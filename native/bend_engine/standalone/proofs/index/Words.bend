# Induction over arbitrary word widths, using actual Base PEXT/PDEP definitions.
import Base
import ./Bits.bend as Bits
import ./Nats.bend as Nats
import ../u64/Words.bend as Old

def pad_value(+n: Nat, +w: Word(n)) ->
  {Word.to_nat(1n+n, Word.shr.pad(n, w)) == Word.to_nat(n, w) : Nat}:
  match n:
    case 0n:
      match w:
        case WNil{}: {==}
    case 1n+p:
      match w:
        case WCon{False{}, t}:
          %pad_value(p, t) : {Nat.double(Word.to_nat(1n+p, Word.shr.pad(p, t))) == Nat.double(_) : Nat}
          {==}
        case WCon{True{}, t}:
          %pad_value(p, t) : {1n+Nat.double(Word.to_nat(1n+p, Word.shr.pad(p, t))) == 1n+Nat.double(_) : Nat}
          {==}

def zero_small(+n: Nat, +w: Word(n), e: {Nat.is_lt(Word.to_nat(n, w), 1n) == True{} : Bool}) ->
  {w == Word.zero(n) : Word(n)}:
  match n:
    case 0n:
      match w:
        case WNil{}: {==}
    case 1n+p:
      match w:
        case WCon{False{}, t}:
          %zero_small(p, t, Nats.double_lt_one(Word.to_nat(p, t), e)) : {WCon{False{}, t} == WCon{False{}, _} : Word(1n+p)}
          {==}
        case WCon{True{}, t}:
          Empty.absurd({WCon{True{}, t} == Word.zero(1n+p) : Word(1n+p)},
            Nats.odd_not_one(Word.to_nat(p, t), e))

def clip_id(+n: Nat, +k: Nat, +w: Word(n),
  e: {Nat.is_lt(Word.to_nat(n, w), Bits.power(k)) == True{} : Bool}) ->
  {Bits.clip(n, k, w) == w : Word(n)}:
  match n:
    case 0n:
      match w:
        case WNil{}: {==}
    case 1n+p:
      match k:
        case 0n:
          Equal.sym(Word(1n+p), w, Word.zero(1n+p), zero_small(1n+p, w, e))
        case 1n+q:
          match w:
            case WCon{False{}, t}:
              Equal.cong(Word(p), Word(1n+p), x => WCon{False{}, x}, Bits.clip(p, q, t), t,
                clip_id(p, q, t, Nats.half_even(Word.to_nat(p, t), Bits.power(q), e)))
            case WCon{True{}, t}:
              Equal.cong(Word(p), Word(1n+p), x => WCon{True{}, x}, Bits.clip(p, q, t), t,
                clip_id(p, q, t, Nats.half_odd(Word.to_nat(p, t), Bits.power(q), e)))

# Trimming one physical high bit does not affect a compact width at most n.
def clip_pad_trim(+n: Nat, +k: Nat, +w: Word(1n+n),
  e: {Nat.is_le(k, n) == True{} : Bool}) ->
  {Word.shr.pad(n, Bits.clip(n, k, Word.trim(n, w))) == Bits.clip(1n+n, k, w) : Word(1n+n)}:
  match n:
    case 0n:
      match k:
        case 0n: {==}
        case 1n+q:
          Old.false_true({Word.shr.pad(0n, Bits.clip(0n, 1n+q, Word.trim(0n, w))) == Bits.clip(1n, 1n+q, w) : Word(1n)}, e)
    case 1n+p:
      match k:
        case 0n: Old.pad_zero(1n+p)
        case 1n+q:
          match w:
            case WCon{b, t}:
              Equal.cong(Word(1n+p), Word(2n+p), x => WCon{b, x},
                Word.shr.pad(p, Bits.clip(p, q, Word.trim(p, t))), Bits.clip(1n+p, q, t),
                clip_pad_trim(p, q, t, e))

# This total identity describes out-of-range compact inputs too: high bits
# are discarded, not reconstructed. The bounded law below derives equality.
def reverse(+n: Nat, +mask: Word(n), +a: Word(n)) ->
  {Word.pext(n, Word.pdep(n, mask, a), mask) == Bits.clip(n, Word.count(n, mask), a) : Word(n)}:
  match n:
    case 0n: {==}
    case 1n+p:
      match mask:
        case WCon{False{}, m}:
          Equal.trans(Word(1n+p),
            Word.shr.pad(p, Word.pext(p, Word.pdep(p, m, Word.trim(p, a)), m)),
            Word.shr.pad(p, Bits.clip(p, Word.count(p, m), Word.trim(p, a))),
            Bits.clip(1n+p, Word.count(p, m), a),
            Equal.cong(Word(p), Word(1n+p), x => Word.shr.pad(p, x),
              Word.pext(p, Word.pdep(p, m, Word.trim(p, a)), m),
              Bits.clip(p, Word.count(p, m), Word.trim(p, a)), reverse(p, m, Word.trim(p, a))),
            clip_pad_trim(p, Word.count(p, m), a, Old.count_bound(p, m)))
        case WCon{True{}, m}:
          match a:
            case WCon{b, t}:
              Equal.cong(Word(p), Word(1n+p), x => WCon{b, x},
                Word.pext(p, Word.pdep(p, m, t), m), Bits.clip(p, Word.count(p, m), t), reverse(p, m, t))

def reverse_bounded(+n: Nat, +mask: Word(n), +a: Word(n),
  e: {Nat.is_lt(Word.to_nat(n, a), Bits.power(Word.count(n, mask))) == True{} : Bool}) ->
  {Word.pext(n, Word.pdep(n, mask, a), mask) == a : Word(n)}:
  Equal.trans(Word(n), Word.pext(n, Word.pdep(n, mask, a), mask),
    Bits.clip(n, Word.count(n, mask), a), a, reverse(n, mask, a), clip_id(n, Word.count(n, mask), a, e))

def extract_bound(+n: Nat, +a: Word(n), +mask: Word(n)) ->
  {Nat.is_lt(Word.to_nat(n, Word.pext(n, a, mask)), Bits.power(Word.count(n, mask))) == True{} : Bool}:
  match n:
    case 0n: {==}
    case 1n+p:
      match a mask:
        case WCon{b, t} WCon{False{}, m}:
          %Equal.sym(Nat, Word.to_nat(1n+p, Word.shr.pad(p, Word.pext(p, t, m))),
            Word.to_nat(p, Word.pext(p, t, m)), pad_value(p, Word.pext(p, t, m))) :
            {Nat.is_lt(_, Bits.power(Word.count(p, m))) == True{} : Bool}
          extract_bound(p, t, m)
        case WCon{False{}, t} WCon{True{}, m}:
          Nats.double_lt(Word.to_nat(p, Word.pext(p, t, m)), Bits.power(Word.count(p, m)), extract_bound(p, t, m))
        case WCon{True{}, t} WCon{True{}, m}:
          Nats.odd_lt(Word.to_nat(p, Word.pext(p, t, m)), Bits.power(Word.count(p, m)), extract_bound(p, t, m))

def deposit_member(+n: Nat, +mask: Word(n), +a: Word(n)) ->
  {Word.and(n, Word.pdep(n, mask, a), mask) == Word.pdep(n, mask, a) : Word(n)}:
  match n:
    case 0n: {==}
    case 1n+p:
      match mask:
        case WCon{False{}, m}:
          Equal.cong(Word(p), Word(1n+p), x => WCon{False{}, x},
            Word.and(p, Word.pdep(p, m, Word.trim(p, a)), m), Word.pdep(p, m, Word.trim(p, a)),
            deposit_member(p, m, Word.trim(p, a)))
        case WCon{True{}, m}:
          match a:
            case WCon{False{}, t}:
              Equal.cong(Word(p), Word(1n+p), x => WCon{False{}, x},
                Word.and(p, Word.pdep(p, m, t), m), Word.pdep(p, m, t), deposit_member(p, m, t))
            case WCon{True{}, t}:
              Equal.cong(Word(p), Word(1n+p), x => WCon{True{}, x},
                Word.and(p, Word.pdep(p, m, t), m), Word.pdep(p, m, t), deposit_member(p, m, t))
