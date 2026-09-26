# Mathematical observation injectivity. No runtime representation changes.
import Base
import ../u64/Words.bend as Old
import ../index/Bits.bend as Bits

def half(n: Nat) -> Nat:
  match n:
    case 0n: 0n
    case 1n: 0n
    case 2n+p: 1n+half(p)

def parity(n: Nat) -> Bool:
  match n:
    case 0n: False{}
    case 1n: True{}
    case 2n+p: parity(p)

def half_even(+n: Nat) -> {half(Nat.double(n)) == n : Nat}:
  match n:
    case 0n: {==}
    case 1n+p:
      %half_even(p) : {1n+half(Nat.double(p)) == 1n+_ : Nat}
      {==}

def half_odd(+n: Nat) -> {half(1n+Nat.double(n)) == n : Nat}:
  match n:
    case 0n: {==}
    case 1n+p:
      %half_odd(p) : {1n+half(1n+Nat.double(p)) == 1n+_ : Nat}
      {==}

def parity_even(n: Nat) -> {parity(Nat.double(n)) == False{} : Bool}:
  match n:
    case 0n: {==}
    case 1n+p: parity_even(p)

def parity_odd(n: Nat) -> {parity(1n+Nat.double(n)) == True{} : Bool}:
  match n:
    case 0n: {==}
    case 1n+p: parity_odd(p)

def half_head(b: Bool, +n: Nat) ->
  {half(Bool.pick(Nat,b,1n+Nat.double(n),Nat.double(n))) == n : Nat}:
  match b:
    case False{}: half_even(n)
    case True{}: half_odd(n)

def parity_head(b: Bool, n: Nat) ->
  {parity(Bool.pick(Nat,b,1n+Nat.double(n),Nat.double(n))) == b : Bool}:
  match b:
    case False{}: parity_even(n)
    case True{}: parity_odd(n)

def head_value(b: Bool, +n: Nat, +w: Word(n)) ->
  {Word.to_nat(1n+n,WCon{b,w}) == Bool.pick(Nat,b,1n+Nat.double(Word.to_nat(n,w)),Nat.double(Word.to_nat(n,w))) : Nat}:
  match b:
    case False{}: {==}
    case True{}: {==}

def tail_value(+b: Bool, +n: Nat, +w: Word(n)) ->
  {half(Word.to_nat(1n+n,WCon{b,w})) == Word.to_nat(n,w) : Nat}:
  %Equal.sym(Nat,Word.to_nat(1n+n,WCon{b,w}),Bool.pick(Nat,b,1n+Nat.double(Word.to_nat(n,w)),Nat.double(Word.to_nat(n,w))),head_value(b,n,w)) :
    {half(_) == Word.to_nat(n,w) : Nat}
  half_head(b,Word.to_nat(n,w))

def bit_value(+b: Bool, +n: Nat, +w: Word(n)) ->
  {parity(Word.to_nat(1n+n,WCon{b,w})) == b : Bool}:
  %Equal.sym(Nat,Word.to_nat(1n+n,WCon{b,w}),Bool.pick(Nat,b,1n+Nat.double(Word.to_nat(n,w)),Nat.double(Word.to_nat(n,w))),head_value(b,n,w)) :
    {parity(_) == b : Bool}
  parity_head(b,Word.to_nat(n,w))

def tail_equal(+n: Nat, +ab: Bool, +at: Word(n), +bb: Bool, +bt: Word(n),
  e: {Word.to_nat(1n+n,WCon{ab,at}) == Word.to_nat(1n+n,WCon{bb,bt}) : Nat}) ->
  {Word.to_nat(n,at) == Word.to_nat(n,bt) : Nat}:
  %tail_value(ab,n,at) : {_ == Word.to_nat(n,bt) : Nat}
  %tail_value(bb,n,bt) : {half(Word.to_nat(1n+n,WCon{ab,at})) == _ : Nat}
  Equal.cong(Nat,Nat,w => half(w),Word.to_nat(1n+n,WCon{ab,at}),Word.to_nat(1n+n,WCon{bb,bt}),e)

def head_equal(+n: Nat, +ab: Bool, +at: Word(n), +bb: Bool, +bt: Word(n),
  e: {Word.to_nat(1n+n,WCon{ab,at}) == Word.to_nat(1n+n,WCon{bb,bt}) : Nat}) -> {ab == bb : Bool}:
  %bit_value(ab,n,at) : {_ == bb : Bool}
  %bit_value(bb,n,bt) : {parity(Word.to_nat(1n+n,WCon{ab,at})) == _ : Bool}
  Equal.cong(Nat,Bool,w => parity(w),Word.to_nat(1n+n,WCon{ab,at}),Word.to_nat(1n+n,WCon{bb,bt}),e)

def injective(+n: Nat, +a: Word(n), +b: Word(n), +e: {Word.to_nat(n,a) == Word.to_nat(n,b) : Nat}) ->
  {a == b : Word(n)}:
  match n:
    case 0n:
      match a b:
        case WNil{} WNil{}: {==}
    case 1n+p:
      match a b:
        case WCon{ab,at} WCon{bb,bt}:
          Equal.trans(Word(1n+p),WCon{ab,at},WCon{bb,at},WCon{bb,bt},
            Equal.cong(Bool,Word(1n+p),h => WCon{h,at},ab,bb,head_equal(p,ab,at,bb,bt,e)),
            Equal.cong(Word(p),Word(1n+p),w => WCon{bb,w},at,bt,injective(p,at,bt,tail_equal(p,ab,at,bb,bt,e))))

def u64_injective(+a: U64, +b: U64, e: {Bits.value(a) == Bits.value(b) : Nat}) -> {a == b : U64}:
  %Old.from_to(a) : {_ == b : U64}
  %Old.from_to(b) : {U64.from_word(U64.to_word(a)) == _ : U64}
  Equal.cong(Word(64n),U64,w => U64.from_word(w),U64.to_word(a),U64.to_word(b),injective(64n,U64.to_word(a),U64.to_word(b),e))
