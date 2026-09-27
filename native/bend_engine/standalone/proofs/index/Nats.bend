# Small structural Nat facts; no foreign arithmetic or bounded machine powers.
import Base
import ./Bits.bend as Bits
import ../u64/Words.bend as Words

def add_zero(a: Nat) -> {Nat.add(a, 0n) == a : Nat}:
  match a:
    case 0n: {==}
    case 1n+p:
      %add_zero(p) : {1n+Nat.add(p, 0n) == 1n+_ : Nat}
      {==}

def add_succ(+a: Nat, +b: Nat) -> {Nat.add(a, 1n+b) == 1n+Nat.add(a, b) : Nat}:
  match a:
    case 0n: {==}
    case 1n+p:
      %add_succ(p, b) : {1n+Nat.add(p, 1n+b) == 1n+_ : Nat}
      {==}

def double_add(+a: Nat) -> {Nat.double(a) == Nat.add(a, a) : Nat}:
  match a:
    case 0n: {==}
    case 1n+p:
      %Equal.sym(Nat, Nat.add(p, 1n+p), 1n+Nat.add(p, p), add_succ(p, p)) : {2n+Nat.double(p) == 1n+_ : Nat}
      %double_add(p) : {2n+Nat.double(p) == 2n+_ : Nat}
      {==}

def positive_double(a: Nat, e: {Nat.is_lt(0n, a) == True{} : Bool}) ->
  {Nat.is_lt(0n, Nat.double(a)) == True{} : Bool}:
  match a:
    case 0n: Words.false_true({Nat.is_lt(0n, 0n) == True{} : Bool}, e)
    case 1n+p: {==}

def power_positive(+k: Nat) -> {Nat.is_lt(0n, Bits.power(k)) == True{} : Bool}:
  match k:
    case 0n: {==}
    case 1n+p:
      positive_double(Bits.power(p), power_positive(p))

# The independently recursive specification is exactly mathematical 2^k.
def power_nat(+k: Nat) -> {Bits.power(k) == Nat.pow(2n, k) : Nat}:
  match k:
    case 0n: {==}
    case 1n+p:
      %Equal.sym(Nat, Nat.add(Nat.pow(2n, p), 0n), Nat.pow(2n, p), add_zero(Nat.pow(2n, p))) : {Nat.double(Bits.power(p)) == Nat.add(Nat.pow(2n, p), _) : Nat}
      %double_add(Nat.pow(2n, p)) : {Nat.double(Bits.power(p)) == _ : Nat}
      %power_nat(p) : {Nat.double(Bits.power(p)) == Nat.double(_) : Nat}
      {==}

# For either low bit b, 2*a+b < 2*limit iff a < limit.
def double_lt(+a: Nat, +limit: Nat, e: {Nat.is_lt(a, limit) == True{} : Bool}) ->
  {Nat.is_lt(Nat.double(a), Nat.double(limit)) == True{} : Bool}:
  match a limit:
    case 0n 0n: Words.false_true({False{} == True{} : Bool}, e)
    case 0n 1n+p: {==}
    case 1n+p 0n: Words.false_true({False{} == True{} : Bool}, e)
    case 1n+p 1n+q: double_lt(p, q, e)

def odd_lt(+a: Nat, +limit: Nat, e: {Nat.is_lt(a, limit) == True{} : Bool}) ->
  {Nat.is_lt(1n+Nat.double(a), Nat.double(limit)) == True{} : Bool}:
  match a limit:
    case 0n 0n: Words.false_true({False{} == True{} : Bool}, e)
    case 0n 1n+p: {==}
    case 1n+p 0n: Words.false_true({False{} == True{} : Bool}, e)
    case 1n+p 1n+q: odd_lt(p, q, e)

def half_even(+a: Nat, +limit: Nat,
  e: {Nat.is_lt(Nat.double(a), Nat.double(limit)) == True{} : Bool}) ->
  {Nat.is_lt(a, limit) == True{} : Bool}:
  match a limit:
    case 0n 0n: e
    case 0n 1n+p: {==}
    case 1n+p 0n: e
    case 1n+p 1n+q: half_even(p, q, e)

def half_odd(+a: Nat, +limit: Nat,
  e: {Nat.is_lt(1n+Nat.double(a), Nat.double(limit)) == True{} : Bool}) ->
  {Nat.is_lt(a, limit) == True{} : Bool}:
  match a limit:
    case 0n 0n: e
    case 0n 1n+p: {==}
    case 1n+p 0n: e
    case 1n+p 1n+q: half_odd(p, q, e)

def double_lt_one(a: Nat,
  e: {Nat.is_lt(Nat.double(a), 1n) == True{} : Bool}) ->
  {Nat.is_lt(a, 1n) == True{} : Bool}:
  match a:
    case 0n: {==}
    case 1n+p: Words.false_true({Nat.is_lt(1n+p, 1n) == True{} : Bool}, e)

def odd_not_one(a: Nat,
  e: {Nat.is_lt(1n+Nat.double(a), 1n) == True{} : Bool}) -> Empty:
  match a:
    case 0n: Words.false_true(Empty, e)
    case 1n+p: Words.false_true(Empty, e)
