# Arithmetic for bounded incrementing fill schedules. No machine-width Nat power.
import Base
import ../u64/Words.bend as Old
import ../prefix/Nats.bend as Nats
import ../relative/Order.bend as Order

def room(n: Nat,w: Word(n)) -> Bool:
  match n:
    case 0n: False{}
    case 1n+p:
      match w:
        case WCon{False{},t}: True{}
        case WCon{True{},t}: room(p,t)

def saturated(+n: Nat,+w: Word(n),e: {room(n,w) == False{} : Bool}) ->
  {w == Word.not(n,Word.zero(n)) : Word(n)}:
  match n:
    case 0n:
      match w:
        case WNil{}: {==}
    case 1n+p:
      match w:
        case WCon{False{},t}:
          Old.false_true({WCon{False{},t} == Word.not(1n+p,Word.zero(1n+p)) : Word(1n+p)},Equal.sym(Bool,True{},False{},e))
        case WCon{True{},t}:
          Equal.cong(Word(p),Word(1n+p),x => WCon{True{},x},t,Word.not(p,Word.zero(p)),saturated(p,t,e))

def increment(+n: Nat,+w: Word(n),e: {room(n,w) == True{} : Bool}) ->
  {Word.to_nat(n,Word.inc(n,w)) == 1n+Word.to_nat(n,w) : Nat}:
  match n:
    case 0n: Old.false_true({0n == 1n : Nat},e)
    case 1n+p:
      match w:
        case WCon{False{},t}: {==}
        case WCon{True{},t}:
          Equal.cong(Nat,Nat,x => Nat.double(x),Word.to_nat(p,Word.inc(p,t)),1n+Word.to_nat(p,t),increment(p,t,e))

def bounded_choice(+w: Word(32n),+b: Bool,same: {room(32n,w) == b : Bool},
  e: {U32.is_lt(U32{w},131072) == True{} : Bool}) -> {b == True{} : Bool}:
  match b:
    case True{}: {==}
    case False{}:
      +eq = saturated(32n,w,same)
      Old.false_true({False{} == True{} : Bool},Equal.trans(Bool,
        U32.is_lt(4294967295,131072),U32.is_lt(U32{w},131072),True{},
        Equal.cong(Word(32n),Bool,x => U32.is_lt(U32{x},131072),Word.not(32n,Word.zero(32n)),w,
          Equal.sym(Word(32n),w,Word.not(32n,Word.zero(32n)),eq)),e))

def bounded_room(+w: Word(32n),e: {U32.is_lt(U32{w},131072) == True{} : Bool}) ->
  {room(32n,w) == True{} : Bool}:
  bounded_choice(w,room(32n,w),{==},e)

def inc_value(+a: U32,e: {U32.is_lt(a,131072) == True{} : Bool}) ->
  {U32.to_nat(U32.inc(a)) == 1n+U32.to_nat(a) : Nat}:
  match a:
    case U32{w}: increment(32n,w,bounded_room(w,e))

def lt_observed(+a: U32,+b: U32) ->
  {U32.is_lt(a,b) == Nat.is_lt(U32.to_nat(a),U32.to_nat(b)) : Bool}:
  match a b:
    case U32{x} U32{y}:
      Equal.cong(Cmp,Bool,c => Cmp.is_lt(c),Word.cmp(32n,x,y),Nat.cmp(Word.to_nat(32n,x),Word.to_nat(32n,y)),Order.observed(32n,x,y))
