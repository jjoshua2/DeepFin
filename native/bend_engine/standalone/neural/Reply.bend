# Model outputs are LOGITS. Gather only exact legal compact slots, then softmax.
# This module is pure Bend; no native chess/softmax fallback or auxiliary head.
import Base
import ../../session_probe/Search.bend as Search
import ../../legal_probe/Chess.bend as Chess
import ../Policy.bend as Policy

def finite(x: F32) -> Bool:
  Bool.not(U32.is_eq(U32.and(F32.bits(x), 2139095040), 2139095040))

def all_finite(xs: List<&2, F32>) -> Bool:
  match xs:
    case Nil{}: True{}
    case Con{x, rest}: Bool.and(finite(x), all_finite(rest))

def failed(ticket: Search.Ticket) -> Search.Reply:
  Search.Ticket{epoch, request, node, b, ms, needs} = ticket
  Search.Reply{epoch, request, node, 1, 0.0, 0.0, 0.0, 0, Array.new(F32, 8n, 0.0)}

def store(xs: List<&2, F32>, +i: U32, a: Array<F32>) -> Array<F32>:
  match xs:
    case Nil{}: a
    case Con{x, rest}: store(rest, U32.inc(i), Array.set(F32, a, i, x))

# Ordered equality is required: priors belong to the ticket's original move order.
def ordered(ms: List<&2, Chess.Ply>, es: List<&2, Policy.Entry>) -> Bool:
  match ms:
    case Nil{}:
      match es:
        case Nil{}: True{}
        case _: False{}
    case Con{m, mt}:
      match es:
        case Nil{}: False{}
        case Con{Policy.Entry{em, full, compact}, et}:
          Bool.and(U32.is_eq(Search.key(m), Search.key(em)),
            Bool.and(U32.is_lt(full, 4672), Bool.and(U32.is_lt(compact, 1858), ordered(mt, et))))

type Seen is Type:
  Seen{values: Array<U32>, valid: Bool}

def seen_read(slot: U32, valid: Bool, r: Array<U32> & U32) -> Seen:
  (seen, old) = r
  Seen{Array.set(U32, seen, slot, 1), Bool.and(valid, U32.is_zero(old))}

def unique(es: List<&2, Policy.Entry>, state: Seen) -> Bool:
  match es:
    case Nil{}:
      Seen{seen, valid} = state
      valid
    case Con{Policy.Entry{m, full, +slot}, rest}:
      Seen{seen, valid} = state
      unique(rest, seen_read(slot, valid, Array.get(U32, seen, slot)))

def picked(i: U32, out: Array<F32>, r: Array<F32> & F32) -> Array<F32> & Array<F32>:
  (raw, x) = r
  (raw, Array.set(F32, out, i, x))

def gather(es: List<&2, Policy.Entry>, +i: U32, r: Array<F32> & Array<F32>) -> Array<F32> & Array<F32>:
  match es:
    case Nil{}: r
    case Con{Policy.Entry{m, full, slot}, rest}:
      (raw, out) = r
      gather(rest, U32.inc(i), picked(i, out, Array.get(F32, raw, slot)))

def max_read(acc: F32, r: Array<F32> & F32) -> Array<F32> & F32:
  (a, x) = r
  (a, F32.max(acc, x))

def maximum(fuel: Nat, +i: U32, r: Array<F32> & F32) -> Array<F32> & F32:
  match fuel:
    case 0n: r
    case 1n+p:
      (a, acc) = r
      maximum(p, U32.inc(i), max_read(acc, Array.get(F32, a, i)))

def exp_read(+i: U32, m: F32, sum: F32, r: Array<F32> & F32) -> Array<F32> & F32:
  (a, x) = r
  +v = F32.exp(F32.sub(x, m))
  (Array.set(F32, a, i, v), F32.add(sum, v))

def exponents(fuel: Nat, +i: U32, +m: F32, r: Array<F32> & F32) -> Array<F32> & F32:
  match fuel:
    case 0n: r
    case 1n+p:
      (a, sum) = r
      exponents(p, U32.inc(i), m, exp_read(i, m, sum, Array.get(F32, a, i)))

def div_read(i: U32, sum: F32, r: Array<F32> & F32) -> Array<F32>:
  (a, x) = r
  Array.set(F32, a, i, F32.div(x, sum))

def normalize(fuel: Nat, +i: U32, +sum: F32, a: Array<F32>) -> Array<F32>:
  match fuel:
    case 0n: a
    case 1n+p: normalize(p, U32.inc(i), sum, div_read(i, sum, Array.get(F32, a, i)))

def divided(n: U32, r: Array<F32> & F32) -> Array<F32>:
  (a, sum) = r
  normalize(U32.to_nat(n), 0, sum, a)

def exponentiated(+n: U32, r: Array<F32> & F32) -> Array<F32>:
  (a, max) = r
  divided(n, exponents(U32.to_nat(n), 0, max, (a, 0.0)))

def softmax(+n: U32, a: Array<F32>) -> Array<F32>:
  exponentiated(n, maximum(U32.to_nat(n), 0, Array.get(F32, a, 0)))

def wdl_end(ticket: Search.Ticket, count: U32, priors: Array<F32>, win: F32, draw: F32,
  r: Array<F32> & F32) -> Search.Reply:
  Search.Ticket{epoch, request, node, b, ms, needs} = ticket
  (wdl, loss) = r
  Search.Reply{epoch, request, node, 0, win, draw, loss, count, priors}

def wdl_draw(ticket: Search.Ticket, count: U32, priors: Array<F32>, win: F32,
  r: Array<F32> & F32) -> Search.Reply:
  (wdl, draw) = r
  wdl_end(ticket, count, priors, win, draw, Array.get(F32, wdl, 2))

def wdl_win(ticket: Search.Ticket, count: U32, priors: Array<F32>, r: Array<F32> & F32) -> Search.Reply:
  (wdl, win) = r
  wdl_draw(ticket, count, priors, win, Array.get(F32, wdl, 1))

def copy_wdl(fuel: Nat, +i: U32, r: Array<F32> & Array<F32>) -> Array<F32> & Array<F32>:
  match fuel:
    case 0n: r
    case 1n+p:
      (raw, wdl) = r
      copy_wdl(p, U32.inc(i), picked(i, wdl, Array.get(F32, raw, U32.add(1858, i))))

def with_wdl(ticket: Search.Ticket, +count: U32, priors: Array<F32>, r: Array<F32> & Array<F32>) -> Search.Reply:
  (raw, wdl) = r
  wdl_win(ticket, count, softmax(count, priors), Array.get(F32, softmax(3, wdl), 0))

def outputs(ticket: Search.Ticket, count: U32, r: Array<F32> & Array<F32>) -> Search.Reply:
  (raw, priors) = r
  with_wdl(ticket, count, priors, copy_wdl(3n, 0, (raw, Array.new(F32, 2n, 0.0))))

def checked(valid: Bool, ticket: Search.Ticket, es: List<&2, Policy.Entry>, count: U32, xs: List<&2, F32>) -> Search.Reply:
  match valid:
    case False{}: failed(ticket)
    case True{}: outputs(ticket, count, gather(es, 0, (store(xs, 0, Array.new(F32, 11n, 0.0)), Array.new(F32, 8n, 0.0))))

def ordered_checked(valid: Bool, ticket: Search.Ticket, +es: List<&2, Policy.Entry>, count: U32, xs: List<&2, F32>) -> Search.Reply:
  match valid:
    case False{}: failed(ticket)
    case True{}: checked(unique(es, Seen{Array.new(U32, 11n, 0), True{}}), ticket, es, count, xs)

def convert(+ticket: Search.Ticket, +es: List<&2, Policy.Entry>, +xs: List<&2, F32>) -> Search.Reply:
  Search.Ticket{epoch, request, node, board, ms, needs} = ticket
  +n = U32.from_nat(List.length(&2, Policy.Entry, es))
  ordered_checked(Bool.and(needs, Bool.and(U32.is_gt(n, 0), Bool.and(U32.is_le(n, 256),
    Bool.and(ordered(ms, es), Bool.and(Nat.is_eq(List.length(&2, F32, xs), 1861n), all_finite(xs)))))), ticket, es, n, xs)
