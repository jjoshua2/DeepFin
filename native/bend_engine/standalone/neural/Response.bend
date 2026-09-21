# Model logits -> exact legal-order search reply, in Bend. No rules are inferred
# from network confidence. Both output heads and all compact slots must be finite.
import Base
import ../../session_probe/Search.bend as Search
import ../../legal_probe/Chess.bend as Chess
import ../Policy.bend as Policy

def float_bits(word: U32) -> F32:
  U32{bits} = word
  F32{bits}

def finite(word: U32) -> Bool:
  Bool.not(U32.is_eq(U32.and(word, 2139095040), 2139095040))

type Decoded is Type:
  Decoded{values: Array<F32>, count: U32, valid: Bool}

def decode(fuel: Nat, words: List<&2, U32>, r: Decoded) -> Decoded:
  match fuel:
    case 0n:
      match words:
        case Nil{}: r
        case _:
          Decoded{a, n, ok} = r
          Decoded{a, n, False{}}
    case 1n+p:
      match words:
        case Nil{}:
          Decoded{a, n, ok} = r
          Decoded{a, n, False{}}
        case Con{+word, rest}:
          Decoded{a, +n, ok} = r
          decode(p, rest, Decoded{Array.set(F32, a, n, float_bits(word)), U32.inc(n), Bool.and(ok, finite(word))})

# Full move identity and order must agree with the selected native ticket.
def matching(ms: List<&2, Chess.Ply>, es: List<&2, Policy.Entry>) -> Bool:
  match ms es:
    case Nil{} Nil{}: True{}
    case Con{m, mt} Con{Policy.Entry{em, full, compact}, et}:
      Bool.and(U32.is_eq(Search.key(m), Search.key(em)),
        Bool.and(U32.is_lt(full, 4672), Bool.and(U32.is_lt(compact, 1858), matching(mt, et))))
    case _ _: False{}

type Gathered is Type:
  Gathered{raw: Array<F32>, legal: Array<F32>, max: F32}

def gathered(i: U32, legal: Array<F32>, maximum: F32, r: Array<F32> & F32) -> Gathered:
  (raw, +value) = r
  Gathered{raw, Array.set(F32, legal, i, value), F32.max(maximum, value)}

def gather(es: List<&2, Policy.Entry>, +i: U32, g: Gathered) -> Gathered:
  match es:
    case Nil{}: g
    case Con{Policy.Entry{move, full, compact}, rest}:
      Gathered{raw, legal, maximum} = g
      gather(rest, U32.inc(i), gathered(i, legal, maximum, Array.get(F32, raw, compact)))

def exp_one(i: U32, maximum: F32, sum: F32, r: Array<F32> & F32) -> Array<F32> & F32:
  (a, value) = r
  +weight = F32.exp(F32.sub(value, maximum))
  (Array.set(F32, a, i, weight), F32.add(sum, weight))

def exps(fuel: Nat, +i: U32, +maximum: F32, r: Array<F32> & F32) -> Array<F32> & F32:
  match fuel:
    case 0n: r
    case 1n+p:
      (a, sum) = r
      exps(p, U32.inc(i), maximum, exp_one(i, maximum, sum, Array.get(F32, a, i)))

def div_one(i: U32, sum: F32, r: Array<F32> & F32) -> Array<F32>:
  (a, value) = r
  Array.set(F32, a, i, F32.div(value, sum))

def divide(fuel: Nat, +i: U32, +sum: F32, a: Array<F32>) -> Array<F32>:
  match fuel:
    case 0n: a
    case 1n+p: divide(p, U32.inc(i), sum, div_one(i, sum, Array.get(F32, a, i)))

def normalized(count: U32, r: Array<F32> & F32) -> Array<F32>:
  (a, sum) = r
  divide(U32.to_nat(count), 0, sum, a)

type Wdl is Data:
  Wdl{win: F32, draw: F32, loss: F32}

def wdl_div(+w: F32, +d: F32, +l: F32) -> Wdl:
  +total = F32.add(F32.add(w, d), l)
  Wdl{F32.div(w, total), F32.div(d, total), F32.div(l, total)}

def wdl_exp(+w: F32, +d: F32, +l: F32) -> Wdl:
  +m = F32.max(w, F32.max(d, l))
  wdl_div(F32.exp(F32.sub(w, m)), F32.exp(F32.sub(d, m)), F32.exp(F32.sub(l, m)))

def read_loss(w: F32, d: F32, r: Array<F32> & F32) -> Wdl:
  (raw, loss) = r
  wdl_exp(w, d, loss)

def read_draw(w: F32, r: Array<F32> & F32) -> Wdl:
  (raw, d) = r
  read_loss(w, d, Array.get(F32, raw, 1860))

def read_win(r: Array<F32> & F32) -> Wdl:
  (raw, w) = r
  read_draw(w, Array.get(F32, raw, 1859))

type Conversion is Type:
  Rejected{}
  Accepted{reply: Search.Reply}

def finish(ticket: Search.Ticket, count: U32, priors: Array<F32>, wdl: Wdl) -> Conversion:
  Search.Ticket{epoch, request, node, board, moves, needed} = ticket
  Wdl{win, draw, loss} = wdl
  Accepted{Search.Reply{epoch, request, node, 0, win, draw, loss, count, priors}}

def output(ticket: Search.Ticket, +count: U32, g: Gathered) -> Conversion:
  Gathered{raw, legal, maximum} = g
  finish(ticket, count, normalized(count, exps(U32.to_nat(count), 0, maximum, (legal, 0.0))),
    read_win(Array.get(F32, raw, 1858)))

def valid_outputs(ok: Bool, ticket: Search.Ticket, +entries: List<&2, Policy.Entry>, r: Decoded) -> Conversion:
  match ok:
    case False{}: Rejected{}
    case True{}:
      Decoded{raw, count, valid} = r
      output(ticket, U32.from_nat(List.length(&2, Policy.Entry, entries)),
        gather(entries, 0, Gathered{raw, Array.new(F32, 8n, 0.0), float_bits(4286578687)}))

def decoded(+ticket: Search.Ticket, +entries: List<&2, Policy.Entry>, r: Decoded) -> Conversion:
  Search.Ticket{epoch, request, node, board, moves, +needed} = ticket
  Decoded{raw, +count, +valid} = r
  +n = U32.from_nat(List.length(&2, Policy.Entry, entries))
  ok = Bool.and(valid, Bool.and(U32.is_eq(count, 1861), Bool.and(needed,
    Bool.and(U32.is_gt(n, 0), Bool.and(U32.is_le(n, 256), matching(moves, entries))))))
  valid_outputs(ok, ticket, entries, Decoded{raw, count, valid})

def convert(ticket: Search.Ticket, entries: List<&2, Policy.Entry>, words: List<&2, U32>) -> Conversion:
  decoded(ticket, entries, decode(1861n, words, Decoded{Array.new(F32, 11n, 0.0), 0, True{}}))
