# A bounded singleton away from home disables both actual wing guards.
import Base
import ../../../legal_probe/Chess.bend as Chess
import ../castle_kings/Spec.bend as K
import ../castle_emission/Spec.bend as E
import ../u64/Words.bend as W

def outside(-A: Type,+p: Nat,e: {Nat.is_lt(64n+p,64n) == True{} : Bool}) -> A:
  match p:
    case 0n: W.false_true(A,e)
    case 1n+q: W.false_true(A,e)

def bit_black(+sq: Nat,
  bound: {Nat.is_lt(sq,64n) == True{} : Bool},
  away: {Nat.is_eq(sq,60n) == False{} : Bool}) ->
  {U64.test_bit(U64.bit(sq),60n) == False{} : Bool}:
  match sq:
    case 0n: {==}
    case 1n: {==}
    case 2n: {==}
    case 3n: {==}
    case 4n: {==}
    case 5n: {==}
    case 6n: {==}
    case 7n: {==}
    case 8n: {==}
    case 9n: {==}
    case 10n: {==}
    case 11n: {==}
    case 12n: {==}
    case 13n: {==}
    case 14n: {==}
    case 15n: {==}
    case 16n: {==}
    case 17n: {==}
    case 18n: {==}
    case 19n: {==}
    case 20n: {==}
    case 21n: {==}
    case 22n: {==}
    case 23n: {==}
    case 24n: {==}
    case 25n: {==}
    case 26n: {==}
    case 27n: {==}
    case 28n: {==}
    case 29n: {==}
    case 30n: {==}
    case 31n: {==}
    case 32n: {==}
    case 33n: {==}
    case 34n: {==}
    case 35n: {==}
    case 36n: {==}
    case 37n: {==}
    case 38n: {==}
    case 39n: {==}
    case 40n: {==}
    case 41n: {==}
    case 42n: {==}
    case 43n: {==}
    case 44n: {==}
    case 45n: {==}
    case 46n: {==}
    case 47n: {==}
    case 48n: {==}
    case 49n: {==}
    case 50n: {==}
    case 51n: {==}
    case 52n: {==}
    case 53n: {==}
    case 54n: {==}
    case 55n: {==}
    case 56n: {==}
    case 57n: {==}
    case 58n: {==}
    case 59n: {==}
    case 60n: W.false_true({True{} == False{} : Bool},Equal.sym(Bool,True{},False{},away))
    case 61n: {==}
    case 62n: {==}
    case 63n: {==}
    case 64n+p: outside({U64.test_bit(U64.bit(64n+p),60n) == False{} : Bool},p,bound)

def bit_white(+sq: Nat,
  bound: {Nat.is_lt(sq,64n) == True{} : Bool},
  away: {Nat.is_eq(sq,4n) == False{} : Bool}) ->
  {U64.test_bit(U64.bit(sq),4n) == False{} : Bool}:
  match sq:
    case 0n: {==}
    case 1n: {==}
    case 2n: {==}
    case 3n: {==}
    case 4n: W.false_true({True{} == False{} : Bool},Equal.sym(Bool,True{},False{},away))
    case 5n: {==}
    case 6n: {==}
    case 7n: {==}
    case 8n: {==}
    case 9n: {==}
    case 10n: {==}
    case 11n: {==}
    case 12n: {==}
    case 13n: {==}
    case 14n: {==}
    case 15n: {==}
    case 16n: {==}
    case 17n: {==}
    case 18n: {==}
    case 19n: {==}
    case 20n: {==}
    case 21n: {==}
    case 22n: {==}
    case 23n: {==}
    case 24n: {==}
    case 25n: {==}
    case 26n: {==}
    case 27n: {==}
    case 28n: {==}
    case 29n: {==}
    case 30n: {==}
    case 31n: {==}
    case 32n: {==}
    case 33n: {==}
    case 34n: {==}
    case 35n: {==}
    case 36n: {==}
    case 37n: {==}
    case 38n: {==}
    case 39n: {==}
    case 40n: {==}
    case 41n: {==}
    case 42n: {==}
    case 43n: {==}
    case 44n: {==}
    case 45n: {==}
    case 46n: {==}
    case 47n: {==}
    case 48n: {==}
    case 49n: {==}
    case 50n: {==}
    case 51n: {==}
    case 52n: {==}
    case 53n: {==}
    case 54n: {==}
    case 55n: {==}
    case 56n: {==}
    case 57n: {==}
    case 58n: {==}
    case 59n: {==}
    case 60n: {==}
    case 61n: {==}
    case 62n: {==}
    case 63n: {==}
    case 64n+p: outside({U64.test_bit(U64.bit(64n+p),4n) == False{} : Bool},p,bound)


def bit_away(+sq: Nat,+white: Bool,
  bound: {Nat.is_lt(sq,64n) == True{} : Bool},
  away: {Nat.is_eq(sq,U32.to_nat(K.src(white))) == False{} : Bool}) ->
  {U64.test_bit(U64.bit(sq),U32.to_nat(K.src(white))) == False{} : Bool}:
  match white:
    case False{}: bit_black(sq,bound,away)
    case True{}: bit_white(sq,bound,away)

def white_value(+white: Bool) -> {U32.is_eq(Bool.to_u32(white),1) == white : Bool}:
  match white:
    case False{}: {==}
    case True{}: {==}

def selected_value(+b: Chess.Board,+white: Bool) ->
  {U64.and(Chess.get_kings(b),Chess.color(b,Bool.to_u32(white))) == K.kings(b,white) : U64}:
  match white:
    case False{}: {==}
    case True{}: {==}

def selected(+b: Chess.Board,+white: Bool,
  turn: {Chess.get_turn(b) == Bool.to_u32(white) : U32}) ->
  {U64.and(Chess.get_kings(b),Chess.color(b,Chess.get_turn(b))) == K.kings(b,white) : U64}:
  Equal.trans(U64,U64.and(Chess.get_kings(b),Chess.color(b,Chess.get_turn(b))),
    U64.and(Chess.get_kings(b),Chess.color(b,Bool.to_u32(white))),K.kings(b,white),
    Equal.cong(U32,U64,x => U64.and(Chess.get_kings(b),Chess.color(b,x)),
      Chess.get_turn(b),Bool.to_u32(white),turn),selected_value(b,white))

def source(+b: Chess.Board,+white: Bool,
  turn: {Chess.get_turn(b) == Bool.to_u32(white) : U32}) ->
  {E.src(b) == K.src(white) : U32}:
  Equal.cong(Bool,U32,x => U32.add(Bool.pick(U32,x,0,56),4),
    U32.is_eq(Chess.get_turn(b),1),white,
    Equal.trans(Bool,U32.is_eq(Chess.get_turn(b),1),U32.is_eq(Bool.to_u32(white),1),white,
      Equal.cong(U32,Bool,x => U32.is_eq(x,1),Chess.get_turn(b),Bool.to_u32(white),turn),white_value(white)))

def king_false(+b: Chess.Board,+white: Bool,+sq: Nat,
  +turn: {Chess.get_turn(b) == Bool.to_u32(white) : U32},
  one: {K.kings(b,white) == U64.bit(sq) : U64},
  bound: {Nat.is_lt(sq,64n) == True{} : Bool},
  away: {Nat.is_eq(sq,U32.to_nat(K.src(white))) == False{} : Bool}) ->
  {E.king_ok(b) == False{} : Bool}:
  %Equal.sym(U32,E.src(b),K.src(white),source(b,white,turn)) :
    {U64.test_bit(U64.and(Chess.get_kings(b),Chess.color(b,Chess.get_turn(b))),U32.to_nat(_)) == False{} : Bool}
  %Equal.sym(U64,U64.and(Chess.get_kings(b),Chess.color(b,Chess.get_turn(b))),K.kings(b,white),selected(b,white,turn)) :
    {U64.test_bit(_,U32.to_nat(K.src(white))) == False{} : Bool}
  %Equal.sym(U64,K.kings(b,white),U64.bit(sq),one) :
    {U64.test_bit(_,U32.to_nat(K.src(white))) == False{} : Bool}
  bit_away(sq,white,bound,away)

def blocked(+rights: Bool,+rook: Bool,+clear: Bool) ->
  {Bool.and(rights,Bool.and(False{},Bool.and(rook,clear))) == False{} : Bool}:
  match rights:
    case False{}: {==}
    case True{}: {==}

def guard_false(+b: Chess.Board,+white: Bool,+sq: Nat,+ks: Bool,
  turn: {Chess.get_turn(b) == Bool.to_u32(white) : U32},
  one: {K.kings(b,white) == U64.bit(sq) : U64},
  bound: {Nat.is_lt(sq,64n) == True{} : Bool},
  away: {Nat.is_eq(sq,U32.to_nat(K.src(white))) == False{} : Bool}) ->
  {E.guard(b,ks) == False{} : Bool}:
  %Equal.sym(Bool,E.king_ok(b),False{},king_false(b,white,sq,turn,one,bound,away)) :
    {Bool.and(E.rights_ok(b,ks),Bool.and(_,Bool.and(E.rook_ok(b,ks),E.path_clear(b,ks)))) == False{} : Bool}
  blocked(E.rights_ok(b,ks),E.rook_ok(b,ks),E.path_clear(b,ks))
