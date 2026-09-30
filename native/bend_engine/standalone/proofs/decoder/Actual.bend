import Base
import ./Spec.bend as S
import ./Bits.bend as Bits
import ./Rows.bend as Rows
import ./Projection.bend as Projection
import ../board/Spec.bend as B
import ../../../legal_probe/Chess.bend as Chess

def bridge(+b: Chess.Board,+sq: U32,bound: {Nat.is_lt(U32.to_nat(sq),64n) == True{} : Bool}) ->
  {S.decoded(b,sq) == S.local_decode(S.row(b,sq)) : S.Cell}:
  Equal.cong(Bool,S.Cell,flag => Bool.pick(S.Cell,flag,
    S.Occupant{Chess.piece(b,sq),U64.test_bit(Chess.get_white(b),U32.to_nat(sq))},S.Empty{}),
    U64.test_bit(Chess.occupied(b),U32.to_nat(sq)),Bool.or(U64.test_bit(Chess.get_white(b),U32.to_nat(sq)),U64.test_bit(Chess.get_black(b),U32.to_nat(sq))),
    Bits.test64_or(Chess.get_white(b),Chess.get_black(b),U32.to_nat(sq),bound))

def decode(+b: Chess.Board,+sq: U32,good: {B.valid(b) == True{} : Bool},
  +bound: {Nat.is_lt(U32.to_nat(sq),64n) == True{} : Bool}) ->
  {S.decoded(b,sq) == S.square(b,sq) : S.Cell}:
  Equal.trans(S.Cell,S.decoded(b,sq),S.local_decode(S.row(b,sq)),S.square(b,sq),
    bridge(b,sq,bound),Rows.decode(S.row(b,sq),Projection.valid(b,sq,good,bound)))

def roundtrip(+b: Chess.Board,+sq: U32,good: {B.valid(b) == True{} : Bool},
  bound: {Nat.is_lt(U32.to_nat(sq),64n) == True{} : Bool}) ->
  {S.encode(S.square(b,sq)) == S.row(b,sq) : B.Row}:
  Rows.roundtrip(S.row(b,sq),Projection.valid(b,sq,good,bound))

def occupied(+b: Chess.Board,+sq: U32,good: {B.valid(b) == True{} : Bool},
  bound: {Nat.is_lt(U32.to_nat(sq),64n) == True{} : Bool},
  present: {U64.test_bit(Chess.occupied(b),U32.to_nat(sq)) == True{} : Bool}) ->
  {S.square(b,sq) == S.Occupant{Chess.piece(b,sq),U64.test_bit(Chess.get_white(b),U32.to_nat(sq))} : S.Cell}:
  Equal.trans(S.Cell,S.square(b,sq),S.decoded(b,sq),S.Occupant{Chess.piece(b,sq),U64.test_bit(Chess.get_white(b),U32.to_nat(sq))},
    Equal.sym(S.Cell,S.decoded(b,sq),S.square(b,sq),decode(b,sq,good,bound)),
    Equal.cong(Bool,S.Cell,flag => Bool.pick(S.Cell,flag,S.Occupant{Chess.piece(b,sq),U64.test_bit(Chess.get_white(b),U32.to_nat(sq))},S.Empty{}),
      U64.test_bit(Chess.occupied(b),U32.to_nat(sq)),True{},present))
