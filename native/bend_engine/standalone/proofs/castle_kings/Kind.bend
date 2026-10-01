import Base
import ./Spec.bend as S
import ./Rows.bend as Rows
import ../board/Spec.bend as B
import ../decoder/Spec.bend as D
import ../decoder/Bits.bend as Bits
import ../decoder/Projection.bend as P
import ../ray/Frame.bend as F
import ../../../legal_probe/Chess.bend as Chess

def get_left(+n: Nat,+k: Word(n),+c: Word(n),+i: Nat,e: {Bits.get(n,Word.and(n,k,c),i) == True{} : Bool}) ->
  {Bits.get(n,k,i) == True{} : Bool}:
  match n:
    case 0n: e
    case 1n+p:
      match k c i:
        case WCon{+kh,kt} WCon{+ch,ct} 0n: F.left(kh,ch,e)
        case WCon{kh,kt} WCon{ch,ct} 1n+j: get_left(p,kt,ct,j,e)

def choose(+kl: Word(32n),+kh: Word(32n),+cl: Word(32n),+ch: Word(32n),+i: Nat,low: Bool,
  e: {Bool.pick(Bool,low,Bits.get(32n,Word.and(32n,kl,cl),i),Bits.get(32n,Word.and(32n,kh,ch),Nat.sub(i,32n))) == True{} : Bool}) ->
  {Bool.pick(Bool,low,Bits.get(32n,kl,i),Bits.get(32n,kh,Nat.sub(i,32n))) == True{} : Bool}:
  match low:
    case True{}: get_left(32n,kl,cl,i,e)
    case False{}: get_left(32n,kh,ch,Nat.sub(i,32n),e)

def get64_left(+k: U64,+c: U64,+i: Nat,e: {Bits.get64(U64.and(k,c),i) == True{} : Bool}) ->
  {Bits.get64(k,i) == True{} : Bool}:
  match k c:
    case U64{U32{kl},U32{kh}} U64{U32{cl},U32{ch}}:
      choose(kl,kh,cl,ch,i,Nat.is_lt(i,32n),e)

def test_left(+k: U64,+c: U64,+i: Nat,+bound: {Nat.is_lt(i,64n) == True{} : Bool},e: {U64.test_bit(U64.and(k,c),i) == True{} : Bool}) ->
  {U64.test_bit(k,i) == True{} : Bool}:
  Equal.trans(Bool,U64.test_bit(k,i),Bits.get64(k,i),True{},Bits.test64(k,i,bound),
    get64_left(k,c,i,Equal.trans(Bool,Bits.get64(U64.and(k,c),i),U64.test_bit(U64.and(k,c),i),True{},
      Equal.sym(Bool,U64.test_bit(U64.and(k,c),i),Bits.get64(U64.and(k,c),i),Bits.test64(U64.and(k,c),i,bound)),e)))

def source_bound(+white: Bool) -> {Nat.is_lt(U32.to_nat(S.src(white)),64n) == True{} : Bool}:
  match white:
    case True{}: {==}
    case False{}: {==}

def source_present(+white: Bool) -> {U64.test_bit(U64.bit(U32.to_nat(S.src(white))),U32.to_nat(S.src(white))) == True{} : Bool}:
  match white:
    case True{}: {==}
    case False{}: {==}

def source_kind(+b: Chess.Board,+white: Bool,good: {B.valid(b) == True{} : Bool},
  one: {S.kings(b,white) == U64.bit(U32.to_nat(S.src(white))) : U64}) ->
  {Chess.piece(b,S.src(white)) == 5 : U32}:
  Rows.kind(D.row(b,S.src(white)),P.valid(b,S.src(white),good,source_bound(white)),
    test_left(Chess.get_kings(b),S.color(b,white),U32.to_nat(S.src(white)),source_bound(white),
      Equal.trans(Bool,U64.test_bit(S.kings(b,white),U32.to_nat(S.src(white))),U64.test_bit(U64.bit(U32.to_nat(S.src(white))),U32.to_nat(S.src(white))),True{},
        Equal.cong(U64,Bool,x => U64.test_bit(x,U32.to_nat(S.src(white))),S.kings(b,white),U64.bit(U32.to_nat(S.src(white))),one),source_present(white))))
