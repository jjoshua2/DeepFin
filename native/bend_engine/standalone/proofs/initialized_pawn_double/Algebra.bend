import Base
import ../decoder/Bits.bend as B
import ../occupancy_insertion/Lattice.bend as L
import ../u64/Words.bend as Old
import ../Mask.bend as Mask
import ../../../legal_probe/Chess.bend as C

def word_absent(+n: Nat,+a: Word(n),b: Word(n),+i: Nat,
  absent: {B.get(n,a,i) == False{} : Bool}) -> {B.get(n,Word.and(n,a,b),i) == False{} : Bool}:
  match n:
    case 0n: {==}
    case 1n+p:
      match a b i:
        case WCon{ah,at} WCon{bh,bt} 0n:
          Equal.cong(Bool,Bool,x => Bool.and(x,bh),ah,False{},absent)
        case WCon{ah,at} WCon{bh,bt} 1n+j: word_absent(p,at,bt,j,absent)

def choose_absent(+al: Word(32n),+ah: Word(32n),bl: Word(32n),bh: Word(32n),+i: Nat,+low: Bool,
  absent: {Bool.pick(Bool,low,B.get(32n,al,i),B.get(32n,ah,Nat.sub(i,32n))) == False{} : Bool}) ->
  {Bool.pick(Bool,low,B.get(32n,Word.and(32n,al,bl),i),B.get(32n,Word.and(32n,ah,bh),Nat.sub(i,32n))) == False{} : Bool}:
  match low:
    case True{}: word_absent(32n,al,bl,i,absent)
    case False{}: word_absent(32n,ah,bh,Nat.sub(i,32n),absent)

def get_absent(+a: U64,b: U64,+i: Nat,absent: {B.get64(a,i) == False{} : Bool}) ->
  {B.get64(U64.and(a,b),i) == False{} : Bool}:
  match a b:
    case U64{U32{al},U32{ah}} U64{U32{bl},U32{bh}}:
      choose_absent(al,ah,bl,bh,i,Nat.is_lt(i,32n),absent)

def capture_absent(+a: U64,+b: U64,+i: Nat,+bound: {Nat.is_lt(i,64n) == True{} : Bool},
  absent: {U64.test_bit(a,i) == False{} : Bool}) -> {U64.test_bit(U64.and(a,b),i) == False{} : Bool}:
  Equal.trans(Bool,U64.test_bit(U64.and(a,b),i),B.get64(U64.and(a,b),i),False{},B.test64(U64.and(a,b),i,bound),
    get_absent(a,b,i,Equal.trans(Bool,B.get64(a,i),U64.test_bit(a,i),False{},
      Equal.sym(Bool,U64.test_bit(a,i),B.get64(a,i),B.test64(a,i,bound)),absent)))

def zero(+i: Nat) -> {U64.test_bit(U64.zero(),i) == False{} : Bool}:
  Equal.cong(U64,Bool,x => U64.is_ne(x,U64.zero()),U64.and(U64.zero(),U64.bit(i)),U64.zero(),Mask.u64_zero_left(U64.bit(i)))

def selected_absent(+on: Bool,+a: U64,+i: Nat,absent: {U64.test_bit(a,i) == False{} : Bool}) ->
  {U64.test_bit(C.select_u64(on,a,U64.zero()),i) == False{} : Bool}:
  match on:
    case False{}: zero(i)
    case True{}: absent

def right(+a: Bool,+b: Bool,
  absent: {a == False{} : Bool},hit: {Bool.or(a,b) == True{} : Bool}) -> {b == True{} : Bool}:
  Equal.trans(Bool,b,Bool.or(a,b),True{},
    Equal.cong(Bool,Bool,x => Bool.or(x,b),False{},a,Equal.sym(Bool,a,False{},absent)),hit)

def selected_true(+on: Bool,+a: U64,+i: Nat,
  hit: {U64.test_bit(C.select_u64(on,a,U64.zero()),i) == True{} : Bool}) -> {on == True{} : Bool}:
  match on:
    case True{}: {==}
    case False{}: Old.false_true({False{} == True{} : Bool},Equal.trans(Bool,False{},U64.test_bit(U64.zero(),i),True{},Equal.sym(Bool,U64.test_bit(U64.zero(),i),False{},zero(i)),hit))

def or_comm(+a: Bool,+b: Bool) -> {Bool.or(a,b) == Bool.or(b,a) : Bool}:
  match a b:
    case False{} False{}: {==}
    case False{} True{}: {==}
    case True{} False{}: {==}
    case True{} True{}: {==}

def second(+one: U64,+two: U64,+pa: U64,+cap: U64,+i: Nat,+bound: {Nat.is_lt(i,64n) == True{} : Bool},
  first_absent: {U64.test_bit(one,i) == False{} : Bool},pawn_absent: {U64.test_bit(pa,i) == False{} : Bool},
  hit: {U64.test_bit(U64.or(U64.or(one,two),U64.and(pa,cap)),i) == True{} : Bool}) -> {U64.test_bit(two,i) == True{} : Bool}:
  right(U64.test_bit(one,i),U64.test_bit(two,i),first_absent,
    Equal.trans(Bool,Bool.or(U64.test_bit(one,i),U64.test_bit(two,i)),U64.test_bit(U64.or(one,two),i),True{},
      Equal.sym(Bool,U64.test_bit(U64.or(one,two),i),Bool.or(U64.test_bit(one,i),U64.test_bit(two,i)),L.test_or(one,two,i)),
      right(U64.test_bit(U64.and(pa,cap),i),U64.test_bit(U64.or(one,two),i),capture_absent(pa,cap,i,bound,pawn_absent),
        Equal.trans(Bool,Bool.or(U64.test_bit(U64.and(pa,cap),i),U64.test_bit(U64.or(one,two),i)),
          Bool.or(U64.test_bit(U64.or(one,two),i),U64.test_bit(U64.and(pa,cap),i)),True{},
          or_comm(U64.test_bit(U64.and(pa,cap),i),U64.test_bit(U64.or(one,two),i)),
          Equal.trans(Bool,Bool.or(U64.test_bit(U64.or(one,two),i),U64.test_bit(U64.and(pa,cap),i)),U64.test_bit(U64.or(U64.or(one,two),U64.and(pa,cap)),i),True{},
            Equal.sym(Bool,U64.test_bit(U64.or(U64.or(one,two),U64.and(pa,cap)),i),Bool.or(U64.test_bit(U64.or(one,two),i),U64.test_bit(U64.and(pa,cap),i)),L.test_or(U64.or(one,two),U64.and(pa,cap),i)),hit)))))
