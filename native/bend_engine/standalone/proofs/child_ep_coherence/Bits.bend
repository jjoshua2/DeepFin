# Pointwise removal/insertion facts for arbitrary two-limb words.
import Base
import ../decoder/Bits.bend as B
import ../occupancy_insertion/Lattice.bend as L
import ../initialized_pawn_double/Algebra.bend as A
import ../../../legal_probe/Chess.bend as C

def and_false(a: Bool) -> {Bool.and(a,False{}) == False{} : Bool}:
  match a:
    case False{}: {==}
    case True{}: {==}

def word_removed(+n: Nat,a: Word(n),+mask: Word(n),+i: Nat,hit: {B.get(n,mask,i) == True{} : Bool}) -> {B.get(n,Word.and(n,a,Word.not(n,mask)),i) == False{} : Bool}:
  match n:
    case 0n: {==}
    case 1n+p:
      match a mask i:
        case WCon{ah,at} WCon{mh,mt} 0n:
          Equal.trans(Bool,Bool.and(ah,Bool.not(mh)),Bool.and(ah,False{}),False{},
            Equal.cong(Bool,Bool,x => Bool.and(ah,Bool.not(x)),mh,True{},hit),and_false(ah))
        case WCon{ah,at} WCon{mh,mt} 1n+j: word_removed(p,at,mt,j,hit)

def choose_removed(+al: Word(32n),+ah: Word(32n),+ml: Word(32n),+mh: Word(32n),+i: Nat,low: Bool,
  hit: {Bool.pick(Bool,low,B.get(32n,ml,i),B.get(32n,mh,Nat.sub(i,32n))) == True{} : Bool}) ->
  {Bool.pick(Bool,low,B.get(32n,Word.and(32n,al,Word.not(32n,ml)),i),B.get(32n,Word.and(32n,ah,Word.not(32n,mh)),Nat.sub(i,32n))) == False{} : Bool}:
  match low:
    case True{}: word_removed(32n,al,ml,i,hit)
    case False{}: word_removed(32n,ah,mh,Nat.sub(i,32n),hit)

def get_removed(a: U64,+mask: U64,+i: Nat,hit: {B.get64(mask,i) == True{} : Bool}) -> {B.get64(U64.and_not(a,mask),i) == False{} : Bool}:
  match a mask:
    case U64{U32{al},U32{ah}} U64{U32{ml},U32{mh}}: choose_removed(al,ah,ml,mh,i,Nat.is_lt(i,32n),hit)

def removed(+a: U64,+mask: U64,+i: Nat,+bound: {Nat.is_lt(i,64n) == True{} : Bool},hit: {U64.test_bit(mask,i) == True{} : Bool}) -> {U64.test_bit(U64.and_not(a,mask),i) == False{} : Bool}:
  Equal.trans(Bool,U64.test_bit(U64.and_not(a,mask),i),B.get64(U64.and_not(a,mask),i),False{},B.test64(U64.and_not(a,mask),i,bound),
    get_removed(a,mask,i,Equal.trans(Bool,B.get64(mask,i),U64.test_bit(mask,i),True{},Equal.sym(Bool,U64.test_bit(mask,i),B.get64(mask,i),B.test64(mask,i,bound)),hit)))

def or_false(+a: U64,+b: U64,+i: Nat,ae: {U64.test_bit(a,i) == False{} : Bool},be: {U64.test_bit(b,i) == False{} : Bool}) -> {U64.test_bit(U64.or(a,b),i) == False{} : Bool}:
  Equal.trans(Bool,U64.test_bit(U64.or(a,b),i),Bool.or(U64.test_bit(a,i),U64.test_bit(b,i)),False{},L.test_or(a,b,i),
    Equal.trans(Bool,Bool.or(U64.test_bit(a,i),U64.test_bit(b,i)),Bool.or(False{},U64.test_bit(b,i)),False{},
      Equal.cong(Bool,Bool,x => Bool.or(x,U64.test_bit(b,i)),U64.test_bit(a,i),False{},ae),be))

def update_absent(+a: U64,+mask: U64,+target: U64,+on: Bool,+i: Nat,+bound: {Nat.is_lt(i,64n) == True{} : Bool},absent: {U64.test_bit(a,i) == False{} : Bool},miss: {U64.test_bit(target,i) == False{} : Bool}) -> {U64.test_bit(C.update_piece(a,mask,target,on),i) == False{} : Bool}:
  or_false(U64.and_not(a,mask),C.select_u64(on,target,U64.zero()),i,A.capture_absent(a,U64.not(mask),i,bound,absent),A.selected_absent(on,target,i,miss))

def update_removed(+a: U64,+mask: U64,+target: U64,+on: Bool,+i: Nat,+bound: {Nat.is_lt(i,64n) == True{} : Bool},hit: {U64.test_bit(mask,i) == True{} : Bool},miss: {U64.test_bit(target,i) == False{} : Bool}) -> {U64.test_bit(C.update_piece(a,mask,target,on),i) == False{} : Bool}:
  or_false(U64.and_not(a,mask),C.select_u64(on,target,U64.zero()),i,removed(a,mask,i,bound,hit),A.selected_absent(on,target,i,miss))

def or_true_right(a: Bool) -> {Bool.or(a,True{}) == True{} : Bool}:
  match a:
    case False{}: {==}
    case True{}: {==}

def update_present(+a: U64,+mask: U64,+target: U64,+i: Nat,hit: {U64.test_bit(target,i) == True{} : Bool}) -> {U64.test_bit(C.update_piece(a,mask,target,True{}),i) == True{} : Bool}:
  Equal.trans(Bool,U64.test_bit(C.update_piece(a,mask,target,True{}),i),Bool.or(U64.test_bit(U64.and_not(a,mask),i),U64.test_bit(target,i)),True{},L.test_or(U64.and_not(a,mask),target,i),
    Equal.trans(Bool,Bool.or(U64.test_bit(U64.and_not(a,mask),i),U64.test_bit(target,i)),Bool.or(U64.test_bit(U64.and_not(a,mask),i),True{}),True{},
      Equal.cong(Bool,Bool,x => Bool.or(U64.test_bit(U64.and_not(a,mask),i),x),U64.test_bit(target,i),True{},hit),or_true_right(U64.test_bit(U64.and_not(a,mask),i))))
