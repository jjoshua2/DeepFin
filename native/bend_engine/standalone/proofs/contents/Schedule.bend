# Relate the traversal used by fill to actual modular U32 addition and the
# previously proved, zero-start carry-rippler recurrence. No new enumerator.
import Base
import ./Spec.bend as Spec
import ../../Subsets.bend as Subsets
import ../LAWS.bend as Sequence
import ../PROOF.bend as SequenceProof

def add_zero(+n: Nat,+a: Word(n)) -> {Word.add(n,a,Word.zero(n)) == a : Word(n)}:
  match n:
    case 0n:
      match a:
        case WNil{}: {==}
    case 1n+p:
      match a:
        case WCon{False{},t}: Equal.cong(Word(p),Word(1n+p),x => WCon{False{},x},Word.add(p,t,Word.zero(p)),t,add_zero(p,t))
        case WCon{True{},t}: Equal.cong(Word(p),Word(1n+p),x => WCon{True{},x},Word.add(p,t,Word.zero(p)),t,add_zero(p,t))

def carry_increment(+n: Nat,+a: Word(n),+b: Word(n)) ->
  {Word.adc(n,a,b,False{},True{}) == Word.inc(n,Word.add(n,a,b)) : Word(n)}:
  match n:
    case 0n: {==}
    case 1n+p:
      match a b:
        case WCon{False{},x} WCon{False{},y}: {==}
        case WCon{False{},x} WCon{True{},y}:
          Equal.cong(Word(p),Word(1n+p),t => WCon{False{},t},Word.adc(p,x,y,False{},True{}),Word.inc(p,Word.add(p,x,y)),carry_increment(p,x,y))
        case WCon{True{},x} WCon{False{},y}:
          Equal.cong(Word(p),Word(1n+p),t => WCon{False{},t},Word.adc(p,x,y,False{},True{}),Word.inc(p,Word.add(p,x,y)),carry_increment(p,x,y))
        case WCon{True{},x} WCon{True{},y}: {==}

def increment_left(+n: Nat,+a: Word(n),+b: Word(n)) ->
  {Word.add(n,Word.inc(n,a),b) == Word.inc(n,Word.add(n,a,b)) : Word(n)}:
  match n:
    case 0n: {==}
    case 1n+p:
      match a b:
        case WCon{False{},x} WCon{False{},y}: {==}
        case WCon{False{},x} WCon{True{},y}:
          Equal.cong(Word(p),Word(1n+p),t => WCon{False{},t},Word.adc(p,x,y,False{},True{}),Word.inc(p,Word.add(p,x,y)),carry_increment(p,x,y))
        case WCon{True{},x} WCon{False{},y}:
          Equal.cong(Word(p),Word(1n+p),t => WCon{False{},t},Word.add(p,Word.inc(p,x),y),Word.inc(p,Word.add(p,x,y)),increment_left(p,x,y))
        case WCon{True{},x} WCon{True{},y}:
          Equal.cong(Word(p),Word(1n+p),t => WCon{True{},t},Word.add(p,Word.inc(p,x),y),Word.adc(p,x,y,False{},True{}),
            Equal.trans(Word(p),Word.add(p,Word.inc(p,x),y),Word.inc(p,Word.add(p,x,y)),Word.adc(p,x,y,False{},True{}),
              increment_left(p,x,y),Equal.sym(Word(p),Word.adc(p,x,y,False{},True{}),Word.inc(p,Word.add(p,x,y)),carry_increment(p,x,y))))

def shift_increment(+a: U32,+b: U32) -> {U32.add(U32.inc(a),b) == U32.add(a,U32.inc(b)) : U32}:
  match a b:
    case U32{x} U32{y}:
      Equal.cong(Word(32n),U32,w => U32{w},Word.add(32n,Word.inc(32n,x),y),Word.add(32n,x,Word.inc(32n,y)),
        Equal.trans(Word(32n),Word.add(32n,Word.inc(32n,x),y),Word.inc(32n,Word.add(32n,x,y)),Word.add(32n,x,Word.inc(32n,y)),
          increment_left(32n,x,y),
          Equal.trans(Word(32n),Word.inc(32n,Word.add(32n,x,y)),Word.inc(32n,Word.add(32n,y,x)),Word.add(32n,x,Word.inc(32n,y)),
            Equal.cong(Word(32n),Word(32n),w => Word.inc(32n,w),Word.add(32n,x,y),Word.add(32n,y,x),Word.add_comm(32n,x,y)),
            Equal.trans(Word(32n),Word.inc(32n,Word.add(32n,y,x)),Word.add(32n,Word.inc(32n,y),x),Word.add(32n,x,Word.inc(32n,y)),
              Equal.sym(Word(32n),Word.add(32n,Word.inc(32n,y),x),Word.inc(32n,Word.add(32n,y,x)),increment_left(32n,y,x)),
              Word.add_comm(32n,Word.inc(32n,y),x)))))

def address(+i: Nat,+at: U32) -> {Spec.address(i,at) == U32.add(at,U32.from_nat(i)) : U32}:
  match i:
    case 0n:
      match at:
        case U32{x}: Equal.cong(Word(32n),U32,w => U32{w},x,Word.add(32n,x,Word.zero(32n)),Equal.sym(Word(32n),Word.add(32n,x,Word.zero(32n)),x,add_zero(32n,x)))
    case 1n+p:
      Equal.trans(U32,Spec.address(p,U32.inc(at)),U32.add(U32.inc(at),U32.from_nat(p)),U32.add(at,U32.inc(U32.from_nat(p))),
        address(p,U32.inc(at)),shift_increment(at,U32.from_nat(p)))

def advance(+n: Nat,+mask: U64,+s: U64) ->
  {Spec.state(n,mask,Subsets.next(s,mask)) == Subsets.next(Spec.state(n,mask,s),mask) : U64}:
  match n:
    case 0n: {==}
    case 1n+p: advance(p,mask,Subsets.next(s,mask))

def from_zero(+n: Nat,+mask: U64) -> {Spec.state(n,mask,U64.zero()) == Sequence.at(n,mask) : U64}:
  match n:
    case 0n: {==}
    case 1n+p:
      Equal.trans(U64,Spec.state(p,mask,Subsets.next(U64.zero(),mask)),Subsets.next(Spec.state(p,mask,U64.zero()),mask),Sequence.at(1n+p,mask),
        advance(p,mask,U64.zero()),Equal.cong(U64,U64,s => Subsets.next(s,mask),Spec.state(p,mask,U64.zero()),Sequence.at(p,mask),from_zero(p,mask)))
