# Specification of one masked successor, proved equal to actual subtraction.
import Base
import ./Spec.bend as Spec
import ./Borrow.bend as Borrow
import ../index/Bits.bend as Bits
import ../index/Nats.bend as Nats
import ../index/Words.bend as Index
import ../u64/Words.bend as Old
import ../../Subsets.bend as Subsets


def hold(+n: Nat, +x: Word(n), +mask: Word(n)) ->
  {Word.and(n,Word.adc(n,Word.and(n,x,mask),mask,True{},False{}),mask) == Word.and(n,x,mask) : Word(n)}:
  match n:
    case 0n: {==}
    case 1n+p:
      match x mask:
        case WCon{False{},t} WCon{False{},m}:
          %hold(p,t,m) : {WCon{False{},Word.and(p,Word.adc(p,Word.and(p,t,m),m,True{},False{}),m)} == WCon{False{},_} : Word(1n+p)}
          {==}
        case WCon{True{},t} WCon{False{},m}:
          %hold(p,t,m) : {WCon{False{},Word.and(p,Word.adc(p,Word.and(p,t,m),m,True{},False{}),m)} == WCon{False{},_} : Word(1n+p)}
          {==}
        case WCon{False{},t} WCon{True{},m}:
          %hold(p,t,m) : {WCon{False{},Word.and(p,Word.adc(p,Word.and(p,t,m),m,True{},False{}),m)} == WCon{False{},_} : Word(1n+p)}
          {==}
        case WCon{True{},t} WCon{True{},m}:
          %hold(p,t,m) : {WCon{True{},Word.and(p,Word.adc(p,Word.and(p,t,m),m,True{},False{}),m)} == WCon{True{},_} : Word(1n+p)}
          {==}

def step(+n: Nat, +x: Word(n), +mask: Word(n)) ->
  {Word.and(n,Word.sub(n,Word.and(n,x,mask),mask),mask) == Spec.increment(n,x,mask) : Word(n)}:
  match n:
    case 0n: {==}
    case 1n+p:
      match x mask:
        case WCon{False{},t} WCon{False{},m}:
          %step(p,t,m) : {WCon{False{},Word.and(p,Word.sub(p,Word.and(p,t,m),m),m)} == WCon{False{},_} : Word(1n+p)}
          {==}
        case WCon{True{},t} WCon{False{},m}:
          %step(p,t,m) : {WCon{False{},Word.and(p,Word.sub(p,Word.and(p,t,m),m),m)} == WCon{False{},_} : Word(1n+p)}
          {==}
        case WCon{False{},t} WCon{True{},m}:
          %hold(p,t,m) : {WCon{True{},Word.and(p,Word.adc(p,Word.and(p,t,m),m,True{},False{}),m)} == WCon{True{},_} : Word(1n+p)}
          {==}
        case WCon{True{},t} WCon{True{},m}:
          %step(p,t,m) : {WCon{False{},Word.and(p,Word.sub(p,Word.and(p,t,m),m),m)} == WCon{False{},_} : Word(1n+p)}
          {==}

def extract_mask(+n: Nat, +x: Word(n), +mask: Word(n)) ->
  {Word.pext(n,Word.and(n,x,mask),mask) == Word.pext(n,x,mask) : Word(n)}:
  match n:
    case 0n: {==}
    case 1n+p:
      match x mask:
        case WCon{False{},t} WCon{False{},m}:
          %extract_mask(p,t,m) : {Word.shr.pad(p,Word.pext(p,Word.and(p,t,m),m)) == Word.shr.pad(p,_) : Word(1n+p)}
          {==}
        case WCon{True{},t} WCon{False{},m}:
          %extract_mask(p,t,m) : {Word.shr.pad(p,Word.pext(p,Word.and(p,t,m),m)) == Word.shr.pad(p,_) : Word(1n+p)}
          {==}
        case WCon{False{},t} WCon{True{},m}:
          %extract_mask(p,t,m) : {WCon{False{},Word.pext(p,Word.and(p,t,m),m)} == WCon{False{},_} : Word(1n+p)}
          {==}
        case WCon{True{},t} WCon{True{},m}:
          %extract_mask(p,t,m) : {WCon{True{},Word.pext(p,Word.and(p,t,m),m)} == WCon{True{},_} : Word(1n+p)}
          {==}


def cycle_inside(+v: Nat, +cap: Nat, e: {Nat.is_lt(1n+v,cap) == True{} : Bool}) ->
  {Spec.cycle(v,cap) == 1n+v : Nat}:
  %Equal.sym(Bool,Nat.is_lt(1n+v,cap),True{},e) : {Bool.pick(Nat,_,1n+v,0n) == 1n+v : Nat}
  {==}

def even(+v: Nat, +cap: Nat, e: {Nat.is_lt(v,cap) == True{} : Bool}) ->
  {Spec.cycle(Nat.double(v),Nat.double(cap)) == 1n+Nat.double(v) : Nat}:
  cycle_inside(Nat.double(v),Nat.double(cap),Nats.odd_lt(v,cap,e))

def compare_double(+v: Nat, +cap: Nat) ->
  {Nat.is_lt(Nat.double(v),Nat.double(cap)) == Nat.is_lt(v,cap) : Bool}:
  match v cap:
    case 0n 0n: {==}
    case 0n 1n+p: {==}
    case 1n+p 0n: {==}
    case 1n+p 1n+q: compare_double(p,q)

def pick_double(c: Bool, +v: Nat) ->
  {Bool.pick(Nat,c,Nat.double(v),0n) == Nat.double(Bool.pick(Nat,c,v,0n)) : Nat}:
  match c:
    case False{}: {==}
    case True{}: {==}

def odd(+v: Nat, +cap: Nat) ->
  {Spec.cycle(1n+Nat.double(v),Nat.double(cap)) == Nat.double(Spec.cycle(v,cap)) : Nat}:
  %Equal.sym(Bool,Nat.is_lt(Nat.double(1n+v),Nat.double(cap)),Nat.is_lt(1n+v,cap),compare_double(1n+v,cap)) :
    {Bool.pick(Nat,_,2n+Nat.double(v),0n) == Nat.double(Spec.cycle(v,cap)) : Nat}
  pick_double(Nat.is_lt(1n+v,cap),1n+v)

def increment_value(+n: Nat, +x: Word(n), +mask: Word(n)) ->
  {Word.to_nat(n,Word.pext(n,Spec.increment(n,x,mask),mask))
    == Spec.cycle(Word.to_nat(n,Word.pext(n,x,mask)),Bits.power(Word.count(n,mask))) : Nat}:
  match n:
    case 0n: {==}
    case 1n+p:
      match x mask:
        case WCon{False{},t} WCon{False{},m}:
          %Equal.sym(Nat,Word.to_nat(1n+p,Word.shr.pad(p,Word.pext(p,Spec.increment(p,t,m),m))),
            Word.to_nat(p,Word.pext(p,Spec.increment(p,t,m),m)),Index.pad_value(p,Word.pext(p,Spec.increment(p,t,m),m))) :
            {_ == Spec.cycle(Word.to_nat(1n+p,Word.shr.pad(p,Word.pext(p,t,m))),Bits.power(Word.count(p,m))) : Nat}
          %Equal.sym(Nat,Word.to_nat(1n+p,Word.shr.pad(p,Word.pext(p,t,m))),
            Word.to_nat(p,Word.pext(p,t,m)),Index.pad_value(p,Word.pext(p,t,m))) :
            {Word.to_nat(p,Word.pext(p,Spec.increment(p,t,m),m)) == Spec.cycle(_,Bits.power(Word.count(p,m))) : Nat}
          increment_value(p,t,m)
        case WCon{True{},t} WCon{False{},m}:
          %Equal.sym(Nat,Word.to_nat(1n+p,Word.shr.pad(p,Word.pext(p,Spec.increment(p,t,m),m))),
            Word.to_nat(p,Word.pext(p,Spec.increment(p,t,m),m)),Index.pad_value(p,Word.pext(p,Spec.increment(p,t,m),m))) :
            {_ == Spec.cycle(Word.to_nat(1n+p,Word.shr.pad(p,Word.pext(p,t,m))),Bits.power(Word.count(p,m))) : Nat}
          %Equal.sym(Nat,Word.to_nat(1n+p,Word.shr.pad(p,Word.pext(p,t,m))),
            Word.to_nat(p,Word.pext(p,t,m)),Index.pad_value(p,Word.pext(p,t,m))) :
            {Word.to_nat(p,Word.pext(p,Spec.increment(p,t,m),m)) == Spec.cycle(_,Bits.power(Word.count(p,m))) : Nat}
          increment_value(p,t,m)
        case WCon{False{},t} WCon{True{},m}:
          %Equal.sym(Word(p),Word.pext(p,Word.and(p,t,m),m),Word.pext(p,t,m),extract_mask(p,t,m)) :
            {1n+Nat.double(Word.to_nat(p,_)) == Spec.cycle(Nat.double(Word.to_nat(p,Word.pext(p,t,m))),Nat.double(Bits.power(Word.count(p,m)))) : Nat}
          Equal.sym(Nat,Spec.cycle(Nat.double(Word.to_nat(p,Word.pext(p,t,m))),Nat.double(Bits.power(Word.count(p,m)))),
            1n+Nat.double(Word.to_nat(p,Word.pext(p,t,m))),
            even(Word.to_nat(p,Word.pext(p,t,m)),Bits.power(Word.count(p,m)),Index.extract_bound(p,t,m)))
        case WCon{True{},t} WCon{True{},m}:
          %Equal.sym(Nat,Spec.cycle(1n+Nat.double(Word.to_nat(p,Word.pext(p,t,m))),Nat.double(Bits.power(Word.count(p,m)))),
            Nat.double(Spec.cycle(Word.to_nat(p,Word.pext(p,t,m)),Bits.power(Word.count(p,m)))),
            odd(Word.to_nat(p,Word.pext(p,t,m)),Bits.power(Word.count(p,m)))) :
            {Nat.double(Word.to_nat(p,Word.pext(p,Spec.increment(p,t,m),m))) == _ : Nat}
          Equal.cong(Nat,Nat,w => Nat.double(w),Word.to_nat(p,Word.pext(p,Spec.increment(p,t,m),m)),
            Spec.cycle(Word.to_nat(p,Word.pext(p,t,m)),Bits.power(Word.count(p,m))),increment_value(p,t,m))

def and_word(+a: U64, +b: U64) ->
  {U64.to_word(U64.and(a,b)) == Word.and(64n,U64.to_word(a),U64.to_word(b)) : Word(64n)}:
  %Old.from_to(a) : {U64.to_word(U64.and(_,b)) == Word.and(64n,U64.to_word(a),U64.to_word(b)) : Word(64n)}
  %Old.from_to(b) : {U64.to_word(U64.and(U64.from_word(U64.to_word(a)),_)) == Word.and(64n,U64.to_word(a),U64.to_word(b)) : Word(64n)}
  %Old.pack_and(U64.to_word(a),U64.to_word(b)) :
    {U64.to_word(_) == Word.and(64n,U64.to_word(a),U64.to_word(b)) : Word(64n)}
  Old.to_from(Word.and(64n,U64.to_word(a),U64.to_word(b)))

def member_word(+s: U64, +mask: U64, e: {U64.and(s,mask) == s : U64}) ->
  {Word.and(64n,U64.to_word(s),U64.to_word(mask)) == U64.to_word(s) : Word(64n)}:
  Equal.trans(Word(64n),Word.and(64n,U64.to_word(s),U64.to_word(mask)),U64.to_word(U64.and(s,mask)),U64.to_word(s),
    Equal.sym(Word(64n),U64.to_word(U64.and(s,mask)),Word.and(64n,U64.to_word(s),U64.to_word(mask)),and_word(s,mask)),
    Equal.cong(U64,Word(64n),w => U64.to_word(w),U64.and(s,mask),s,e))

def actual_step(+s: U64, +mask: U64, e: {U64.and(s,mask) == s : U64}) ->
  {U64.to_word(Subsets.next(s,mask)) == Spec.increment(64n,U64.to_word(s),U64.to_word(mask)) : Word(64n)}:
  %Equal.sym(Word(64n),U64.to_word(U64.and(U64.sub(s,mask),mask)),
    Word.and(64n,U64.to_word(U64.sub(s,mask)),U64.to_word(mask)),and_word(U64.sub(s,mask),mask)) :
    {_ == Spec.increment(64n,U64.to_word(s),U64.to_word(mask)) : Word(64n)}
  %Equal.sym(Word(64n),U64.to_word(U64.sub(s,mask)),Word.sub(64n,U64.to_word(s),U64.to_word(mask)),Borrow.u64_sub(s,mask)) :
    {Word.and(64n,_,U64.to_word(mask)) == Spec.increment(64n,U64.to_word(s),U64.to_word(mask)) : Word(64n)}
  %member_word(s,mask,e) :
    {Word.and(64n,Word.sub(64n,_,U64.to_word(mask)),U64.to_word(mask)) == Spec.increment(64n,U64.to_word(s),U64.to_word(mask)) : Word(64n)}
  step(64n,U64.to_word(s),U64.to_word(mask))

def extract_value(+s: U64, +mask: U64) ->
  {Bits.value(U64.pext(s,mask)) == Word.to_nat(64n,Word.pext(64n,U64.to_word(s),U64.to_word(mask))) : Nat}:
  Equal.cong(Word(64n),Nat,w => Word.to_nat(64n,w),
    U64.to_word(U64.from_word(Word.pext(64n,U64.to_word(s),U64.to_word(mask)))),
    Word.pext(64n,U64.to_word(s),U64.to_word(mask)),Old.to_from(Word.pext(64n,U64.to_word(s),U64.to_word(mask))))

def actual_value(+s: U64, +mask: U64, e: {U64.and(s,mask) == s : U64}) ->
  {Bits.value(U64.pext(Subsets.next(s,mask),mask)) == Spec.cycle(Bits.value(U64.pext(s,mask)),Bits.capacity(mask)) : Nat}:
  %Equal.sym(Nat,Bits.value(U64.pext(Subsets.next(s,mask),mask)),
    Word.to_nat(64n,Word.pext(64n,U64.to_word(Subsets.next(s,mask)),U64.to_word(mask))),extract_value(Subsets.next(s,mask),mask)) :
    {_ == Spec.cycle(Bits.value(U64.pext(s,mask)),Bits.capacity(mask)) : Nat}
  %Equal.sym(Word(64n),U64.to_word(Subsets.next(s,mask)),Spec.increment(64n,U64.to_word(s),U64.to_word(mask)),actual_step(s,mask,e)) :
    {Word.to_nat(64n,Word.pext(64n,_,U64.to_word(mask))) == Spec.cycle(Bits.value(U64.pext(s,mask)),Bits.capacity(mask)) : Nat}
  %Equal.sym(Nat,Bits.value(U64.pext(s,mask)),Word.to_nat(64n,Word.pext(64n,U64.to_word(s),U64.to_word(mask))),extract_value(s,mask)) :
    {Word.to_nat(64n,Word.pext(64n,Spec.increment(64n,U64.to_word(s),U64.to_word(mask)),U64.to_word(mask))) == Spec.cycle(_,Bits.capacity(mask)) : Nat}
  increment_value(64n,U64.to_word(s),U64.to_word(mask))
