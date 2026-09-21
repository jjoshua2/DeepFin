import Base
import ./Spec.bend as Spec
import ../../Subsets.bend as Subsets
import ./Step.bend as Step
import ../LAWS.bend as Sequence
import ../PROOF.bend as SequenceProof
import ../index/Bits.bend as Bits
import ../u64/Words.bend as Old
import ./Value.bend as Value
import ../index/LAWS.bend as IndexLaws
import ../index/PROOF.bend as IndexProof
import ../index/Nats.bend as Nats
import ../Mask.bend as Mask
# Induction on the actual imported Nat-indexed carry-rippler recurrence.

# Narrowing the index, never the mask or the accepted theorem's domain.
def smaller(+n: Nat, +cap: Nat, e: {Nat.is_lt(1n+n,cap) == True{} : Bool}) ->
  {Nat.is_lt(n,cap) == True{} : Bool}:
  match n cap:
    case 0n 0n: e
    case 0n 1n+p: {==}
    case 1n+p 0n: e
    case 1n+p 1n+q: smaller(p,q,e)

def zero_value(+n: Nat) -> {Word.to_nat(n,Word.zero(n)) == 0n : Nat}:
  match n:
    case 0n: {==}
    case 1n+p:
      %zero_value(p) : {Nat.double(Word.to_nat(p,Word.zero(p))) == Nat.double(_) : Nat}
      {==}

def extract_zero(+n: Nat, +mask: Word(n)) ->
  {Word.pext(n,Word.zero(n),mask) == Word.zero(n) : Word(n)}:
  match n:
    case 0n: {==}
    case 1n+p:
      match mask:
        case WCon{False{},m}:
          %Old.pad_zero(p) : {Word.shr.pad(p,Word.pext(p,Word.zero(p),m)) == _ : Word(1n+p)}
          %extract_zero(p,m) : {Word.shr.pad(p,Word.pext(p,Word.zero(p),m)) == Word.shr.pad(p,_) : Word(1n+p)}
          {==}
        case WCon{True{},m}:
          %extract_zero(p,m) : {WCon{False{},Word.pext(p,Word.zero(p),m)} == WCon{False{},_} : Word(1n+p)}
          {==}

def start(+mask: U64) -> {Bits.value(U64.pext(U64.zero(),mask)) == 0n : Nat}:
  %Equal.sym(Nat,Bits.value(U64.pext(U64.zero(),mask)),
    Word.to_nat(64n,Word.pext(64n,U64.to_word(U64.zero()),U64.to_word(mask))),Step.extract_value(U64.zero(),mask)) : {_ == 0n : Nat}
  %Equal.sym(Word(64n),Word.pext(64n,Word.zero(64n),U64.to_word(mask)),Word.zero(64n),extract_zero(64n,U64.to_word(mask))) :
    {Word.to_nat(64n,_) == 0n : Nat}
  zero_value(64n)

def ordinal(+n: Nat, +mask: U64, +e: {Nat.is_lt(n,Bits.capacity(mask)) == True{} : Bool}) ->
  {Bits.value(U64.pext(Sequence.at(n,mask),mask)) == n : Nat}:
  match n:
    case 0n: start(mask)
    case 1n+p:
      Equal.trans(Nat,Bits.value(U64.pext(Sequence.at(1n+p,mask),mask)),
        Spec.cycle(Bits.value(U64.pext(Sequence.at(p,mask),mask)),Bits.capacity(mask)),1n+p,
        Step.actual_value(Sequence.at(p,mask),mask,Sequence.sequence_in_mask(p,mask)),
        Equal.trans(Nat,Spec.cycle(Bits.value(U64.pext(Sequence.at(p,mask),mask)),Bits.capacity(mask)),
          Spec.cycle(p,Bits.capacity(mask)),1n+p,
          Equal.cong(Nat,Nat,i => Spec.cycle(i,Bits.capacity(mask)),
            Bits.value(U64.pext(Sequence.at(p,mask),mask)),p,ordinal(p,mask,smaller(p,Bits.capacity(mask),e))),
          Step.cycle_inside(p,Bits.capacity(mask),e)))


def states_equal(+a: U64, +b: U64, +mask: U64,
  am: {U64.and(a,mask) == a : U64}, bm: {U64.and(b,mask) == b : U64},
  e: {Bits.value(U64.pext(a,mask)) == Bits.value(U64.pext(b,mask)) : Nat}) -> {a == b : U64}:
  IndexLaws.masked_extract_injective(a,b,mask,am,bm,Value.u64_injective(U64.pext(a,mask),U64.pext(b,mask),e))

def nonduplicating(+i: Nat, +j: Nat, +mask: U64,
  ib: {Nat.is_lt(i,Bits.capacity(mask)) == True{} : Bool},
  jb: {Nat.is_lt(j,Bits.capacity(mask)) == True{} : Bool},
  e: {Sequence.at(i,mask) == Sequence.at(j,mask) : U64}) -> {i == j : Nat}:
  %ordinal(i,mask,ib) : {_ == j : Nat}
  %ordinal(j,mask,jb) : {Bits.value(U64.pext(Sequence.at(i,mask),mask)) == _ : Nat}
  Equal.cong(U64,Nat,s => Bits.value(U64.pext(s,mask)),Sequence.at(i,mask),Sequence.at(j,mask),e)

def covering(+s: U64, +mask: U64, member: {U64.and(s,mask) == s : U64}) ->
  {Sequence.at(Bits.value(U64.pext(s,mask)),mask) == s : U64}:
  states_equal(Sequence.at(Bits.value(U64.pext(s,mask)),mask),s,mask,
    Sequence.sequence_in_mask(Bits.value(U64.pext(s,mask)),mask),member,
    ordinal(Bits.value(U64.pext(s,mask)),mask,IndexLaws.extraction_bound(s,mask)))

def lt_successor(n: Nat) -> {Nat.is_lt(n,1n+n) == True{} : Bool}:
  match n:
    case 0n: {==}
    case 1n+p: lt_successor(p)

def no_lt_self(n: Nat) -> {Nat.is_lt(n,n) == False{} : Bool}:
  match n:
    case 0n: {==}
    case 1n+p: no_lt_self(p)

def last_bound(+p: Nat, +mask: U64, e: {1n+p == Bits.capacity(mask) : Nat}) ->
  {Nat.is_lt(p,Bits.capacity(mask)) == True{} : Bool}:
  %e : {Nat.is_lt(p,_) == True{} : Bool}
  lt_successor(p)

def cycle_last(+p: Nat, +cap: Nat, e: {1n+p == cap : Nat}) -> {Spec.cycle(p,cap) == 0n : Nat}:
  %e : {Spec.cycle(p,_) == 0n : Nat}
  %Equal.sym(Bool,Nat.is_lt(1n+p,1n+p),False{},no_lt_self(1n+p)) :
    {Bool.pick(Nat,_,1n+p,0n) == 0n : Nat}
  {==}

def wrap_value(+cap: Nat, +mask: U64, +e: {cap == Bits.capacity(mask) : Nat}) ->
  {Bits.value(U64.pext(Sequence.at(cap,mask),mask)) == 0n : Nat}:
  match cap:
    case 0n:
      Old.false_true({Bits.value(U64.pext(Sequence.at(0n,mask),mask)) == 0n : Nat},
        Equal.trans(Bool,False{},Nat.is_lt(0n,Bits.capacity(mask)),True{},
          Equal.cong(Nat,Bool,c => Nat.is_lt(0n,c),0n,Bits.capacity(mask),e),Nats.power_positive(Bits.width(mask))))
    case 1n+p:
      Equal.trans(Nat,Bits.value(U64.pext(Sequence.at(1n+p,mask),mask)),
        Spec.cycle(Bits.value(U64.pext(Sequence.at(p,mask),mask)),Bits.capacity(mask)),0n,
        Step.actual_value(Sequence.at(p,mask),mask,Sequence.sequence_in_mask(p,mask)),
        Equal.trans(Nat,Spec.cycle(Bits.value(U64.pext(Sequence.at(p,mask),mask)),Bits.capacity(mask)),
          Spec.cycle(p,Bits.capacity(mask)),0n,
          Equal.cong(Nat,Nat,i => Spec.cycle(i,Bits.capacity(mask)),
            Bits.value(U64.pext(Sequence.at(p,mask),mask)),p,ordinal(p,mask,last_bound(p,mask,e))),
          cycle_last(p,Bits.capacity(mask),e)))

def wrap(+mask: U64) -> {Sequence.at(Bits.capacity(mask),mask) == U64.zero() : U64}:
  states_equal(Sequence.at(Bits.capacity(mask),mask),U64.zero(),mask,
    Sequence.sequence_in_mask(Bits.capacity(mask),mask),Mask.u64_zero_left(mask),
    Equal.trans(Nat,Bits.value(U64.pext(Sequence.at(Bits.capacity(mask),mask),mask)),0n,Bits.value(U64.pext(U64.zero(),mask)),
      wrap_value(Bits.capacity(mask),mask,{==}),Equal.sym(Nat,Bits.value(U64.pext(U64.zero(),mask)),0n,start(mask))))

def periodic(+i: Nat, +mask: U64) ->
  {Sequence.at(Nat.add(i,Bits.capacity(mask)),mask) == Sequence.at(i,mask) : U64}:
  match i:
    case 0n: wrap(mask)
    case 1n+p:
      Equal.cong(U64,U64,s => Subsets.next(s,mask),Sequence.at(Nat.add(p,Bits.capacity(mask)),mask),Sequence.at(p,mask),periodic(p,mask))
