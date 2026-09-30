# Relate the actual U32 lookup to full PEXT and the existing Nat recurrence.
import Base
import ./Projection.bend as Projection
import ../layout/Spec.bend as Layout
import ../layout/Geometry.bend as Geometry
import ../layout/Consequences.bend as Consequences
import ../layout/Facts.bend as Facts
import ../index/Bits.bend as Bits
import ../index/LAWS.bend as IndexLaws
import ../index/PROOF.bend as IndexProof
import ../successor/LAWS.bend as Successor
import ../successor/PROOF.bend as SuccessorProof
import ../successor/Ordinal.bend as Ordinal
import ../successor/Value.bend as Value
import ../LAWS.bend as Sequence
import ../PROOF.bend as SequenceProof
import ../u64/LAWS.bend as U64Laws
import ../u64/PROOF.bend as U64Proof
import ../../../bitboard_probe/Sliders.bend as Sliders

# The new generic result needs only a 32-bit compact width, not chess geometry.
def compact_exact(+occ: U64, +mask: U64,
  small: {Nat.is_le(Bits.width(mask),32n) == True{} : Bool}) ->
  {U32.to_nat(Sliders.pext_index(occ,mask)) == Bits.value(U64.pext(occ,mask)) : Nat}:
  Projection.low_exact(U64.pext(occ,mask),Bits.width(mask),small,IndexLaws.extraction_bound(occ,mask))

def chess_width(+key: U32, +valid: {Nat.is_lt(U32.to_nat(key),128n) == True{} : Bool},
  +counts: {U64.popcount(Layout.mask(key)) == U32.from_nat(Geometry.population(U32.to_nat(key))) : U32}) ->
  {Nat.is_le(Bits.width(Layout.mask(key)),32n) == True{} : Bool}:
  %IndexProof.count_exact(Layout.mask(key)) : {Nat.is_le(_,32n) == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat(U64.popcount(Layout.mask(key))),Geometry.population(U32.to_nat(key)),
    Consequences.observed(key,valid,counts)) :
    {Nat.is_le(_,32n) == True{} : Bool}
  Projection.lt_le(Geometry.population(U32.to_nat(key)),32n,
    Projection.lt_trans(Geometry.population(U32.to_nat(key)),13n,32n,
      Facts.small(U32.to_nat(key),key,Facts.actual(key,valid)),{==}))

def chess_exact(+key: U32, +occ: U64, valid: {Nat.is_lt(U32.to_nat(key),128n) == True{} : Bool},
  +counts: {U64.popcount(Layout.mask(key)) == U32.from_nat(Geometry.population(U32.to_nat(key))) : U32}) ->
  {U32.to_nat(Sliders.pext_index(occ,Layout.mask(key))) == Bits.value(U64.pext(occ,Layout.mask(key))) : Nat}:
  compact_exact(occ,Layout.mask(key),chess_width(key,valid,counts))

# Mathematical value is preserved by actual U32-to-U64 zero extension.
def join_zero_value(+n: Nat, +m: Nat, +lo: Word(n)) ->
  {Word.to_nat(Nat.add(n,m),Word.join(n,m,lo,Word.zero(m))) == Word.to_nat(n,lo) : Nat}:
  match n:
    case 0n:
      match lo:
        case WNil{}: Ordinal.zero_value(m)
    case 1n+p:
      match lo:
        case WCon{False{},t}:
          Equal.cong(Nat,Nat,v => Nat.double(v),
            Word.to_nat(Nat.add(p,m),Word.join(p,m,t,Word.zero(m))),Word.to_nat(p,t),join_zero_value(p,m,t))
        case WCon{True{},t}:
          Equal.cong(Nat,Nat,v => 1n+Nat.double(v),
            Word.to_nat(Nat.add(p,m),Word.join(p,m,t,Word.zero(m))),Word.to_nat(p,t),join_zero_value(p,m,t))

def widen_value(a: U32) -> {Bits.value(U64.from_u32(a)) == U32.to_nat(a) : Nat}:
  match a:
    case U32{lo}: join_zero_value(32n,32n,lo)

def chess_word(+key: U32, +occ: U64, valid: {Nat.is_lt(U32.to_nat(key),128n) == True{} : Bool},
  +counts: {U64.popcount(Layout.mask(key)) == U32.from_nat(Geometry.population(U32.to_nat(key))) : U32}) ->
  {U64.from_u32(Sliders.pext_index(occ,Layout.mask(key))) == U64.pext(occ,Layout.mask(key)) : U64}:
  Value.u64_injective(U64.from_u32(Sliders.pext_index(occ,Layout.mask(key))),U64.pext(occ,Layout.mask(key)),
    Equal.trans(Nat,Bits.value(U64.from_u32(Sliders.pext_index(occ,Layout.mask(key)))),
      U32.to_nat(Sliders.pext_index(occ,Layout.mask(key))),Bits.value(U64.pext(occ,Layout.mask(key))),
      widen_value(Sliders.pext_index(occ,Layout.mask(key))),chess_exact(key,occ,valid,counts)))

def ordinal(+key: U32, +i: Nat, valid: {Nat.is_lt(U32.to_nat(key),128n) == True{} : Bool},
  bounded: {Nat.is_lt(i,Bits.capacity(Layout.mask(key))) == True{} : Bool},
  +counts: {U64.popcount(Layout.mask(key)) == U32.from_nat(Geometry.population(U32.to_nat(key))) : U32}) ->
  {U32.to_nat(Sliders.pext_index(Sequence.at(i,Layout.mask(key)),Layout.mask(key))) == i : Nat}:
  Equal.trans(Nat,U32.to_nat(Sliders.pext_index(Sequence.at(i,Layout.mask(key)),Layout.mask(key))),
    Bits.value(U64.pext(Sequence.at(i,Layout.mask(key)),Layout.mask(key))),i,
    chess_exact(key,Sequence.at(i,Layout.mask(key)),valid,counts),Successor.sequence_ordinal(i,Layout.mask(key),bounded))

def covering(+key: U32, +s: U64, valid: {Nat.is_lt(U32.to_nat(key),128n) == True{} : Bool},
  member: {U64.and(s,Layout.mask(key)) == s : U64},
  +counts: {U64.popcount(Layout.mask(key)) == U32.from_nat(Geometry.population(U32.to_nat(key))) : U32}) ->
  {Sequence.at(U32.to_nat(Sliders.pext_index(s,Layout.mask(key))),Layout.mask(key)) == s : U64}:
  %Equal.sym(Nat,U32.to_nat(Sliders.pext_index(s,Layout.mask(key))),Bits.value(U64.pext(s,Layout.mask(key))),chess_exact(key,s,valid,counts)) :
    {Sequence.at(_,Layout.mask(key)) == s : U64}
  Ordinal.covering(s,Layout.mask(key),member)

def injective(+key: U32, +a: U64, +b: U64, +valid: {Nat.is_lt(U32.to_nat(key),128n) == True{} : Bool},
  am: {U64.and(a,Layout.mask(key)) == a : U64}, bm: {U64.and(b,Layout.mask(key)) == b : U64},
  same: {Sliders.pext_index(a,Layout.mask(key)) == Sliders.pext_index(b,Layout.mask(key)) : U32},
  +counts: {U64.popcount(Layout.mask(key)) == U32.from_nat(Geometry.population(U32.to_nat(key))) : U32}) -> {a == b : U64}:
  IndexLaws.masked_extract_injective(a,b,Layout.mask(key),am,bm,
    Equal.trans(U64,U64.pext(a,Layout.mask(key)),U64.from_u32(Sliders.pext_index(a,Layout.mask(key))),U64.pext(b,Layout.mask(key)),
      Equal.sym(U64,U64.from_u32(Sliders.pext_index(a,Layout.mask(key))),U64.pext(a,Layout.mask(key)),chess_word(key,a,valid,counts)),
      Equal.trans(U64,U64.from_u32(Sliders.pext_index(a,Layout.mask(key))),U64.from_u32(Sliders.pext_index(b,Layout.mask(key))),U64.pext(b,Layout.mask(key)),
        Equal.cong(U32,U64,x => U64.from_u32(x),Sliders.pext_index(a,Layout.mask(key)),Sliders.pext_index(b,Layout.mask(key)),same),
        chess_word(key,b,valid,counts))))

def redeposit(+key: U32, +occ: U64, valid: {Nat.is_lt(U32.to_nat(key),128n) == True{} : Bool},
  +counts: {U64.popcount(Layout.mask(key)) == U32.from_nat(Geometry.population(U32.to_nat(key))) : U32}) ->
  {U64.pdep(U64.from_u32(Sliders.pext_index(occ,Layout.mask(key))),Layout.mask(key)) == U64.and(occ,Layout.mask(key)) : U64}:
  %Equal.sym(U64,U64.from_u32(Sliders.pext_index(occ,Layout.mask(key))),U64.pext(occ,Layout.mask(key)),chess_word(key,occ,valid,counts)) :
    {U64.pdep(_,Layout.mask(key)) == U64.and(occ,Layout.mask(key)) : U64}
  U64Laws.pext_pdep_roundtrip(occ,Layout.mask(key))
