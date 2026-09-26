import Base
import ./LAWS.bend as Laws
import ./Bits.bend as Bits
import ./Nats.bend as Nats
import ./Count.bend as Count
import ./Words.bend as Words
import ../LAWS.bend as SubsetLaws
import ../PROOF.bend as SubsetProof
import ../u64/Words.bend as Old

def count_exact(+mask: U64) -> {U32.to_nat(U64.popcount(mask)) == Bits.width(mask) : Nat}:
  Count.exact(Bits.width(mask), Old.count_bound(64n, U64.to_word(mask)))

def Laws.capacity_matches_popcount(mask):
  Equal.trans(Nat, Bits.capacity(mask), Nat.pow(2n, Bits.width(mask)), Nat.pow(2n, U32.to_nat(U64.popcount(mask))),
    Nats.power_nat(Bits.width(mask)),
    Equal.cong(Nat, Nat, k => Nat.pow(2n, k), Bits.width(mask), U32.to_nat(U64.popcount(mask)),
      Equal.sym(Nat, U32.to_nat(U64.popcount(mask)), Bits.width(mask), count_exact(mask))))

def Laws.extraction_bound(x, mask):
  %Equal.sym(Word(64n), U64.to_word(U64.from_word(Word.pext(64n, U64.to_word(x), U64.to_word(mask)))),
    Word.pext(64n, U64.to_word(x), U64.to_word(mask)),
    Old.to_from(Word.pext(64n, U64.to_word(x), U64.to_word(mask)))) :
    {Nat.is_lt(Word.to_nat(64n, _), Bits.capacity(mask)) == True{} : Bool}
  Words.extract_bound(64n, U64.to_word(x), U64.to_word(mask))

def Laws.compact_projection(x, mask):
  %Equal.sym(Word(64n), U64.to_word(U64.from_word(Word.pdep(64n, U64.to_word(mask), U64.to_word(x)))),
    Word.pdep(64n, U64.to_word(mask), U64.to_word(x)),
    Old.to_from(Word.pdep(64n, U64.to_word(mask), U64.to_word(x)))) :
    {U64.from_word(Word.pext(64n, _, U64.to_word(mask)))
      == U64.from_word(Bits.clip(64n, Bits.width(mask), U64.to_word(x))) : U64}
  Equal.cong(Word(64n), U64, w => U64.from_word(w),
    Word.pext(64n, Word.pdep(64n, U64.to_word(mask), U64.to_word(x)), U64.to_word(mask)),
    Bits.clip(64n, Bits.width(mask), U64.to_word(x)),
    Words.reverse(64n, U64.to_word(mask), U64.to_word(x)))

def Laws.compact_roundtrip(x, mask, bounded):
  Equal.trans(U64, U64.pext(U64.pdep(x, mask), mask),
    U64.from_word(Bits.clip(64n, Bits.width(mask), U64.to_word(x))), x,
    Laws.compact_projection(x, mask),
    Equal.trans(U64, U64.from_word(Bits.clip(64n, Bits.width(mask), U64.to_word(x))),
      U64.from_word(U64.to_word(x)), x,
      Equal.cong(Word(64n), U64, w => U64.from_word(w), Bits.clip(64n, Bits.width(mask), U64.to_word(x)), U64.to_word(x),
        Words.clip_id(64n, Bits.width(mask), U64.to_word(x), bounded)), Old.from_to(x)))

def Laws.deposit_in_mask(x, mask):
  %Old.from_to(mask) : {U64.and(U64.pdep(x, mask), _) == U64.pdep(x, mask) : U64}
  %Old.pack_and(Word.pdep(64n, U64.to_word(mask), U64.to_word(x)), U64.to_word(mask)) :
    {_ == U64.pdep(x, mask) : U64}
  Equal.cong(Word(64n), U64, w => U64.from_word(w),
    Word.and(64n, Word.pdep(64n, U64.to_word(mask), U64.to_word(x)), U64.to_word(mask)),
    Word.pdep(64n, U64.to_word(mask), U64.to_word(x)),
    Words.deposit_member(64n, U64.to_word(mask), U64.to_word(x)))

def Laws.deposit_injective(a, b, mask, a_bound, b_bound, same):
  Equal.trans(U64, a, U64.pext(U64.pdep(a, mask), mask), b,
    Equal.sym(U64, U64.pext(U64.pdep(a, mask), mask), a, Laws.compact_roundtrip(a, mask, a_bound)),
    Equal.trans(U64, U64.pext(U64.pdep(a, mask), mask), U64.pext(U64.pdep(b, mask), mask), b,
      Equal.cong(U64, U64, s => U64.pext(s, mask), U64.pdep(a, mask), U64.pdep(b, mask), same),
      Laws.compact_roundtrip(b, mask, b_bound)))

def Laws.masked_extract_injective(a, b, mask, a_member, b_member, same):
  Equal.trans(U64, a, U64.pdep(U64.pext(a, mask), mask), b,
    Equal.sym(U64, U64.pdep(U64.pext(a, mask), mask), a, SubsetProof.masked_roundtrip(a, mask, a_member)),
    Equal.trans(U64, U64.pdep(U64.pext(a, mask), mask), U64.pdep(U64.pext(b, mask), mask), b,
      Equal.cong(U64, U64, s => U64.pdep(s, mask), U64.pext(a, mask), U64.pext(b, mask), same),
      SubsetProof.masked_roundtrip(b, mask, b_member)))

def Laws.compact_coverage(x, mask, bounded):
  (U64.pdep(x, mask), (Laws.deposit_in_mask(x, mask), Laws.compact_roundtrip(x, mask, bounded)))

def Laws.sequence_index_bound(n, mask):
  Laws.extraction_bound(SubsetLaws.at(n, mask), mask)
