import Base
import ./LAWS.bend as Laws
import ./Words.bend as Words

def Laws.pext_pdep_roundtrip(x, mask):
  %Words.from_to(x) : {U64.pdep(U64.pext(x, mask), mask) == U64.and(_, mask) : U64}
  %Words.from_to(mask) : {U64.pdep(U64.pext(x, mask), mask) == U64.and(U64.from_word(U64.to_word(x)), _) : U64}
  %Words.pack_and(U64.to_word(x), U64.to_word(mask)) : {U64.pdep(U64.pext(x, mask), mask) == _ : U64}
  %Words.pdep_pext(64n, U64.to_word(x), U64.to_word(mask)) : {U64.pdep(U64.pext(x, mask), mask) == U64.from_word(_) : U64}
  %Words.to_from(Word.pext(64n, U64.to_word(x), U64.to_word(mask))) : {U64.pdep(U64.pext(x, mask), mask) == U64.from_word(Word.pdep(64n, U64.to_word(mask), _)) : U64}
  {==}

def Laws.popcount_bound(x):
  Words.count_u32_bound(Word.count(64n, U64.to_word(x)), Words.count_bound(64n, U64.to_word(x)))

def Laws.popcount_zero():
  {==}

def Laws.popcount_full():
  {==}

def Laws.popcount_bit63():
  {==}

def Laws.word_roundtrip(x):
  Words.from_to(x)

def Laws.word_fullwidth(w):
  Words.to_from(w)

def Laws.parts_roundtrip(x):
  match x:
    case U64{lo, hi}:
      {==}

def Laws.from_parts_low(hi, lo):
  {==}

def Laws.from_parts_high(hi, lo):
  {==}

def Laws.widening_low(x):
  {==}

def Laws.widening_high_zero(x):
  {==}

def Laws.pext_zero_mask(x):
  %Words.pext_zero_mask(64n, U64.to_word(x)) : {U64.pext(x, U64.zero()) == U64.from_word(_) : U64}
  {==}

def Laws.pdep_zero_mask(x):
  %Words.pdep_zero_mask(64n, U64.to_word(x)) : {U64.pdep(x, U64.zero()) == U64.from_word(_) : U64}
  {==}

def Laws.xor_twice(x, mask):
  match x mask:
    case U64{U32{lo}, U32{hi}} U64{U32{ml}, U32{mh}}:
      %Words.xor_twice(32n, lo, ml) : {U64.xor(U64.xor(U64{U32{lo}, U32{hi}}, U64{U32{ml}, U32{mh}}), U64{U32{ml}, U32{mh}}) == U64{U32{_}, U32{hi}} : U64}
      %Words.xor_twice(32n, hi, mh) : {U64.xor(U64.xor(U64{U32{lo}, U32{hi}}, U64{U32{ml}, U32{mh}}), U64{U32{ml}, U32{mh}}) == U64{U32{Word.xor(32n, Word.xor(32n, lo, ml), ml)}, U32{_}} : U64}
      {==}

def Laws.not_twice(x):
  match x:
    case U64{U32{lo}, U32{hi}}:
      %Words.not_twice(32n, lo) : {U64.not(U64.not(U64{U32{lo}, U32{hi}})) == U64{U32{_}, U32{hi}} : U64}
      %Words.not_twice(32n, hi) : {U64.not(U64.not(U64{U32{lo}, U32{hi}})) == U64{U32{Word.not(32n, Word.not(32n, lo))}, U32{_}} : U64}
      {==}
