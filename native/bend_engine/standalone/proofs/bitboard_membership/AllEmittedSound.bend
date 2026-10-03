import Base
import ../../../legal_probe/Chess.bend as Chess
import ../u64/Words.bend as Old
import ../bitboard_ctz_structural/WordCtzCore.bend as Ctz
import ../bitboard_ctz_structural/TestBitBridge.bend as TestBit
import ../successor/Step.bend as Step
import ./PrefixDecomposition.bend as Prefix
import ./FrequencyIteration.bend as Frequency
import ./IndexEquality.bend as IndexEquality
import ./Cardinality.bend as Cardinality

def all_below64(+xs: List<&2,U32>) -> Bool:
  match xs:
    case Nil{}: True{}
    case Con{head,tail}:
      Bool.and(Nat.is_lt(U32.to_nat(head),64n),all_below64(tail))

def bool_and_true(+a: Bool,+b: Bool,
  +ea: {a == True{} : Bool},+eb: {b == True{} : Bool}) ->
  {Bool.and(a,b) == True{} : Bool}:
  match a b:
    case True{} True{}: {==}
    case True{} False{}: Old.false_true({False{} == True{} : Bool},eb)
    case False{} _: Old.false_true({False{} == True{} : Bool},ea)

def bool_and_true_left(+a: Bool,+b: Bool,
  +e: {Bool.and(a,b) == True{} : Bool}) -> {a == True{} : Bool}:
  match a b:
    case True{} _: {==}
    case False{} _: Old.false_true({False{} == True{} : Bool},e)

def bool_and_true_right(+a: Bool,+b: Bool,
  +e: {Bool.and(a,b) == True{} : Bool}) -> {b == True{} : Bool}:
  match a b:
    case True{} True{}: {==}
    case True{} False{}: Old.false_true({False{} == True{} : Bool},e)
    case False{} _: Old.false_true({b == True{} : Bool},e)

def singleton_below64(+sq: U32,
  +bound: {Nat.is_lt(U32.to_nat(sq),64n) == True{} : Bool}) ->
  {all_below64(sq <> Nil{}) == True{} : Bool}:
  bool_and_true(Nat.is_lt(U32.to_nat(sq),64n),True{},bound,{==})

def all_below64_append(+xs: List<&2,U32>,+ys: List<&2,U32>,
  +xs_below: {all_below64(xs) == True{} : Bool},
  +ys_below: {all_below64(ys) == True{} : Bool}) ->
  {all_below64(Prefix.append(xs,ys)) == True{} : Bool}:
  match xs:
    case Nil{}: ys_below
    case Con{head,tail}:
      +parts: {Bool.and(Nat.is_lt(U32.to_nat(head),64n),all_below64(tail)) == True{} : Bool} = xs_below
      +head_below: {Nat.is_lt(U32.to_nat(head),64n) == True{} : Bool} =
        bool_and_true_left(Nat.is_lt(U32.to_nat(head),64n),all_below64(tail),parts)
      +tail_below: {all_below64(tail) == True{} : Bool} =
        bool_and_true_right(Nat.is_lt(U32.to_nat(head),64n),all_below64(tail),parts)
      +appended_tail: {all_below64(Prefix.append(tail,ys)) == True{} : Bool} =
        all_below64_append(tail,ys,tail_below,ys_below)
      bool_and_true(Nat.is_lt(U32.to_nat(head),64n),all_below64(Prefix.append(tail,ys)),
        head_below,appended_tail)

def emitted_prefix_below64(+n: Nat,+empty: Bool,+bb: U64,
  +same: {empty == U64.is_zero(bb) : Bool}) ->
  {all_below64(Prefix.emitted_prefix(n,empty,bb)) == True{} : Bool}:
  match n:
    case 0n: {==}
    case 1n+p:
      match empty:
        case True{}: {==}
        case False{}:
          +nonzero: {U64.is_zero(bb) == False{} : Bool} =
            Equal.trans(Bool,U64.is_zero(bb),empty,False{},
              Equal.sym(Bool,empty,U64.is_zero(bb),same),{==})
          +rest = U64.clear_lsb(bb)
          +tail = Prefix.emitted_prefix(p,U64.is_zero(rest),rest)
          +tail_below: {all_below64(tail) == True{} : Bool} =
            emitted_prefix_below64(p,U64.is_zero(rest),rest,{==})
          +head_below: {Nat.is_lt(U32.to_nat(U64.ctz(bb)),64n) == True{} : Bool} =
            Ctz.actual_ctz_range(bb,nonzero)
          +head_list_below: {all_below64(U64.ctz(bb) <> Nil{}) == True{} : Bool} =
            singleton_below64(U64.ctz(bb),head_below)
          all_below64_append(tail,U64.ctz(bb) <> Nil{},tail_below,head_list_below)

def actual_bit_squares_below64(+bb: U64) ->
  {all_below64(Chess.bit_squares(64n,U64.is_zero(bb),bb,Nil{})) == True{} : Bool}:
  +prefix = Prefix.emitted_prefix(64n,U64.is_zero(bb),bb)
  +as_prefix: {Chess.bit_squares(64n,U64.is_zero(bb),bb,Nil{}) ==
    Prefix.append(prefix,Nil{}) : List<&2,U32>} =
    Prefix.bit_squares_as_prefix(64n,U64.is_zero(bb),bb,Nil{})
  +prefix_below: {all_below64(prefix) == True{} : Bool} =
    emitted_prefix_below64(64n,U64.is_zero(bb),bb,{==})
  +appended_below: {all_below64(Prefix.append(prefix,Nil{})) == True{} : Bool} =
    all_below64_append(prefix,Nil{},prefix_below,{==})
  Equal.trans(Bool,
    all_below64(Chess.bit_squares(64n,U64.is_zero(bb),bb,Nil{})),
    all_below64(Prefix.append(prefix,Nil{})),True{},
    Equal.cong(List<&2,U32>,Bool,xs => all_below64(xs),
      Chess.bit_squares(64n,U64.is_zero(bb),bb,Nil{}),Prefix.append(prefix,Nil{}),as_prefix),
    appended_below)

def all_bits_set(+xs: List<&2,U32>,+bb: U64) -> Bool:
  match xs:
    case Nil{}: True{}
    case Con{head,tail}:
      Bool.and(U64.test_bit(bb,U32.to_nat(head)),all_bits_set(tail,bb))

def u32_eq_self(+sq: U32) -> {U32.is_eq(sq,sq) == True{} : Bool}:
  IndexEquality.u32_eq_refl(sq)

def occurrences_cons_self(+head: U32,+tail: List<&2,U32>) ->
  {Prefix.occurrences(Con{head,tail},head) ==
    Nat.add(1n,Prefix.occurrences(tail,head)) : Nat}:
  Equal.cong(Nat,Nat,n => Nat.add(n,Prefix.occurrences(tail,head)),
    Bool.pick(Nat,U32.is_eq(head,head),1n,0n),1n,
    Equal.cong(Bool,Nat,b => Bool.pick(Nat,b,1n,0n),
      U32.is_eq(head,head),True{},u32_eq_self(head)))

def nat_add_right_successor_nonzero(+left: Nat,+tail: Nat) ->
  {Nat.is_eq(Nat.add(left,Nat.add(1n,tail)),0n) == False{} : Bool}:
  match left:
    case 0n: {==}
    case 1n+p: {==}

def occurrences_cons_self_nonzero(+head: U32,+tail: List<&2,U32>) ->
  {Nat.is_eq(Prefix.occurrences(Con{head,tail},head),0n) == False{} : Bool}:
  Equal.trans(Bool,
    Nat.is_eq(Prefix.occurrences(Con{head,tail},head),0n),
    Nat.is_eq(Nat.add(1n,Prefix.occurrences(tail,head)),0n),False{},
    Equal.cong(Nat,Bool,n => Nat.is_eq(n,0n),
      Prefix.occurrences(Con{head,tail},head),
      Nat.add(1n,Prefix.occurrences(tail,head)),occurrences_cons_self(head,tail)),
    nat_add_right_successor_nonzero(0n,Prefix.occurrences(tail,head)))

def actual_occurrences_cons_nonzero(+bb: U64,+prefix: List<&2,U32>,
  +head: U32,+tail: List<&2,U32>,
  +decomposition: {Chess.bit_squares(64n,U64.is_zero(bb),bb,Nil{}) ==
    Prefix.append(prefix,Con{head,tail}) : List<&2,U32>}) ->
  {Nat.is_eq(Prefix.occurrences(Chess.bit_squares(64n,U64.is_zero(bb),bb,Nil{}),head),0n) ==
    False{} : Bool}:
  +whole_count: {Prefix.occurrences(Chess.bit_squares(64n,U64.is_zero(bb),bb,Nil{}),head) ==
    Prefix.occurrences(Prefix.append(prefix,Con{head,tail}),head) : Nat} =
    Equal.cong(List<&2,U32>,Nat,xs => Prefix.occurrences(xs,head),
      Chess.bit_squares(64n,U64.is_zero(bb),bb,Nil{}),
      Prefix.append(prefix,Con{head,tail}),decomposition)
  +split_count: {Prefix.occurrences(Prefix.append(prefix,Con{head,tail}),head) ==
    Nat.add(Prefix.occurrences(prefix,head),Prefix.occurrences(Con{head,tail},head)) : Nat} =
    Prefix.occurrences_append(prefix,Con{head,tail},head)
  +cons_count: {Prefix.occurrences(Con{head,tail},head) ==
    Nat.add(1n,Prefix.occurrences(tail,head)) : Nat} =
    occurrences_cons_self(head,tail)
  +expanded_count: {Nat.add(Prefix.occurrences(prefix,head),Prefix.occurrences(Con{head,tail},head)) ==
    Nat.add(Prefix.occurrences(prefix,head),Nat.add(1n,Prefix.occurrences(tail,head))) : Nat} =
    Equal.cong(Nat,Nat,n => Nat.add(Prefix.occurrences(prefix,head),n),
      Prefix.occurrences(Con{head,tail},head),
      Nat.add(1n,Prefix.occurrences(tail,head)),cons_count)
  +count_positive: {Nat.is_eq(Nat.add(Prefix.occurrences(prefix,head),
    Nat.add(1n,Prefix.occurrences(tail,head))),0n) == False{} : Bool} =
    nat_add_right_successor_nonzero(Prefix.occurrences(prefix,head),Prefix.occurrences(tail,head))
  Equal.trans(Bool,
    Nat.is_eq(Prefix.occurrences(Chess.bit_squares(64n,U64.is_zero(bb),bb,Nil{}),head),0n),
    Nat.is_eq(Nat.add(Prefix.occurrences(prefix,head),
      Nat.add(1n,Prefix.occurrences(tail,head))),0n),False{},
    Equal.cong(Nat,Bool,n => Nat.is_eq(n,0n),
      Prefix.occurrences(Chess.bit_squares(64n,U64.is_zero(bb),bb,Nil{}),head),
      Nat.add(Prefix.occurrences(prefix,head),Nat.add(1n,Prefix.occurrences(tail,head))),
      Equal.trans(Nat,
        Prefix.occurrences(Chess.bit_squares(64n,U64.is_zero(bb),bb,Nil{}),head),
        Prefix.occurrences(Prefix.append(prefix,Con{head,tail}),head),
        Nat.add(Prefix.occurrences(prefix,head),Nat.add(1n,Prefix.occurrences(tail,head))),
        whole_count,Equal.trans(Nat,
          Prefix.occurrences(Prefix.append(prefix,Con{head,tail}),head),
          Nat.add(Prefix.occurrences(prefix,head),Prefix.occurrences(Con{head,tail},head)),
          Nat.add(Prefix.occurrences(prefix,head),Nat.add(1n,Prefix.occurrences(tail,head))),
          split_count,expanded_count))),
    count_positive)

def frequency_positive_means_bit(+count: Nat,+present: Bool,
  +frequency: {count == Bool.pick(Nat,present,1n,0n) : Nat},
  +positive: {Nat.is_eq(count,0n) == False{} : Bool}) ->
  {present == True{} : Bool}:
  match present:
    case True{}: {==}
    case False{}:
      +picked_positive: {Nat.is_eq(Bool.pick(Nat,False{},1n,0n),0n) == False{} : Bool} =
        Equal.trans(Bool,
          Nat.is_eq(Bool.pick(Nat,False{},1n,0n),0n),
          Nat.is_eq(count,0n),False{},
          Equal.sym(Bool,Nat.is_eq(count,0n),
            Nat.is_eq(Bool.pick(Nat,False{},1n,0n),0n),
            Equal.cong(Nat,Bool,n => Nat.is_eq(n,0n),count,
              Bool.pick(Nat,False{},1n,0n),frequency)),
          positive)
      Old.false_true({False{} == True{} : Bool},
        Equal.sym(Bool,True{},False{},picked_positive))

def actual_list_bits_from_frequency(+bb: U64,+remaining: List<&2,U32>,
  +prefix: List<&2,U32>,
  +decomposition: {Chess.bit_squares(64n,U64.is_zero(bb),bb,Nil{}) ==
    Prefix.append(prefix,remaining) : List<&2,U32>},
  +remaining_below: {all_below64(remaining) == True{} : Bool}) ->
  {all_bits_set(remaining,bb) == True{} : Bool}:
  match remaining:
    case Nil{}: {==}
    case Con{head,tail}:
      +parts: {Bool.and(Nat.is_lt(U32.to_nat(head),64n),all_below64(tail)) == True{} : Bool} = remaining_below
      +head_below: {Nat.is_lt(U32.to_nat(head),64n) == True{} : Bool} =
        bool_and_true_left(Nat.is_lt(U32.to_nat(head),64n),all_below64(tail),parts)
      +tail_below: {all_below64(tail) == True{} : Bool} =
        bool_and_true_right(Nat.is_lt(U32.to_nat(head),64n),all_below64(tail),parts)
      +whole_occurs: {Nat.is_eq(Prefix.occurrences(Chess.bit_squares(64n,U64.is_zero(bb),bb,Nil{}),head),0n) == False{} : Bool} =
        actual_occurrences_cons_nonzero(bb,prefix,head,tail,decomposition)
      +frequency: {Prefix.occurrences(Chess.bit_squares(64n,U64.is_zero(bb),bb,Nil{}),head) ==
        Bool.pick(Nat,U64.test_bit(bb,U32.to_nat(head)),1n,0n) : Nat} =
        Frequency.actual_bit_squares_frequency(bb,head,head_below)
      +head_set: {U64.test_bit(bb,U32.to_nat(head)) == True{} : Bool} =
        frequency_positive_means_bit(
          Prefix.occurrences(Chess.bit_squares(64n,U64.is_zero(bb),bb,Nil{}),head),
          U64.test_bit(bb,U32.to_nat(head)),frequency,whole_occurs)
      +next_prefix = Prefix.append(prefix,head <> Nil{})
      +next_decomposition: {Chess.bit_squares(64n,U64.is_zero(bb),bb,Nil{}) ==
        Prefix.append(next_prefix,tail) : List<&2,U32>} =
        Equal.trans(List<&2,U32>,
          Chess.bit_squares(64n,U64.is_zero(bb),bb,Nil{}),
          Prefix.append(prefix,Con{head,tail}),Prefix.append(next_prefix,tail),
          decomposition,Prefix.append_cons_right(prefix,head,tail))
      +tail_set: {all_bits_set(tail,bb) == True{} : Bool} =
        actual_list_bits_from_frequency(bb,tail,next_prefix,next_decomposition,tail_below)
      bool_and_true(U64.test_bit(bb,U32.to_nat(head)),all_bits_set(tail,bb),head_set,tail_set)

def actual_bit_squares_all_set(+bb: U64) ->
  {all_bits_set(Chess.bit_squares(64n,U64.is_zero(bb),bb,Nil{}),bb) == True{} : Bool}:
  actual_list_bits_from_frequency(bb,Chess.bit_squares(64n,U64.is_zero(bb),bb,Nil{}),Nil{},
    {==},actual_bit_squares_below64(bb))

def actual_bit_squares_all_sound(+bb: U64) ->
  {Bool.and(
    all_below64(Chess.bit_squares(64n,U64.is_zero(bb),bb,Nil{})),
    all_bits_set(Chess.bit_squares(64n,U64.is_zero(bb),bb,Nil{}),bb)) == True{} : Bool}:
  bool_and_true(
    all_below64(Chess.bit_squares(64n,U64.is_zero(bb),bb,Nil{})),
    all_bits_set(Chess.bit_squares(64n,U64.is_zero(bb),bb,Nil{}),bb),
    actual_bit_squares_below64(bb),actual_bit_squares_all_set(bb))

def u64_bit_64_zero() -> {U64.bit(64n) == U64.zero() : U64}:
  {==}

def nat_not_lt_implies_reverse_le(+a: Nat,+b: Nat,
  +not_lt: {Nat.is_lt(a,b) == False{} : Bool}) ->
  {Nat.is_le(b,a) == True{} : Bool}:
  match a b:
    case 0n 0n: {==}
    case 0n 1n+q:
      Old.false_true({False{} == True{} : Bool},
        Equal.sym(Bool,True{},False{},not_lt))
    case 1n+p 0n: {==}
    case 1n+p 1n+q:
      Equal.trans(Bool,Nat.is_le(1n+q,1n+p),Nat.is_le(q,p),True{},
        Cardinality.le_succ_pair(q,p),nat_not_lt_implies_reverse_le(p,q,not_lt))

def nat_min_right_if_le(+a: Nat,+b: Nat,
  +b_le_a: {Nat.is_le(b,a) == True{} : Bool}) ->
  {Nat.min(a,b) == b : Nat}:
  match a b:
    case 0n 0n: {==}
    case 0n 1n+q:
      Old.false_true({0n == 1n+q : Nat},b_le_a)
    case 1n+p 0n: {==}
    case 1n+p 1n+q:
      +tail_le: {Nat.is_le(q,p) == True{} : Bool} =
        Equal.trans(Bool,Nat.is_le(q,p),Nat.is_le(1n+q,1n+p),True{},
          Equal.sym(Bool,Nat.is_le(1n+q,1n+p),Nat.is_le(q,p),
            Cardinality.le_succ_pair(q,p)),b_le_a)
      Equal.cong(Nat,Nat,n => 1n+n,Nat.min(p,q),q,
        nat_min_right_if_le(p,q,tail_le))

def nat_lt_le_impossible(+a: Nat,+b: Nat,
  +a_lt_b: {Nat.is_lt(a,b) == True{} : Bool},
  +b_le_a: {Nat.is_le(b,a) == True{} : Bool}) ->
  {False{} == True{} : Bool}:
  match a b:
    case 0n 0n: Old.false_true({False{} == True{} : Bool},a_lt_b)
    case 0n 1n+q: Old.false_true({False{} == True{} : Bool},b_le_a)
    case 1n+p 0n: Old.false_true({False{} == True{} : Bool},a_lt_b)
    case 1n+p 1n+q:
      +tail_ge: {Nat.is_le(q,p) == True{} : Bool} =
        Equal.trans(Bool,Nat.is_le(q,p),Nat.is_le(1n+q,1n+p),True{},
          Equal.sym(Bool,Nat.is_le(1n+q,1n+p),Nat.is_le(q,p),
            Cardinality.le_succ_pair(q,p)),b_le_a)
      nat_lt_le_impossible(p,q,a_lt_b,tail_ge)

def nat_eq_false_lt_ge(+a: Nat,+b: Nat,+limit: Nat,
  +a_below: {Nat.is_lt(a,limit) == True{} : Bool},
  +limit_below_b: {Nat.is_le(limit,b) == True{} : Bool}) ->
  {Nat.is_eq(a,b) == False{} : Bool}:
  match a b limit:
    case 0n _ 0n: Old.false_true({Nat.is_eq(a,b) == False{} : Bool},a_below)
    case 1n+p _ 0n: Old.false_true({Nat.is_eq(a,b) == False{} : Bool},a_below)
    case 0n 0n 1n+t: Old.false_true({Nat.is_eq(a,b) == False{} : Bool},limit_below_b)
    case 0n 1n+q 1n+t: {==}
    case 1n+p 0n 1n+t: {==}
    case 1n+p 1n+q 1n+t:
      +tail_ge: {Nat.is_le(t,q) == True{} : Bool} =
        Equal.trans(Bool,Nat.is_le(t,q),Nat.is_le(1n+t,1n+q),True{},
          Equal.sym(Bool,Nat.is_le(1n+t,1n+q),Nat.is_le(t,q),
            Cardinality.le_succ_pair(t,q)),limit_below_b)
      nat_eq_false_lt_ge(p,q,t,a_below,tail_ge)

def u32_eq_false_below_ge(+a: U32,+b: U32,
  +a_below: {Nat.is_lt(U32.to_nat(a),64n) == True{} : Bool},
  +sixty_four_le_b: {Nat.is_le(64n,U32.to_nat(b)) == True{} : Bool}) ->
  {U32.is_eq(a,b) == False{} : Bool}:
  Equal.trans(Bool,U32.is_eq(a,b),
    Nat.is_eq(U32.to_nat(a),U32.to_nat(b)),False{},
    IndexEquality.u32_eq_nat_eq(a,b),
    nat_eq_false_lt_ge(U32.to_nat(a),U32.to_nat(b),64n,a_below,sixty_four_le_b))

def occurrences_outside_zero(+xs: List<&2,U32>,+query: U32,
  +xs_below: {all_below64(xs) == True{} : Bool},
  +sixty_four_le_query: {Nat.is_le(64n,U32.to_nat(query)) == True{} : Bool}) ->
  {Prefix.occurrences(xs,query) == 0n : Nat}:
  match xs:
    case Nil{}: {==}
    case Con{head,tail}:
      +parts: {Bool.and(Nat.is_lt(U32.to_nat(head),64n),all_below64(tail)) == True{} : Bool} = xs_below
      +head_below: {Nat.is_lt(U32.to_nat(head),64n) == True{} : Bool} =
        bool_and_true_left(Nat.is_lt(U32.to_nat(head),64n),all_below64(tail),parts)
      +tail_below: {all_below64(tail) == True{} : Bool} =
        bool_and_true_right(Nat.is_lt(U32.to_nat(head),64n),all_below64(tail),parts)
      +head_not_query: {U32.is_eq(head,query) == False{} : Bool} =
        u32_eq_false_below_ge(head,query,head_below,sixty_four_le_query)
      +head_pick_zero: {Bool.pick(Nat,U32.is_eq(head,query),1n,0n) == 0n : Nat} =
        Equal.cong(Bool,Nat,b => Bool.pick(Nat,b,1n,0n),
          U32.is_eq(head,query),False{},head_not_query)
      +tail_zero: {Prefix.occurrences(tail,query) == 0n : Nat} =
        occurrences_outside_zero(tail,query,tail_below,sixty_four_le_query)
      Equal.trans(Nat,
        Prefix.occurrences(Con{head,tail},query),
        Nat.add(Bool.pick(Nat,U32.is_eq(head,query),1n,0n),Prefix.occurrences(tail,query)),
        0n,{==},
        Equal.trans(Nat,
          Nat.add(Bool.pick(Nat,U32.is_eq(head,query),1n,0n),Prefix.occurrences(tail,query)),
          Nat.add(0n,Prefix.occurrences(tail,query)),0n,
          Equal.cong(Nat,Nat,n => Nat.add(n,Prefix.occurrences(tail,query)),
            Bool.pick(Nat,U32.is_eq(head,query),1n,0n),0n,head_pick_zero),
          Equal.trans(Nat,
            Nat.add(0n,Prefix.occurrences(tail,query)),Nat.add(0n,0n),0n,
            Equal.cong(Nat,Nat,n => Nat.add(0n,n),Prefix.occurrences(tail,query),0n,tail_zero),
            {==})))

def u64_bit_outside(+index: Nat,
  +sixty_four_le_index: {Nat.is_le(64n,index) == True{} : Bool}) ->
  {U64.bit(index) == U64.zero() : U64}:
  Equal.trans(U64,U64.bit(index),U64.bit(64n),U64.zero(),
    Equal.cong(Nat,U64,n => U64.shln.go(n,U64.one()),
      Nat.min(index,64n),64n,
      nat_min_right_if_le(index,64n,sixty_four_le_index)),
    u64_bit_64_zero())

def u64_and_zero_right(+bb: U64) -> {U64.and(bb,U64.zero()) == U64.zero() : U64}:
  +word_and: {Word.and(64n,U64.to_word(bb),Word.zero(64n)) == Word.zero(64n) : Word(64n)} =
    TestBit.and_zero_right(64n,U64.to_word(bb))
  +zero_u64_word: {U64.to_word(U64.zero()) == Word.zero(64n) : Word(64n)} = Ctz.zero_word()
  +word_and_actual: {U64.to_word(U64.and(bb,U64.zero())) ==
    Word.and(64n,U64.to_word(bb),U64.to_word(U64.zero())) : Word(64n)} = Step.and_word(bb,U64.zero())
  +word_and_zero: {Word.and(64n,U64.to_word(bb),U64.to_word(U64.zero())) ==
    Word.and(64n,U64.to_word(bb),Word.zero(64n)) : Word(64n)} =
    Equal.cong(Word(64n),Word(64n),w => Word.and(64n,U64.to_word(bb),w),
      U64.to_word(U64.zero()),Word.zero(64n),zero_u64_word)
  +and_word_as_u64: {U64.and(bb,U64.zero()) ==
    U64.from_word(Word.and(64n,U64.to_word(bb),U64.to_word(U64.zero()))) : U64} =
    Equal.trans(U64,U64.and(bb,U64.zero()),
      U64.from_word(U64.to_word(U64.and(bb,U64.zero()))),
      U64.from_word(Word.and(64n,U64.to_word(bb),U64.to_word(U64.zero()))),
      Equal.sym(U64,U64.from_word(U64.to_word(U64.and(bb,U64.zero()))),
        U64.and(bb,U64.zero()),Old.from_to(U64.and(bb,U64.zero()))),
      Equal.cong(Word(64n),U64,w => U64.from_word(w),
        U64.to_word(U64.and(bb,U64.zero())),
        Word.and(64n,U64.to_word(bb),U64.to_word(U64.zero())),word_and_actual))
  +word_and_zero_as_u64: {
    U64.from_word(Word.and(64n,U64.to_word(bb),U64.to_word(U64.zero()))) ==
    U64.from_word(Word.and(64n,U64.to_word(bb),Word.zero(64n))) : U64} =
    Equal.cong(Word(64n),U64,w => U64.from_word(w),
      Word.and(64n,U64.to_word(bb),U64.to_word(U64.zero())),
      Word.and(64n,U64.to_word(bb),Word.zero(64n)),word_and_zero)
  +last_zero: {U64.from_word(Word.zero(64n)) == U64.zero() : U64} =
    Equal.trans(U64,U64.from_word(Word.zero(64n)),
      U64.from_word(U64.to_word(U64.zero())),U64.zero(),
      Equal.cong(Word(64n),U64,w => U64.from_word(w),Word.zero(64n),
        U64.to_word(U64.zero()),Equal.sym(Word(64n),U64.to_word(U64.zero()),Word.zero(64n),zero_u64_word)),
      Old.from_to(U64.zero()))
  Equal.trans(U64,U64.and(bb,U64.zero()),
    U64.from_word(Word.and(64n,U64.to_word(bb),U64.to_word(U64.zero()))),
    U64.zero(),and_word_as_u64,
    Equal.trans(U64,
      U64.from_word(Word.and(64n,U64.to_word(bb),U64.to_word(U64.zero()))),
      U64.from_word(Word.and(64n,U64.to_word(bb),Word.zero(64n))),
      U64.zero(),word_and_zero_as_u64,
      Equal.trans(U64,U64.from_word(Word.and(64n,U64.to_word(bb),Word.zero(64n))),
        U64.from_word(Word.zero(64n)),U64.zero(),
        Equal.cong(Word(64n),U64,w => U64.from_word(w),
          Word.and(64n,U64.to_word(bb),Word.zero(64n)),Word.zero(64n),word_and),
        last_zero)))

def u64_test_bit_outside(+bb: U64,+index: Nat,
  +sixty_four_le_index: {Nat.is_le(64n,index) == True{} : Bool}) ->
  {U64.test_bit(bb,index) == False{} : Bool}:
  +and_zero: {U64.and(bb,U64.zero()) == U64.zero() : U64} = u64_and_zero_right(bb)
  Equal.trans(Bool,U64.test_bit(bb,index),
    U64.is_ne(U64.and(bb,U64.zero()),U64.zero()),
    False{},
    Equal.trans(Bool,U64.test_bit(bb,index),
      U64.is_ne(U64.and(bb,U64.bit(index)),U64.zero()),
      U64.is_ne(U64.and(bb,U64.zero()),U64.zero()),
      {==},Equal.cong(U64,Bool,x => U64.is_ne(U64.and(bb,x),U64.zero()),
        U64.bit(index),U64.zero(),u64_bit_outside(index,sixty_four_le_index))),
    Equal.trans(Bool,
      U64.is_ne(U64.and(bb,U64.zero()),U64.zero()),
      U64.is_ne(U64.zero(),U64.zero()),False{},
      Equal.cong(U64,Bool,x => U64.is_ne(x,U64.zero()),
        U64.and(bb,U64.zero()),U64.zero(),and_zero),
      {==}))

def actual_bit_squares_frequency_classified(+bb: U64,+query: U32,+below: Bool,
  +same: {below == Nat.is_lt(U32.to_nat(query),64n) : Bool}) ->
  {Prefix.occurrences(Chess.bit_squares(64n,U64.is_zero(bb),bb,Nil{}),query) ==
    Bool.pick(Nat,U64.test_bit(bb,U32.to_nat(query)),1n,0n) : Nat}:
  match below:
    case True{}:
      +bound: {Nat.is_lt(U32.to_nat(query),64n) == True{} : Bool} =
        Equal.trans(Bool,Nat.is_lt(U32.to_nat(query),64n),below,True{},
          Equal.sym(Bool,below,Nat.is_lt(U32.to_nat(query),64n),same),{==})
      Frequency.actual_bit_squares_frequency(bb,query,bound)
    case False{}:
      +not_below: {Nat.is_lt(U32.to_nat(query),64n) == False{} : Bool} =
        Equal.trans(Bool,Nat.is_lt(U32.to_nat(query),64n),below,False{},
          Equal.sym(Bool,below,Nat.is_lt(U32.to_nat(query),64n),same),{==})
      +sixty_four_le_query: {Nat.is_le(64n,U32.to_nat(query)) == True{} : Bool} =
        nat_not_lt_implies_reverse_le(U32.to_nat(query),64n,not_below)
      +whole_below: {all_below64(Chess.bit_squares(64n,U64.is_zero(bb),bb,Nil{})) == True{} : Bool} =
        actual_bit_squares_below64(bb)
      +occurrences_zero: {Prefix.occurrences(Chess.bit_squares(64n,U64.is_zero(bb),bb,Nil{}),query) == 0n : Nat} =
        occurrences_outside_zero(Chess.bit_squares(64n,U64.is_zero(bb),bb,Nil{}),
          query,whole_below,sixty_four_le_query)
      +bit_false: {U64.test_bit(bb,U32.to_nat(query)) == False{} : Bool} =
        u64_test_bit_outside(bb,U32.to_nat(query),sixty_four_le_query)
      Equal.trans(Nat,
        Prefix.occurrences(Chess.bit_squares(64n,U64.is_zero(bb),bb,Nil{}),query),
        0n,Bool.pick(Nat,U64.test_bit(bb,U32.to_nat(query)),1n,0n),
        occurrences_zero,
        Equal.sym(Nat,Bool.pick(Nat,U64.test_bit(bb,U32.to_nat(query)),1n,0n),0n,
          Equal.cong(Bool,Nat,b => Bool.pick(Nat,b,1n,0n),
            U64.test_bit(bb,U32.to_nat(query)),False{},bit_false)))

def actual_bit_squares_frequency_all_u32(+bb: U64,+query: U32) ->
  {Prefix.occurrences(Chess.bit_squares(64n,U64.is_zero(bb),bb,Nil{}),query) ==
    Bool.pick(Nat,U64.test_bit(bb,U32.to_nat(query)),1n,0n) : Nat}:
  actual_bit_squares_frequency_classified(bb,query,
    Nat.is_lt(U32.to_nat(query),64n),{==})
