import Base
import ../../../legal_probe/Chess.bend as Chess
import ../u64/Words.bend as Old
import ./MembershipIteration.bend as M
import ./PrefixDecomposition.bend as Prefix
import ./ActualClearBit.bend as ActualClear
import ./RemainingBudget.bend as Budget
import ./Cardinality.bend as Cardinality
import ./IndexEquality.bend as IndexEquality
import ../bitboard_ctz_structural/TestBitBridge.bend as CtzBit
import ../ray/Traversal.bend as NatEq

def nat_is_eq_sym(+a: Nat,+b: Nat) -> {Nat.is_eq(a,b) == Nat.is_eq(b,a) : Bool}:
  match a b:
    case 0n 0n: {==}
    case 0n 1n+q: {==}
    case 1n+p 0n: {==}
    case 1n+p 1n+q: nat_is_eq_sym(p,q)

def frequency_bits_step(+present: Bool,+selected: Bool,
  +selected_present: {Bool.and(present,selected) == selected : Bool}) ->
  {Nat.add(Bool.pick(Nat,Bool.and(present,Bool.not(selected)),1n,0n),
    Bool.pick(Nat,selected,1n,0n)) == Bool.pick(Nat,present,1n,0n) : Nat}:
  match present selected:
    case False{} False{}: {==}
    case False{} True{}: Old.false_true({Nat.add(0n,1n) == 0n : Nat},selected_present)
    case True{} False{}: {==}
    case True{} True{}: {==}

def selected_bit_present(+bb: U64,+index: Nat,+ctz_index: Nat,
  +selected: Bool,+present: Bool,
  +selected_eq: {selected == Nat.is_eq(index,ctz_index) : Bool},
  +present_actual: {present == U64.test_bit(bb,index) : Bool},
  +ctz_same: {ctz_index == U32.to_nat(U64.ctz(bb)) : Nat},
  +nonzero: {U64.is_zero(bb) == False{} : Bool}) ->
  {Bool.and(present,selected) == selected : Bool}:
  match selected:
    case False{}: Prefix.and_false_right(present)
    case True{}:
      +selected_nat: {Nat.is_eq(index,ctz_index) == True{} : Bool} =
        Equal.trans(Bool,Nat.is_eq(index,ctz_index),selected,True{},
          Equal.sym(Bool,selected,Nat.is_eq(index,ctz_index),selected_eq),{==})
      +same_index: {index == ctz_index : Nat} = NatEq.nat_eq(index,ctz_index,selected_nat)
      +ctz_bit: {U64.test_bit(bb,ctz_index) == True{} : Bool} =
        Equal.trans(Bool,U64.test_bit(bb,ctz_index),
          U64.test_bit(bb,U32.to_nat(U64.ctz(bb))),True{},
          Equal.cong(Nat,Bool,i => U64.test_bit(bb,i),ctz_index,U32.to_nat(U64.ctz(bb)),ctz_same),
          CtzBit.actual_ctz_selects_bit(bb,nonzero))
      +query_bit: {present == True{} : Bool} =
        Equal.trans(Bool,present,U64.test_bit(bb,ctz_index),True{},
          Equal.trans(Bool,present,U64.test_bit(bb,index),U64.test_bit(bb,ctz_index),
            present_actual,
            Equal.cong(Nat,Bool,i => U64.test_bit(bb,i),index,ctz_index,same_index)),ctz_bit)
      Equal.cong(Bool,Bool,b => Bool.and(b,True{}),present,True{},query_bit)

def emitted_frequency_go(+n: Nat,+empty: Bool,+bb: U64,+query: U32,
  +bound: {Nat.is_lt(U32.to_nat(query),64n) == True{} : Bool},
  +budget: {Nat.is_lt(Word.count(64n,U64.to_word(bb)),Nat.add(1n,n)) == True{} : Bool},
  +same: {empty == U64.is_zero(bb) : Bool}) ->
  {Prefix.occurrences(Prefix.emitted_prefix(n,empty,bb),query) ==
    Bool.pick(Nat,U64.test_bit(bb,U32.to_nat(query)),1n,0n) : Nat}:
  match n:
    case 0n:
      Equal.trans(Nat,
        Prefix.occurrences(Prefix.emitted_prefix(0n,U64.is_zero(bb),bb),query),
        0n,Bool.pick(Nat,U64.test_bit(bb,U32.to_nat(query)),1n,0n),
        {==},
        Equal.trans(Nat,0n,Bool.pick(Nat,False{},1n,0n),
          Bool.pick(Nat,U64.test_bit(bb,U32.to_nat(query)),1n,0n),
          {==},
          Equal.sym(Nat,Bool.pick(Nat,U64.test_bit(bb,U32.to_nat(query)),1n,0n),
            Bool.pick(Nat,False{},1n,0n),
          Equal.cong(Bool,Nat,b => Bool.pick(Nat,b,1n,0n),
            U64.test_bit(bb,U32.to_nat(query)),False{},
            M.count_zero_u64_testbit_false(bb,U32.to_nat(query),bound,
              Cardinality.count_lt_one_zero(Word.count(64n,U64.to_word(bb)),budget))))))
    case 1n+p:
      match empty:
        case True{}:
          +zero: {U64.is_zero(bb) == True{} : Bool} =
            Equal.trans(Bool,U64.is_zero(bb),empty,True{},Equal.sym(Bool,empty,U64.is_zero(bb),same),{==})
          Equal.trans(Nat,
            Prefix.occurrences(Prefix.emitted_prefix(1n+p,True{},bb),query),
            0n,Bool.pick(Nat,U64.test_bit(bb,U32.to_nat(query)),1n,0n),
            {==},
            Equal.trans(Nat,0n,Bool.pick(Nat,False{},1n,0n),
              Bool.pick(Nat,U64.test_bit(bb,U32.to_nat(query)),1n,0n),
              {==},
              Equal.sym(Nat,Bool.pick(Nat,U64.test_bit(bb,U32.to_nat(query)),1n,0n),
                Bool.pick(Nat,False{},1n,0n),
              Equal.cong(Bool,Nat,b => Bool.pick(Nat,b,1n,0n),
                U64.test_bit(bb,U32.to_nat(query)),False{},
                M.zero_u64_testbit_false(bb,U32.to_nat(query),bound,zero)))))
        case False{}:
          +nonzero: {U64.is_zero(bb) == False{} : Bool} =
            Equal.trans(Bool,U64.is_zero(bb),empty,False{},Equal.sym(Bool,empty,U64.is_zero(bb),same),{==})
          +rest = U64.clear_lsb(bb)
          +next_budget: {Nat.is_lt(Word.count(64n,U64.to_word(rest)),Nat.add(1n,p)) == True{} : Bool} =
            Budget.clear_count_budget(bb,1n+p,budget,nonzero)
          +tail = Prefix.emitted_prefix(p,U64.is_zero(rest),rest)
          +index = U32.to_nat(query)
          +ctz_index = U32.to_nat(U64.ctz(bb))
          +selected = Nat.is_eq(index,ctz_index)
          +head_eq = U32.is_eq(U64.ctz(bb),query)
          +present = U64.test_bit(bb,index)
          +tail_bit = U64.test_bit(rest,index)
          +tail_freq = Prefix.occurrences(tail,query)
          +head_freq = Bool.pick(Nat,head_eq,1n,0n)
          +tail_count: {tail_freq == Bool.pick(Nat,tail_bit,1n,0n) : Nat} =
            emitted_frequency_go(p,U64.is_zero(rest),rest,query,bound,next_budget,{==})
          +clear_bit: {tail_bit == Bool.and(present,Bool.not(selected)) : Bool} =
            ActualClear.actual_clear_lsb_bit(bb,index,bound,nonzero)
          +head_pick: {head_freq == Bool.pick(Nat,selected,1n,0n) : Nat} =
            Equal.cong(Bool,Nat,b => Bool.pick(Nat,b,1n,0n),head_eq,selected,
              Equal.trans(Bool,head_eq,Nat.is_eq(ctz_index,index),selected,
              IndexEquality.u32_eq_nat_eq(U64.ctz(bb),query),
                Equal.sym(Bool,selected,Nat.is_eq(ctz_index,index),nat_is_eq_sym(index,ctz_index))))
          +appended: {Prefix.occurrences(Prefix.append(tail,U64.ctz(bb) <> Nil{}),query) ==
            Nat.add(Prefix.occurrences(tail,query),head_freq) : Nat} =
            Equal.trans(Nat,
              Prefix.occurrences(Prefix.append(tail,U64.ctz(bb) <> Nil{}),query),
              Nat.add(Prefix.occurrences(tail,query),Prefix.occurrences(U64.ctz(bb) <> Nil{},query)),
              Nat.add(Prefix.occurrences(tail,query),head_freq),
              Prefix.occurrences_append(tail,U64.ctz(bb) <> Nil{},query),
              Equal.cong(Nat,Nat,x => Nat.add(Prefix.occurrences(tail,query),x),
                Prefix.occurrences(U64.ctz(bb) <> Nil{},query),head_freq,
                M.occurrence_one(U64.ctz(bb),query)))
          Equal.trans(Nat,
            Prefix.occurrences(Prefix.emitted_prefix(1n+p,False{},bb),query),
            Nat.add(Prefix.occurrences(tail,query),head_freq),
            Bool.pick(Nat,present,1n,0n),
            Equal.trans(Nat,
              Prefix.occurrences(Prefix.emitted_prefix(1n+p,False{},bb),query),
              Prefix.occurrences(Prefix.append(tail,U64.ctz(bb) <> Nil{}),query),
              Nat.add(Prefix.occurrences(tail,query),head_freq),{==},appended),
            Equal.trans(Nat,
              Nat.add(Prefix.occurrences(tail,query),head_freq),
              Nat.add(Bool.pick(Nat,tail_bit,1n,0n),head_freq),
              Bool.pick(Nat,present,1n,0n),
              Equal.cong(Nat,Nat,x => Nat.add(x,head_freq),Prefix.occurrences(tail,query),
                Bool.pick(Nat,tail_bit,1n,0n),tail_count),
              Equal.trans(Nat,
              Nat.add(Bool.pick(Nat,tail_bit,1n,0n),head_freq),
              Nat.add(Bool.pick(Nat,Bool.and(present,Bool.not(selected)),1n,0n),head_freq),
                Bool.pick(Nat,present,1n,0n),
                Equal.cong(Bool,Nat,x => Nat.add(Bool.pick(Nat,x,1n,0n),head_freq),
                  tail_bit,Bool.and(present,Bool.not(selected)),clear_bit),
                Equal.trans(Nat,
                  Nat.add(Bool.pick(Nat,Bool.and(present,Bool.not(selected)),1n,0n),head_freq),
                  Nat.add(Bool.pick(Nat,Bool.and(present,Bool.not(selected)),1n,0n),
                    Bool.pick(Nat,selected,1n,0n)),
                  Bool.pick(Nat,present,1n,0n),
                  Equal.cong(Nat,Nat,x => Nat.add(Bool.pick(Nat,Bool.and(present,Bool.not(selected)),1n,0n),x),
                    head_freq,Bool.pick(Nat,selected,1n,0n),head_pick),
                  frequency_bits_step(present,selected,
                    selected_bit_present(bb,index,ctz_index,selected,present,
                      {==},{==},{==},nonzero))))))

def emitted_frequency(+n: Nat,+bb: U64,+query: U32,
  +bound: {Nat.is_lt(U32.to_nat(query),64n) == True{} : Bool},
  +budget: {Nat.is_lt(Word.count(64n,U64.to_word(bb)),Nat.add(1n,n)) == True{} : Bool}) ->
  {Prefix.occurrences(Prefix.emitted_prefix(n,U64.is_zero(bb),bb),query) ==
    Bool.pick(Nat,U64.test_bit(bb,U32.to_nat(query)),1n,0n) : Nat}:
  emitted_frequency_go(n,U64.is_zero(bb),bb,query,bound,budget,{==})

def actual_bit_squares_frequency(+bb: U64,+query: U32,
  +bound: {Nat.is_lt(U32.to_nat(query),64n) == True{} : Bool}) ->
  {Prefix.occurrences(Chess.bit_squares(64n,U64.is_zero(bb),bb,Nil{}),query) ==
    Bool.pick(Nat,U64.test_bit(bb,U32.to_nat(query)),1n,0n) : Nat}:
  Equal.trans(Nat,
    Prefix.occurrences(Chess.bit_squares(64n,U64.is_zero(bb),bb,Nil{}),query),
    Nat.add(Prefix.occurrences(Prefix.emitted_prefix(64n,U64.is_zero(bb),bb),query),0n),
    Bool.pick(Nat,U64.test_bit(bb,U32.to_nat(query)),1n,0n),
    Prefix.bit_squares_occurrences_acc(64n,U64.is_zero(bb),bb,Nil{},query),
    Equal.trans(Nat,
      Nat.add(Prefix.occurrences(Prefix.emitted_prefix(64n,U64.is_zero(bb),bb),query),0n),
      Prefix.occurrences(Prefix.emitted_prefix(64n,U64.is_zero(bb),bb),query),
      Bool.pick(Nat,U64.test_bit(bb,U32.to_nat(query)),1n,0n),
      Cardinality.nat_add_zero_right(
        Prefix.occurrences(Prefix.emitted_prefix(64n,U64.is_zero(bb),bb),query)),
      emitted_frequency(64n,bb,query,bound,
        Cardinality.count_lt_width_succ(64n,U64.to_word(bb)))))
