import Base
import ../ray/Algebra.bend as A
import ../occupancy_insertion/Lattice.bend as L

def disjoint(removed: U64,old_attack: U64) -> Type:
  {U64.and(removed,old_attack) == U64.zero() : U64}

def partition_bool(+removed: Bool,+bit: Bool) ->
  {Bool.or(Bool.and(Bool.not(removed),bit),Bool.and(removed,bit)) == bit : Bool}:
  match removed bit:
    case False{} False{}: {==}
    case False{} True{}: {==}
    case True{} False{}: {==}
    case True{} True{}: {==}

def partition_word(+n: Nat,+removed: Word(n),+bit: Word(n)) ->
  {Word.or(n,Word.and(n,Word.not(n,removed),bit),Word.and(n,removed,bit)) == bit : Word(n)}:
  match n:
    case 0n:
      match bit:
        case WNil{}: {==}
    case 1n+p:
      match removed bit:
        case WCon{rh,rt} WCon{bh,bt}:
          Equal.trans(Word(1n+p),
            WCon{Bool.or(Bool.and(Bool.not(rh),bh),Bool.and(rh,bh)),
              Word.or(p,Word.and(p,Word.not(p,rt),bt),Word.and(p,rt,bt))},
            WCon{bh,Word.or(p,Word.and(p,Word.not(p,rt),bt),Word.and(p,rt,bt))},
            WCon{bh,bt},
            Equal.cong(Bool,Word(1n+p),h => WCon{h,Word.or(p,Word.and(p,Word.not(p,rt),bt),Word.and(p,rt,bt))},
              Bool.or(Bool.and(Bool.not(rh),bh),Bool.and(rh,bh)),bh,partition_bool(rh,bh)),
            Equal.cong(Word(p),Word(1n+p),t => WCon{bh,t},
              Word.or(p,Word.and(p,Word.not(p,rt),bt),Word.and(p,rt,bt)),bt,partition_word(p,rt,bt)))

def partition(+removed: U64,+bit: U64) ->
  {U64.or(U64.and(U64.not(removed),bit),U64.and(removed,bit)) == bit : U64}:
  match removed bit:
    case U64{U32{rl},U32{rh}} U64{U32{bl},U32{bh}}:
      Equal.trans(U64,
        U64{U32{Word.or(32n,Word.and(32n,Word.not(32n,rl),bl),Word.and(32n,rl,bl))},
          U32{Word.or(32n,Word.and(32n,Word.not(32n,rh),bh),Word.and(32n,rh,bh))}},
        U64{U32{bl},U32{Word.or(32n,Word.and(32n,Word.not(32n,rh),bh),Word.and(32n,rh,bh))}},
        U64{U32{bl},U32{bh}},
        Equal.cong(Word(32n),U64,l => U64{U32{l},U32{Word.or(32n,Word.and(32n,Word.not(32n,rh),bh),Word.and(32n,rh,bh))}},
          Word.or(32n,Word.and(32n,Word.not(32n,rl),bl),Word.and(32n,rl,bl)),bl,partition_word(32n,rl,bl)),
        Equal.cong(Word(32n),U64,h => U64{U32{bl},U32{h}},
          Word.or(32n,Word.and(32n,Word.not(32n,rh),bh),Word.and(32n,rh,bh)),bh,partition_word(32n,rh,bh)))

def word_left_zero(+n: Nat,+b: Word(n)) ->
  {Word.and(n,Word.zero(n),b) == Word.zero(n) : Word(n)}:
  match n:
    case 0n: {==}
    case 1n+p:
      match b:
        case WCon{h,t}:
          Equal.cong(Word(p),Word(1n+p),tail => WCon{False{},tail},
            Word.and(p,Word.zero(p),t),Word.zero(p),word_left_zero(p,t))

def left_zero(+b: U64) -> {U64.and(U64.zero(),b) == U64.zero() : U64}:
  match b:
    case U64{U32{lo},U32{hi}}:
      Equal.trans(U64,
        U64{U32{Word.and(32n,Word.zero(32n),lo)},U32{Word.and(32n,Word.zero(32n),hi)}},
        U64{0,U32{Word.and(32n,Word.zero(32n),hi)}},U64.zero(),
        Equal.cong(Word(32n),U64,l => U64{U32{l},U32{Word.and(32n,Word.zero(32n),hi)}},
          Word.and(32n,Word.zero(32n),lo),Word.zero(32n),word_left_zero(32n,lo)),
        Equal.cong(Word(32n),U64,h => U64{0,U32{h}},
          Word.and(32n,Word.zero(32n),hi),Word.zero(32n),word_left_zero(32n,hi)))

def right_member(+bit: U64,+tail: U64) -> L.subset(tail,U64.or(bit,tail)):
  Equal.trans(U64,U64.and(U64.or(bit,tail),tail),
    U64.and(U64.or(bit,tail),U64.and(tail,tail)),tail,
    Equal.cong(U64,U64,w => U64.and(U64.or(bit,tail),w),tail,U64.and(tail,tail),
      Equal.sym(U64,U64.and(tail,tail),tail,L.reflexive(tail))),
    Equal.trans(U64,U64.and(U64.or(bit,tail),U64.and(tail,tail)),U64.and(tail,tail),tail,
      A.extend_left(bit,tail,tail),L.reflexive(tail)))

def disjoint_subset(+removed: U64,+small: U64,+big: U64,
  included: L.subset(small,big),empty: disjoint(removed,big)) -> disjoint(removed,small):
  Equal.trans(U64,U64.and(removed,small),U64.and(removed,U64.and(big,small)),U64.zero(),
    Equal.cong(U64,U64,w => U64.and(removed,w),small,U64.and(big,small),
      Equal.sym(U64,U64.and(big,small),small,included)),
    Equal.trans(U64,U64.and(removed,U64.and(big,small)),U64.and(U64.and(removed,big),small),U64.zero(),
      Equal.sym(U64,U64.and(U64.and(removed,big),small),U64.and(removed,U64.and(big,small)),A.and_assoc(removed,big,small)),
      Equal.trans(U64,U64.and(U64.and(removed,big),small),U64.and(U64.zero(),small),U64.zero(),
        Equal.cong(U64,U64,w => U64.and(w,small),U64.and(removed,big),U64.zero(),empty),left_zero(small))))

def keep_complement(+removed: U64,+bit: U64,empty: disjoint(removed,bit)) ->
  {U64.and(U64.not(removed),bit) == bit : U64}:
  Equal.trans(U64,U64.and(U64.not(removed),bit),U64.or(U64.and(U64.not(removed),bit),U64.zero()),bit,
    Equal.sym(U64,U64.or(U64.and(U64.not(removed),bit),U64.zero()),U64.and(U64.not(removed),bit),
      A.zero(U64.and(U64.not(removed),bit))),
    Equal.trans(U64,U64.or(U64.and(U64.not(removed),bit),U64.zero()),
      U64.or(U64.and(U64.not(removed),bit),U64.and(removed,bit)),bit,
      Equal.cong(U64,U64,w => U64.or(U64.and(U64.not(removed),bit),w),U64.zero(),U64.and(removed,bit),
        Equal.sym(U64,U64.and(removed,bit),U64.zero(),empty)),partition(removed,bit)))
