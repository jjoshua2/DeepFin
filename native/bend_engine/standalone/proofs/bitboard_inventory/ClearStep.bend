# Actual U64.clear_lsb decreases Base Word.count by one exactly when nonzero.
import Base
import ../successor/Borrow.bend as Borrow
import ../successor/Step.bend as Step

def one(+n: Nat) -> Word(n):
  match n:
    case 0n: WNil{}
    case 1n+p: WCon{True{},Word.zero(p)}

def clear_model(+n: Nat,+w: Word(n)) -> Word(n):
  match n:
    case 0n: WNil{}
    case 1n+p:
      match w:
        case WCon{False{},t}: WCon{False{},clear_model(p,t)}
        case WCon{True{},t}: WCon{False{},t}

def and_self(+n: Nat,+w: Word(n)) -> {Word.and(n,w,w) == w : Word(n)}:
  match n:
    case 0n:
      match w:
        case WNil{}: {==}
    case 1n+p:
      match w:
        case WCon{False{},t}:
          Equal.cong(Word(p),Word(1n+p),x => WCon{False{},x},Word.and(p,t,t),t,and_self(p,t))
        case WCon{True{},t}:
          Equal.cong(Word(p),Word(1n+p),x => WCon{True{},x},Word.and(p,t,t),t,and_self(p,t))

def sub_one(+n: Nat,+w: Word(n)) -> {Word.sub(n,w,one(n)) == Borrow.dec(n,w) : Word(n)}:
  match n:
    case 0n:
      match w:
        case WNil{}: {==}
    case 1n+p: Borrow.sub_one(p,w)

def clear_matches(+n: Nat,+w: Word(n)) ->
  {Word.and(n,w,Word.sub(n,w,one(n))) == clear_model(n,w) : Word(n)}:
  match n:
    case 0n:
      match w:
        case WNil{}: {==}
    case 1n+p:
      match w:
        case WCon{False{},t}:
          Equal.trans(Word(1n+p),
            Word.and(1n+p,w,Word.sub(1n+p,w,one(1n+p))),
            WCon{False{},Word.and(p,t,Word.sub(p,t,one(p)))},
            clear_model(1n+p,w),
            Equal.trans(Word(1n+p),
              Word.and(1n+p,w,Word.sub(1n+p,w,one(1n+p))),
              WCon{False{},Word.and(p,t,Borrow.dec(p,t))},
              WCon{False{},Word.and(p,t,Word.sub(p,t,one(p)))},
              Equal.cong(Word(1n+p),Word(1n+p),x => Word.and(1n+p,w,x),
                Word.sub(1n+p,w,one(1n+p)),Borrow.dec(1n+p,w),sub_one(1n+p,w)),
              Equal.cong(Word(p),Word(1n+p),x => WCon{False{},x},
                Word.and(p,t,Borrow.dec(p,t)),Word.and(p,t,Word.sub(p,t,one(p))),
                Equal.cong(Word(p),Word(p),x => Word.and(p,t,x),Borrow.dec(p,t),Word.sub(p,t,one(p)),
                  Equal.sym(Word(p),Word.sub(p,t,one(p)),Borrow.dec(p,t),sub_one(p,t))))),
            Equal.cong(Word(p),Word(1n+p),x => WCon{False{},x},
              Word.and(p,t,Word.sub(p,t,one(p))),clear_model(p,t),clear_matches(p,t)))
        case WCon{True{},t}:
          Equal.trans(Word(1n+p),
            Word.and(1n+p,w,Word.sub(1n+p,w,one(1n+p))),
            WCon{False{},Word.and(p,t,t)},clear_model(1n+p,w),
            Equal.cong(Word(1n+p),Word(1n+p),x => Word.and(1n+p,w,x),
              Word.sub(1n+p,w,one(1n+p)),Borrow.dec(1n+p,w),sub_one(1n+p,w)),
            Equal.cong(Word(p),Word(1n+p),x => WCon{False{},x},Word.and(p,t,t),t,and_self(p,t)))

def is_zero_word(+n: Nat,+w: Word(n)) -> Bool:
  match n:
    case 0n: True{}
    case 1n+p:
      match w:
        case WCon{False{},t}: is_zero_word(p,t)
        case WCon{True{},t}: False{}

def count_delta(+n: Nat,+w: Word(n)) ->
  {Word.count(n,w) == Nat.add(Bool.pick(Nat,is_zero_word(n,w),0n,1n),Word.count(n,clear_model(n,w))) : Nat}:
  match n:
    case 0n:
      match w:
        case WNil{}: {==}
    case 1n+p:
      match w:
        case WCon{False{},t}: count_delta(p,t)
        case WCon{True{},t}: {==}


def one_word() ->
  {U64.to_word(U64.one()) == one(64n) : Word(64n)}:
  {==}

def sub_word(+a: U64) ->
  {Word.sub(64n,U64.to_word(a),one(64n)) == U64.to_word(U64.sub(a,U64.one())) : Word(64n)}:
  Equal.trans(Word(64n),
    Word.sub(64n,U64.to_word(a),one(64n)),
    Word.sub(64n,U64.to_word(a),U64.to_word(U64.one())),
    U64.to_word(U64.sub(a,U64.one())),
    Equal.cong(Word(64n),Word(64n),x => Word.sub(64n,U64.to_word(a),x),
      one(64n),U64.to_word(U64.one()),Equal.sym(Word(64n),U64.to_word(U64.one()),one(64n),one_word())),
    Equal.sym(Word(64n),U64.to_word(U64.sub(a,U64.one())),
      Word.sub(64n,U64.to_word(a),U64.to_word(U64.one())),Borrow.u64_sub(a,U64.one())))

def clear_to_and(+a: U64) ->
  {U64.to_word(U64.clear_lsb(a)) == Word.and(64n,U64.to_word(a),U64.to_word(U64.sub(a,U64.one()))) : Word(64n)}:
  Step.and_word(a,U64.sub(a,U64.one()))

def clear_actual(+a: U64) ->
  {U64.to_word(U64.clear_lsb(a)) == clear_model(64n,U64.to_word(a)) : Word(64n)}:
  Equal.trans(Word(64n),
    U64.to_word(U64.clear_lsb(a)),
    Word.and(64n,U64.to_word(a),Word.sub(64n,U64.to_word(a),one(64n))),
    clear_model(64n,U64.to_word(a)),
    Equal.trans(Word(64n),
      U64.to_word(U64.clear_lsb(a)),
      Word.and(64n,U64.to_word(a),U64.to_word(U64.sub(a,U64.one()))),
      Word.and(64n,U64.to_word(a),Word.sub(64n,U64.to_word(a),one(64n))),
      clear_to_and(a),
      Equal.cong(Word(64n),Word(64n),x => Word.and(64n,U64.to_word(a),x),
        U64.to_word(U64.sub(a,U64.one())),Word.sub(64n,U64.to_word(a),one(64n)),Equal.sym(Word(64n),
          Word.sub(64n,U64.to_word(a),one(64n)),U64.to_word(U64.sub(a,U64.one())),sub_word(a)))),
    clear_matches(64n,U64.to_word(a)))

def clear_lsb_count_step(+a: U64) ->
  {Word.count(64n,U64.to_word(a)) == Nat.add(Bool.pick(Nat,is_zero_word(64n,U64.to_word(a)),0n,1n),
    Word.count(64n,U64.to_word(U64.clear_lsb(a)))) : Nat}:
  Equal.trans(Nat,
    Word.count(64n,U64.to_word(a)),
    Nat.add(Bool.pick(Nat,is_zero_word(64n,U64.to_word(a)),0n,1n),Word.count(64n,clear_model(64n,U64.to_word(a)))),
    Nat.add(Bool.pick(Nat,is_zero_word(64n,U64.to_word(a)),0n,1n),Word.count(64n,U64.to_word(U64.clear_lsb(a)))),
    count_delta(64n,U64.to_word(a)),
    Equal.cong(Word(64n),Nat,w => Nat.add(Bool.pick(Nat,is_zero_word(64n,U64.to_word(a)),0n,1n),Word.count(64n,w)),
      clear_model(64n,U64.to_word(a)),U64.to_word(U64.clear_lsb(a)),Equal.sym(Word(64n),U64.to_word(U64.clear_lsb(a)),
        clear_model(64n,U64.to_word(a)),clear_actual(a))))


def clear_lsb_popcount_step(+a: U64) ->
  {U64.popcount(a) == U32.from_nat(Nat.add(Bool.pick(Nat,is_zero_word(64n,U64.to_word(a)),0n,1n),
    Word.count(64n,U64.to_word(U64.clear_lsb(a))))) : U32}:
  Equal.trans(U32,
    U64.popcount(a),
    U32.from_nat(Word.count(64n,U64.to_word(a))),
    U32.from_nat(Nat.add(Bool.pick(Nat,is_zero_word(64n,U64.to_word(a)),0n,1n),
      Word.count(64n,U64.to_word(U64.clear_lsb(a))))),
    {==},
    Equal.cong(Nat,U32,n => U32.from_nat(n),Word.count(64n,U64.to_word(a)),
      Nat.add(Bool.pick(Nat,is_zero_word(64n,U64.to_word(a)),0n,1n),
        Word.count(64n,U64.to_word(U64.clear_lsb(a)))),clear_lsb_count_step(a)))
