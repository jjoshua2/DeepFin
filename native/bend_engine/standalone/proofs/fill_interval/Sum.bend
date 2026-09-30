# Mathematical value of actual addition, under its checked outgoing-carry bound.
import Base
import ../relative/Add.bend as Add
import ../prefix/Nats.bend as Nats
import ../u64/Words.bend as Old

def distribute(+a: Nat,+b: Nat) -> {Nat.double(Nat.add(a,b)) == Nat.add(Nat.double(a),Nat.double(b)) : Nat}:
  match a:
    case 0n: {==}
    case 1n+p:
      Equal.cong(Nat,Nat,x => 2n+x,Nat.double(Nat.add(p,b)),Nat.add(Nat.double(p),Nat.double(b)),distribute(p,b))

def right_one(+x: Nat,+y: Nat) -> {1n+Nat.double(Nat.add(x,y)) == Nat.add(Nat.double(x),1n+Nat.double(y)) : Nat}:
  Equal.trans(Nat,1n+Nat.double(Nat.add(x,y)),1n+Nat.add(Nat.double(x),Nat.double(y)),Nat.add(Nat.double(x),1n+Nat.double(y)),
    Equal.cong(Nat,Nat,z => 1n+z,Nat.double(Nat.add(x,y)),Nat.add(Nat.double(x),Nat.double(y)),distribute(x,y)),
    Equal.sym(Nat,Nat.add(Nat.double(x),1n+Nat.double(y)),1n+Nat.add(Nat.double(x),Nat.double(y)),Nats.add_suc(Nat.double(x),Nat.double(y))))

def digits(+a: Bool,+b: Bool,+c: Bool,+x: Nat,+y: Nat) ->
  {Nat.add(Add.bit(Add.sum_bit(a,b,c)),Nat.double(Nat.add(Add.bit(Add.carry_bit(a,b,c)),Nat.add(x,y)))) ==
    Nat.add(Add.bit(c),Nat.add(Nat.add(Add.bit(a),Nat.double(x)),Nat.add(Add.bit(b),Nat.double(y)))) : Nat}:
  match a b c:
    case False{} False{} False{}: distribute(x,y)
    case False{} False{} True{}:
      Equal.cong(Nat,Nat,z => 1n+z,Nat.double(Nat.add(x,y)),Nat.add(Nat.double(x),Nat.double(y)),distribute(x,y))
    case False{} True{} False{}:
      Equal.trans(Nat,1n+Nat.double(Nat.add(x,y)),1n+Nat.add(Nat.double(x),Nat.double(y)),Nat.add(Nat.double(x),1n+Nat.double(y)),
        Equal.cong(Nat,Nat,z => 1n+z,Nat.double(Nat.add(x,y)),Nat.add(Nat.double(x),Nat.double(y)),distribute(x,y)),
        Equal.sym(Nat,Nat.add(Nat.double(x),1n+Nat.double(y)),1n+Nat.add(Nat.double(x),Nat.double(y)),Nats.add_suc(Nat.double(x),Nat.double(y))))
    case False{} True{} True{}:
      Equal.cong(Nat,Nat,z => 1n+z,1n+Nat.double(Nat.add(x,y)),Nat.add(Nat.double(x),1n+Nat.double(y)),right_one(x,y))
    case True{} False{} False{}:
      Equal.cong(Nat,Nat,z => 1n+z,Nat.double(Nat.add(x,y)),Nat.add(Nat.double(x),Nat.double(y)),distribute(x,y))
    case True{} False{} True{}:
      Equal.cong(Nat,Nat,z => 2n+z,Nat.double(Nat.add(x,y)),Nat.add(Nat.double(x),Nat.double(y)),distribute(x,y))
    case True{} True{} False{}:
      Equal.cong(Nat,Nat,z => 1n+z,1n+Nat.double(Nat.add(x,y)),Nat.add(Nat.double(x),1n+Nat.double(y)),right_one(x,y))
    case True{} True{} True{}:
      Equal.cong(Nat,Nat,z => 2n+z,1n+Nat.double(Nat.add(x,y)),Nat.add(Nat.double(x),1n+Nat.double(y)),right_one(x,y))

def word(+n: Nat,+a: Word(n),+b: Word(n),+c: Bool,
  e: {Add.carry(n,a,b,c) == False{} : Bool}) ->
  {Word.to_nat(n,Word.adc(n,a,b,False{},c)) == Nat.add(Add.bit(c),Nat.add(Word.to_nat(n,a),Word.to_nat(n,b))) : Nat}:
  match n:
    case 0n:
      match c:
        case False{}: {==}
        case True{}: Old.false_true({0n == 1n : Nat},Equal.sym(Bool,True{},False{},e))
    case 1n+p:
      match a b c:
        case WCon{False{},at} WCon{False{},bt} False{}:
          Equal.trans(Nat,Word.to_nat(1n+p,Word.adc(1n+p,WCon{False{},at},WCon{False{},bt},False{},False{})),
            Nat.add(Add.bit(False{}),Nat.double(Nat.add(Add.bit(False{}),Nat.add(Word.to_nat(p,at),Word.to_nat(p,bt))))),Nat.add(Add.bit(False{}),Nat.add(Word.to_nat(1n+p,WCon{False{},at}),Word.to_nat(1n+p,WCon{False{},bt}))),
            Equal.cong(Nat,Nat,z => Nat.add(Add.bit(False{}),Nat.double(z)),
              Word.to_nat(p,Word.adc(p,at,bt,False{},False{})),Nat.add(Add.bit(False{}),Nat.add(Word.to_nat(p,at),Word.to_nat(p,bt))),word(p,at,bt,False{},e)),
            digits(False{},False{},False{},Word.to_nat(p,at),Word.to_nat(p,bt)))
        case WCon{False{},at} WCon{False{},bt} True{}:
          Equal.trans(Nat,Word.to_nat(1n+p,Word.adc(1n+p,WCon{False{},at},WCon{False{},bt},False{},True{})),
            Nat.add(Add.bit(True{}),Nat.double(Nat.add(Add.bit(False{}),Nat.add(Word.to_nat(p,at),Word.to_nat(p,bt))))),Nat.add(Add.bit(True{}),Nat.add(Word.to_nat(1n+p,WCon{False{},at}),Word.to_nat(1n+p,WCon{False{},bt}))),
            Equal.cong(Nat,Nat,z => Nat.add(Add.bit(True{}),Nat.double(z)),
              Word.to_nat(p,Word.adc(p,at,bt,False{},False{})),Nat.add(Add.bit(False{}),Nat.add(Word.to_nat(p,at),Word.to_nat(p,bt))),word(p,at,bt,False{},e)),
            digits(False{},False{},True{},Word.to_nat(p,at),Word.to_nat(p,bt)))
        case WCon{False{},at} WCon{True{},bt} False{}:
          Equal.trans(Nat,Word.to_nat(1n+p,Word.adc(1n+p,WCon{False{},at},WCon{True{},bt},False{},False{})),
            Nat.add(Add.bit(True{}),Nat.double(Nat.add(Add.bit(False{}),Nat.add(Word.to_nat(p,at),Word.to_nat(p,bt))))),Nat.add(Add.bit(False{}),Nat.add(Word.to_nat(1n+p,WCon{False{},at}),Word.to_nat(1n+p,WCon{True{},bt}))),
            Equal.cong(Nat,Nat,z => Nat.add(Add.bit(True{}),Nat.double(z)),
              Word.to_nat(p,Word.adc(p,at,bt,False{},False{})),Nat.add(Add.bit(False{}),Nat.add(Word.to_nat(p,at),Word.to_nat(p,bt))),word(p,at,bt,False{},e)),
            digits(False{},True{},False{},Word.to_nat(p,at),Word.to_nat(p,bt)))
        case WCon{False{},at} WCon{True{},bt} True{}:
          Equal.trans(Nat,Word.to_nat(1n+p,Word.adc(1n+p,WCon{False{},at},WCon{True{},bt},False{},True{})),
            Nat.add(Add.bit(False{}),Nat.double(Nat.add(Add.bit(True{}),Nat.add(Word.to_nat(p,at),Word.to_nat(p,bt))))),Nat.add(Add.bit(True{}),Nat.add(Word.to_nat(1n+p,WCon{False{},at}),Word.to_nat(1n+p,WCon{True{},bt}))),
            Equal.cong(Nat,Nat,z => Nat.add(Add.bit(False{}),Nat.double(z)),
              Word.to_nat(p,Word.adc(p,at,bt,False{},True{})),Nat.add(Add.bit(True{}),Nat.add(Word.to_nat(p,at),Word.to_nat(p,bt))),word(p,at,bt,True{},e)),
            digits(False{},True{},True{},Word.to_nat(p,at),Word.to_nat(p,bt)))
        case WCon{True{},at} WCon{False{},bt} False{}:
          Equal.trans(Nat,Word.to_nat(1n+p,Word.adc(1n+p,WCon{True{},at},WCon{False{},bt},False{},False{})),
            Nat.add(Add.bit(True{}),Nat.double(Nat.add(Add.bit(False{}),Nat.add(Word.to_nat(p,at),Word.to_nat(p,bt))))),Nat.add(Add.bit(False{}),Nat.add(Word.to_nat(1n+p,WCon{True{},at}),Word.to_nat(1n+p,WCon{False{},bt}))),
            Equal.cong(Nat,Nat,z => Nat.add(Add.bit(True{}),Nat.double(z)),
              Word.to_nat(p,Word.adc(p,at,bt,False{},False{})),Nat.add(Add.bit(False{}),Nat.add(Word.to_nat(p,at),Word.to_nat(p,bt))),word(p,at,bt,False{},e)),
            digits(True{},False{},False{},Word.to_nat(p,at),Word.to_nat(p,bt)))
        case WCon{True{},at} WCon{False{},bt} True{}:
          Equal.trans(Nat,Word.to_nat(1n+p,Word.adc(1n+p,WCon{True{},at},WCon{False{},bt},False{},True{})),
            Nat.add(Add.bit(False{}),Nat.double(Nat.add(Add.bit(True{}),Nat.add(Word.to_nat(p,at),Word.to_nat(p,bt))))),Nat.add(Add.bit(True{}),Nat.add(Word.to_nat(1n+p,WCon{True{},at}),Word.to_nat(1n+p,WCon{False{},bt}))),
            Equal.cong(Nat,Nat,z => Nat.add(Add.bit(False{}),Nat.double(z)),
              Word.to_nat(p,Word.adc(p,at,bt,False{},True{})),Nat.add(Add.bit(True{}),Nat.add(Word.to_nat(p,at),Word.to_nat(p,bt))),word(p,at,bt,True{},e)),
            digits(True{},False{},True{},Word.to_nat(p,at),Word.to_nat(p,bt)))
        case WCon{True{},at} WCon{True{},bt} False{}:
          Equal.trans(Nat,Word.to_nat(1n+p,Word.adc(1n+p,WCon{True{},at},WCon{True{},bt},False{},False{})),
            Nat.add(Add.bit(False{}),Nat.double(Nat.add(Add.bit(True{}),Nat.add(Word.to_nat(p,at),Word.to_nat(p,bt))))),Nat.add(Add.bit(False{}),Nat.add(Word.to_nat(1n+p,WCon{True{},at}),Word.to_nat(1n+p,WCon{True{},bt}))),
            Equal.cong(Nat,Nat,z => Nat.add(Add.bit(False{}),Nat.double(z)),
              Word.to_nat(p,Word.adc(p,at,bt,False{},True{})),Nat.add(Add.bit(True{}),Nat.add(Word.to_nat(p,at),Word.to_nat(p,bt))),word(p,at,bt,True{},e)),
            digits(True{},True{},False{},Word.to_nat(p,at),Word.to_nat(p,bt)))
        case WCon{True{},at} WCon{True{},bt} True{}:
          Equal.trans(Nat,Word.to_nat(1n+p,Word.adc(1n+p,WCon{True{},at},WCon{True{},bt},False{},True{})),
            Nat.add(Add.bit(True{}),Nat.double(Nat.add(Add.bit(True{}),Nat.add(Word.to_nat(p,at),Word.to_nat(p,bt))))),Nat.add(Add.bit(True{}),Nat.add(Word.to_nat(1n+p,WCon{True{},at}),Word.to_nat(1n+p,WCon{True{},bt}))),
            Equal.cong(Nat,Nat,z => Nat.add(Add.bit(True{}),Nat.double(z)),
              Word.to_nat(p,Word.adc(p,at,bt,False{},True{})),Nat.add(Add.bit(True{}),Nat.add(Word.to_nat(p,at),Word.to_nat(p,bt))),word(p,at,bt,True{},e)),
            digits(True{},True{},True{},Word.to_nat(p,at),Word.to_nat(p,bt)))

def u32(+a: U32,+b: U32,e: {Add.carry32(a,b) == False{} : Bool}) ->
  {U32.to_nat(U32.add(a,b)) == Nat.add(U32.to_nat(a),U32.to_nat(b)) : Nat}:
  match a b:
    case U32{x} U32{y}: word(32n,x,y,False{},e)
