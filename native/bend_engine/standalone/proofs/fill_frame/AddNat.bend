# Numeric interpretation of actual addition with a certified absent final carry.
import Base
import ../relative/Add.bend as Add
import ../prefix/Nats.bend as NatProof
import ../u64/Words.bend as Old

def double_add(+x: Nat,+y: Nat) ->
  {Nat.double(Nat.add(x,y)) == Nat.add(Nat.double(x),Nat.double(y)) : Nat}:
  match x:
    case 0n: {==}
    case 1n+p:
      Equal.cong(Nat,Nat,n=>2n+n,Nat.double(Nat.add(p,y)),Nat.add(Nat.double(p),Nat.double(y)),double_add(p,y))

def step(+a: Bool,+b: Bool,+c: Bool,+x: Nat,+y: Nat) ->
  {Nat.add(Add.bit(Add.sum_bit(a,b,c)),Nat.double(Nat.add(Add.bit(Add.carry_bit(a,b,c)),Nat.add(x,y)))) ==
    Nat.add(Add.bit(c),Nat.add(Nat.add(Add.bit(a),Nat.double(x)),Nat.add(Add.bit(b),Nat.double(y)))) : Nat}:
  match a b c:
    case False{} False{} False{}:
      Equal.cong(Nat,Nat,n=>0n+n,Nat.double(Nat.add(x,y)),Nat.add(Nat.double(x),Nat.double(y)),double_add(x,y))
    case False{} False{} True{}:
      Equal.cong(Nat,Nat,n=>1n+n,Nat.double(Nat.add(x,y)),Nat.add(Nat.double(x),Nat.double(y)),double_add(x,y))
    case False{} True{} False{}:
      Equal.trans(Nat,1n+Nat.double(Nat.add(x,y)),1n+Nat.add(Nat.double(x),Nat.double(y)),
        0n+Nat.add(Nat.double(x),1n+Nat.double(y)),
        Equal.cong(Nat,Nat,n=>1n+n,Nat.double(Nat.add(x,y)),Nat.add(Nat.double(x),Nat.double(y)),double_add(x,y)),
        Equal.sym(Nat,0n+Nat.add(Nat.double(x),1n+Nat.double(y)),1n+Nat.add(Nat.double(x),Nat.double(y)),
          Equal.cong(Nat,Nat,n=>0n+n,Nat.add(Nat.double(x),1n+Nat.double(y)),1n+Nat.add(Nat.double(x),Nat.double(y)),
            NatProof.add_suc(Nat.double(x),Nat.double(y)))))
    case False{} True{} True{}:
      Equal.trans(Nat,2n+Nat.double(Nat.add(x,y)),2n+Nat.add(Nat.double(x),Nat.double(y)),
        1n+Nat.add(Nat.double(x),1n+Nat.double(y)),
        Equal.cong(Nat,Nat,n=>2n+n,Nat.double(Nat.add(x,y)),Nat.add(Nat.double(x),Nat.double(y)),double_add(x,y)),
        Equal.sym(Nat,1n+Nat.add(Nat.double(x),1n+Nat.double(y)),2n+Nat.add(Nat.double(x),Nat.double(y)),
          Equal.cong(Nat,Nat,n=>1n+n,Nat.add(Nat.double(x),1n+Nat.double(y)),1n+Nat.add(Nat.double(x),Nat.double(y)),
            NatProof.add_suc(Nat.double(x),Nat.double(y)))))
    case True{} False{} False{}:
      Equal.cong(Nat,Nat,n=>1n+n,Nat.double(Nat.add(x,y)),Nat.add(Nat.double(x),Nat.double(y)),double_add(x,y))
    case True{} False{} True{}:
      Equal.cong(Nat,Nat,n=>2n+n,Nat.double(Nat.add(x,y)),Nat.add(Nat.double(x),Nat.double(y)),double_add(x,y))
    case True{} True{} False{}:
      Equal.trans(Nat,2n+Nat.double(Nat.add(x,y)),2n+Nat.add(Nat.double(x),Nat.double(y)),
        1n+Nat.add(Nat.double(x),1n+Nat.double(y)),
        Equal.cong(Nat,Nat,n=>2n+n,Nat.double(Nat.add(x,y)),Nat.add(Nat.double(x),Nat.double(y)),double_add(x,y)),
        Equal.sym(Nat,1n+Nat.add(Nat.double(x),1n+Nat.double(y)),2n+Nat.add(Nat.double(x),Nat.double(y)),
          Equal.cong(Nat,Nat,n=>1n+n,Nat.add(Nat.double(x),1n+Nat.double(y)),1n+Nat.add(Nat.double(x),Nat.double(y)),
            NatProof.add_suc(Nat.double(x),Nat.double(y)))))
    case True{} True{} True{}:
      Equal.trans(Nat,3n+Nat.double(Nat.add(x,y)),3n+Nat.add(Nat.double(x),Nat.double(y)),
        2n+Nat.add(Nat.double(x),1n+Nat.double(y)),
        Equal.cong(Nat,Nat,n=>3n+n,Nat.double(Nat.add(x,y)),Nat.add(Nat.double(x),Nat.double(y)),double_add(x,y)),
        Equal.sym(Nat,2n+Nat.add(Nat.double(x),1n+Nat.double(y)),3n+Nat.add(Nat.double(x),Nat.double(y)),
          Equal.cong(Nat,Nat,n=>2n+n,Nat.add(Nat.double(x),1n+Nat.double(y)),1n+Nat.add(Nat.double(x),Nat.double(y)),
            NatProof.add_suc(Nat.double(x),Nat.double(y)))))

def word(+n: Nat,+a: Word(n),+b: Word(n),+c: Bool,
  e: {Add.carry(n,a,b,c) == False{} : Bool}) ->
  {Word.to_nat(n,Word.adc(n,a,b,False{},c)) == Nat.add(Add.bit(c),Nat.add(Word.to_nat(n,a),Word.to_nat(n,b))) : Nat}:
  match n:
    case 0n:
      match c:
        case False{}: {==}
        case True{}:
          Old.false_true({0n == 1n : Nat},Equal.sym(Bool,True{},False{},e))
    case 1n+p:
      match a b c:
        case WCon{False{},x} WCon{False{},y} False{}:
          Equal.trans(Nat,Word.to_nat(1n+p,Word.adc(1n+p,WCon{False{},x},WCon{False{},y},False{},False{})),
            0n+Nat.double(Nat.add(Add.bit(False{}),Nat.add(Word.to_nat(p,x),Word.to_nat(p,y)))),
            Nat.add(Add.bit(False{}),Nat.add(Word.to_nat(1n+p,WCon{False{},x}),Word.to_nat(1n+p,WCon{False{},y}))),
            Equal.cong(Nat,Nat,t=>0n+Nat.double(t),Word.to_nat(p,Word.adc(p,x,y,False{},False{})),
              Nat.add(Add.bit(False{}),Nat.add(Word.to_nat(p,x),Word.to_nat(p,y))),word(p,x,y,False{},e)),
            step(False{},False{},False{},Word.to_nat(p,x),Word.to_nat(p,y)))
        case WCon{False{},x} WCon{False{},y} True{}:
          Equal.trans(Nat,Word.to_nat(1n+p,Word.adc(1n+p,WCon{False{},x},WCon{False{},y},False{},True{})),
            1n+Nat.double(Nat.add(Add.bit(False{}),Nat.add(Word.to_nat(p,x),Word.to_nat(p,y)))),
            Nat.add(Add.bit(True{}),Nat.add(Word.to_nat(1n+p,WCon{False{},x}),Word.to_nat(1n+p,WCon{False{},y}))),
            Equal.cong(Nat,Nat,t=>1n+Nat.double(t),Word.to_nat(p,Word.adc(p,x,y,False{},False{})),
              Nat.add(Add.bit(False{}),Nat.add(Word.to_nat(p,x),Word.to_nat(p,y))),word(p,x,y,False{},e)),
            step(False{},False{},True{},Word.to_nat(p,x),Word.to_nat(p,y)))
        case WCon{False{},x} WCon{True{},y} False{}:
          Equal.trans(Nat,Word.to_nat(1n+p,Word.adc(1n+p,WCon{False{},x},WCon{True{},y},False{},False{})),
            1n+Nat.double(Nat.add(Add.bit(False{}),Nat.add(Word.to_nat(p,x),Word.to_nat(p,y)))),
            Nat.add(Add.bit(False{}),Nat.add(Word.to_nat(1n+p,WCon{False{},x}),Word.to_nat(1n+p,WCon{True{},y}))),
            Equal.cong(Nat,Nat,t=>1n+Nat.double(t),Word.to_nat(p,Word.adc(p,x,y,False{},False{})),
              Nat.add(Add.bit(False{}),Nat.add(Word.to_nat(p,x),Word.to_nat(p,y))),word(p,x,y,False{},e)),
            step(False{},True{},False{},Word.to_nat(p,x),Word.to_nat(p,y)))
        case WCon{False{},x} WCon{True{},y} True{}:
          Equal.trans(Nat,Word.to_nat(1n+p,Word.adc(1n+p,WCon{False{},x},WCon{True{},y},False{},True{})),
            0n+Nat.double(Nat.add(Add.bit(True{}),Nat.add(Word.to_nat(p,x),Word.to_nat(p,y)))),
            Nat.add(Add.bit(True{}),Nat.add(Word.to_nat(1n+p,WCon{False{},x}),Word.to_nat(1n+p,WCon{True{},y}))),
            Equal.cong(Nat,Nat,t=>0n+Nat.double(t),Word.to_nat(p,Word.adc(p,x,y,False{},True{})),
              Nat.add(Add.bit(True{}),Nat.add(Word.to_nat(p,x),Word.to_nat(p,y))),word(p,x,y,True{},e)),
            step(False{},True{},True{},Word.to_nat(p,x),Word.to_nat(p,y)))
        case WCon{True{},x} WCon{False{},y} False{}:
          Equal.trans(Nat,Word.to_nat(1n+p,Word.adc(1n+p,WCon{True{},x},WCon{False{},y},False{},False{})),
            1n+Nat.double(Nat.add(Add.bit(False{}),Nat.add(Word.to_nat(p,x),Word.to_nat(p,y)))),
            Nat.add(Add.bit(False{}),Nat.add(Word.to_nat(1n+p,WCon{True{},x}),Word.to_nat(1n+p,WCon{False{},y}))),
            Equal.cong(Nat,Nat,t=>1n+Nat.double(t),Word.to_nat(p,Word.adc(p,x,y,False{},False{})),
              Nat.add(Add.bit(False{}),Nat.add(Word.to_nat(p,x),Word.to_nat(p,y))),word(p,x,y,False{},e)),
            step(True{},False{},False{},Word.to_nat(p,x),Word.to_nat(p,y)))
        case WCon{True{},x} WCon{False{},y} True{}:
          Equal.trans(Nat,Word.to_nat(1n+p,Word.adc(1n+p,WCon{True{},x},WCon{False{},y},False{},True{})),
            0n+Nat.double(Nat.add(Add.bit(True{}),Nat.add(Word.to_nat(p,x),Word.to_nat(p,y)))),
            Nat.add(Add.bit(True{}),Nat.add(Word.to_nat(1n+p,WCon{True{},x}),Word.to_nat(1n+p,WCon{False{},y}))),
            Equal.cong(Nat,Nat,t=>0n+Nat.double(t),Word.to_nat(p,Word.adc(p,x,y,False{},True{})),
              Nat.add(Add.bit(True{}),Nat.add(Word.to_nat(p,x),Word.to_nat(p,y))),word(p,x,y,True{},e)),
            step(True{},False{},True{},Word.to_nat(p,x),Word.to_nat(p,y)))
        case WCon{True{},x} WCon{True{},y} False{}:
          Equal.trans(Nat,Word.to_nat(1n+p,Word.adc(1n+p,WCon{True{},x},WCon{True{},y},False{},False{})),
            0n+Nat.double(Nat.add(Add.bit(True{}),Nat.add(Word.to_nat(p,x),Word.to_nat(p,y)))),
            Nat.add(Add.bit(False{}),Nat.add(Word.to_nat(1n+p,WCon{True{},x}),Word.to_nat(1n+p,WCon{True{},y}))),
            Equal.cong(Nat,Nat,t=>0n+Nat.double(t),Word.to_nat(p,Word.adc(p,x,y,False{},True{})),
              Nat.add(Add.bit(True{}),Nat.add(Word.to_nat(p,x),Word.to_nat(p,y))),word(p,x,y,True{},e)),
            step(True{},True{},False{},Word.to_nat(p,x),Word.to_nat(p,y)))
        case WCon{True{},x} WCon{True{},y} True{}:
          Equal.trans(Nat,Word.to_nat(1n+p,Word.adc(1n+p,WCon{True{},x},WCon{True{},y},False{},True{})),
            1n+Nat.double(Nat.add(Add.bit(True{}),Nat.add(Word.to_nat(p,x),Word.to_nat(p,y)))),
            Nat.add(Add.bit(True{}),Nat.add(Word.to_nat(1n+p,WCon{True{},x}),Word.to_nat(1n+p,WCon{True{},y}))),
            Equal.cong(Nat,Nat,t=>1n+Nat.double(t),Word.to_nat(p,Word.adc(p,x,y,False{},True{})),
              Nat.add(Add.bit(True{}),Nat.add(Word.to_nat(p,x),Word.to_nat(p,y))),word(p,x,y,True{},e)),
            step(True{},True{},True{},Word.to_nat(p,x),Word.to_nat(p,y)))

def exact(+a: U32,+b: U32,e: {Add.carry32(a,b) == False{} : Bool}) ->
  {U32.to_nat(U32.add(a,b)) == Nat.add(U32.to_nat(a),U32.to_nat(b)) : Nat}:
  match a b:
    case U32{x} U32{y}: word(32n,x,y,False{},e)
