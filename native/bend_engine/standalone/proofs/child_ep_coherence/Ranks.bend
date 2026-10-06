# Only the 64 source indices are split; bitboards and tables remain arbitrary.
import Base
import ../decoder/Bits.bend as B
import ./Spec.bend as S
import ../initialized_pawn_double/Spec.bend as D
import ../layout/Domain.bend as Domain
import ../u64/Words.bend as Old
def start_rank(+s: U32,w: Bool) -> Bool: U32.is_eq(U32.div(s,8),D.start(w))

def finite_white(+s: U32,n: Nat,same: {U32.to_nat(s) == n : Nat},below: {Nat.is_lt(n,64n) == True{} : Bool},rank: {start_rank(s,True{}) == True{} : Bool}) -> {S.backrank(D.second(s,True{})) == False{} : Bool}:
  match n:
    case 0n:
      +eq=Domain.injective(s,0,same)
      %Equal.sym(U32,s,0,eq) : {S.backrank(D.second(_,True{})) == False{} : Bool}
      Old.false_true({S.backrank(D.second(0,True{})) == False{} : Bool},
        Equal.trans(Bool,start_rank(0,True{}),start_rank(s,True{}),True{},
          Equal.cong(U32,Bool,x => start_rank(x,True{}),0,s,Equal.sym(U32,s,0,eq)),rank))
    case 1n:
      +eq=Domain.injective(s,1,same)
      %Equal.sym(U32,s,1,eq) : {S.backrank(D.second(_,True{})) == False{} : Bool}
      Old.false_true({S.backrank(D.second(1,True{})) == False{} : Bool},
        Equal.trans(Bool,start_rank(1,True{}),start_rank(s,True{}),True{},
          Equal.cong(U32,Bool,x => start_rank(x,True{}),1,s,Equal.sym(U32,s,1,eq)),rank))
    case 2n:
      +eq=Domain.injective(s,2,same)
      %Equal.sym(U32,s,2,eq) : {S.backrank(D.second(_,True{})) == False{} : Bool}
      Old.false_true({S.backrank(D.second(2,True{})) == False{} : Bool},
        Equal.trans(Bool,start_rank(2,True{}),start_rank(s,True{}),True{},
          Equal.cong(U32,Bool,x => start_rank(x,True{}),2,s,Equal.sym(U32,s,2,eq)),rank))
    case 3n:
      +eq=Domain.injective(s,3,same)
      %Equal.sym(U32,s,3,eq) : {S.backrank(D.second(_,True{})) == False{} : Bool}
      Old.false_true({S.backrank(D.second(3,True{})) == False{} : Bool},
        Equal.trans(Bool,start_rank(3,True{}),start_rank(s,True{}),True{},
          Equal.cong(U32,Bool,x => start_rank(x,True{}),3,s,Equal.sym(U32,s,3,eq)),rank))
    case 4n:
      +eq=Domain.injective(s,4,same)
      %Equal.sym(U32,s,4,eq) : {S.backrank(D.second(_,True{})) == False{} : Bool}
      Old.false_true({S.backrank(D.second(4,True{})) == False{} : Bool},
        Equal.trans(Bool,start_rank(4,True{}),start_rank(s,True{}),True{},
          Equal.cong(U32,Bool,x => start_rank(x,True{}),4,s,Equal.sym(U32,s,4,eq)),rank))
    case 5n:
      +eq=Domain.injective(s,5,same)
      %Equal.sym(U32,s,5,eq) : {S.backrank(D.second(_,True{})) == False{} : Bool}
      Old.false_true({S.backrank(D.second(5,True{})) == False{} : Bool},
        Equal.trans(Bool,start_rank(5,True{}),start_rank(s,True{}),True{},
          Equal.cong(U32,Bool,x => start_rank(x,True{}),5,s,Equal.sym(U32,s,5,eq)),rank))
    case 6n:
      +eq=Domain.injective(s,6,same)
      %Equal.sym(U32,s,6,eq) : {S.backrank(D.second(_,True{})) == False{} : Bool}
      Old.false_true({S.backrank(D.second(6,True{})) == False{} : Bool},
        Equal.trans(Bool,start_rank(6,True{}),start_rank(s,True{}),True{},
          Equal.cong(U32,Bool,x => start_rank(x,True{}),6,s,Equal.sym(U32,s,6,eq)),rank))
    case 7n:
      +eq=Domain.injective(s,7,same)
      %Equal.sym(U32,s,7,eq) : {S.backrank(D.second(_,True{})) == False{} : Bool}
      Old.false_true({S.backrank(D.second(7,True{})) == False{} : Bool},
        Equal.trans(Bool,start_rank(7,True{}),start_rank(s,True{}),True{},
          Equal.cong(U32,Bool,x => start_rank(x,True{}),7,s,Equal.sym(U32,s,7,eq)),rank))
    case 8n:
      +eq=Domain.injective(s,8,same)
      %Equal.sym(U32,s,8,eq) : {S.backrank(D.second(_,True{})) == False{} : Bool}
      {==}
    case 9n:
      +eq=Domain.injective(s,9,same)
      %Equal.sym(U32,s,9,eq) : {S.backrank(D.second(_,True{})) == False{} : Bool}
      {==}
    case 10n:
      +eq=Domain.injective(s,10,same)
      %Equal.sym(U32,s,10,eq) : {S.backrank(D.second(_,True{})) == False{} : Bool}
      {==}
    case 11n:
      +eq=Domain.injective(s,11,same)
      %Equal.sym(U32,s,11,eq) : {S.backrank(D.second(_,True{})) == False{} : Bool}
      {==}
    case 12n:
      +eq=Domain.injective(s,12,same)
      %Equal.sym(U32,s,12,eq) : {S.backrank(D.second(_,True{})) == False{} : Bool}
      {==}
    case 13n:
      +eq=Domain.injective(s,13,same)
      %Equal.sym(U32,s,13,eq) : {S.backrank(D.second(_,True{})) == False{} : Bool}
      {==}
    case 14n:
      +eq=Domain.injective(s,14,same)
      %Equal.sym(U32,s,14,eq) : {S.backrank(D.second(_,True{})) == False{} : Bool}
      {==}
    case 15n:
      +eq=Domain.injective(s,15,same)
      %Equal.sym(U32,s,15,eq) : {S.backrank(D.second(_,True{})) == False{} : Bool}
      {==}
    case 16n:
      +eq=Domain.injective(s,16,same)
      %Equal.sym(U32,s,16,eq) : {S.backrank(D.second(_,True{})) == False{} : Bool}
      Old.false_true({S.backrank(D.second(16,True{})) == False{} : Bool},
        Equal.trans(Bool,start_rank(16,True{}),start_rank(s,True{}),True{},
          Equal.cong(U32,Bool,x => start_rank(x,True{}),16,s,Equal.sym(U32,s,16,eq)),rank))
    case 17n:
      +eq=Domain.injective(s,17,same)
      %Equal.sym(U32,s,17,eq) : {S.backrank(D.second(_,True{})) == False{} : Bool}
      Old.false_true({S.backrank(D.second(17,True{})) == False{} : Bool},
        Equal.trans(Bool,start_rank(17,True{}),start_rank(s,True{}),True{},
          Equal.cong(U32,Bool,x => start_rank(x,True{}),17,s,Equal.sym(U32,s,17,eq)),rank))
    case 18n:
      +eq=Domain.injective(s,18,same)
      %Equal.sym(U32,s,18,eq) : {S.backrank(D.second(_,True{})) == False{} : Bool}
      Old.false_true({S.backrank(D.second(18,True{})) == False{} : Bool},
        Equal.trans(Bool,start_rank(18,True{}),start_rank(s,True{}),True{},
          Equal.cong(U32,Bool,x => start_rank(x,True{}),18,s,Equal.sym(U32,s,18,eq)),rank))
    case 19n:
      +eq=Domain.injective(s,19,same)
      %Equal.sym(U32,s,19,eq) : {S.backrank(D.second(_,True{})) == False{} : Bool}
      Old.false_true({S.backrank(D.second(19,True{})) == False{} : Bool},
        Equal.trans(Bool,start_rank(19,True{}),start_rank(s,True{}),True{},
          Equal.cong(U32,Bool,x => start_rank(x,True{}),19,s,Equal.sym(U32,s,19,eq)),rank))
    case 20n:
      +eq=Domain.injective(s,20,same)
      %Equal.sym(U32,s,20,eq) : {S.backrank(D.second(_,True{})) == False{} : Bool}
      Old.false_true({S.backrank(D.second(20,True{})) == False{} : Bool},
        Equal.trans(Bool,start_rank(20,True{}),start_rank(s,True{}),True{},
          Equal.cong(U32,Bool,x => start_rank(x,True{}),20,s,Equal.sym(U32,s,20,eq)),rank))
    case 21n:
      +eq=Domain.injective(s,21,same)
      %Equal.sym(U32,s,21,eq) : {S.backrank(D.second(_,True{})) == False{} : Bool}
      Old.false_true({S.backrank(D.second(21,True{})) == False{} : Bool},
        Equal.trans(Bool,start_rank(21,True{}),start_rank(s,True{}),True{},
          Equal.cong(U32,Bool,x => start_rank(x,True{}),21,s,Equal.sym(U32,s,21,eq)),rank))
    case 22n:
      +eq=Domain.injective(s,22,same)
      %Equal.sym(U32,s,22,eq) : {S.backrank(D.second(_,True{})) == False{} : Bool}
      Old.false_true({S.backrank(D.second(22,True{})) == False{} : Bool},
        Equal.trans(Bool,start_rank(22,True{}),start_rank(s,True{}),True{},
          Equal.cong(U32,Bool,x => start_rank(x,True{}),22,s,Equal.sym(U32,s,22,eq)),rank))
    case 23n:
      +eq=Domain.injective(s,23,same)
      %Equal.sym(U32,s,23,eq) : {S.backrank(D.second(_,True{})) == False{} : Bool}
      Old.false_true({S.backrank(D.second(23,True{})) == False{} : Bool},
        Equal.trans(Bool,start_rank(23,True{}),start_rank(s,True{}),True{},
          Equal.cong(U32,Bool,x => start_rank(x,True{}),23,s,Equal.sym(U32,s,23,eq)),rank))
    case 24n:
      +eq=Domain.injective(s,24,same)
      %Equal.sym(U32,s,24,eq) : {S.backrank(D.second(_,True{})) == False{} : Bool}
      Old.false_true({S.backrank(D.second(24,True{})) == False{} : Bool},
        Equal.trans(Bool,start_rank(24,True{}),start_rank(s,True{}),True{},
          Equal.cong(U32,Bool,x => start_rank(x,True{}),24,s,Equal.sym(U32,s,24,eq)),rank))
    case 25n:
      +eq=Domain.injective(s,25,same)
      %Equal.sym(U32,s,25,eq) : {S.backrank(D.second(_,True{})) == False{} : Bool}
      Old.false_true({S.backrank(D.second(25,True{})) == False{} : Bool},
        Equal.trans(Bool,start_rank(25,True{}),start_rank(s,True{}),True{},
          Equal.cong(U32,Bool,x => start_rank(x,True{}),25,s,Equal.sym(U32,s,25,eq)),rank))
    case 26n:
      +eq=Domain.injective(s,26,same)
      %Equal.sym(U32,s,26,eq) : {S.backrank(D.second(_,True{})) == False{} : Bool}
      Old.false_true({S.backrank(D.second(26,True{})) == False{} : Bool},
        Equal.trans(Bool,start_rank(26,True{}),start_rank(s,True{}),True{},
          Equal.cong(U32,Bool,x => start_rank(x,True{}),26,s,Equal.sym(U32,s,26,eq)),rank))
    case 27n:
      +eq=Domain.injective(s,27,same)
      %Equal.sym(U32,s,27,eq) : {S.backrank(D.second(_,True{})) == False{} : Bool}
      Old.false_true({S.backrank(D.second(27,True{})) == False{} : Bool},
        Equal.trans(Bool,start_rank(27,True{}),start_rank(s,True{}),True{},
          Equal.cong(U32,Bool,x => start_rank(x,True{}),27,s,Equal.sym(U32,s,27,eq)),rank))
    case 28n:
      +eq=Domain.injective(s,28,same)
      %Equal.sym(U32,s,28,eq) : {S.backrank(D.second(_,True{})) == False{} : Bool}
      Old.false_true({S.backrank(D.second(28,True{})) == False{} : Bool},
        Equal.trans(Bool,start_rank(28,True{}),start_rank(s,True{}),True{},
          Equal.cong(U32,Bool,x => start_rank(x,True{}),28,s,Equal.sym(U32,s,28,eq)),rank))
    case 29n:
      +eq=Domain.injective(s,29,same)
      %Equal.sym(U32,s,29,eq) : {S.backrank(D.second(_,True{})) == False{} : Bool}
      Old.false_true({S.backrank(D.second(29,True{})) == False{} : Bool},
        Equal.trans(Bool,start_rank(29,True{}),start_rank(s,True{}),True{},
          Equal.cong(U32,Bool,x => start_rank(x,True{}),29,s,Equal.sym(U32,s,29,eq)),rank))
    case 30n:
      +eq=Domain.injective(s,30,same)
      %Equal.sym(U32,s,30,eq) : {S.backrank(D.second(_,True{})) == False{} : Bool}
      Old.false_true({S.backrank(D.second(30,True{})) == False{} : Bool},
        Equal.trans(Bool,start_rank(30,True{}),start_rank(s,True{}),True{},
          Equal.cong(U32,Bool,x => start_rank(x,True{}),30,s,Equal.sym(U32,s,30,eq)),rank))
    case 31n:
      +eq=Domain.injective(s,31,same)
      %Equal.sym(U32,s,31,eq) : {S.backrank(D.second(_,True{})) == False{} : Bool}
      Old.false_true({S.backrank(D.second(31,True{})) == False{} : Bool},
        Equal.trans(Bool,start_rank(31,True{}),start_rank(s,True{}),True{},
          Equal.cong(U32,Bool,x => start_rank(x,True{}),31,s,Equal.sym(U32,s,31,eq)),rank))
    case 32n:
      +eq=Domain.injective(s,32,same)
      %Equal.sym(U32,s,32,eq) : {S.backrank(D.second(_,True{})) == False{} : Bool}
      Old.false_true({S.backrank(D.second(32,True{})) == False{} : Bool},
        Equal.trans(Bool,start_rank(32,True{}),start_rank(s,True{}),True{},
          Equal.cong(U32,Bool,x => start_rank(x,True{}),32,s,Equal.sym(U32,s,32,eq)),rank))
    case 33n:
      +eq=Domain.injective(s,33,same)
      %Equal.sym(U32,s,33,eq) : {S.backrank(D.second(_,True{})) == False{} : Bool}
      Old.false_true({S.backrank(D.second(33,True{})) == False{} : Bool},
        Equal.trans(Bool,start_rank(33,True{}),start_rank(s,True{}),True{},
          Equal.cong(U32,Bool,x => start_rank(x,True{}),33,s,Equal.sym(U32,s,33,eq)),rank))
    case 34n:
      +eq=Domain.injective(s,34,same)
      %Equal.sym(U32,s,34,eq) : {S.backrank(D.second(_,True{})) == False{} : Bool}
      Old.false_true({S.backrank(D.second(34,True{})) == False{} : Bool},
        Equal.trans(Bool,start_rank(34,True{}),start_rank(s,True{}),True{},
          Equal.cong(U32,Bool,x => start_rank(x,True{}),34,s,Equal.sym(U32,s,34,eq)),rank))
    case 35n:
      +eq=Domain.injective(s,35,same)
      %Equal.sym(U32,s,35,eq) : {S.backrank(D.second(_,True{})) == False{} : Bool}
      Old.false_true({S.backrank(D.second(35,True{})) == False{} : Bool},
        Equal.trans(Bool,start_rank(35,True{}),start_rank(s,True{}),True{},
          Equal.cong(U32,Bool,x => start_rank(x,True{}),35,s,Equal.sym(U32,s,35,eq)),rank))
    case 36n:
      +eq=Domain.injective(s,36,same)
      %Equal.sym(U32,s,36,eq) : {S.backrank(D.second(_,True{})) == False{} : Bool}
      Old.false_true({S.backrank(D.second(36,True{})) == False{} : Bool},
        Equal.trans(Bool,start_rank(36,True{}),start_rank(s,True{}),True{},
          Equal.cong(U32,Bool,x => start_rank(x,True{}),36,s,Equal.sym(U32,s,36,eq)),rank))
    case 37n:
      +eq=Domain.injective(s,37,same)
      %Equal.sym(U32,s,37,eq) : {S.backrank(D.second(_,True{})) == False{} : Bool}
      Old.false_true({S.backrank(D.second(37,True{})) == False{} : Bool},
        Equal.trans(Bool,start_rank(37,True{}),start_rank(s,True{}),True{},
          Equal.cong(U32,Bool,x => start_rank(x,True{}),37,s,Equal.sym(U32,s,37,eq)),rank))
    case 38n:
      +eq=Domain.injective(s,38,same)
      %Equal.sym(U32,s,38,eq) : {S.backrank(D.second(_,True{})) == False{} : Bool}
      Old.false_true({S.backrank(D.second(38,True{})) == False{} : Bool},
        Equal.trans(Bool,start_rank(38,True{}),start_rank(s,True{}),True{},
          Equal.cong(U32,Bool,x => start_rank(x,True{}),38,s,Equal.sym(U32,s,38,eq)),rank))
    case 39n:
      +eq=Domain.injective(s,39,same)
      %Equal.sym(U32,s,39,eq) : {S.backrank(D.second(_,True{})) == False{} : Bool}
      Old.false_true({S.backrank(D.second(39,True{})) == False{} : Bool},
        Equal.trans(Bool,start_rank(39,True{}),start_rank(s,True{}),True{},
          Equal.cong(U32,Bool,x => start_rank(x,True{}),39,s,Equal.sym(U32,s,39,eq)),rank))
    case 40n:
      +eq=Domain.injective(s,40,same)
      %Equal.sym(U32,s,40,eq) : {S.backrank(D.second(_,True{})) == False{} : Bool}
      Old.false_true({S.backrank(D.second(40,True{})) == False{} : Bool},
        Equal.trans(Bool,start_rank(40,True{}),start_rank(s,True{}),True{},
          Equal.cong(U32,Bool,x => start_rank(x,True{}),40,s,Equal.sym(U32,s,40,eq)),rank))
    case 41n:
      +eq=Domain.injective(s,41,same)
      %Equal.sym(U32,s,41,eq) : {S.backrank(D.second(_,True{})) == False{} : Bool}
      Old.false_true({S.backrank(D.second(41,True{})) == False{} : Bool},
        Equal.trans(Bool,start_rank(41,True{}),start_rank(s,True{}),True{},
          Equal.cong(U32,Bool,x => start_rank(x,True{}),41,s,Equal.sym(U32,s,41,eq)),rank))
    case 42n:
      +eq=Domain.injective(s,42,same)
      %Equal.sym(U32,s,42,eq) : {S.backrank(D.second(_,True{})) == False{} : Bool}
      Old.false_true({S.backrank(D.second(42,True{})) == False{} : Bool},
        Equal.trans(Bool,start_rank(42,True{}),start_rank(s,True{}),True{},
          Equal.cong(U32,Bool,x => start_rank(x,True{}),42,s,Equal.sym(U32,s,42,eq)),rank))
    case 43n:
      +eq=Domain.injective(s,43,same)
      %Equal.sym(U32,s,43,eq) : {S.backrank(D.second(_,True{})) == False{} : Bool}
      Old.false_true({S.backrank(D.second(43,True{})) == False{} : Bool},
        Equal.trans(Bool,start_rank(43,True{}),start_rank(s,True{}),True{},
          Equal.cong(U32,Bool,x => start_rank(x,True{}),43,s,Equal.sym(U32,s,43,eq)),rank))
    case 44n:
      +eq=Domain.injective(s,44,same)
      %Equal.sym(U32,s,44,eq) : {S.backrank(D.second(_,True{})) == False{} : Bool}
      Old.false_true({S.backrank(D.second(44,True{})) == False{} : Bool},
        Equal.trans(Bool,start_rank(44,True{}),start_rank(s,True{}),True{},
          Equal.cong(U32,Bool,x => start_rank(x,True{}),44,s,Equal.sym(U32,s,44,eq)),rank))
    case 45n:
      +eq=Domain.injective(s,45,same)
      %Equal.sym(U32,s,45,eq) : {S.backrank(D.second(_,True{})) == False{} : Bool}
      Old.false_true({S.backrank(D.second(45,True{})) == False{} : Bool},
        Equal.trans(Bool,start_rank(45,True{}),start_rank(s,True{}),True{},
          Equal.cong(U32,Bool,x => start_rank(x,True{}),45,s,Equal.sym(U32,s,45,eq)),rank))
    case 46n:
      +eq=Domain.injective(s,46,same)
      %Equal.sym(U32,s,46,eq) : {S.backrank(D.second(_,True{})) == False{} : Bool}
      Old.false_true({S.backrank(D.second(46,True{})) == False{} : Bool},
        Equal.trans(Bool,start_rank(46,True{}),start_rank(s,True{}),True{},
          Equal.cong(U32,Bool,x => start_rank(x,True{}),46,s,Equal.sym(U32,s,46,eq)),rank))
    case 47n:
      +eq=Domain.injective(s,47,same)
      %Equal.sym(U32,s,47,eq) : {S.backrank(D.second(_,True{})) == False{} : Bool}
      Old.false_true({S.backrank(D.second(47,True{})) == False{} : Bool},
        Equal.trans(Bool,start_rank(47,True{}),start_rank(s,True{}),True{},
          Equal.cong(U32,Bool,x => start_rank(x,True{}),47,s,Equal.sym(U32,s,47,eq)),rank))
    case 48n:
      +eq=Domain.injective(s,48,same)
      %Equal.sym(U32,s,48,eq) : {S.backrank(D.second(_,True{})) == False{} : Bool}
      Old.false_true({S.backrank(D.second(48,True{})) == False{} : Bool},
        Equal.trans(Bool,start_rank(48,True{}),start_rank(s,True{}),True{},
          Equal.cong(U32,Bool,x => start_rank(x,True{}),48,s,Equal.sym(U32,s,48,eq)),rank))
    case 49n:
      +eq=Domain.injective(s,49,same)
      %Equal.sym(U32,s,49,eq) : {S.backrank(D.second(_,True{})) == False{} : Bool}
      Old.false_true({S.backrank(D.second(49,True{})) == False{} : Bool},
        Equal.trans(Bool,start_rank(49,True{}),start_rank(s,True{}),True{},
          Equal.cong(U32,Bool,x => start_rank(x,True{}),49,s,Equal.sym(U32,s,49,eq)),rank))
    case 50n:
      +eq=Domain.injective(s,50,same)
      %Equal.sym(U32,s,50,eq) : {S.backrank(D.second(_,True{})) == False{} : Bool}
      Old.false_true({S.backrank(D.second(50,True{})) == False{} : Bool},
        Equal.trans(Bool,start_rank(50,True{}),start_rank(s,True{}),True{},
          Equal.cong(U32,Bool,x => start_rank(x,True{}),50,s,Equal.sym(U32,s,50,eq)),rank))
    case 51n:
      +eq=Domain.injective(s,51,same)
      %Equal.sym(U32,s,51,eq) : {S.backrank(D.second(_,True{})) == False{} : Bool}
      Old.false_true({S.backrank(D.second(51,True{})) == False{} : Bool},
        Equal.trans(Bool,start_rank(51,True{}),start_rank(s,True{}),True{},
          Equal.cong(U32,Bool,x => start_rank(x,True{}),51,s,Equal.sym(U32,s,51,eq)),rank))
    case 52n:
      +eq=Domain.injective(s,52,same)
      %Equal.sym(U32,s,52,eq) : {S.backrank(D.second(_,True{})) == False{} : Bool}
      Old.false_true({S.backrank(D.second(52,True{})) == False{} : Bool},
        Equal.trans(Bool,start_rank(52,True{}),start_rank(s,True{}),True{},
          Equal.cong(U32,Bool,x => start_rank(x,True{}),52,s,Equal.sym(U32,s,52,eq)),rank))
    case 53n:
      +eq=Domain.injective(s,53,same)
      %Equal.sym(U32,s,53,eq) : {S.backrank(D.second(_,True{})) == False{} : Bool}
      Old.false_true({S.backrank(D.second(53,True{})) == False{} : Bool},
        Equal.trans(Bool,start_rank(53,True{}),start_rank(s,True{}),True{},
          Equal.cong(U32,Bool,x => start_rank(x,True{}),53,s,Equal.sym(U32,s,53,eq)),rank))
    case 54n:
      +eq=Domain.injective(s,54,same)
      %Equal.sym(U32,s,54,eq) : {S.backrank(D.second(_,True{})) == False{} : Bool}
      Old.false_true({S.backrank(D.second(54,True{})) == False{} : Bool},
        Equal.trans(Bool,start_rank(54,True{}),start_rank(s,True{}),True{},
          Equal.cong(U32,Bool,x => start_rank(x,True{}),54,s,Equal.sym(U32,s,54,eq)),rank))
    case 55n:
      +eq=Domain.injective(s,55,same)
      %Equal.sym(U32,s,55,eq) : {S.backrank(D.second(_,True{})) == False{} : Bool}
      Old.false_true({S.backrank(D.second(55,True{})) == False{} : Bool},
        Equal.trans(Bool,start_rank(55,True{}),start_rank(s,True{}),True{},
          Equal.cong(U32,Bool,x => start_rank(x,True{}),55,s,Equal.sym(U32,s,55,eq)),rank))
    case 56n:
      +eq=Domain.injective(s,56,same)
      %Equal.sym(U32,s,56,eq) : {S.backrank(D.second(_,True{})) == False{} : Bool}
      Old.false_true({S.backrank(D.second(56,True{})) == False{} : Bool},
        Equal.trans(Bool,start_rank(56,True{}),start_rank(s,True{}),True{},
          Equal.cong(U32,Bool,x => start_rank(x,True{}),56,s,Equal.sym(U32,s,56,eq)),rank))
    case 57n:
      +eq=Domain.injective(s,57,same)
      %Equal.sym(U32,s,57,eq) : {S.backrank(D.second(_,True{})) == False{} : Bool}
      Old.false_true({S.backrank(D.second(57,True{})) == False{} : Bool},
        Equal.trans(Bool,start_rank(57,True{}),start_rank(s,True{}),True{},
          Equal.cong(U32,Bool,x => start_rank(x,True{}),57,s,Equal.sym(U32,s,57,eq)),rank))
    case 58n:
      +eq=Domain.injective(s,58,same)
      %Equal.sym(U32,s,58,eq) : {S.backrank(D.second(_,True{})) == False{} : Bool}
      Old.false_true({S.backrank(D.second(58,True{})) == False{} : Bool},
        Equal.trans(Bool,start_rank(58,True{}),start_rank(s,True{}),True{},
          Equal.cong(U32,Bool,x => start_rank(x,True{}),58,s,Equal.sym(U32,s,58,eq)),rank))
    case 59n:
      +eq=Domain.injective(s,59,same)
      %Equal.sym(U32,s,59,eq) : {S.backrank(D.second(_,True{})) == False{} : Bool}
      Old.false_true({S.backrank(D.second(59,True{})) == False{} : Bool},
        Equal.trans(Bool,start_rank(59,True{}),start_rank(s,True{}),True{},
          Equal.cong(U32,Bool,x => start_rank(x,True{}),59,s,Equal.sym(U32,s,59,eq)),rank))
    case 60n:
      +eq=Domain.injective(s,60,same)
      %Equal.sym(U32,s,60,eq) : {S.backrank(D.second(_,True{})) == False{} : Bool}
      Old.false_true({S.backrank(D.second(60,True{})) == False{} : Bool},
        Equal.trans(Bool,start_rank(60,True{}),start_rank(s,True{}),True{},
          Equal.cong(U32,Bool,x => start_rank(x,True{}),60,s,Equal.sym(U32,s,60,eq)),rank))
    case 61n:
      +eq=Domain.injective(s,61,same)
      %Equal.sym(U32,s,61,eq) : {S.backrank(D.second(_,True{})) == False{} : Bool}
      Old.false_true({S.backrank(D.second(61,True{})) == False{} : Bool},
        Equal.trans(Bool,start_rank(61,True{}),start_rank(s,True{}),True{},
          Equal.cong(U32,Bool,x => start_rank(x,True{}),61,s,Equal.sym(U32,s,61,eq)),rank))
    case 62n:
      +eq=Domain.injective(s,62,same)
      %Equal.sym(U32,s,62,eq) : {S.backrank(D.second(_,True{})) == False{} : Bool}
      Old.false_true({S.backrank(D.second(62,True{})) == False{} : Bool},
        Equal.trans(Bool,start_rank(62,True{}),start_rank(s,True{}),True{},
          Equal.cong(U32,Bool,x => start_rank(x,True{}),62,s,Equal.sym(U32,s,62,eq)),rank))
    case 63n:
      +eq=Domain.injective(s,63,same)
      %Equal.sym(U32,s,63,eq) : {S.backrank(D.second(_,True{})) == False{} : Bool}
      Old.false_true({S.backrank(D.second(63,True{})) == False{} : Bool},
        Equal.trans(Bool,start_rank(63,True{}),start_rank(s,True{}),True{},
          Equal.cong(U32,Bool,x => start_rank(x,True{}),63,s,Equal.sym(U32,s,63,eq)),rank))
    case 64n+p: B.impossible({S.backrank(D.second(s,True{})) == False{} : Bool},p,below)

def finite_black(+s: U32,n: Nat,same: {U32.to_nat(s) == n : Nat},below: {Nat.is_lt(n,64n) == True{} : Bool},rank: {start_rank(s,False{}) == True{} : Bool}) -> {S.backrank(D.second(s,False{})) == False{} : Bool}:
  match n:
    case 0n:
      +eq=Domain.injective(s,0,same)
      %Equal.sym(U32,s,0,eq) : {S.backrank(D.second(_,False{})) == False{} : Bool}
      Old.false_true({S.backrank(D.second(0,False{})) == False{} : Bool},
        Equal.trans(Bool,start_rank(0,False{}),start_rank(s,False{}),True{},
          Equal.cong(U32,Bool,x => start_rank(x,False{}),0,s,Equal.sym(U32,s,0,eq)),rank))
    case 1n:
      +eq=Domain.injective(s,1,same)
      %Equal.sym(U32,s,1,eq) : {S.backrank(D.second(_,False{})) == False{} : Bool}
      Old.false_true({S.backrank(D.second(1,False{})) == False{} : Bool},
        Equal.trans(Bool,start_rank(1,False{}),start_rank(s,False{}),True{},
          Equal.cong(U32,Bool,x => start_rank(x,False{}),1,s,Equal.sym(U32,s,1,eq)),rank))
    case 2n:
      +eq=Domain.injective(s,2,same)
      %Equal.sym(U32,s,2,eq) : {S.backrank(D.second(_,False{})) == False{} : Bool}
      Old.false_true({S.backrank(D.second(2,False{})) == False{} : Bool},
        Equal.trans(Bool,start_rank(2,False{}),start_rank(s,False{}),True{},
          Equal.cong(U32,Bool,x => start_rank(x,False{}),2,s,Equal.sym(U32,s,2,eq)),rank))
    case 3n:
      +eq=Domain.injective(s,3,same)
      %Equal.sym(U32,s,3,eq) : {S.backrank(D.second(_,False{})) == False{} : Bool}
      Old.false_true({S.backrank(D.second(3,False{})) == False{} : Bool},
        Equal.trans(Bool,start_rank(3,False{}),start_rank(s,False{}),True{},
          Equal.cong(U32,Bool,x => start_rank(x,False{}),3,s,Equal.sym(U32,s,3,eq)),rank))
    case 4n:
      +eq=Domain.injective(s,4,same)
      %Equal.sym(U32,s,4,eq) : {S.backrank(D.second(_,False{})) == False{} : Bool}
      Old.false_true({S.backrank(D.second(4,False{})) == False{} : Bool},
        Equal.trans(Bool,start_rank(4,False{}),start_rank(s,False{}),True{},
          Equal.cong(U32,Bool,x => start_rank(x,False{}),4,s,Equal.sym(U32,s,4,eq)),rank))
    case 5n:
      +eq=Domain.injective(s,5,same)
      %Equal.sym(U32,s,5,eq) : {S.backrank(D.second(_,False{})) == False{} : Bool}
      Old.false_true({S.backrank(D.second(5,False{})) == False{} : Bool},
        Equal.trans(Bool,start_rank(5,False{}),start_rank(s,False{}),True{},
          Equal.cong(U32,Bool,x => start_rank(x,False{}),5,s,Equal.sym(U32,s,5,eq)),rank))
    case 6n:
      +eq=Domain.injective(s,6,same)
      %Equal.sym(U32,s,6,eq) : {S.backrank(D.second(_,False{})) == False{} : Bool}
      Old.false_true({S.backrank(D.second(6,False{})) == False{} : Bool},
        Equal.trans(Bool,start_rank(6,False{}),start_rank(s,False{}),True{},
          Equal.cong(U32,Bool,x => start_rank(x,False{}),6,s,Equal.sym(U32,s,6,eq)),rank))
    case 7n:
      +eq=Domain.injective(s,7,same)
      %Equal.sym(U32,s,7,eq) : {S.backrank(D.second(_,False{})) == False{} : Bool}
      Old.false_true({S.backrank(D.second(7,False{})) == False{} : Bool},
        Equal.trans(Bool,start_rank(7,False{}),start_rank(s,False{}),True{},
          Equal.cong(U32,Bool,x => start_rank(x,False{}),7,s,Equal.sym(U32,s,7,eq)),rank))
    case 8n:
      +eq=Domain.injective(s,8,same)
      %Equal.sym(U32,s,8,eq) : {S.backrank(D.second(_,False{})) == False{} : Bool}
      Old.false_true({S.backrank(D.second(8,False{})) == False{} : Bool},
        Equal.trans(Bool,start_rank(8,False{}),start_rank(s,False{}),True{},
          Equal.cong(U32,Bool,x => start_rank(x,False{}),8,s,Equal.sym(U32,s,8,eq)),rank))
    case 9n:
      +eq=Domain.injective(s,9,same)
      %Equal.sym(U32,s,9,eq) : {S.backrank(D.second(_,False{})) == False{} : Bool}
      Old.false_true({S.backrank(D.second(9,False{})) == False{} : Bool},
        Equal.trans(Bool,start_rank(9,False{}),start_rank(s,False{}),True{},
          Equal.cong(U32,Bool,x => start_rank(x,False{}),9,s,Equal.sym(U32,s,9,eq)),rank))
    case 10n:
      +eq=Domain.injective(s,10,same)
      %Equal.sym(U32,s,10,eq) : {S.backrank(D.second(_,False{})) == False{} : Bool}
      Old.false_true({S.backrank(D.second(10,False{})) == False{} : Bool},
        Equal.trans(Bool,start_rank(10,False{}),start_rank(s,False{}),True{},
          Equal.cong(U32,Bool,x => start_rank(x,False{}),10,s,Equal.sym(U32,s,10,eq)),rank))
    case 11n:
      +eq=Domain.injective(s,11,same)
      %Equal.sym(U32,s,11,eq) : {S.backrank(D.second(_,False{})) == False{} : Bool}
      Old.false_true({S.backrank(D.second(11,False{})) == False{} : Bool},
        Equal.trans(Bool,start_rank(11,False{}),start_rank(s,False{}),True{},
          Equal.cong(U32,Bool,x => start_rank(x,False{}),11,s,Equal.sym(U32,s,11,eq)),rank))
    case 12n:
      +eq=Domain.injective(s,12,same)
      %Equal.sym(U32,s,12,eq) : {S.backrank(D.second(_,False{})) == False{} : Bool}
      Old.false_true({S.backrank(D.second(12,False{})) == False{} : Bool},
        Equal.trans(Bool,start_rank(12,False{}),start_rank(s,False{}),True{},
          Equal.cong(U32,Bool,x => start_rank(x,False{}),12,s,Equal.sym(U32,s,12,eq)),rank))
    case 13n:
      +eq=Domain.injective(s,13,same)
      %Equal.sym(U32,s,13,eq) : {S.backrank(D.second(_,False{})) == False{} : Bool}
      Old.false_true({S.backrank(D.second(13,False{})) == False{} : Bool},
        Equal.trans(Bool,start_rank(13,False{}),start_rank(s,False{}),True{},
          Equal.cong(U32,Bool,x => start_rank(x,False{}),13,s,Equal.sym(U32,s,13,eq)),rank))
    case 14n:
      +eq=Domain.injective(s,14,same)
      %Equal.sym(U32,s,14,eq) : {S.backrank(D.second(_,False{})) == False{} : Bool}
      Old.false_true({S.backrank(D.second(14,False{})) == False{} : Bool},
        Equal.trans(Bool,start_rank(14,False{}),start_rank(s,False{}),True{},
          Equal.cong(U32,Bool,x => start_rank(x,False{}),14,s,Equal.sym(U32,s,14,eq)),rank))
    case 15n:
      +eq=Domain.injective(s,15,same)
      %Equal.sym(U32,s,15,eq) : {S.backrank(D.second(_,False{})) == False{} : Bool}
      Old.false_true({S.backrank(D.second(15,False{})) == False{} : Bool},
        Equal.trans(Bool,start_rank(15,False{}),start_rank(s,False{}),True{},
          Equal.cong(U32,Bool,x => start_rank(x,False{}),15,s,Equal.sym(U32,s,15,eq)),rank))
    case 16n:
      +eq=Domain.injective(s,16,same)
      %Equal.sym(U32,s,16,eq) : {S.backrank(D.second(_,False{})) == False{} : Bool}
      Old.false_true({S.backrank(D.second(16,False{})) == False{} : Bool},
        Equal.trans(Bool,start_rank(16,False{}),start_rank(s,False{}),True{},
          Equal.cong(U32,Bool,x => start_rank(x,False{}),16,s,Equal.sym(U32,s,16,eq)),rank))
    case 17n:
      +eq=Domain.injective(s,17,same)
      %Equal.sym(U32,s,17,eq) : {S.backrank(D.second(_,False{})) == False{} : Bool}
      Old.false_true({S.backrank(D.second(17,False{})) == False{} : Bool},
        Equal.trans(Bool,start_rank(17,False{}),start_rank(s,False{}),True{},
          Equal.cong(U32,Bool,x => start_rank(x,False{}),17,s,Equal.sym(U32,s,17,eq)),rank))
    case 18n:
      +eq=Domain.injective(s,18,same)
      %Equal.sym(U32,s,18,eq) : {S.backrank(D.second(_,False{})) == False{} : Bool}
      Old.false_true({S.backrank(D.second(18,False{})) == False{} : Bool},
        Equal.trans(Bool,start_rank(18,False{}),start_rank(s,False{}),True{},
          Equal.cong(U32,Bool,x => start_rank(x,False{}),18,s,Equal.sym(U32,s,18,eq)),rank))
    case 19n:
      +eq=Domain.injective(s,19,same)
      %Equal.sym(U32,s,19,eq) : {S.backrank(D.second(_,False{})) == False{} : Bool}
      Old.false_true({S.backrank(D.second(19,False{})) == False{} : Bool},
        Equal.trans(Bool,start_rank(19,False{}),start_rank(s,False{}),True{},
          Equal.cong(U32,Bool,x => start_rank(x,False{}),19,s,Equal.sym(U32,s,19,eq)),rank))
    case 20n:
      +eq=Domain.injective(s,20,same)
      %Equal.sym(U32,s,20,eq) : {S.backrank(D.second(_,False{})) == False{} : Bool}
      Old.false_true({S.backrank(D.second(20,False{})) == False{} : Bool},
        Equal.trans(Bool,start_rank(20,False{}),start_rank(s,False{}),True{},
          Equal.cong(U32,Bool,x => start_rank(x,False{}),20,s,Equal.sym(U32,s,20,eq)),rank))
    case 21n:
      +eq=Domain.injective(s,21,same)
      %Equal.sym(U32,s,21,eq) : {S.backrank(D.second(_,False{})) == False{} : Bool}
      Old.false_true({S.backrank(D.second(21,False{})) == False{} : Bool},
        Equal.trans(Bool,start_rank(21,False{}),start_rank(s,False{}),True{},
          Equal.cong(U32,Bool,x => start_rank(x,False{}),21,s,Equal.sym(U32,s,21,eq)),rank))
    case 22n:
      +eq=Domain.injective(s,22,same)
      %Equal.sym(U32,s,22,eq) : {S.backrank(D.second(_,False{})) == False{} : Bool}
      Old.false_true({S.backrank(D.second(22,False{})) == False{} : Bool},
        Equal.trans(Bool,start_rank(22,False{}),start_rank(s,False{}),True{},
          Equal.cong(U32,Bool,x => start_rank(x,False{}),22,s,Equal.sym(U32,s,22,eq)),rank))
    case 23n:
      +eq=Domain.injective(s,23,same)
      %Equal.sym(U32,s,23,eq) : {S.backrank(D.second(_,False{})) == False{} : Bool}
      Old.false_true({S.backrank(D.second(23,False{})) == False{} : Bool},
        Equal.trans(Bool,start_rank(23,False{}),start_rank(s,False{}),True{},
          Equal.cong(U32,Bool,x => start_rank(x,False{}),23,s,Equal.sym(U32,s,23,eq)),rank))
    case 24n:
      +eq=Domain.injective(s,24,same)
      %Equal.sym(U32,s,24,eq) : {S.backrank(D.second(_,False{})) == False{} : Bool}
      Old.false_true({S.backrank(D.second(24,False{})) == False{} : Bool},
        Equal.trans(Bool,start_rank(24,False{}),start_rank(s,False{}),True{},
          Equal.cong(U32,Bool,x => start_rank(x,False{}),24,s,Equal.sym(U32,s,24,eq)),rank))
    case 25n:
      +eq=Domain.injective(s,25,same)
      %Equal.sym(U32,s,25,eq) : {S.backrank(D.second(_,False{})) == False{} : Bool}
      Old.false_true({S.backrank(D.second(25,False{})) == False{} : Bool},
        Equal.trans(Bool,start_rank(25,False{}),start_rank(s,False{}),True{},
          Equal.cong(U32,Bool,x => start_rank(x,False{}),25,s,Equal.sym(U32,s,25,eq)),rank))
    case 26n:
      +eq=Domain.injective(s,26,same)
      %Equal.sym(U32,s,26,eq) : {S.backrank(D.second(_,False{})) == False{} : Bool}
      Old.false_true({S.backrank(D.second(26,False{})) == False{} : Bool},
        Equal.trans(Bool,start_rank(26,False{}),start_rank(s,False{}),True{},
          Equal.cong(U32,Bool,x => start_rank(x,False{}),26,s,Equal.sym(U32,s,26,eq)),rank))
    case 27n:
      +eq=Domain.injective(s,27,same)
      %Equal.sym(U32,s,27,eq) : {S.backrank(D.second(_,False{})) == False{} : Bool}
      Old.false_true({S.backrank(D.second(27,False{})) == False{} : Bool},
        Equal.trans(Bool,start_rank(27,False{}),start_rank(s,False{}),True{},
          Equal.cong(U32,Bool,x => start_rank(x,False{}),27,s,Equal.sym(U32,s,27,eq)),rank))
    case 28n:
      +eq=Domain.injective(s,28,same)
      %Equal.sym(U32,s,28,eq) : {S.backrank(D.second(_,False{})) == False{} : Bool}
      Old.false_true({S.backrank(D.second(28,False{})) == False{} : Bool},
        Equal.trans(Bool,start_rank(28,False{}),start_rank(s,False{}),True{},
          Equal.cong(U32,Bool,x => start_rank(x,False{}),28,s,Equal.sym(U32,s,28,eq)),rank))
    case 29n:
      +eq=Domain.injective(s,29,same)
      %Equal.sym(U32,s,29,eq) : {S.backrank(D.second(_,False{})) == False{} : Bool}
      Old.false_true({S.backrank(D.second(29,False{})) == False{} : Bool},
        Equal.trans(Bool,start_rank(29,False{}),start_rank(s,False{}),True{},
          Equal.cong(U32,Bool,x => start_rank(x,False{}),29,s,Equal.sym(U32,s,29,eq)),rank))
    case 30n:
      +eq=Domain.injective(s,30,same)
      %Equal.sym(U32,s,30,eq) : {S.backrank(D.second(_,False{})) == False{} : Bool}
      Old.false_true({S.backrank(D.second(30,False{})) == False{} : Bool},
        Equal.trans(Bool,start_rank(30,False{}),start_rank(s,False{}),True{},
          Equal.cong(U32,Bool,x => start_rank(x,False{}),30,s,Equal.sym(U32,s,30,eq)),rank))
    case 31n:
      +eq=Domain.injective(s,31,same)
      %Equal.sym(U32,s,31,eq) : {S.backrank(D.second(_,False{})) == False{} : Bool}
      Old.false_true({S.backrank(D.second(31,False{})) == False{} : Bool},
        Equal.trans(Bool,start_rank(31,False{}),start_rank(s,False{}),True{},
          Equal.cong(U32,Bool,x => start_rank(x,False{}),31,s,Equal.sym(U32,s,31,eq)),rank))
    case 32n:
      +eq=Domain.injective(s,32,same)
      %Equal.sym(U32,s,32,eq) : {S.backrank(D.second(_,False{})) == False{} : Bool}
      Old.false_true({S.backrank(D.second(32,False{})) == False{} : Bool},
        Equal.trans(Bool,start_rank(32,False{}),start_rank(s,False{}),True{},
          Equal.cong(U32,Bool,x => start_rank(x,False{}),32,s,Equal.sym(U32,s,32,eq)),rank))
    case 33n:
      +eq=Domain.injective(s,33,same)
      %Equal.sym(U32,s,33,eq) : {S.backrank(D.second(_,False{})) == False{} : Bool}
      Old.false_true({S.backrank(D.second(33,False{})) == False{} : Bool},
        Equal.trans(Bool,start_rank(33,False{}),start_rank(s,False{}),True{},
          Equal.cong(U32,Bool,x => start_rank(x,False{}),33,s,Equal.sym(U32,s,33,eq)),rank))
    case 34n:
      +eq=Domain.injective(s,34,same)
      %Equal.sym(U32,s,34,eq) : {S.backrank(D.second(_,False{})) == False{} : Bool}
      Old.false_true({S.backrank(D.second(34,False{})) == False{} : Bool},
        Equal.trans(Bool,start_rank(34,False{}),start_rank(s,False{}),True{},
          Equal.cong(U32,Bool,x => start_rank(x,False{}),34,s,Equal.sym(U32,s,34,eq)),rank))
    case 35n:
      +eq=Domain.injective(s,35,same)
      %Equal.sym(U32,s,35,eq) : {S.backrank(D.second(_,False{})) == False{} : Bool}
      Old.false_true({S.backrank(D.second(35,False{})) == False{} : Bool},
        Equal.trans(Bool,start_rank(35,False{}),start_rank(s,False{}),True{},
          Equal.cong(U32,Bool,x => start_rank(x,False{}),35,s,Equal.sym(U32,s,35,eq)),rank))
    case 36n:
      +eq=Domain.injective(s,36,same)
      %Equal.sym(U32,s,36,eq) : {S.backrank(D.second(_,False{})) == False{} : Bool}
      Old.false_true({S.backrank(D.second(36,False{})) == False{} : Bool},
        Equal.trans(Bool,start_rank(36,False{}),start_rank(s,False{}),True{},
          Equal.cong(U32,Bool,x => start_rank(x,False{}),36,s,Equal.sym(U32,s,36,eq)),rank))
    case 37n:
      +eq=Domain.injective(s,37,same)
      %Equal.sym(U32,s,37,eq) : {S.backrank(D.second(_,False{})) == False{} : Bool}
      Old.false_true({S.backrank(D.second(37,False{})) == False{} : Bool},
        Equal.trans(Bool,start_rank(37,False{}),start_rank(s,False{}),True{},
          Equal.cong(U32,Bool,x => start_rank(x,False{}),37,s,Equal.sym(U32,s,37,eq)),rank))
    case 38n:
      +eq=Domain.injective(s,38,same)
      %Equal.sym(U32,s,38,eq) : {S.backrank(D.second(_,False{})) == False{} : Bool}
      Old.false_true({S.backrank(D.second(38,False{})) == False{} : Bool},
        Equal.trans(Bool,start_rank(38,False{}),start_rank(s,False{}),True{},
          Equal.cong(U32,Bool,x => start_rank(x,False{}),38,s,Equal.sym(U32,s,38,eq)),rank))
    case 39n:
      +eq=Domain.injective(s,39,same)
      %Equal.sym(U32,s,39,eq) : {S.backrank(D.second(_,False{})) == False{} : Bool}
      Old.false_true({S.backrank(D.second(39,False{})) == False{} : Bool},
        Equal.trans(Bool,start_rank(39,False{}),start_rank(s,False{}),True{},
          Equal.cong(U32,Bool,x => start_rank(x,False{}),39,s,Equal.sym(U32,s,39,eq)),rank))
    case 40n:
      +eq=Domain.injective(s,40,same)
      %Equal.sym(U32,s,40,eq) : {S.backrank(D.second(_,False{})) == False{} : Bool}
      Old.false_true({S.backrank(D.second(40,False{})) == False{} : Bool},
        Equal.trans(Bool,start_rank(40,False{}),start_rank(s,False{}),True{},
          Equal.cong(U32,Bool,x => start_rank(x,False{}),40,s,Equal.sym(U32,s,40,eq)),rank))
    case 41n:
      +eq=Domain.injective(s,41,same)
      %Equal.sym(U32,s,41,eq) : {S.backrank(D.second(_,False{})) == False{} : Bool}
      Old.false_true({S.backrank(D.second(41,False{})) == False{} : Bool},
        Equal.trans(Bool,start_rank(41,False{}),start_rank(s,False{}),True{},
          Equal.cong(U32,Bool,x => start_rank(x,False{}),41,s,Equal.sym(U32,s,41,eq)),rank))
    case 42n:
      +eq=Domain.injective(s,42,same)
      %Equal.sym(U32,s,42,eq) : {S.backrank(D.second(_,False{})) == False{} : Bool}
      Old.false_true({S.backrank(D.second(42,False{})) == False{} : Bool},
        Equal.trans(Bool,start_rank(42,False{}),start_rank(s,False{}),True{},
          Equal.cong(U32,Bool,x => start_rank(x,False{}),42,s,Equal.sym(U32,s,42,eq)),rank))
    case 43n:
      +eq=Domain.injective(s,43,same)
      %Equal.sym(U32,s,43,eq) : {S.backrank(D.second(_,False{})) == False{} : Bool}
      Old.false_true({S.backrank(D.second(43,False{})) == False{} : Bool},
        Equal.trans(Bool,start_rank(43,False{}),start_rank(s,False{}),True{},
          Equal.cong(U32,Bool,x => start_rank(x,False{}),43,s,Equal.sym(U32,s,43,eq)),rank))
    case 44n:
      +eq=Domain.injective(s,44,same)
      %Equal.sym(U32,s,44,eq) : {S.backrank(D.second(_,False{})) == False{} : Bool}
      Old.false_true({S.backrank(D.second(44,False{})) == False{} : Bool},
        Equal.trans(Bool,start_rank(44,False{}),start_rank(s,False{}),True{},
          Equal.cong(U32,Bool,x => start_rank(x,False{}),44,s,Equal.sym(U32,s,44,eq)),rank))
    case 45n:
      +eq=Domain.injective(s,45,same)
      %Equal.sym(U32,s,45,eq) : {S.backrank(D.second(_,False{})) == False{} : Bool}
      Old.false_true({S.backrank(D.second(45,False{})) == False{} : Bool},
        Equal.trans(Bool,start_rank(45,False{}),start_rank(s,False{}),True{},
          Equal.cong(U32,Bool,x => start_rank(x,False{}),45,s,Equal.sym(U32,s,45,eq)),rank))
    case 46n:
      +eq=Domain.injective(s,46,same)
      %Equal.sym(U32,s,46,eq) : {S.backrank(D.second(_,False{})) == False{} : Bool}
      Old.false_true({S.backrank(D.second(46,False{})) == False{} : Bool},
        Equal.trans(Bool,start_rank(46,False{}),start_rank(s,False{}),True{},
          Equal.cong(U32,Bool,x => start_rank(x,False{}),46,s,Equal.sym(U32,s,46,eq)),rank))
    case 47n:
      +eq=Domain.injective(s,47,same)
      %Equal.sym(U32,s,47,eq) : {S.backrank(D.second(_,False{})) == False{} : Bool}
      Old.false_true({S.backrank(D.second(47,False{})) == False{} : Bool},
        Equal.trans(Bool,start_rank(47,False{}),start_rank(s,False{}),True{},
          Equal.cong(U32,Bool,x => start_rank(x,False{}),47,s,Equal.sym(U32,s,47,eq)),rank))
    case 48n:
      +eq=Domain.injective(s,48,same)
      %Equal.sym(U32,s,48,eq) : {S.backrank(D.second(_,False{})) == False{} : Bool}
      {==}
    case 49n:
      +eq=Domain.injective(s,49,same)
      %Equal.sym(U32,s,49,eq) : {S.backrank(D.second(_,False{})) == False{} : Bool}
      {==}
    case 50n:
      +eq=Domain.injective(s,50,same)
      %Equal.sym(U32,s,50,eq) : {S.backrank(D.second(_,False{})) == False{} : Bool}
      {==}
    case 51n:
      +eq=Domain.injective(s,51,same)
      %Equal.sym(U32,s,51,eq) : {S.backrank(D.second(_,False{})) == False{} : Bool}
      {==}
    case 52n:
      +eq=Domain.injective(s,52,same)
      %Equal.sym(U32,s,52,eq) : {S.backrank(D.second(_,False{})) == False{} : Bool}
      {==}
    case 53n:
      +eq=Domain.injective(s,53,same)
      %Equal.sym(U32,s,53,eq) : {S.backrank(D.second(_,False{})) == False{} : Bool}
      {==}
    case 54n:
      +eq=Domain.injective(s,54,same)
      %Equal.sym(U32,s,54,eq) : {S.backrank(D.second(_,False{})) == False{} : Bool}
      {==}
    case 55n:
      +eq=Domain.injective(s,55,same)
      %Equal.sym(U32,s,55,eq) : {S.backrank(D.second(_,False{})) == False{} : Bool}
      {==}
    case 56n:
      +eq=Domain.injective(s,56,same)
      %Equal.sym(U32,s,56,eq) : {S.backrank(D.second(_,False{})) == False{} : Bool}
      Old.false_true({S.backrank(D.second(56,False{})) == False{} : Bool},
        Equal.trans(Bool,start_rank(56,False{}),start_rank(s,False{}),True{},
          Equal.cong(U32,Bool,x => start_rank(x,False{}),56,s,Equal.sym(U32,s,56,eq)),rank))
    case 57n:
      +eq=Domain.injective(s,57,same)
      %Equal.sym(U32,s,57,eq) : {S.backrank(D.second(_,False{})) == False{} : Bool}
      Old.false_true({S.backrank(D.second(57,False{})) == False{} : Bool},
        Equal.trans(Bool,start_rank(57,False{}),start_rank(s,False{}),True{},
          Equal.cong(U32,Bool,x => start_rank(x,False{}),57,s,Equal.sym(U32,s,57,eq)),rank))
    case 58n:
      +eq=Domain.injective(s,58,same)
      %Equal.sym(U32,s,58,eq) : {S.backrank(D.second(_,False{})) == False{} : Bool}
      Old.false_true({S.backrank(D.second(58,False{})) == False{} : Bool},
        Equal.trans(Bool,start_rank(58,False{}),start_rank(s,False{}),True{},
          Equal.cong(U32,Bool,x => start_rank(x,False{}),58,s,Equal.sym(U32,s,58,eq)),rank))
    case 59n:
      +eq=Domain.injective(s,59,same)
      %Equal.sym(U32,s,59,eq) : {S.backrank(D.second(_,False{})) == False{} : Bool}
      Old.false_true({S.backrank(D.second(59,False{})) == False{} : Bool},
        Equal.trans(Bool,start_rank(59,False{}),start_rank(s,False{}),True{},
          Equal.cong(U32,Bool,x => start_rank(x,False{}),59,s,Equal.sym(U32,s,59,eq)),rank))
    case 60n:
      +eq=Domain.injective(s,60,same)
      %Equal.sym(U32,s,60,eq) : {S.backrank(D.second(_,False{})) == False{} : Bool}
      Old.false_true({S.backrank(D.second(60,False{})) == False{} : Bool},
        Equal.trans(Bool,start_rank(60,False{}),start_rank(s,False{}),True{},
          Equal.cong(U32,Bool,x => start_rank(x,False{}),60,s,Equal.sym(U32,s,60,eq)),rank))
    case 61n:
      +eq=Domain.injective(s,61,same)
      %Equal.sym(U32,s,61,eq) : {S.backrank(D.second(_,False{})) == False{} : Bool}
      Old.false_true({S.backrank(D.second(61,False{})) == False{} : Bool},
        Equal.trans(Bool,start_rank(61,False{}),start_rank(s,False{}),True{},
          Equal.cong(U32,Bool,x => start_rank(x,False{}),61,s,Equal.sym(U32,s,61,eq)),rank))
    case 62n:
      +eq=Domain.injective(s,62,same)
      %Equal.sym(U32,s,62,eq) : {S.backrank(D.second(_,False{})) == False{} : Bool}
      Old.false_true({S.backrank(D.second(62,False{})) == False{} : Bool},
        Equal.trans(Bool,start_rank(62,False{}),start_rank(s,False{}),True{},
          Equal.cong(U32,Bool,x => start_rank(x,False{}),62,s,Equal.sym(U32,s,62,eq)),rank))
    case 63n:
      +eq=Domain.injective(s,63,same)
      %Equal.sym(U32,s,63,eq) : {S.backrank(D.second(_,False{})) == False{} : Bool}
      Old.false_true({S.backrank(D.second(63,False{})) == False{} : Bool},
        Equal.trans(Bool,start_rank(63,False{}),start_rank(s,False{}),True{},
          Equal.cong(U32,Bool,x => start_rank(x,False{}),63,s,Equal.sym(U32,s,63,eq)),rank))
    case 64n+p: B.impossible({S.backrank(D.second(s,False{})) == False{} : Bool},p,below)

def second(+s: U32,+w: Bool,below: {Nat.is_lt(U32.to_nat(s),64n) == True{} : Bool},rank: {start_rank(s,w) == True{} : Bool}) -> {S.backrank(D.second(s,w)) == False{} : Bool}:
  match w:
    case True{}: finite_white(s,U32.to_nat(s),{==},below,rank)
    case False{}: finite_black(s,U32.to_nat(s),{==},below,rank)
