# Finite start-file certificate contains only source geometry, never board planes.
import Base
import ./Ranks.bend as Ranks
import ../layout/Domain.bend as Domain
import ../decoder/Bits.bend as B
import ../u64/Words.bend as Old

type File is Data:
  A{}
  B{}
  C{}
  D{}
  E{}
  F{}
  G{}
  H{}

def source(file: File,w: Bool) -> U32:
  match file w:
    case A{} True{}: 8
    case A{} False{}: 48
    case B{} True{}: 9
    case B{} False{}: 49
    case C{} True{}: 10
    case C{} False{}: 50
    case D{} True{}: 11
    case D{} False{}: 51
    case E{} True{}: 12
    case E{} False{}: 52
    case F{} True{}: 13
    case F{} False{}: 53
    case G{} True{}: 14
    case G{} False{}: 54
    case H{} True{}: 15
    case H{} False{}: 55

def finite_white(+s: U32,n: Nat,same: {U32.to_nat(s) == n : Nat},below: {Nat.is_lt(n,64n) == True{} : Bool},rank: {Ranks.start_rank(s,True{}) == True{} : Bool}) -> Exists(File,f => {s == source(f,True{}) : U32}):
  match n:
    case 0n:
      +eq=Domain.injective(s,0,same)
      Old.false_true(Exists(File,f => {s == source(f,True{}) : U32}),Equal.trans(Bool,Ranks.start_rank(0,True{}),Ranks.start_rank(s,True{}),True{},Equal.cong(U32,Bool,x => Ranks.start_rank(x,True{}),0,s,Equal.sym(U32,s,0,eq)),rank))
    case 1n:
      +eq=Domain.injective(s,1,same)
      Old.false_true(Exists(File,f => {s == source(f,True{}) : U32}),Equal.trans(Bool,Ranks.start_rank(1,True{}),Ranks.start_rank(s,True{}),True{},Equal.cong(U32,Bool,x => Ranks.start_rank(x,True{}),1,s,Equal.sym(U32,s,1,eq)),rank))
    case 2n:
      +eq=Domain.injective(s,2,same)
      Old.false_true(Exists(File,f => {s == source(f,True{}) : U32}),Equal.trans(Bool,Ranks.start_rank(2,True{}),Ranks.start_rank(s,True{}),True{},Equal.cong(U32,Bool,x => Ranks.start_rank(x,True{}),2,s,Equal.sym(U32,s,2,eq)),rank))
    case 3n:
      +eq=Domain.injective(s,3,same)
      Old.false_true(Exists(File,f => {s == source(f,True{}) : U32}),Equal.trans(Bool,Ranks.start_rank(3,True{}),Ranks.start_rank(s,True{}),True{},Equal.cong(U32,Bool,x => Ranks.start_rank(x,True{}),3,s,Equal.sym(U32,s,3,eq)),rank))
    case 4n:
      +eq=Domain.injective(s,4,same)
      Old.false_true(Exists(File,f => {s == source(f,True{}) : U32}),Equal.trans(Bool,Ranks.start_rank(4,True{}),Ranks.start_rank(s,True{}),True{},Equal.cong(U32,Bool,x => Ranks.start_rank(x,True{}),4,s,Equal.sym(U32,s,4,eq)),rank))
    case 5n:
      +eq=Domain.injective(s,5,same)
      Old.false_true(Exists(File,f => {s == source(f,True{}) : U32}),Equal.trans(Bool,Ranks.start_rank(5,True{}),Ranks.start_rank(s,True{}),True{},Equal.cong(U32,Bool,x => Ranks.start_rank(x,True{}),5,s,Equal.sym(U32,s,5,eq)),rank))
    case 6n:
      +eq=Domain.injective(s,6,same)
      Old.false_true(Exists(File,f => {s == source(f,True{}) : U32}),Equal.trans(Bool,Ranks.start_rank(6,True{}),Ranks.start_rank(s,True{}),True{},Equal.cong(U32,Bool,x => Ranks.start_rank(x,True{}),6,s,Equal.sym(U32,s,6,eq)),rank))
    case 7n:
      +eq=Domain.injective(s,7,same)
      Old.false_true(Exists(File,f => {s == source(f,True{}) : U32}),Equal.trans(Bool,Ranks.start_rank(7,True{}),Ranks.start_rank(s,True{}),True{},Equal.cong(U32,Bool,x => Ranks.start_rank(x,True{}),7,s,Equal.sym(U32,s,7,eq)),rank))
    case 8n:
      +eq=Domain.injective(s,8,same)
      (A{},eq)
    case 9n:
      +eq=Domain.injective(s,9,same)
      (B{},eq)
    case 10n:
      +eq=Domain.injective(s,10,same)
      (C{},eq)
    case 11n:
      +eq=Domain.injective(s,11,same)
      (D{},eq)
    case 12n:
      +eq=Domain.injective(s,12,same)
      (E{},eq)
    case 13n:
      +eq=Domain.injective(s,13,same)
      (F{},eq)
    case 14n:
      +eq=Domain.injective(s,14,same)
      (G{},eq)
    case 15n:
      +eq=Domain.injective(s,15,same)
      (H{},eq)
    case 16n:
      +eq=Domain.injective(s,16,same)
      Old.false_true(Exists(File,f => {s == source(f,True{}) : U32}),Equal.trans(Bool,Ranks.start_rank(16,True{}),Ranks.start_rank(s,True{}),True{},Equal.cong(U32,Bool,x => Ranks.start_rank(x,True{}),16,s,Equal.sym(U32,s,16,eq)),rank))
    case 17n:
      +eq=Domain.injective(s,17,same)
      Old.false_true(Exists(File,f => {s == source(f,True{}) : U32}),Equal.trans(Bool,Ranks.start_rank(17,True{}),Ranks.start_rank(s,True{}),True{},Equal.cong(U32,Bool,x => Ranks.start_rank(x,True{}),17,s,Equal.sym(U32,s,17,eq)),rank))
    case 18n:
      +eq=Domain.injective(s,18,same)
      Old.false_true(Exists(File,f => {s == source(f,True{}) : U32}),Equal.trans(Bool,Ranks.start_rank(18,True{}),Ranks.start_rank(s,True{}),True{},Equal.cong(U32,Bool,x => Ranks.start_rank(x,True{}),18,s,Equal.sym(U32,s,18,eq)),rank))
    case 19n:
      +eq=Domain.injective(s,19,same)
      Old.false_true(Exists(File,f => {s == source(f,True{}) : U32}),Equal.trans(Bool,Ranks.start_rank(19,True{}),Ranks.start_rank(s,True{}),True{},Equal.cong(U32,Bool,x => Ranks.start_rank(x,True{}),19,s,Equal.sym(U32,s,19,eq)),rank))
    case 20n:
      +eq=Domain.injective(s,20,same)
      Old.false_true(Exists(File,f => {s == source(f,True{}) : U32}),Equal.trans(Bool,Ranks.start_rank(20,True{}),Ranks.start_rank(s,True{}),True{},Equal.cong(U32,Bool,x => Ranks.start_rank(x,True{}),20,s,Equal.sym(U32,s,20,eq)),rank))
    case 21n:
      +eq=Domain.injective(s,21,same)
      Old.false_true(Exists(File,f => {s == source(f,True{}) : U32}),Equal.trans(Bool,Ranks.start_rank(21,True{}),Ranks.start_rank(s,True{}),True{},Equal.cong(U32,Bool,x => Ranks.start_rank(x,True{}),21,s,Equal.sym(U32,s,21,eq)),rank))
    case 22n:
      +eq=Domain.injective(s,22,same)
      Old.false_true(Exists(File,f => {s == source(f,True{}) : U32}),Equal.trans(Bool,Ranks.start_rank(22,True{}),Ranks.start_rank(s,True{}),True{},Equal.cong(U32,Bool,x => Ranks.start_rank(x,True{}),22,s,Equal.sym(U32,s,22,eq)),rank))
    case 23n:
      +eq=Domain.injective(s,23,same)
      Old.false_true(Exists(File,f => {s == source(f,True{}) : U32}),Equal.trans(Bool,Ranks.start_rank(23,True{}),Ranks.start_rank(s,True{}),True{},Equal.cong(U32,Bool,x => Ranks.start_rank(x,True{}),23,s,Equal.sym(U32,s,23,eq)),rank))
    case 24n:
      +eq=Domain.injective(s,24,same)
      Old.false_true(Exists(File,f => {s == source(f,True{}) : U32}),Equal.trans(Bool,Ranks.start_rank(24,True{}),Ranks.start_rank(s,True{}),True{},Equal.cong(U32,Bool,x => Ranks.start_rank(x,True{}),24,s,Equal.sym(U32,s,24,eq)),rank))
    case 25n:
      +eq=Domain.injective(s,25,same)
      Old.false_true(Exists(File,f => {s == source(f,True{}) : U32}),Equal.trans(Bool,Ranks.start_rank(25,True{}),Ranks.start_rank(s,True{}),True{},Equal.cong(U32,Bool,x => Ranks.start_rank(x,True{}),25,s,Equal.sym(U32,s,25,eq)),rank))
    case 26n:
      +eq=Domain.injective(s,26,same)
      Old.false_true(Exists(File,f => {s == source(f,True{}) : U32}),Equal.trans(Bool,Ranks.start_rank(26,True{}),Ranks.start_rank(s,True{}),True{},Equal.cong(U32,Bool,x => Ranks.start_rank(x,True{}),26,s,Equal.sym(U32,s,26,eq)),rank))
    case 27n:
      +eq=Domain.injective(s,27,same)
      Old.false_true(Exists(File,f => {s == source(f,True{}) : U32}),Equal.trans(Bool,Ranks.start_rank(27,True{}),Ranks.start_rank(s,True{}),True{},Equal.cong(U32,Bool,x => Ranks.start_rank(x,True{}),27,s,Equal.sym(U32,s,27,eq)),rank))
    case 28n:
      +eq=Domain.injective(s,28,same)
      Old.false_true(Exists(File,f => {s == source(f,True{}) : U32}),Equal.trans(Bool,Ranks.start_rank(28,True{}),Ranks.start_rank(s,True{}),True{},Equal.cong(U32,Bool,x => Ranks.start_rank(x,True{}),28,s,Equal.sym(U32,s,28,eq)),rank))
    case 29n:
      +eq=Domain.injective(s,29,same)
      Old.false_true(Exists(File,f => {s == source(f,True{}) : U32}),Equal.trans(Bool,Ranks.start_rank(29,True{}),Ranks.start_rank(s,True{}),True{},Equal.cong(U32,Bool,x => Ranks.start_rank(x,True{}),29,s,Equal.sym(U32,s,29,eq)),rank))
    case 30n:
      +eq=Domain.injective(s,30,same)
      Old.false_true(Exists(File,f => {s == source(f,True{}) : U32}),Equal.trans(Bool,Ranks.start_rank(30,True{}),Ranks.start_rank(s,True{}),True{},Equal.cong(U32,Bool,x => Ranks.start_rank(x,True{}),30,s,Equal.sym(U32,s,30,eq)),rank))
    case 31n:
      +eq=Domain.injective(s,31,same)
      Old.false_true(Exists(File,f => {s == source(f,True{}) : U32}),Equal.trans(Bool,Ranks.start_rank(31,True{}),Ranks.start_rank(s,True{}),True{},Equal.cong(U32,Bool,x => Ranks.start_rank(x,True{}),31,s,Equal.sym(U32,s,31,eq)),rank))
    case 32n:
      +eq=Domain.injective(s,32,same)
      Old.false_true(Exists(File,f => {s == source(f,True{}) : U32}),Equal.trans(Bool,Ranks.start_rank(32,True{}),Ranks.start_rank(s,True{}),True{},Equal.cong(U32,Bool,x => Ranks.start_rank(x,True{}),32,s,Equal.sym(U32,s,32,eq)),rank))
    case 33n:
      +eq=Domain.injective(s,33,same)
      Old.false_true(Exists(File,f => {s == source(f,True{}) : U32}),Equal.trans(Bool,Ranks.start_rank(33,True{}),Ranks.start_rank(s,True{}),True{},Equal.cong(U32,Bool,x => Ranks.start_rank(x,True{}),33,s,Equal.sym(U32,s,33,eq)),rank))
    case 34n:
      +eq=Domain.injective(s,34,same)
      Old.false_true(Exists(File,f => {s == source(f,True{}) : U32}),Equal.trans(Bool,Ranks.start_rank(34,True{}),Ranks.start_rank(s,True{}),True{},Equal.cong(U32,Bool,x => Ranks.start_rank(x,True{}),34,s,Equal.sym(U32,s,34,eq)),rank))
    case 35n:
      +eq=Domain.injective(s,35,same)
      Old.false_true(Exists(File,f => {s == source(f,True{}) : U32}),Equal.trans(Bool,Ranks.start_rank(35,True{}),Ranks.start_rank(s,True{}),True{},Equal.cong(U32,Bool,x => Ranks.start_rank(x,True{}),35,s,Equal.sym(U32,s,35,eq)),rank))
    case 36n:
      +eq=Domain.injective(s,36,same)
      Old.false_true(Exists(File,f => {s == source(f,True{}) : U32}),Equal.trans(Bool,Ranks.start_rank(36,True{}),Ranks.start_rank(s,True{}),True{},Equal.cong(U32,Bool,x => Ranks.start_rank(x,True{}),36,s,Equal.sym(U32,s,36,eq)),rank))
    case 37n:
      +eq=Domain.injective(s,37,same)
      Old.false_true(Exists(File,f => {s == source(f,True{}) : U32}),Equal.trans(Bool,Ranks.start_rank(37,True{}),Ranks.start_rank(s,True{}),True{},Equal.cong(U32,Bool,x => Ranks.start_rank(x,True{}),37,s,Equal.sym(U32,s,37,eq)),rank))
    case 38n:
      +eq=Domain.injective(s,38,same)
      Old.false_true(Exists(File,f => {s == source(f,True{}) : U32}),Equal.trans(Bool,Ranks.start_rank(38,True{}),Ranks.start_rank(s,True{}),True{},Equal.cong(U32,Bool,x => Ranks.start_rank(x,True{}),38,s,Equal.sym(U32,s,38,eq)),rank))
    case 39n:
      +eq=Domain.injective(s,39,same)
      Old.false_true(Exists(File,f => {s == source(f,True{}) : U32}),Equal.trans(Bool,Ranks.start_rank(39,True{}),Ranks.start_rank(s,True{}),True{},Equal.cong(U32,Bool,x => Ranks.start_rank(x,True{}),39,s,Equal.sym(U32,s,39,eq)),rank))
    case 40n:
      +eq=Domain.injective(s,40,same)
      Old.false_true(Exists(File,f => {s == source(f,True{}) : U32}),Equal.trans(Bool,Ranks.start_rank(40,True{}),Ranks.start_rank(s,True{}),True{},Equal.cong(U32,Bool,x => Ranks.start_rank(x,True{}),40,s,Equal.sym(U32,s,40,eq)),rank))
    case 41n:
      +eq=Domain.injective(s,41,same)
      Old.false_true(Exists(File,f => {s == source(f,True{}) : U32}),Equal.trans(Bool,Ranks.start_rank(41,True{}),Ranks.start_rank(s,True{}),True{},Equal.cong(U32,Bool,x => Ranks.start_rank(x,True{}),41,s,Equal.sym(U32,s,41,eq)),rank))
    case 42n:
      +eq=Domain.injective(s,42,same)
      Old.false_true(Exists(File,f => {s == source(f,True{}) : U32}),Equal.trans(Bool,Ranks.start_rank(42,True{}),Ranks.start_rank(s,True{}),True{},Equal.cong(U32,Bool,x => Ranks.start_rank(x,True{}),42,s,Equal.sym(U32,s,42,eq)),rank))
    case 43n:
      +eq=Domain.injective(s,43,same)
      Old.false_true(Exists(File,f => {s == source(f,True{}) : U32}),Equal.trans(Bool,Ranks.start_rank(43,True{}),Ranks.start_rank(s,True{}),True{},Equal.cong(U32,Bool,x => Ranks.start_rank(x,True{}),43,s,Equal.sym(U32,s,43,eq)),rank))
    case 44n:
      +eq=Domain.injective(s,44,same)
      Old.false_true(Exists(File,f => {s == source(f,True{}) : U32}),Equal.trans(Bool,Ranks.start_rank(44,True{}),Ranks.start_rank(s,True{}),True{},Equal.cong(U32,Bool,x => Ranks.start_rank(x,True{}),44,s,Equal.sym(U32,s,44,eq)),rank))
    case 45n:
      +eq=Domain.injective(s,45,same)
      Old.false_true(Exists(File,f => {s == source(f,True{}) : U32}),Equal.trans(Bool,Ranks.start_rank(45,True{}),Ranks.start_rank(s,True{}),True{},Equal.cong(U32,Bool,x => Ranks.start_rank(x,True{}),45,s,Equal.sym(U32,s,45,eq)),rank))
    case 46n:
      +eq=Domain.injective(s,46,same)
      Old.false_true(Exists(File,f => {s == source(f,True{}) : U32}),Equal.trans(Bool,Ranks.start_rank(46,True{}),Ranks.start_rank(s,True{}),True{},Equal.cong(U32,Bool,x => Ranks.start_rank(x,True{}),46,s,Equal.sym(U32,s,46,eq)),rank))
    case 47n:
      +eq=Domain.injective(s,47,same)
      Old.false_true(Exists(File,f => {s == source(f,True{}) : U32}),Equal.trans(Bool,Ranks.start_rank(47,True{}),Ranks.start_rank(s,True{}),True{},Equal.cong(U32,Bool,x => Ranks.start_rank(x,True{}),47,s,Equal.sym(U32,s,47,eq)),rank))
    case 48n:
      +eq=Domain.injective(s,48,same)
      Old.false_true(Exists(File,f => {s == source(f,True{}) : U32}),Equal.trans(Bool,Ranks.start_rank(48,True{}),Ranks.start_rank(s,True{}),True{},Equal.cong(U32,Bool,x => Ranks.start_rank(x,True{}),48,s,Equal.sym(U32,s,48,eq)),rank))
    case 49n:
      +eq=Domain.injective(s,49,same)
      Old.false_true(Exists(File,f => {s == source(f,True{}) : U32}),Equal.trans(Bool,Ranks.start_rank(49,True{}),Ranks.start_rank(s,True{}),True{},Equal.cong(U32,Bool,x => Ranks.start_rank(x,True{}),49,s,Equal.sym(U32,s,49,eq)),rank))
    case 50n:
      +eq=Domain.injective(s,50,same)
      Old.false_true(Exists(File,f => {s == source(f,True{}) : U32}),Equal.trans(Bool,Ranks.start_rank(50,True{}),Ranks.start_rank(s,True{}),True{},Equal.cong(U32,Bool,x => Ranks.start_rank(x,True{}),50,s,Equal.sym(U32,s,50,eq)),rank))
    case 51n:
      +eq=Domain.injective(s,51,same)
      Old.false_true(Exists(File,f => {s == source(f,True{}) : U32}),Equal.trans(Bool,Ranks.start_rank(51,True{}),Ranks.start_rank(s,True{}),True{},Equal.cong(U32,Bool,x => Ranks.start_rank(x,True{}),51,s,Equal.sym(U32,s,51,eq)),rank))
    case 52n:
      +eq=Domain.injective(s,52,same)
      Old.false_true(Exists(File,f => {s == source(f,True{}) : U32}),Equal.trans(Bool,Ranks.start_rank(52,True{}),Ranks.start_rank(s,True{}),True{},Equal.cong(U32,Bool,x => Ranks.start_rank(x,True{}),52,s,Equal.sym(U32,s,52,eq)),rank))
    case 53n:
      +eq=Domain.injective(s,53,same)
      Old.false_true(Exists(File,f => {s == source(f,True{}) : U32}),Equal.trans(Bool,Ranks.start_rank(53,True{}),Ranks.start_rank(s,True{}),True{},Equal.cong(U32,Bool,x => Ranks.start_rank(x,True{}),53,s,Equal.sym(U32,s,53,eq)),rank))
    case 54n:
      +eq=Domain.injective(s,54,same)
      Old.false_true(Exists(File,f => {s == source(f,True{}) : U32}),Equal.trans(Bool,Ranks.start_rank(54,True{}),Ranks.start_rank(s,True{}),True{},Equal.cong(U32,Bool,x => Ranks.start_rank(x,True{}),54,s,Equal.sym(U32,s,54,eq)),rank))
    case 55n:
      +eq=Domain.injective(s,55,same)
      Old.false_true(Exists(File,f => {s == source(f,True{}) : U32}),Equal.trans(Bool,Ranks.start_rank(55,True{}),Ranks.start_rank(s,True{}),True{},Equal.cong(U32,Bool,x => Ranks.start_rank(x,True{}),55,s,Equal.sym(U32,s,55,eq)),rank))
    case 56n:
      +eq=Domain.injective(s,56,same)
      Old.false_true(Exists(File,f => {s == source(f,True{}) : U32}),Equal.trans(Bool,Ranks.start_rank(56,True{}),Ranks.start_rank(s,True{}),True{},Equal.cong(U32,Bool,x => Ranks.start_rank(x,True{}),56,s,Equal.sym(U32,s,56,eq)),rank))
    case 57n:
      +eq=Domain.injective(s,57,same)
      Old.false_true(Exists(File,f => {s == source(f,True{}) : U32}),Equal.trans(Bool,Ranks.start_rank(57,True{}),Ranks.start_rank(s,True{}),True{},Equal.cong(U32,Bool,x => Ranks.start_rank(x,True{}),57,s,Equal.sym(U32,s,57,eq)),rank))
    case 58n:
      +eq=Domain.injective(s,58,same)
      Old.false_true(Exists(File,f => {s == source(f,True{}) : U32}),Equal.trans(Bool,Ranks.start_rank(58,True{}),Ranks.start_rank(s,True{}),True{},Equal.cong(U32,Bool,x => Ranks.start_rank(x,True{}),58,s,Equal.sym(U32,s,58,eq)),rank))
    case 59n:
      +eq=Domain.injective(s,59,same)
      Old.false_true(Exists(File,f => {s == source(f,True{}) : U32}),Equal.trans(Bool,Ranks.start_rank(59,True{}),Ranks.start_rank(s,True{}),True{},Equal.cong(U32,Bool,x => Ranks.start_rank(x,True{}),59,s,Equal.sym(U32,s,59,eq)),rank))
    case 60n:
      +eq=Domain.injective(s,60,same)
      Old.false_true(Exists(File,f => {s == source(f,True{}) : U32}),Equal.trans(Bool,Ranks.start_rank(60,True{}),Ranks.start_rank(s,True{}),True{},Equal.cong(U32,Bool,x => Ranks.start_rank(x,True{}),60,s,Equal.sym(U32,s,60,eq)),rank))
    case 61n:
      +eq=Domain.injective(s,61,same)
      Old.false_true(Exists(File,f => {s == source(f,True{}) : U32}),Equal.trans(Bool,Ranks.start_rank(61,True{}),Ranks.start_rank(s,True{}),True{},Equal.cong(U32,Bool,x => Ranks.start_rank(x,True{}),61,s,Equal.sym(U32,s,61,eq)),rank))
    case 62n:
      +eq=Domain.injective(s,62,same)
      Old.false_true(Exists(File,f => {s == source(f,True{}) : U32}),Equal.trans(Bool,Ranks.start_rank(62,True{}),Ranks.start_rank(s,True{}),True{},Equal.cong(U32,Bool,x => Ranks.start_rank(x,True{}),62,s,Equal.sym(U32,s,62,eq)),rank))
    case 63n:
      +eq=Domain.injective(s,63,same)
      Old.false_true(Exists(File,f => {s == source(f,True{}) : U32}),Equal.trans(Bool,Ranks.start_rank(63,True{}),Ranks.start_rank(s,True{}),True{},Equal.cong(U32,Bool,x => Ranks.start_rank(x,True{}),63,s,Equal.sym(U32,s,63,eq)),rank))
    case 64n+p: B.impossible(Exists(File,f => {s == source(f,True{}) : U32}),p,below)

def finite_black(+s: U32,n: Nat,same: {U32.to_nat(s) == n : Nat},below: {Nat.is_lt(n,64n) == True{} : Bool},rank: {Ranks.start_rank(s,False{}) == True{} : Bool}) -> Exists(File,f => {s == source(f,False{}) : U32}):
  match n:
    case 0n:
      +eq=Domain.injective(s,0,same)
      Old.false_true(Exists(File,f => {s == source(f,False{}) : U32}),Equal.trans(Bool,Ranks.start_rank(0,False{}),Ranks.start_rank(s,False{}),True{},Equal.cong(U32,Bool,x => Ranks.start_rank(x,False{}),0,s,Equal.sym(U32,s,0,eq)),rank))
    case 1n:
      +eq=Domain.injective(s,1,same)
      Old.false_true(Exists(File,f => {s == source(f,False{}) : U32}),Equal.trans(Bool,Ranks.start_rank(1,False{}),Ranks.start_rank(s,False{}),True{},Equal.cong(U32,Bool,x => Ranks.start_rank(x,False{}),1,s,Equal.sym(U32,s,1,eq)),rank))
    case 2n:
      +eq=Domain.injective(s,2,same)
      Old.false_true(Exists(File,f => {s == source(f,False{}) : U32}),Equal.trans(Bool,Ranks.start_rank(2,False{}),Ranks.start_rank(s,False{}),True{},Equal.cong(U32,Bool,x => Ranks.start_rank(x,False{}),2,s,Equal.sym(U32,s,2,eq)),rank))
    case 3n:
      +eq=Domain.injective(s,3,same)
      Old.false_true(Exists(File,f => {s == source(f,False{}) : U32}),Equal.trans(Bool,Ranks.start_rank(3,False{}),Ranks.start_rank(s,False{}),True{},Equal.cong(U32,Bool,x => Ranks.start_rank(x,False{}),3,s,Equal.sym(U32,s,3,eq)),rank))
    case 4n:
      +eq=Domain.injective(s,4,same)
      Old.false_true(Exists(File,f => {s == source(f,False{}) : U32}),Equal.trans(Bool,Ranks.start_rank(4,False{}),Ranks.start_rank(s,False{}),True{},Equal.cong(U32,Bool,x => Ranks.start_rank(x,False{}),4,s,Equal.sym(U32,s,4,eq)),rank))
    case 5n:
      +eq=Domain.injective(s,5,same)
      Old.false_true(Exists(File,f => {s == source(f,False{}) : U32}),Equal.trans(Bool,Ranks.start_rank(5,False{}),Ranks.start_rank(s,False{}),True{},Equal.cong(U32,Bool,x => Ranks.start_rank(x,False{}),5,s,Equal.sym(U32,s,5,eq)),rank))
    case 6n:
      +eq=Domain.injective(s,6,same)
      Old.false_true(Exists(File,f => {s == source(f,False{}) : U32}),Equal.trans(Bool,Ranks.start_rank(6,False{}),Ranks.start_rank(s,False{}),True{},Equal.cong(U32,Bool,x => Ranks.start_rank(x,False{}),6,s,Equal.sym(U32,s,6,eq)),rank))
    case 7n:
      +eq=Domain.injective(s,7,same)
      Old.false_true(Exists(File,f => {s == source(f,False{}) : U32}),Equal.trans(Bool,Ranks.start_rank(7,False{}),Ranks.start_rank(s,False{}),True{},Equal.cong(U32,Bool,x => Ranks.start_rank(x,False{}),7,s,Equal.sym(U32,s,7,eq)),rank))
    case 8n:
      +eq=Domain.injective(s,8,same)
      Old.false_true(Exists(File,f => {s == source(f,False{}) : U32}),Equal.trans(Bool,Ranks.start_rank(8,False{}),Ranks.start_rank(s,False{}),True{},Equal.cong(U32,Bool,x => Ranks.start_rank(x,False{}),8,s,Equal.sym(U32,s,8,eq)),rank))
    case 9n:
      +eq=Domain.injective(s,9,same)
      Old.false_true(Exists(File,f => {s == source(f,False{}) : U32}),Equal.trans(Bool,Ranks.start_rank(9,False{}),Ranks.start_rank(s,False{}),True{},Equal.cong(U32,Bool,x => Ranks.start_rank(x,False{}),9,s,Equal.sym(U32,s,9,eq)),rank))
    case 10n:
      +eq=Domain.injective(s,10,same)
      Old.false_true(Exists(File,f => {s == source(f,False{}) : U32}),Equal.trans(Bool,Ranks.start_rank(10,False{}),Ranks.start_rank(s,False{}),True{},Equal.cong(U32,Bool,x => Ranks.start_rank(x,False{}),10,s,Equal.sym(U32,s,10,eq)),rank))
    case 11n:
      +eq=Domain.injective(s,11,same)
      Old.false_true(Exists(File,f => {s == source(f,False{}) : U32}),Equal.trans(Bool,Ranks.start_rank(11,False{}),Ranks.start_rank(s,False{}),True{},Equal.cong(U32,Bool,x => Ranks.start_rank(x,False{}),11,s,Equal.sym(U32,s,11,eq)),rank))
    case 12n:
      +eq=Domain.injective(s,12,same)
      Old.false_true(Exists(File,f => {s == source(f,False{}) : U32}),Equal.trans(Bool,Ranks.start_rank(12,False{}),Ranks.start_rank(s,False{}),True{},Equal.cong(U32,Bool,x => Ranks.start_rank(x,False{}),12,s,Equal.sym(U32,s,12,eq)),rank))
    case 13n:
      +eq=Domain.injective(s,13,same)
      Old.false_true(Exists(File,f => {s == source(f,False{}) : U32}),Equal.trans(Bool,Ranks.start_rank(13,False{}),Ranks.start_rank(s,False{}),True{},Equal.cong(U32,Bool,x => Ranks.start_rank(x,False{}),13,s,Equal.sym(U32,s,13,eq)),rank))
    case 14n:
      +eq=Domain.injective(s,14,same)
      Old.false_true(Exists(File,f => {s == source(f,False{}) : U32}),Equal.trans(Bool,Ranks.start_rank(14,False{}),Ranks.start_rank(s,False{}),True{},Equal.cong(U32,Bool,x => Ranks.start_rank(x,False{}),14,s,Equal.sym(U32,s,14,eq)),rank))
    case 15n:
      +eq=Domain.injective(s,15,same)
      Old.false_true(Exists(File,f => {s == source(f,False{}) : U32}),Equal.trans(Bool,Ranks.start_rank(15,False{}),Ranks.start_rank(s,False{}),True{},Equal.cong(U32,Bool,x => Ranks.start_rank(x,False{}),15,s,Equal.sym(U32,s,15,eq)),rank))
    case 16n:
      +eq=Domain.injective(s,16,same)
      Old.false_true(Exists(File,f => {s == source(f,False{}) : U32}),Equal.trans(Bool,Ranks.start_rank(16,False{}),Ranks.start_rank(s,False{}),True{},Equal.cong(U32,Bool,x => Ranks.start_rank(x,False{}),16,s,Equal.sym(U32,s,16,eq)),rank))
    case 17n:
      +eq=Domain.injective(s,17,same)
      Old.false_true(Exists(File,f => {s == source(f,False{}) : U32}),Equal.trans(Bool,Ranks.start_rank(17,False{}),Ranks.start_rank(s,False{}),True{},Equal.cong(U32,Bool,x => Ranks.start_rank(x,False{}),17,s,Equal.sym(U32,s,17,eq)),rank))
    case 18n:
      +eq=Domain.injective(s,18,same)
      Old.false_true(Exists(File,f => {s == source(f,False{}) : U32}),Equal.trans(Bool,Ranks.start_rank(18,False{}),Ranks.start_rank(s,False{}),True{},Equal.cong(U32,Bool,x => Ranks.start_rank(x,False{}),18,s,Equal.sym(U32,s,18,eq)),rank))
    case 19n:
      +eq=Domain.injective(s,19,same)
      Old.false_true(Exists(File,f => {s == source(f,False{}) : U32}),Equal.trans(Bool,Ranks.start_rank(19,False{}),Ranks.start_rank(s,False{}),True{},Equal.cong(U32,Bool,x => Ranks.start_rank(x,False{}),19,s,Equal.sym(U32,s,19,eq)),rank))
    case 20n:
      +eq=Domain.injective(s,20,same)
      Old.false_true(Exists(File,f => {s == source(f,False{}) : U32}),Equal.trans(Bool,Ranks.start_rank(20,False{}),Ranks.start_rank(s,False{}),True{},Equal.cong(U32,Bool,x => Ranks.start_rank(x,False{}),20,s,Equal.sym(U32,s,20,eq)),rank))
    case 21n:
      +eq=Domain.injective(s,21,same)
      Old.false_true(Exists(File,f => {s == source(f,False{}) : U32}),Equal.trans(Bool,Ranks.start_rank(21,False{}),Ranks.start_rank(s,False{}),True{},Equal.cong(U32,Bool,x => Ranks.start_rank(x,False{}),21,s,Equal.sym(U32,s,21,eq)),rank))
    case 22n:
      +eq=Domain.injective(s,22,same)
      Old.false_true(Exists(File,f => {s == source(f,False{}) : U32}),Equal.trans(Bool,Ranks.start_rank(22,False{}),Ranks.start_rank(s,False{}),True{},Equal.cong(U32,Bool,x => Ranks.start_rank(x,False{}),22,s,Equal.sym(U32,s,22,eq)),rank))
    case 23n:
      +eq=Domain.injective(s,23,same)
      Old.false_true(Exists(File,f => {s == source(f,False{}) : U32}),Equal.trans(Bool,Ranks.start_rank(23,False{}),Ranks.start_rank(s,False{}),True{},Equal.cong(U32,Bool,x => Ranks.start_rank(x,False{}),23,s,Equal.sym(U32,s,23,eq)),rank))
    case 24n:
      +eq=Domain.injective(s,24,same)
      Old.false_true(Exists(File,f => {s == source(f,False{}) : U32}),Equal.trans(Bool,Ranks.start_rank(24,False{}),Ranks.start_rank(s,False{}),True{},Equal.cong(U32,Bool,x => Ranks.start_rank(x,False{}),24,s,Equal.sym(U32,s,24,eq)),rank))
    case 25n:
      +eq=Domain.injective(s,25,same)
      Old.false_true(Exists(File,f => {s == source(f,False{}) : U32}),Equal.trans(Bool,Ranks.start_rank(25,False{}),Ranks.start_rank(s,False{}),True{},Equal.cong(U32,Bool,x => Ranks.start_rank(x,False{}),25,s,Equal.sym(U32,s,25,eq)),rank))
    case 26n:
      +eq=Domain.injective(s,26,same)
      Old.false_true(Exists(File,f => {s == source(f,False{}) : U32}),Equal.trans(Bool,Ranks.start_rank(26,False{}),Ranks.start_rank(s,False{}),True{},Equal.cong(U32,Bool,x => Ranks.start_rank(x,False{}),26,s,Equal.sym(U32,s,26,eq)),rank))
    case 27n:
      +eq=Domain.injective(s,27,same)
      Old.false_true(Exists(File,f => {s == source(f,False{}) : U32}),Equal.trans(Bool,Ranks.start_rank(27,False{}),Ranks.start_rank(s,False{}),True{},Equal.cong(U32,Bool,x => Ranks.start_rank(x,False{}),27,s,Equal.sym(U32,s,27,eq)),rank))
    case 28n:
      +eq=Domain.injective(s,28,same)
      Old.false_true(Exists(File,f => {s == source(f,False{}) : U32}),Equal.trans(Bool,Ranks.start_rank(28,False{}),Ranks.start_rank(s,False{}),True{},Equal.cong(U32,Bool,x => Ranks.start_rank(x,False{}),28,s,Equal.sym(U32,s,28,eq)),rank))
    case 29n:
      +eq=Domain.injective(s,29,same)
      Old.false_true(Exists(File,f => {s == source(f,False{}) : U32}),Equal.trans(Bool,Ranks.start_rank(29,False{}),Ranks.start_rank(s,False{}),True{},Equal.cong(U32,Bool,x => Ranks.start_rank(x,False{}),29,s,Equal.sym(U32,s,29,eq)),rank))
    case 30n:
      +eq=Domain.injective(s,30,same)
      Old.false_true(Exists(File,f => {s == source(f,False{}) : U32}),Equal.trans(Bool,Ranks.start_rank(30,False{}),Ranks.start_rank(s,False{}),True{},Equal.cong(U32,Bool,x => Ranks.start_rank(x,False{}),30,s,Equal.sym(U32,s,30,eq)),rank))
    case 31n:
      +eq=Domain.injective(s,31,same)
      Old.false_true(Exists(File,f => {s == source(f,False{}) : U32}),Equal.trans(Bool,Ranks.start_rank(31,False{}),Ranks.start_rank(s,False{}),True{},Equal.cong(U32,Bool,x => Ranks.start_rank(x,False{}),31,s,Equal.sym(U32,s,31,eq)),rank))
    case 32n:
      +eq=Domain.injective(s,32,same)
      Old.false_true(Exists(File,f => {s == source(f,False{}) : U32}),Equal.trans(Bool,Ranks.start_rank(32,False{}),Ranks.start_rank(s,False{}),True{},Equal.cong(U32,Bool,x => Ranks.start_rank(x,False{}),32,s,Equal.sym(U32,s,32,eq)),rank))
    case 33n:
      +eq=Domain.injective(s,33,same)
      Old.false_true(Exists(File,f => {s == source(f,False{}) : U32}),Equal.trans(Bool,Ranks.start_rank(33,False{}),Ranks.start_rank(s,False{}),True{},Equal.cong(U32,Bool,x => Ranks.start_rank(x,False{}),33,s,Equal.sym(U32,s,33,eq)),rank))
    case 34n:
      +eq=Domain.injective(s,34,same)
      Old.false_true(Exists(File,f => {s == source(f,False{}) : U32}),Equal.trans(Bool,Ranks.start_rank(34,False{}),Ranks.start_rank(s,False{}),True{},Equal.cong(U32,Bool,x => Ranks.start_rank(x,False{}),34,s,Equal.sym(U32,s,34,eq)),rank))
    case 35n:
      +eq=Domain.injective(s,35,same)
      Old.false_true(Exists(File,f => {s == source(f,False{}) : U32}),Equal.trans(Bool,Ranks.start_rank(35,False{}),Ranks.start_rank(s,False{}),True{},Equal.cong(U32,Bool,x => Ranks.start_rank(x,False{}),35,s,Equal.sym(U32,s,35,eq)),rank))
    case 36n:
      +eq=Domain.injective(s,36,same)
      Old.false_true(Exists(File,f => {s == source(f,False{}) : U32}),Equal.trans(Bool,Ranks.start_rank(36,False{}),Ranks.start_rank(s,False{}),True{},Equal.cong(U32,Bool,x => Ranks.start_rank(x,False{}),36,s,Equal.sym(U32,s,36,eq)),rank))
    case 37n:
      +eq=Domain.injective(s,37,same)
      Old.false_true(Exists(File,f => {s == source(f,False{}) : U32}),Equal.trans(Bool,Ranks.start_rank(37,False{}),Ranks.start_rank(s,False{}),True{},Equal.cong(U32,Bool,x => Ranks.start_rank(x,False{}),37,s,Equal.sym(U32,s,37,eq)),rank))
    case 38n:
      +eq=Domain.injective(s,38,same)
      Old.false_true(Exists(File,f => {s == source(f,False{}) : U32}),Equal.trans(Bool,Ranks.start_rank(38,False{}),Ranks.start_rank(s,False{}),True{},Equal.cong(U32,Bool,x => Ranks.start_rank(x,False{}),38,s,Equal.sym(U32,s,38,eq)),rank))
    case 39n:
      +eq=Domain.injective(s,39,same)
      Old.false_true(Exists(File,f => {s == source(f,False{}) : U32}),Equal.trans(Bool,Ranks.start_rank(39,False{}),Ranks.start_rank(s,False{}),True{},Equal.cong(U32,Bool,x => Ranks.start_rank(x,False{}),39,s,Equal.sym(U32,s,39,eq)),rank))
    case 40n:
      +eq=Domain.injective(s,40,same)
      Old.false_true(Exists(File,f => {s == source(f,False{}) : U32}),Equal.trans(Bool,Ranks.start_rank(40,False{}),Ranks.start_rank(s,False{}),True{},Equal.cong(U32,Bool,x => Ranks.start_rank(x,False{}),40,s,Equal.sym(U32,s,40,eq)),rank))
    case 41n:
      +eq=Domain.injective(s,41,same)
      Old.false_true(Exists(File,f => {s == source(f,False{}) : U32}),Equal.trans(Bool,Ranks.start_rank(41,False{}),Ranks.start_rank(s,False{}),True{},Equal.cong(U32,Bool,x => Ranks.start_rank(x,False{}),41,s,Equal.sym(U32,s,41,eq)),rank))
    case 42n:
      +eq=Domain.injective(s,42,same)
      Old.false_true(Exists(File,f => {s == source(f,False{}) : U32}),Equal.trans(Bool,Ranks.start_rank(42,False{}),Ranks.start_rank(s,False{}),True{},Equal.cong(U32,Bool,x => Ranks.start_rank(x,False{}),42,s,Equal.sym(U32,s,42,eq)),rank))
    case 43n:
      +eq=Domain.injective(s,43,same)
      Old.false_true(Exists(File,f => {s == source(f,False{}) : U32}),Equal.trans(Bool,Ranks.start_rank(43,False{}),Ranks.start_rank(s,False{}),True{},Equal.cong(U32,Bool,x => Ranks.start_rank(x,False{}),43,s,Equal.sym(U32,s,43,eq)),rank))
    case 44n:
      +eq=Domain.injective(s,44,same)
      Old.false_true(Exists(File,f => {s == source(f,False{}) : U32}),Equal.trans(Bool,Ranks.start_rank(44,False{}),Ranks.start_rank(s,False{}),True{},Equal.cong(U32,Bool,x => Ranks.start_rank(x,False{}),44,s,Equal.sym(U32,s,44,eq)),rank))
    case 45n:
      +eq=Domain.injective(s,45,same)
      Old.false_true(Exists(File,f => {s == source(f,False{}) : U32}),Equal.trans(Bool,Ranks.start_rank(45,False{}),Ranks.start_rank(s,False{}),True{},Equal.cong(U32,Bool,x => Ranks.start_rank(x,False{}),45,s,Equal.sym(U32,s,45,eq)),rank))
    case 46n:
      +eq=Domain.injective(s,46,same)
      Old.false_true(Exists(File,f => {s == source(f,False{}) : U32}),Equal.trans(Bool,Ranks.start_rank(46,False{}),Ranks.start_rank(s,False{}),True{},Equal.cong(U32,Bool,x => Ranks.start_rank(x,False{}),46,s,Equal.sym(U32,s,46,eq)),rank))
    case 47n:
      +eq=Domain.injective(s,47,same)
      Old.false_true(Exists(File,f => {s == source(f,False{}) : U32}),Equal.trans(Bool,Ranks.start_rank(47,False{}),Ranks.start_rank(s,False{}),True{},Equal.cong(U32,Bool,x => Ranks.start_rank(x,False{}),47,s,Equal.sym(U32,s,47,eq)),rank))
    case 48n:
      +eq=Domain.injective(s,48,same)
      (A{},eq)
    case 49n:
      +eq=Domain.injective(s,49,same)
      (B{},eq)
    case 50n:
      +eq=Domain.injective(s,50,same)
      (C{},eq)
    case 51n:
      +eq=Domain.injective(s,51,same)
      (D{},eq)
    case 52n:
      +eq=Domain.injective(s,52,same)
      (E{},eq)
    case 53n:
      +eq=Domain.injective(s,53,same)
      (F{},eq)
    case 54n:
      +eq=Domain.injective(s,54,same)
      (G{},eq)
    case 55n:
      +eq=Domain.injective(s,55,same)
      (H{},eq)
    case 56n:
      +eq=Domain.injective(s,56,same)
      Old.false_true(Exists(File,f => {s == source(f,False{}) : U32}),Equal.trans(Bool,Ranks.start_rank(56,False{}),Ranks.start_rank(s,False{}),True{},Equal.cong(U32,Bool,x => Ranks.start_rank(x,False{}),56,s,Equal.sym(U32,s,56,eq)),rank))
    case 57n:
      +eq=Domain.injective(s,57,same)
      Old.false_true(Exists(File,f => {s == source(f,False{}) : U32}),Equal.trans(Bool,Ranks.start_rank(57,False{}),Ranks.start_rank(s,False{}),True{},Equal.cong(U32,Bool,x => Ranks.start_rank(x,False{}),57,s,Equal.sym(U32,s,57,eq)),rank))
    case 58n:
      +eq=Domain.injective(s,58,same)
      Old.false_true(Exists(File,f => {s == source(f,False{}) : U32}),Equal.trans(Bool,Ranks.start_rank(58,False{}),Ranks.start_rank(s,False{}),True{},Equal.cong(U32,Bool,x => Ranks.start_rank(x,False{}),58,s,Equal.sym(U32,s,58,eq)),rank))
    case 59n:
      +eq=Domain.injective(s,59,same)
      Old.false_true(Exists(File,f => {s == source(f,False{}) : U32}),Equal.trans(Bool,Ranks.start_rank(59,False{}),Ranks.start_rank(s,False{}),True{},Equal.cong(U32,Bool,x => Ranks.start_rank(x,False{}),59,s,Equal.sym(U32,s,59,eq)),rank))
    case 60n:
      +eq=Domain.injective(s,60,same)
      Old.false_true(Exists(File,f => {s == source(f,False{}) : U32}),Equal.trans(Bool,Ranks.start_rank(60,False{}),Ranks.start_rank(s,False{}),True{},Equal.cong(U32,Bool,x => Ranks.start_rank(x,False{}),60,s,Equal.sym(U32,s,60,eq)),rank))
    case 61n:
      +eq=Domain.injective(s,61,same)
      Old.false_true(Exists(File,f => {s == source(f,False{}) : U32}),Equal.trans(Bool,Ranks.start_rank(61,False{}),Ranks.start_rank(s,False{}),True{},Equal.cong(U32,Bool,x => Ranks.start_rank(x,False{}),61,s,Equal.sym(U32,s,61,eq)),rank))
    case 62n:
      +eq=Domain.injective(s,62,same)
      Old.false_true(Exists(File,f => {s == source(f,False{}) : U32}),Equal.trans(Bool,Ranks.start_rank(62,False{}),Ranks.start_rank(s,False{}),True{},Equal.cong(U32,Bool,x => Ranks.start_rank(x,False{}),62,s,Equal.sym(U32,s,62,eq)),rank))
    case 63n:
      +eq=Domain.injective(s,63,same)
      Old.false_true(Exists(File,f => {s == source(f,False{}) : U32}),Equal.trans(Bool,Ranks.start_rank(63,False{}),Ranks.start_rank(s,False{}),True{},Equal.cong(U32,Bool,x => Ranks.start_rank(x,False{}),63,s,Equal.sym(U32,s,63,eq)),rank))
    case 64n+p: B.impossible(Exists(File,f => {s == source(f,False{}) : U32}),p,below)

def certify(+s: U32,w: Bool,below: {Nat.is_lt(U32.to_nat(s),64n) == True{} : Bool},rank: {Ranks.start_rank(s,w) == True{} : Bool}) -> Exists(File,f => {s == source(f,w) : U32}):
  match w:
    case True{}: finite_white(s,U32.to_nat(s),{==},below,rank)
    case False{}: finite_black(s,U32.to_nat(s),{==},below,rank)
