# Complete Word constructor elimination avoids impossible scalar-pattern tails.
import Base
import ../../Position.bend as Position

def slash(l: Position.Layout) -> Position.Layout:
  Position.Layout{b,file,+rank,valid} = l
  Position.Layout{b,0,U32.sub(rank,1),Bool.and(valid,Bool.and(U32.is_eq(file,8),U32.is_gt(rank,0)))}

def other(+c: U32,l: Position.Layout) -> Position.Layout:
  Position.digit(Bool.and(U32.is_ge(c,49),U32.is_le(c,56)),c,l)

def bits(+b0: Bool,+b1: Bool,+b2: Bool,+b3: Bool,+b4: Bool,+b5: Bool,+b6: Bool,+b7: Bool,+b8: Bool,+b9: Bool,+b10: Bool,+b11: Bool,+b12: Bool,+b13: Bool,+b14: Bool,+b15: Bool,+b16: Bool,+b17: Bool,+b18: Bool,+b19: Bool,+b20: Bool,+b21: Bool,+b22: Bool,+b23: Bool,+b24: Bool,+b25: Bool,+b26: Bool,+b27: Bool,+b28: Bool,+b29: Bool,+b30: Bool,+b31: Bool,+l: Position.Layout,-P: Position.Layout -> Type,
  left: P(slash(l)),right: P(other(U32{WCon{b0,WCon{b1,WCon{b2,WCon{b3,WCon{b4,WCon{b5,WCon{b6,WCon{b7,WCon{b8,WCon{b9,WCon{b10,WCon{b11,WCon{b12,WCon{b13,WCon{b14,WCon{b15,WCon{b16,WCon{b17,WCon{b18,WCon{b19,WCon{b20,WCon{b21,WCon{b22,WCon{b23,WCon{b24,WCon{b25,WCon{b26,WCon{b27,WCon{b28,WCon{b29,WCon{b30,WCon{b31,WNil{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}},l))) -> P(Position.layout_char(U32{WCon{b0,WCon{b1,WCon{b2,WCon{b3,WCon{b4,WCon{b5,WCon{b6,WCon{b7,WCon{b8,WCon{b9,WCon{b10,WCon{b11,WCon{b12,WCon{b13,WCon{b14,WCon{b15,WCon{b16,WCon{b17,WCon{b18,WCon{b19,WCon{b20,WCon{b21,WCon{b22,WCon{b23,WCon{b24,WCon{b25,WCon{b26,WCon{b27,WCon{b28,WCon{b29,WCon{b30,WCon{b31,WNil{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}},l)):
  match b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 b20 b21 b22 b23 b24 b25 b26 b27 b28 b29 b30 b31:
    case True{} True{} True{} True{} False{} True{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{}:
      match l:
        case Position.Layout{b,file,rank,valid}: left
    case False{} _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _: right
    case True{} False{} _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _: right
    case True{} True{} False{} _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _: right
    case True{} True{} True{} False{} _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _: right
    case True{} True{} True{} True{} True{} _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _: right
    case True{} True{} True{} True{} False{} False{} _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _: right
    case True{} True{} True{} True{} False{} True{} True{} _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _: right
    case True{} True{} True{} True{} False{} True{} False{} True{} _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _: right
    case True{} True{} True{} True{} False{} True{} False{} False{} True{} _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _: right
    case True{} True{} True{} True{} False{} True{} False{} False{} False{} True{} _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _: right
    case True{} True{} True{} True{} False{} True{} False{} False{} False{} False{} True{} _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _: right
    case True{} True{} True{} True{} False{} True{} False{} False{} False{} False{} False{} True{} _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _: right
    case True{} True{} True{} True{} False{} True{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _: right
    case True{} True{} True{} True{} False{} True{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _: right
    case True{} True{} True{} True{} False{} True{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _: right
    case True{} True{} True{} True{} False{} True{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _: right
    case True{} True{} True{} True{} False{} True{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _ _ _ _ _ _ _ _ _ _ _ _: right
    case True{} True{} True{} True{} False{} True{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _ _ _ _ _ _ _ _ _ _ _: right
    case True{} True{} True{} True{} False{} True{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _ _ _ _ _ _ _ _ _ _: right
    case True{} True{} True{} True{} False{} True{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _ _ _ _ _ _ _ _ _: right
    case True{} True{} True{} True{} False{} True{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _ _ _ _ _ _ _ _: right
    case True{} True{} True{} True{} False{} True{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _ _ _ _ _ _ _: right
    case True{} True{} True{} True{} False{} True{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _ _ _ _ _ _: right
    case True{} True{} True{} True{} False{} True{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _ _ _ _ _: right
    case True{} True{} True{} True{} False{} True{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _ _ _ _: right
    case True{} True{} True{} True{} False{} True{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _ _ _: right
    case True{} True{} True{} True{} False{} True{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _ _: right
    case True{} True{} True{} True{} False{} True{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _ _: right
    case True{} True{} True{} True{} False{} True{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _ _: right
    case True{} True{} True{} True{} False{} True{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _ _: right
    case True{} True{} True{} True{} False{} True{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{} _: right
    case True{} True{} True{} True{} False{} True{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} False{} True{}: right


def cases(+c: U32,+l: Position.Layout,-P: Position.Layout -> Type,
  left: P(slash(l)),right: P(other(c,l))) -> P(Position.layout_char(c,l)):
  match c:
    case U32{WCon{+b0,WCon{+b1,WCon{+b2,WCon{+b3,WCon{+b4,WCon{+b5,WCon{+b6,WCon{+b7,WCon{+b8,WCon{+b9,WCon{+b10,WCon{+b11,WCon{+b12,WCon{+b13,WCon{+b14,WCon{+b15,WCon{+b16,WCon{+b17,WCon{+b18,WCon{+b19,WCon{+b20,WCon{+b21,WCon{+b22,WCon{+b23,WCon{+b24,WCon{+b25,WCon{+b26,WCon{+b27,WCon{+b28,WCon{+b29,WCon{+b30,WCon{+b31,WNil{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}:
      bits(b0,b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,b12,b13,b14,b15,b16,b17,b18,b19,b20,b21,b22,b23,b24,b25,b26,b27,b28,b29,b30,b31,l,P,left,right)
