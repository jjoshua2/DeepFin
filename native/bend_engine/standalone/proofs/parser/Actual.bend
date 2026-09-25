# Structural properties of the imported real parser; no surrogate transition loop.
import Base
import ./Spec.bend as S
import ./Dispatch.bend as Dispatch
import ../../Position.bend as Position
import ../../../legal_probe/Chess.bend as Chess
import ../board/Actual.bend as B
import ../board/Spec.bend as BS
import ../u64/Words.bend as Old

def digit_invalid(+d: Bool,+c: U32,+l: Position.Layout,
  e: {S.live(l) == False{} : Bool}) -> {S.live(Position.digit(d,c,l)) == False{} : Bool}:
  match d l:
    case True{} Position.Layout{b,file,rank,+valid}:
      Equal.cong(Bool,Bool,v => Bool.and(v,U32.is_le(U32.add(file,U32.sub(c,48)),8)),valid,False{},e)
    case False{} Position.Layout{b,+file,rank,+valid}:
      Equal.cong(Bool,Bool,v => Bool.and(v,Bool.and(U32.is_lt(Position.kind(U32.or(c,32)),6),U32.is_lt(file,8))),valid,False{},e)

def slash_invalid(+l: Position.Layout,e: {S.live(l) == False{} : Bool}) ->
  {S.live(Dispatch.slash(l)) == False{} : Bool}:
  match l:
    case Position.Layout{b,file,rank,+valid}:
      Equal.cong(Bool,Bool,v => Bool.and(v,Bool.and(U32.is_eq(file,8),U32.is_gt(rank,0))),valid,False{},e)

def char_invalid(+c: U32,+l: Position.Layout,+e: {S.live(l) == False{} : Bool}) ->
  {S.live(Position.layout_char(c,l)) == False{} : Bool}:
  Dispatch.cases(c,l,x => {S.live(x) == False{} : Bool},
    slash_invalid(l,e),digit_invalid(Bool.and(U32.is_ge(c,49),U32.is_le(c,56)),c,l,e))

def invalid(+s: String,+l: Position.Layout,e: {S.live(l) == False{} : Bool}) ->
  {S.live(Position.layout(s,l)) == False{} : Bool}:
  match s:
    case SNil{}: e
    case SCon{Chr{+c},rest}: invalid(rest,Position.layout_char(c,l),char_invalid(c,l,e))

def append(+a: String,+b: String,+l: Position.Layout) ->
  {Position.layout(String.append(a,b),l) == Position.layout(b,Position.layout(a,l)) : Position.Layout}:
  match a:
    case SNil{}: {==}
    case SCon{Chr{c},rest}: append(rest,b,Position.layout_char(c,l))

def rejected(+l: Position.Layout,e: {S.live(l) == False{} : Bool}) ->
  {Position.layout_result(l) == None{} : Maybe<&2,Chess.Board>}:
  match l:
    case Position.Layout{b,file,rank,+valid}:
      Equal.cong(Bool,Maybe<&2,Chess.Board>,v => Position.board_when(Bool.and(v,Bool.and(U32.is_eq(file,8),U32.is_zero(rank))),b),valid,False{},e)

def start_live(+s: String,+l: Position.Layout,
  e: {S.accepted(Position.layout_result(Position.layout(s,l))) == True{} : Bool}) -> {S.live(l) == True{} : Bool}:
  match l:
    case Position.Layout{b,file,rank,True{}}: {==}
    case Position.Layout{+b,+file,+rank,False{}}:
      Old.false_true({False{} == True{} : Bool},
        Equal.trans(Bool,False{},S.accepted(Position.layout_result(Position.layout(s,Position.Layout{b,file,rank,False{}}))),True{},
          Equal.sym(Bool,S.accepted(Position.layout_result(Position.layout(s,Position.Layout{b,file,rank,False{}}))),False{},
            Equal.cong(Maybe<&2,Chess.Board>,Bool,r => S.accepted(r),
              Position.layout_result(Position.layout(s,Position.Layout{b,file,rank,False{}})),None{},
              rejected(Position.layout(s,Position.Layout{b,file,rank,False{}}),invalid(s,Position.Layout{b,file,rank,False{}},{==})))),e))

def accepted_prefix(+prefix: String,+suffix: String,+l: Position.Layout,
  e: {S.accepted(Position.layout_result(Position.layout(String.append(prefix,suffix),l))) == True{} : Bool}) ->
  {S.live(Position.layout(prefix,l)) == True{} : Bool}:
  start_live(suffix,Position.layout(prefix,l),
    Equal.trans(Bool,S.accepted(Position.layout_result(Position.layout(suffix,Position.layout(prefix,l)))),
      S.accepted(Position.layout_result(Position.layout(String.append(prefix,suffix),l))),True{},
      Equal.cong(Position.Layout,Bool,x => S.accepted(Position.layout_result(x)),
        Position.layout(suffix,Position.layout(prefix,l)),Position.layout(String.append(prefix,suffix),l),
        Equal.sym(Position.Layout,Position.layout(String.append(prefix,suffix),l),Position.layout(suffix,Position.layout(prefix,l)),append(prefix,suffix,l))),e))
