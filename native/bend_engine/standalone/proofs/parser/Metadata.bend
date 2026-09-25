# Metadata framing for the actual placement parser.
import Base
import ./Spec.bend as S
import ./Dispatch.bend as Dispatch
import ../../Position.bend as Position
import ../../../legal_probe/Chess.bend as Chess
import ../board/Actual.bend as B
import ../board/Spec.bend as BS
import ../u64/Words.bend as Old

def digit_metadata(+d: Bool,+c: U32,+l: Position.Layout) ->
  {S.metadata(Position.digit(d,c,l)) == S.metadata(l) : U32 & (U32 & U32)}:
  match d l:
    case True{} Position.Layout{b,file,rank,valid}: {==}
    case False{} Position.Layout{+b,file,rank,valid}:
      B.put_metadata(b,Position.kind(U32.or(c,32)),U32.is_lt(c,97),U32.add(file,U32.mul(rank,8)))

def slash_metadata(+l: Position.Layout) ->
  {S.metadata(Dispatch.slash(l)) == S.metadata(l) : U32 & (U32 & U32)}:
  match l:
    case Position.Layout{b,file,rank,valid}: {==}

def char_metadata(+c: U32,+l: Position.Layout) ->
  {S.metadata(Position.layout_char(c,l)) == S.metadata(l) : U32 & (U32 & U32)}:
  Dispatch.cases(c,l,x => {S.metadata(x) == S.metadata(l) : U32 & (U32 & U32)},
    slash_metadata(l),digit_metadata(Bool.and(U32.is_ge(c,49),U32.is_le(c,56)),c,l))

def metadata(+s: String,+l: Position.Layout) ->
  {S.metadata(Position.layout(s,l)) == S.metadata(l) : U32 & (U32 & U32)}:
  match s:
    case SNil{}: {==}
    case SCon{Chr{+c},+rest}:
      Equal.trans(U32 & (U32 & U32),S.metadata(Position.layout(rest,Position.layout_char(c,l))),
        S.metadata(Position.layout_char(c,l)),S.metadata(l),metadata(rest,Position.layout_char(c,l)),char_metadata(c,l))
