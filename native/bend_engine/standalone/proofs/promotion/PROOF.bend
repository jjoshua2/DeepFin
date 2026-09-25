import Base
import ./LAWS.bend as Laws
import ./Encoding.bend as Encoding
import ./Actual.bend as Actual

def Laws.promotion_choice_roundtrip(choice):
  Encoding.roundtrip(choice)

def Laws.promotion_choices_exact(src,dst,flag,tail):
  Encoding.generated(src,dst,flag,tail)

def Laws.promotion_make_move_exact_update(b,src,dst,choice):
  Actual.exact(b,src,dst,choice)

def Laws.promotion_make_move_preserves_representation(b,src,dst,choice,good):
  Actual.preserves(b,src,dst,choice,good)
