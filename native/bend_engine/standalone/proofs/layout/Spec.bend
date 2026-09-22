# Read-only scalar observations of the exact expressions in Tables.tables.
# Not a replacement builder: no Array or affine refinement claim.
import Base
import ../../Tables.bend as Tables

def mask(+key: U32) -> U64:
  Tables.slider(U32.mod(key,64),U32.is_ge(key,64),U64.zero(),True{})

def size(key: U32) -> U32:
  U32.shln(1,U32.to_nat(U64.popcount(mask(key))))

def max_population(key: U32) -> U32:
  Bool.pick(U32,U32.is_ge(key,64),9,12)

def max_size(key: U32) -> U32:
  Bool.pick(U32,U32.is_ge(key,64),512,4096)
