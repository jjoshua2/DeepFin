# Only numeric regions; no expected read value or external oracle.
import Base

def outside(+q: U32) -> Bool:
  Bool.or(U32.is_lt(q,256),U32.is_le(512,q))
