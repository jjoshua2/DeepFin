import Base
import ../prefix/Spec.bend as Prefix
import ../layout/Spec.bend as Layout

# The actual U32 end expression. This is not a table of expected addresses.
def finish(+k: Nat) -> U32:
  U32.add(Prefix.prefix(k),Layout.size(U32.from_nat(k)))

def normalize(x: U32) -> U32:
  U32.and(x,U32.sub(131072,1))

def member(+k: Nat, +x: U32) -> Type:
  {U32.is_le(Prefix.prefix(k),x) == True{} : Bool}
  & {U32.is_lt(x,finish(k)) == True{} : Bool}

def interior(+x: U32) -> Type:
  {U32.is_le(512,x) == True{} : Bool}
  & {U32.is_lt(x,131072) == True{} : Bool}
  & {normalize(x) == x : U32}

# Destruct certificates in parameter helpers, not computed-value matches.
def bound(+x: U32, e: interior(x)) -> {U32.is_lt(x,131072) == True{} : Bool}:
  (lo,(hi,norm)) = e
  hi

def normalized(+x: U32, e: interior(x)) -> {normalize(x) == x : U32}:
  (lo,(hi,norm)) = e
  norm
