import Base

# PR 2 architecture probe:
#
# Bend owns the control flow and consumes only compact scalar results. The
# native C++ side owns tensor construction and AOTInductor/LibTorch execution.
# This keeps large tensor buffers out of Bend's term representation, matching
# the intended engine architecture.

def NN.eval(seed: U32) -> IO(List<U32>):
  import "./aoti_call.c"

def checksum_words(xs: List<U32>, acc: U32) -> U32:
  match xs:
    case Nil{}:
      acc
    case Con{word, rest}:
      checksum_words(
        rest,
        U32.xor(U32.mul(acc, 2654435761), word)
      )

def list_len(xs: List<U32>, acc: U32) -> U32:
  match xs:
    case Nil{}:
      acc
    case Con{_, rest}:
      list_len(rest, U32.add(acc, 1))

def report(+seed: U32, words: List<U32>) -> IO(Unit):
  # The foreign value is affine, so consume it once after duplicating through
  # an explicit reusable binder.
  +shared = words
  IO.print(
    "seed=" ++ U32.show(seed)
    ++ " count=" ++ U32.show(list_len(shared, 0))
    ++ " checksum=" ++ U32.show(checksum_words(shared, 0))
  )

def probe(seed: U32) -> IO(Unit):
  do IO<Unit>:
    words : List<U32> <- NN.eval(seed)
    report(seed, words)

def main() -> IO(Unit):
  do IO<Unit>:
    a : Unit <- probe(0)
    b : Unit <- probe(1)
    c : Unit <- probe(7)
    return Unit{}
