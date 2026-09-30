# Batch token is NOT a search ticket. Bend retains row identities and cancellation.
import Base

type BatchSubmitted is Type:
  BatchSubmitted{input: Array<F32>, token: U32}
type BatchPolled is Type:
  BatchPolled{output: Array<F32>, status: U32}

def BatchAsync.enabled() -> IO(Bool):
  import "./async_call.c"
def BatchAsync.submit(rows: U32, input: Array<F32>) -> IO(BatchSubmitted):
  import "./async_call.c"
def BatchAsync.poll(token: U32, rows: U32, output: Array<F32>) -> IO(BatchPolled):
  import "./async_call.c"
# Wait for notification or the native one-millisecond budget; do not take output.
def BatchAsync.wait(token: U32) -> IO(Unit):
  import "./async_call.c"
def BatchAsync.shutdown() -> IO(Unit):
  import "./async_call.c"
