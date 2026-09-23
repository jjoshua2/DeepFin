# CPU-singleton opt-in. Submission copies inputs; only take publishes a result.
# A cancelled identity remains reserved until drain reports physical completion.
import Base

type Identity is Data:
  Identity{token: U32, epoch: U32, request: U32, node: U32}

type AsyncSubmitted is Type:
  AsyncSubmitted{input: Array<F32>, token: U32}

type AsyncPolled is Type:
  AsyncPolled{output: Array<F32>, status: U32}

def Async.enabled() -> IO(Bool):
  import "./async_call.c"

def Async.submit(epoch: U32, request: U32, node: U32, count: U32, input: Array<F32>) -> IO(AsyncSubmitted):
  import "./async_call.c"

def Async.poll(token: U32, epoch: U32, request: U32, node: U32, output: Array<F32>) -> IO(AsyncPolled):
  import "./async_call.c"

def Async.cancel(token: U32, epoch: U32, request: U32, node: U32) -> IO(Unit):
  import "./async_call.c"

def Async.drain(token: U32, epoch: U32, request: U32, node: U32) -> IO(U32):
  import "./async_call.c"

def Async.shutdown() -> IO(Unit):
  import "./async_call.c"

def cancel_identity(id: Identity) -> IO(Unit):
  Identity{token, epoch, request, node} = id
  Async.cancel(token, epoch, request, node)

def drain_identity(id: Identity) -> IO(U32):
  Identity{token, epoch, request, node} = id
  Async.drain(token, epoch, request, node)

def show(id: Identity) -> String:
  Identity{token, epoch, request, node} = id
  "token=" ++ U32.show(token) ++ " epoch=" ++ U32.show(epoch) ++ " request=" ++ U32.show(request) ++ " node=" ++ U32.show(node)
